"""Latent Meanflow Model (LMM): same Stage-0/1 + latent residual path as LDM, MeanFlow on ``z_R``."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
from lightning import LightningModule

from .components.ldm.denoiser import LitEma
from .components.meanflow.meanflow_core import MeanFlowCore
from .components.meanflow.meanflow_paper_core import MeanFlowPaperCore
from .components.ldm.denoiser.lmm_infer import generate_latent_one_step
from .latent_residual_inputs import build_latent_target_and_context_dict
from .temperature_field_losses import TemperatureFieldLosses


class LatentMeanFlowLitModule(LightningModule):
    """MeanFlow training on latent ``z_R`` with frozen residual VAE (+ optional context encoder)."""

    def __init__(
        self,
        mf_unet,
        meanflow_core,
        autoencoder,
        context_encoder=None,
        ae_load_state_file: Optional[str] = None,
        trainable_parts: Optional[List[str]] = None,
        lr: float = 1e-4,
        lr_warmup: int = 0,
        use_ema: bool = True,
        loss_type: str = "l2",
        pde_lambda: float = 0.0,
        pde_mode: Optional[str] = None,
        temp_pde_coef: float = 0.0,
        temp_energy_coef: float = 0.0,
        temp_pde_num_supercells: int = 8,
        use_meanflow_paper_core: bool = False,
        meanflow_paper: Optional[Dict[str, Any]] = None,
        control_metric_weights: Optional[Dict[str, float]] = None,
        lr_scheduler_metric_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["mf_unet", "meanflow_core", "autoencoder", "context_encoder"])

        self.loss_type = loss_type
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.pde_lambda = pde_lambda
        self.pde_mode = pde_mode
        self.temp_pde_coef = float(temp_pde_coef)
        self.temp_energy_coef = float(temp_energy_coef)
        self.temp_pde_num_supercells = int(temp_pde_num_supercells)
        self.use_meanflow_paper_core = bool(use_meanflow_paper_core)
        self.control_metric_weights = {
            "loss": 0.0,
            "legacy_adaptive_l2": 0.0,
            "rmse": 1.0,
            "r2": 0.0,
        }
        if control_metric_weights is not None:
            self.control_metric_weights.update({k: float(v) for k, v in control_metric_weights.items()})
        # Backward compatibility with older configs.
        if lr_scheduler_metric_weights is not None:
            self.control_metric_weights.update({k: float(v) for k, v in lr_scheduler_metric_weights.items()})

        self.meanflow_core = (
            MeanFlowPaperCore(**(meanflow_paper or {}))
            if self.use_meanflow_paper_core
            else meanflow_core
        )
        self._legacy_meanflow_core = MeanFlowCore()
        self.mf_unet = mf_unet
        self.autoencoder = autoencoder.requires_grad_(False)
        if ae_load_state_file is not None:
            ckpt = torch.load(ae_load_state_file, map_location="cpu", weights_only=False)
            self.autoencoder.load_state_dict(ckpt["state_dict"], strict=False)

        self.conditional = context_encoder is not None
        self.context_encoder = context_encoder

        self.use_ema = use_ema
        if self.use_ema:
            self.mf_unet_ema = LitEma(self.mf_unet)

        self._field_losses = TemperatureFieldLosses()

        if trainable_parts is not None and len(trainable_parts) > 0:
            self.set_trainable_layers(trainable_parts)
            if self.use_ema:
                self.mf_unet_ema = LitEma(self.mf_unet)

        n_mf = sum(p.numel() for p in self.mf_unet.parameters() if p.requires_grad)
        n_ae = sum(p.numel() for p in self.autoencoder.parameters() if p.requires_grad)
        print(f"[LatentMeanFlowLitModule] trainable mf_unet params: {n_mf:,}; trainable autoencoder params: {n_ae:,}")

    def set_trainable_layers(self, trainable_parts: List[str]):
        for _, param in self.named_parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if any(tp in name for tp in trainable_parts):
                param.requires_grad = True
                print(f"Unfreezing parameter: {name}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.mf_unet_ema.store(self.mf_unet.parameters())
            self.mf_unet_ema.copy_to(self.mf_unet)
            if context is not None:
                print(f"{context}: Switched mf_unet to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.mf_unet_ema.restore(self.mf_unet.parameters())
                if context is not None:
                    print(f"{context}: Restored mf_unet training weights")

    def _build_context_dict(self, x, z) -> Dict[str, Any]:
        context_dict: Dict[str, Any] = {"T_c": x}
        if self.conditional:
            encoder_context = self.context_encoder([(z, [0]), (x, [0])])
            if isinstance(encoder_context, dict):
                context_dict.update(encoder_context)
            else:
                context_dict["encoder_context"] = encoder_context
        return context_dict

    def build_latent_and_context(self, batch):
        """Match ``LatentDiffusion.shared_step`` data construction (residual VAE path)."""
        if self.autoencoder.ae_flag != "residual":
            raise ValueError("LMM v1 is defined for residual ``ae_flag`` only.")
        return build_latent_target_and_context_dict(
            self.autoencoder, self.context_encoder, self.conditional, batch
        )

    def _physics_addon(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        context: Dict[str, Any],
    ) -> torch.Tensor:
        """PDE / energy on a **decoded temperature** field.

        **Design lock (L-F):** decode the **one-step** latent ``single_step_generate(x_t, t, r, u_theta)``
        with ``u_theta = mf_unet(x_t, t, r, context)`` — same attachment spirit as using an explicit
        denoiser output in LDM, without decoding raw velocity.
        """
        addon = torch.zeros((), device=x_t.device, dtype=x_t.dtype)
        if self.pde_lambda > 0 and self.pde_mode == "uv":
            raise NotImplementedError("LMM v1 does not implement UV PDE on latent MFUNet outputs.")

        if self.pde_mode == "temp" and (self.temp_pde_coef > 0 or self.temp_energy_coef > 0):
            u_theta = self.mf_unet(x_t, t, r, context=context)
            z_hat = self.meanflow_core.single_step_generate(x_t, t, r, u_theta)
            T_f = self.autoencoder.decode(z_hat)
            if not isinstance(context, dict) or "T_c" not in context:
                raise ValueError("For temperature PDE loss, context must be a dict containing key 'T_c'")
            T_c = context["T_c"]
            if self.temp_pde_coef > 0:
                addon = addon + self.temp_pde_coef * self._field_losses.temperature_pde_loss(
                    T_f, T_c, self.temp_pde_num_supercells
                )
            if self.temp_energy_coef > 0:
                addon = addon + self.temp_energy_coef * self._field_losses.temperature_energy_loss(T_f, T_c)
        return addon

    @staticmethod
    def _compute_rmse_r2(u_pred: torch.Tensor, u_tgt: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            pred = u_pred.detach().float()
            tgt = u_tgt.detach().float()
            err = pred - tgt
            rmse = torch.sqrt(torch.mean(err * err))
            # Match current xs.r2(pred, tgt) semantics exactly:
            # denominator uses variance of the first argument (pred).
            ss_res = torch.sum(err * err)
            ss_tot = torch.sum((pred - torch.mean(pred)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot)
            rmse = rmse.to(device=u_pred.device, dtype=u_pred.dtype)
            r2 = r2.to(device=u_pred.device, dtype=u_pred.dtype)
        return {"rmse": rmse, "r2": r2}

    def _meanflow_train_loss(self, z0: torch.Tensor, context: Dict[str, Any], create_graph: bool):
        train_targets = self.meanflow_core.compute_train_targets(z0)
        x_t = train_targets["x_t"]
        t = train_targets["t"]
        r = train_targets["r"]
        v_target = train_targets["v_target"]

        def backbone(x_state, time_r, time_t):
            return self.mf_unet(x_state, time_t, time_r, context=context)

        error = self.meanflow_core.compute_teacher_error(
            backbone_model=backbone,
            x_t=x_t,
            t=t,
            r=r,
            v_target=v_target,
            create_graph=create_graph,
            return_details=True,
        )
        error, u_pred, u_tgt = error
        mf_loss = self.meanflow_core.adaptive_l2_loss(error)
        phys = self._physics_addon(x_t, t, r, context)
        total_loss = mf_loss + phys

        with torch.no_grad():
            metrics = self._compute_rmse_r2(u_pred, u_tgt)
            if self.use_meanflow_paper_core:
                metrics["legacy_adaptive_l2"] = self._legacy_meanflow_core.adaptive_l2_loss(error.detach())

        return total_loss, metrics

    def shared_step(self, batch, create_graph: bool):
        latent_target, context_dict = self.build_latent_and_context(batch)
        return self._meanflow_train_loss(latent_target, context_dict, create_graph=create_graph)

    def training_step(self, batch, batch_idx):
        loss, metrics = self.shared_step(batch, create_graph=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/rmse", metrics["rmse"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/r2", metrics["r2"], on_step=False, on_epoch=True, sync_dist=True)
        if "legacy_adaptive_l2" in metrics:
            self.log(
                "train/legacy_adaptive_l2",
                metrics["legacy_adaptive_l2"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.shared_step(batch, create_graph=False)
        with self.ema_scope():
            loss_ema, metrics_ema = self.shared_step(batch, create_graph=False)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log("val/loss", loss, **log_params, sync_dist=True)
        self.log("val/loss_ema", loss_ema, **log_params, sync_dist=True)
        self.log("val/rmse", metrics["rmse"], **log_params, sync_dist=True)
        self.log("val/r2", metrics["r2"], **log_params, sync_dist=True)
        # Composite control monitor used by scheduler/early-stop/checkpoint selection.
        # Lower is better; better r2 reduces score via the negative sign.
        legacy_l2 = metrics.get("legacy_adaptive_l2")
        if legacy_l2 is None:
            # Fallback for setups where legacy metric is not emitted.
            legacy_l2 = torch.zeros_like(loss)
        control_score = (
            self.control_metric_weights["loss"] * loss
            + self.control_metric_weights["legacy_adaptive_l2"] * legacy_l2
            + self.control_metric_weights["rmse"] * metrics["rmse"]
            - self.control_metric_weights["r2"] * metrics["r2"]
        )
        self.log("val/control_score", control_score, **log_params, sync_dist=True)
        if "legacy_adaptive_l2" in metrics:
            self.log("val/legacy_adaptive_l2", metrics["legacy_adaptive_l2"], **log_params, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, metrics = self.shared_step(batch, create_graph=False)
        with self.ema_scope():
            loss_ema, metrics_ema = self.shared_step(batch, create_graph=False)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log("test/loss", loss, **log_params, sync_dist=True)
        self.log("test/loss_ema", loss_ema, **log_params, sync_dist=True)
        self.log("test/rmse", metrics["rmse"], **log_params, sync_dist=True)
        self.log("test/r2", metrics["r2"], **log_params, sync_dist=True)
        if "legacy_adaptive_l2" in metrics:
            self.log("test/legacy_adaptive_l2", metrics["legacy_adaptive_l2"], **log_params, sync_dist=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.mf_unet_ema(self.mf_unet)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-3)
        monitor = "val/control_score"
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": reduce_lr, "monitor": monitor, "frequency": 1},
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        if self.lr_warmup > 0 and self.trainer.global_step < self.lr_warmup:
            lr_scale = (self.trainer.global_step + 1) / self.lr_warmup
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)

    @torch.no_grad()
    def predict_latent_residual(self, low_res, static, y_hr):
        """Encode ``R = y - ŷ_up``, run latent one-step MeanFlow, decode to pixel residual."""
        residual, _ = self.autoencoder.preprocess_batch([low_res, y_hr, static])
        z_enc = self.autoencoder.encode(residual)[0]
        context = self._build_context_dict(low_res, static)
        z_hat = generate_latent_one_step(
            mf_unet=self.mf_unet,
            meanflow_core=self.meanflow_core,
            context=context,
            shape=z_enc.shape,
            device=low_res.device,
            dtype=low_res.dtype,
        )
        return self.autoencoder.decode(z_hat)

    @torch.no_grad()
    def predict_final(self, low_res, static, y_hr):
        """``R̂`` then fusion ``ŷ = ŷ_up + R̂`` for ``ae_flag == residual`` (same as LDM inference)."""
        r_hat = self.predict_latent_residual(low_res, static, y_hr)
        merged = self.autoencoder.nn_lr_and_merge_with_static(low_res, static)
        y_up = self.autoencoder.unet(merged)
        return y_up + r_hat
