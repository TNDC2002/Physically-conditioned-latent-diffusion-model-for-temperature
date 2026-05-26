"""Latent Meanflow Model (LMM): same Stage-0/1 + latent residual path as LDM, MeanFlow on ``z_R``."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
from lightning import LightningModule

from .components.ldm.denoiser import LitEma
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
        temp_pde_num_supercells: int = 8,
        anisotropic_transport_coef: float = 0.0,
        at_lambda_mag: float = 1.0,
        at_lambda_dir: float = 1.0,
        at_loss_eps: float = 1e-12,
        at_direction_loss: str = "cosine",
        at_dx: float = 2000.0,
        at_dy: float = -2000.0,
        at_qmag_quantile: Optional[float] = None,
        at_qmag_min: Optional[float] = None,
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
        self.temp_pde_num_supercells = int(temp_pde_num_supercells)
        self.anisotropic_transport_coef = float(anisotropic_transport_coef)
        self.at_lambda_mag = float(at_lambda_mag)
        self.at_lambda_dir = float(at_lambda_dir)
        self.at_loss_eps = float(at_loss_eps)
        self.at_direction_loss = str(at_direction_loss)
        self.at_dx = float(at_dx)
        self.at_dy = float(at_dy)
        self.at_qmag_quantile = (
            None if at_qmag_quantile is None else float(at_qmag_quantile)
        )
        self.at_qmag_min = None if at_qmag_min is None else float(at_qmag_min)
        self.use_meanflow_paper_core = bool(use_meanflow_paper_core)
        # All 0.0 by default. Non-zero control_metric_weights only in configs/experiment/*.yaml.
        self.control_metric_weights = {
            "loss": 0.0,
            "legacy_adaptive_l2": 0.0,
            "mf_loss_f64": 0.0,
            "mf_minus_1": 0.0,
            "loss_total_f64": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "temp_pde_pure": 0.0,
            "at_mag_pure": 0.0,
            "at_dir_pure": 0.0,
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
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Physics on **decoded residual** ``R̂`` (one-step latent decode).

        **Design lock (L-F):** decode ``single_step_generate(x_t, t, r, mf_unet(...))``.

        - Diffusive–advective PDE (optional): ``R̂`` vs normalized ERA5 ``T_c``.
        - Anisotropic Transport (optional): ``R̂`` vs normalized COSMO-CLM ``T_hr``.

        Returns ``(scaled_addon, pure_metrics)`` with unscaled terms for logging / control score.
        """
        addon = torch.zeros((), device=x_t.device, dtype=x_t.dtype)
        pure_metrics: Dict[str, torch.Tensor] = {}
        if self.pde_lambda > 0 and self.pde_mode == "uv":
            raise NotImplementedError("LMM v1 does not implement UV PDE on latent MFUNet outputs.")

        need_decode = (self.pde_mode == "temp" and self.temp_pde_coef > 0) or (
            self.anisotropic_transport_coef > 0
        )
        if need_decode:
            if not isinstance(context, dict):
                raise ValueError("Physics losses require context dict with T_c and/or T_hr")
            u_theta = self.mf_unet(x_t, t, r, context=context)
            z_hat = self.meanflow_core.single_step_generate(x_t, t, r, u_theta)
            T_f = self.autoencoder.decode(z_hat)

            if self.pde_mode == "temp" and self.temp_pde_coef > 0:
                if "T_c" not in context:
                    raise ValueError("temp_pde_coef > 0 requires context['T_c'] (normalized ERA5)")
                pde_raw = self._field_losses.temperature_pde_loss(
                    T_f, context["T_c"], self.temp_pde_num_supercells
                )
                addon = addon + self.temp_pde_coef * pde_raw
                pure_metrics["temp_pde_pure"] = pde_raw.detach()

            if self.anisotropic_transport_coef > 0:
                if "T_hr" not in context:
                    raise ValueError(
                        "anisotropic_transport_coef > 0 requires context['T_hr'] "
                        "(normalized COSMO-CLM high-res target)"
                    )
                at = self._field_losses.anisotropic_transport_loss(
                    T_f,
                    context["T_hr"],
                    dx=self.at_dx,
                    dy=self.at_dy,
                    eps=self.at_loss_eps,
                    lambda_mag=self.at_lambda_mag,
                    lambda_dir=self.at_lambda_dir,
                    direction_kind=self.at_direction_loss,
                    qmag_quantile=self.at_qmag_quantile,
                    qmag_min=self.at_qmag_min,
                )
                addon = addon + self.anisotropic_transport_coef * at["L_total"]
                pure_metrics["at_mag_pure"] = at["L_mag"].detach()
                pure_metrics["at_dir_pure"] = at["L_dir"].detach()
                pure_metrics["at_dir_pure_cosine"] = at["L_dir_cosine"].detach()
                pure_metrics["at_dir_pure_unit_mse"] = at["L_dir_unit_mse"].detach()
                if "at_mask_frac" in at:
                    pure_metrics["at_mask_frac"] = at["at_mask_frac"].detach()
        return addon, pure_metrics

    @property
    def _uses_legacy_meanflow_loss(self) -> bool:
        """Old ``MeanFlowCore`` is the training loss (not the paper core)."""
        return not self.use_meanflow_paper_core

    @staticmethod
    def _mf_adaptive_l2_f64(core, error: torch.Tensor) -> torch.Tensor:
        """MeanFlow adaptive loss in float64 (monitoring only; training stays float32)."""
        return core.adaptive_l2_loss(error.detach().double())

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
        phys, pure_phys = self._physics_addon(x_t, t, r, context)
        total_loss = mf_loss + phys

        with torch.no_grad():
            metrics = self._compute_rmse_r2(u_pred, u_tgt)
            mf_loss_f64 = self._mf_adaptive_l2_f64(self.meanflow_core, error)
            mf_minus_1 = mf_loss_f64 - 1.0
            phys_det = phys.detach()
            metrics["mf_loss"] = mf_loss.detach()
            metrics["mf_loss_f64"] = mf_loss_f64
            metrics["mf_minus_1"] = mf_minus_1
            metrics["mf_minus_1_x1e8"] = mf_minus_1 * 1.0e8
            metrics["phys_loss"] = phys_det
            metrics["loss_total_f64"] = mf_loss_f64 + phys_det.double()
            if self._uses_legacy_meanflow_loss:
                metrics["legacy_adaptive_l2"] = mf_loss.detach()
            # Same pattern as rmse/r2: always defined so callers use metrics["..."] directly.
            z_phys = torch.zeros((), device=metrics["rmse"].device, dtype=metrics["rmse"].dtype)
            metrics["temp_pde_pure"] = pure_phys.get("temp_pde_pure", z_phys)
            metrics["at_mag_pure"] = pure_phys.get("at_mag_pure", z_phys)
            metrics["at_dir_pure"] = pure_phys.get("at_dir_pure", z_phys)
            metrics["at_dir_pure_cosine"] = pure_phys.get("at_dir_pure_cosine", z_phys)
            metrics["at_dir_pure_unit_mse"] = pure_phys.get("at_dir_pure_unit_mse", z_phys)
            metrics["at_mask_frac"] = pure_phys.get("at_mask_frac", z_phys)

        return total_loss, metrics

    def _compute_control_score(
        self,
        loss: torch.Tensor,
        metrics: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Composite validation monitor (lower is better). Enable ``mf_minus_1`` / ``loss_total_f64`` via config."""
        w = self.control_metric_weights
        score = (
            w.get("loss", 0.0) * loss
            + w.get("rmse", 0.0) * metrics["rmse"]
            - w.get("r2", 0.0) * metrics["r2"]
            + w.get("temp_pde_pure", 0.0) * metrics["temp_pde_pure"]
            + w.get("at_mag_pure", 0.0) * metrics["at_mag_pure"]
            + w.get("at_dir_pure", 0.0) * metrics["at_dir_pure"]
            + w.get("mf_loss_f64", 0.0) * metrics["mf_loss_f64"]
            + w.get("mf_minus_1", 0.0) * metrics["mf_minus_1"]
            + w.get("loss_total_f64", 0.0) * metrics["loss_total_f64"]
        )
        legacy_l2 = metrics.get("legacy_adaptive_l2")
        if legacy_l2 is not None:
            score = score + w.get("legacy_adaptive_l2", 0.0) * legacy_l2
        return score

    def _log_mf_monitors(
        self,
        prefix: str,
        metrics: Dict[str, torch.Tensor],
        *,
        prog_bar: bool = False,
    ) -> None:
        """MeanFlow deviation + physics only (skip flat ~1.0 scalars and EMA duplicates)."""
        log_params = {"on_step": False, "on_epoch": True, "sync_dist": True}
        self.log(f"{prefix}/mf_minus_1", metrics["mf_minus_1"], **log_params)
        self.log(
            f"{prefix}/mf_minus_1_x1e8",
            metrics["mf_minus_1_x1e8"],
            **log_params,
            prog_bar=prog_bar,
        )
        self.log(f"{prefix}/phys_loss", metrics["phys_loss"], **log_params)

    def shared_step(self, batch, create_graph: bool):
        latent_target, context_dict = self.build_latent_and_context(batch)
        return self._meanflow_train_loss(latent_target, context_dict, create_graph=create_graph)

    def training_step(self, batch, batch_idx):
        loss, metrics = self.shared_step(batch, create_graph=True)
        log_params = {"on_step": False, "on_epoch": True, "sync_dist": True}
        self.log("train/rmse", metrics["rmse"], **log_params)
        self.log("train/r2", metrics["r2"], **log_params)
        self._log_mf_monitors("train", metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.shared_step(batch, create_graph=False)

        log_params = {"on_step": False, "on_epoch": True, "prog_bar": False, "sync_dist": True}
        self.log("val/rmse", metrics["rmse"], **log_params)
        self.log("val/r2", metrics["r2"], **log_params)
        self.log("val/at_mag_pure", metrics["at_mag_pure"], **log_params)
        self.log("val/at_dir_pure_cosine", metrics["at_dir_pure_cosine"], **log_params)
        self.log("val/at_dir_pure_unit_mse", metrics["at_dir_pure_unit_mse"], **log_params)
        self._log_mf_monitors("val", metrics, prog_bar=True)
        control_score = self._compute_control_score(loss, metrics)
        self.log(
            "val/control_score",
            control_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        _, metrics = self.shared_step(batch, create_graph=False)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True, "sync_dist": True}
        self.log("test/rmse", metrics["rmse"], **log_params)
        self.log("test/r2", metrics["r2"], **log_params)
        self.log("test/at_mag_pure", metrics["at_mag_pure"], **log_params)
        self.log("test/at_dir_pure_cosine", metrics["at_dir_pure_cosine"], **log_params)
        self.log("test/at_dir_pure_unit_mse", metrics["at_dir_pure_unit_mse"], **log_params)
        self._log_mf_monitors("test", metrics)

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
