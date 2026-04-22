from typing import Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule

from .components.ldm.denoiser.meanflow_infer import generate_residual_one_step


class HrResidualMeanFlowLitModule(LightningModule):
    def __init__(
        self,
        mf_unet,
        meanflow_core,
        legacy_autoencoder,
        context_encoder=None,
        stage1_encoder_ckpt: Optional[str] = None,
        freeze_stage1_encoder: bool = False,
        lr: float = 1e-4,
        lr_warmup: int = 0,
    ):
        super().__init__()
        self.mf_unet = mf_unet
        self.meanflow_core = meanflow_core
        self.legacy_autoencoder = legacy_autoencoder.requires_grad_(False)
        self.context_encoder = context_encoder
        self.lr = lr
        self.lr_warmup = lr_warmup

        if stage1_encoder_ckpt is not None and self.context_encoder is not None:
            ckpt = torch.load(stage1_encoder_ckpt, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            self.context_encoder.autoencoder[0].load_state_dict(state, strict=False)

        if freeze_stage1_encoder and self.context_encoder is not None:
            self.context_encoder.autoencoder[0].requires_grad_(False)

    def _merge_lowres_with_static(self, low_res, static, target_size):
        low_res_up = F.interpolate(low_res, size=target_size, mode="nearest")
        return torch.cat([low_res_up, static], dim=1)

    def _stage2_upscale(self, low_res, static, y_hr):
        if hasattr(self.legacy_autoencoder, "nn_lr_and_merge_with_static"):
            merged = self.legacy_autoencoder.nn_lr_and_merge_with_static(low_res, static)
        else:
            merged = self._merge_lowres_with_static(low_res, static, y_hr.shape[-2:])
        return self.legacy_autoencoder.unet(merged)

    def build_context(self, low_res, static):
        if self.context_encoder is None:
            return None
        zeros = torch.zeros((low_res.shape[0], 1), device=low_res.device, dtype=low_res.dtype)
        return self.context_encoder([(static, zeros), (low_res, zeros)])

    def shared_step(self, batch, create_graph: bool):
        if len(batch) == 4:
            low_res, y_hr, static, _ = batch
        else:
            low_res, y_hr, static = batch

        with torch.no_grad():
            y_up = self._stage2_upscale(low_res, static, y_hr)

        r_gt = y_hr - y_up
        z_ctx = self.build_context(low_res, static)

        train_targets = self.meanflow_core.compute_train_targets(r_gt)
        x_t = train_targets["x_t"]
        t = train_targets["t"]
        r = train_targets["r"]
        v_target = train_targets["v_target"]

        def UNET(x_state, time_r, time_t):
            return self.mf_unet(x_state, time_t, time_r, context=z_ctx)


        error = self.meanflow_core.compute_teacher_error(
            backbone_model=UNET,
            x_t=x_t,
            t=t,
            r=r,
            v_target=v_target,
            create_graph=create_graph,
        )
        loss = self.meanflow_core.adaptive_l2_loss(error)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, create_graph=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, create_graph=False)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    @torch.no_grad()
    def predict_residual(self, low_res, static, shape):
        z_ctx = self.build_context(low_res, static)
        return generate_residual_one_step(
            mf_unet=self.mf_unet,
            meanflow_core=self.meanflow_core,
            context=z_ctx,
            shape=shape,
            device=low_res.device,
            dtype=low_res.dtype,
        )

    @torch.no_grad()
    def predict_final(self, low_res, static, y_shape):
        y_up = self._stage2_upscale(low_res, static, torch.zeros(y_shape, device=low_res.device, dtype=low_res.dtype))
        r_hat = self.predict_residual(low_res, static, shape=y_up.shape)
        return y_up + r_hat

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.25, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val/loss",
                "frequency": 1,
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        if self.lr_warmup > 0 and self.trainer.global_step < self.lr_warmup:
            lr_scale = (self.trainer.global_step + 1) / self.lr_warmup
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)
