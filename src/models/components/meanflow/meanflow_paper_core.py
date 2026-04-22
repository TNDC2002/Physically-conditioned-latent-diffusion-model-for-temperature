# MeanFlow training math aligned with the reference implementation in ``MeanFlow/loss.py``
# (paper code: https://github.com/... — cloned under repo ``MeanFlow/``).
# Blocks below intentionally mirror that file with minimal edits; CFG / label paths are omitted
# because LMM uses MF-UNet + context dicts instead of class-conditioned SiT.

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F


Tensor = torch.Tensor


class MeanFlowPaperCore:
    """MeanFlow core matching ``MeanFlow/loss.py`` (SILoss) non-CFG behavior for MF-UNet."""

    def __init__(
        self,
        path_type: str = "linear",
        weighting: str = "adaptive",
        time_sampler: str = "logit_normal",
        time_mu: float = -0.4,
        time_sigma: float = 1.0,
        ratio_r_not_equal_t: float = 0.75,
        adaptive_p: float = 1.0,
    ) -> None:
        self.weighting = weighting
        self.path_type = path_type
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.adaptive_p = adaptive_p

    # --- duplicated from MeanFlow/loss.py (SILoss) ---------------------------------

    def interpolant(self, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Define interpolation function"""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def sample_time_steps(self, batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Sample time steps (r, t) according to the configured sampler"""
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")

        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]

        fraction_equal = 1.0 - self.ratio_r_not_equal_t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        r = torch.where(equal_mask, t, r)

        return r, t

    # --- public API (same surface as ``MeanFlowCore`` where LMM depends on it) -----

    def sample_t_r(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tuple[Tensor, Tensor]:
        r, t = self.sample_time_steps(batch_size, device)
        return t.to(dtype=dtype), r.to(dtype=dtype)

    def build_bridge_state(self, x0: Tensor, t: Tensor, noise: Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_t, sigma_t, _, _ = self.interpolant(t.view(-1, 1, 1, 1))
        x_t = alpha_t * x0 + sigma_t * noise
        return x_t, {"noise": noise}

    def compute_train_targets(self, x0: Tensor):
        batch_size = x0.shape[0]
        device = x0.device
        dtype = x0.dtype
        r, t = self.sample_time_steps(batch_size, device)
        r = r.to(dtype=dtype)
        t = t.to(dtype=dtype)

        noises = torch.randn_like(x0)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * x0 + sigma_t * noises
        v_t = d_alpha_t * x0 + d_sigma_t * noises

        return {
            "x_t": z_t,
            "t": t,
            "r": r,
            "v_target": v_t,
        }

    def single_step_generate(self, x_t: Tensor, t: Tensor, r: Tensor, u_theta: Tensor):
        delta = (t - r).view(-1, 1, 1, 1)
        return x_t - delta * u_theta

    def meanflow_teacher(self, v_target: Tensor, t: Tensor, r: Tensor, dudt: Tensor):
        delta = (t - r).view(-1, 1, 1, 1)
        return v_target - delta * dudt

    def compute_teacher_error(
        self,
        backbone_model: Callable[..., Tensor],
        x_t: Tensor,
        t: Tensor,
        r: Tensor,
        v_target: Tensor,
        create_graph: bool,
    ):
        # Same primals/tangents as ``MeanFlow/loss.py`` non-CFG branch (lines 206–215); reference
        # uses ``torch.func.jvp`` and a separate ``u = model(...)`` — one ``jvp`` here matches that
        # math and matches ``MeanFlowCore`` (autograd JVP for ``create_graph``).
        def fn_current(z: Tensor, cur_r: Tensor, cur_t: Tensor) -> Tensor:
            return backbone_model(z, cur_r, cur_t)

        primals = (x_t, r, t)
        tangents = (v_target, torch.zeros_like(r), torch.ones_like(t))
        u_theta, dudt = torch.autograd.functional.jvp(
            fn_current,
            primals,
            tangents,
            create_graph=create_graph,
        )
        u_target = self.meanflow_teacher(v_target, t, r, dudt)
        return u_theta - u_target.detach()

    def adaptive_l2_loss(self, error: Tensor, gamma: float = 0.5, c: float = 1e-3):
        # Paper reference (``MeanFlow/loss.py``): per-sample sum of squared error + adaptive weights.
        # Note: ``gamma`` / ``c`` are accepted for call-site compatibility with ``MeanFlowCore`` but
        # the paper implementation uses fixed ``c=1e-3`` and power ``adaptive_p`` (default 1.0).
        del gamma  # paper uses self.adaptive_p instead of (1 - gamma)
        loss_mid = torch.sum((error**2).reshape(error.shape[0], -1), dim=-1)
        if self.weighting == "adaptive":
            weights = 1.0 / (loss_mid.detach() + c).pow(self.adaptive_p)
            loss = weights * loss_mid
        else:
            loss = loss_mid
        return loss.mean()

    def residual_loss(self, r_hat: Tensor, r_gt: Tensor):
        return F.mse_loss(r_hat, r_gt)
