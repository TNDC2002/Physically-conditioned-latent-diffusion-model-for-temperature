"""Temperature / wind field physics losses shared by LDM-style training paths.

Extracted from ``LatentDiffusion`` so LMM can apply the same PDE / energy terms
without subclassing the diffusion module. API mirrors the original methods.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class TemperatureFieldLosses:
    """Stateless helpers; no registered parameters."""

    def mass_conservation_loss(self, wind_field: torch.Tensor) -> torch.Tensor:
        wind_field = torch.clamp(wind_field, min=-10.0, max=10.0)
        u = wind_field[:, 0:1, :, :]
        v = wind_field[:, 1:2, :, :]
        du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / 2.0
        dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / 2.0
        du_dx = du_dx[:, :, 1:-1, :]
        dv_dy = dv_dy[:, :, :, 1:-1]
        divergence = du_dx + dv_dy
        return torch.mean(divergence**2)

    def _compute_gradients_torch_batch(self, T: torch.Tensor, dx=1.0, dy=1.0, eps=1e-4):
        if T.dim() == 4:
            T = T[:, 0, :, :]

        B, H, W = T.shape
        dTdx = torch.zeros_like(T)
        dTdy = torch.zeros_like(T)

        if W > 2:
            dTdx[:, :, 1:-1] = (T[:, :, 2:] - T[:, :, :-2]) / (2.0 * dx)
        if H > 2:
            dTdy[:, 1:-1, :] = (T[:, 2:, :] - T[:, :-2, :]) / (2.0 * dy)

        if W > 1:
            dTdx[:, :, 0] = (T[:, :, 1] - T[:, :, 0]) / dx
            dTdx[:, :, -1] = (T[:, :, -1] - T[:, :, -2]) / dx
        if H > 1:
            dTdy[:, 0, :] = (T[:, 1, :] - T[:, 0, :]) / dy
            dTdy[:, -1, :] = (T[:, -1, :] - T[:, -2, :]) / dy

        return dTdx, dTdy

    def _compute_block_flux_ratio(self, T_block, dTdx_block, dTdy_block, eps=1e-4):
        device = T_block.device
        B, blockH, blockW = T_block.shape

        if blockH < 1 or blockW < 1:
            return torch.zeros((B,), device=device, dtype=T_block.dtype)

        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        adv_vals = []
        diff_vals = []

        def boundary_adv_diff(i_coords, j_coords, nx, ny):
            grad_x = dTdx_block[
                batch_idx, i_coords.unsqueeze(0).expand(B, -1), j_coords.unsqueeze(0).expand(B, -1)
            ]
            grad_y = dTdy_block[
                batch_idx, i_coords.unsqueeze(0).expand(B, -1), j_coords.unsqueeze(0).expand(B, -1)
            ]
            Tvals = T_block[
                batch_idx, i_coords.unsqueeze(0).expand(B, -1), j_coords.unsqueeze(0).expand(B, -1)
            ]

            grad_norm = torch.sqrt(grad_x**2 + grad_y**2) + eps
            g_hat_x = grad_x / grad_norm
            g_hat_y = grad_y / grad_norm
            dot = g_hat_x * nx + g_hat_y * ny
            adv_ = Tvals * dot
            diff_ = grad_norm
            return adv_, diff_

        i_top = torch.zeros(blockW, device=device, dtype=torch.long)
        j_top = torch.arange(blockW, device=device, dtype=torch.long)
        a_top, d_top = boundary_adv_diff(i_top, j_top, nx=0.0, ny=-1.0)
        adv_vals.append(a_top)
        diff_vals.append(d_top)

        i_bot = torch.full((blockW,), blockH - 1, device=device, dtype=torch.long)
        j_bot = torch.arange(blockW, device=device, dtype=torch.long)
        a_bot, d_bot = boundary_adv_diff(i_bot, j_bot, nx=0.0, ny=1.0)
        adv_vals.append(a_bot)
        diff_vals.append(d_bot)

        if blockH > 2:
            i_left = torch.arange(1, blockH - 1, device=device, dtype=torch.long)
            j_left = torch.zeros(blockH - 2, device=device, dtype=torch.long)
            a_left, d_left = boundary_adv_diff(i_left, j_left, nx=-1.0, ny=0.0)
            adv_vals.append(a_left)
            diff_vals.append(d_left)

        if blockH > 2:
            i_right = torch.arange(1, blockH - 1, device=device, dtype=torch.long)
            j_right = torch.full((blockH - 2,), blockW - 1, device=device, dtype=torch.long)
            a_right, d_right = boundary_adv_diff(i_right, j_right, nx=1.0, ny=0.0)
            adv_vals.append(a_right)
            diff_vals.append(d_right)

        adv_all = torch.cat(adv_vals, dim=1)
        diff_all = torch.cat(diff_vals, dim=1)

        adv_mean = torch.mean(adv_all, dim=1)
        diff_mean = torch.mean(diff_all, dim=1)

        ratio = adv_mean / (diff_mean + eps)
        return ratio

    def _compute_supercell_flux_ratio_field_batch(
        self, T_in: torch.Tensor, num_supercells: int, dx=1.0, dy=1.0, eps=1e-4
    ):
        if T_in.dim() == 4:
            T_in = T_in[:, 0, :, :]

        B, H, W = T_in.shape

        if H % num_supercells != 0 or W % num_supercells != 0:
            raise ValueError("H and W must be evenly divisible by num_supercells")

        block_size_h = H // num_supercells
        block_size_w = W // num_supercells

        if block_size_h < 2 or block_size_w < 2:
            raise ValueError("Each supercell must have at least four pixels (2x2 minimum)")

        ratio_field = torch.zeros((B, num_supercells, num_supercells), device=T_in.device, dtype=T_in.dtype)

        for i in range(num_supercells):
            r0, r1 = i * block_size_h, (i + 1) * block_size_h
            for j in range(num_supercells):
                c0, c1 = j * block_size_w, (j + 1) * block_size_w

                T_block = T_in[:, r0:r1, c0:c1]
                dTdx_block, dTdy_block = self._compute_gradients_torch_batch(T_block, dx=dx, dy=dy, eps=eps)

                flux_ratio = self._compute_block_flux_ratio(T_block, dTdx_block, dTdy_block, eps=eps)
                ratio_field[:, i, j] = flux_ratio

        return ratio_field

    def temperature_pde_loss(self, T_f: torch.Tensor, T_c: torch.Tensor, num_supercells: int) -> torch.Tensor:
        R_f = self._compute_supercell_flux_ratio_field_batch(
            T_f, num_supercells=num_supercells, dx=1.0, dy=1.0, eps=1e-4
        )
        R_c = self._compute_supercell_flux_ratio_field_batch(
            T_c, num_supercells=num_supercells, dx=1.0, dy=1.0, eps=1e-4
        )

        loss_map = torch.abs(R_f - R_c)
        return torch.mean(loss_map)

    def _anisotropic_flux_q(
        self, T: torch.Tensor, dx: float = 1.0, dy: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q = -J @ grad T with recovered J = grad T grad T^T (same as flux notebook)."""
        dTdx, dTdy = self._compute_gradients_torch_batch(T, dx=dx, dy=dy)
        J_xx = dTdx * dTdx
        J_xy = dTdx * dTdy
        J_yy = dTdy * dTdy
        qx = -(J_xx * dTdx + J_xy * dTdy)
        qy = -(J_xy * dTdx + J_yy * dTdy)
        return qx, qy

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return x.mean()
        m = mask.to(dtype=x.dtype)
        denom = m.sum().clamp(min=1.0)
        return (x * m).sum() / denom

    def _build_anisotropic_qmag_mask(
        self,
        mag_gt: torch.Tensor,
        *,
        qmag_quantile: float | None = None,
        qmag_min: float | None = None,
    ) -> torch.Tensor | None:
        """Build GT-only mask for anisotropic loss (modes combine with AND).

        - ``qmag_quantile``: keep pixels with ``|q_gt|`` at/above per-sample quantile.
        - ``qmag_min``: keep pixels with ``|q_gt| > qmag_min`` (drop ``|q_gt| <= qmag_min``).

        Mask uses ground truth only: low ``|q_pred|`` on meaningful ``|q_gt|`` still counts.
        """
        mask: torch.Tensor | None = None

        if qmag_quantile is not None:
            q = float(qmag_quantile)
            if q <= 0.0 or q >= 1.0:
                raise ValueError(f"qmag_quantile must be in (0, 1), got {qmag_quantile!r}")
            flat = mag_gt.reshape(mag_gt.shape[0], -1)
            thr = torch.quantile(flat, q, dim=1, keepdim=True)
            thr = thr.view(mag_gt.shape[0], *([1] * (mag_gt.ndim - 1)))
            q_mask = mag_gt >= thr
            mask = q_mask if mask is None else mask & q_mask

        if qmag_min is not None:
            m = float(qmag_min)
            if m < 0.0:
                raise ValueError(f"qmag_min must be >= 0, got {qmag_min!r}")
            min_mask = mag_gt > m
            mask = min_mask if mask is None else mask & min_mask

        return mask

    def anisotropic_transport_loss(
        self,
        T_pred: torch.Tensor,
        T_gt: torch.Tensor,
        *,
        dx: float = 2000.0,
        dy: float = -2000.0,
        eps: float = 1e-12,
        lambda_mag: float = 1.0,
        lambda_dir: float = 1.0,
        direction_kind: str = "cosine",
        qmag_quantile: float | None = None,
        qmag_min: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Anisotropic Transport Loss: L = lambda_mag * L_mag + lambda_dir * L_dir.

        - L_mag = MSE(log(|q_pred|+eps), log(|q_gt|+eps))
        - L_dir = mean(1 - cos(q_pred, q_gt)) or unit-vector MSE on q_hat

        GT masking (optional, combined with AND):
        - ``qmag_quantile`` (e.g. 0.5): per-sample quantile on ``|q_gt|``.
        - ``qmag_min`` (e.g. 1e-12): drop pixels with ``|q_gt| <= qmag_min``.

        ``T_pred``: reconstructed residual R (decoded one-step latent), normalized.
        ``T_gt``: normalized COSMO-CLM high-res target (batch ``y``).
        """
        qx_p, qy_p = self._anisotropic_flux_q(T_pred, dx=dx, dy=dy)
        qx_g, qy_g = self._anisotropic_flux_q(T_gt, dx=dx, dy=dy)
        mag_p = torch.hypot(qx_p, qy_p)
        mag_g = torch.hypot(qx_g, qy_g)
        mask = self._build_anisotropic_qmag_mask(
            mag_g, qmag_quantile=qmag_quantile, qmag_min=qmag_min
        )

        l_mag = self._masked_mean(
            (torch.log(mag_p + eps) - torch.log(mag_g + eps)) ** 2, mask
        )

        dot = qx_p * qx_g + qy_p * qy_g
        cos = dot / (mag_p * mag_g + eps)
        l_dir_cosine = self._masked_mean(1.0 - cos, mask)
        px, py = qx_p / (mag_p + eps), qy_p / (mag_p + eps)
        gx, gy = qx_g / (mag_g + eps), qy_g / (mag_g + eps)
        unit_sq = (px - gx) ** 2 + (py - gy) ** 2
        l_dir_unit_mse = self._masked_mean(unit_sq, mask)

        if direction_kind == "cosine":
            l_dir = l_dir_cosine
        elif direction_kind == "unit_mse":
            l_dir = l_dir_unit_mse
        else:
            raise ValueError(
                f"direction_kind must be 'cosine' or 'unit_mse', got {direction_kind!r}"
            )

        l_total = lambda_mag * l_mag + lambda_dir * l_dir
        out: dict[str, torch.Tensor] = {
            "L_mag": l_mag,
            "L_dir": l_dir,
            "L_dir_cosine": l_dir_cosine,
            "L_dir_unit_mse": l_dir_unit_mse,
            "L_total": l_total,
        }
        if mask is not None:
            out["at_mask_frac"] = mask.float().mean()
        return out
