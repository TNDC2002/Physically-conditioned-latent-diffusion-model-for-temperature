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

    def temperature_energy_loss(self, T_f: torch.Tensor, T_c: torch.Tensor) -> torch.Tensor:
        T_f_down = F.adaptive_avg_pool2d(T_f, T_c.shape[-2:])
        return torch.mean((T_f_down - T_c) ** 2)
