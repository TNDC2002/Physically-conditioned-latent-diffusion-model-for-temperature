import torch
import torch.nn.functional as F


class MeanFlowCore:
    """Core MeanFlow utilities reused by training and inference."""

    def __init__(
        self,
        flow_ratio: float = 0.5,
        time_dist: str = "lognorm",
        time_mu: float = -0.4,
        time_sigma: float = 1.0,
    ) -> None:
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.time_mu = time_mu
        self.time_sigma = time_sigma

    def sample_t_r(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        if self.time_dist == "uniform":
            samples = torch.rand(batch_size, 2, device=device, dtype=dtype)
        elif self.time_dist == "lognorm":
            normal = torch.randn(batch_size, 2, device=device, dtype=dtype)
            samples = torch.sigmoid(normal * self.time_sigma + self.time_mu)
        else:
            raise ValueError(f"Unsupported time_dist: {self.time_dist}")

        t = torch.maximum(samples[:, 0], samples[:, 1])
        r = torch.minimum(samples[:, 0], samples[:, 1])

        num_equal = int(self.flow_ratio * batch_size)
        if num_equal > 0:
            perm = torch.randperm(batch_size, device=device)[:num_equal]
            r[perm] = t[perm]
        return t, r

    def build_bridge_state(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        t_view = t.view(-1, 1, 1, 1)
        x_t = (1.0 - t_view) * x0 + t_view * noise
        return x_t, {"noise": noise}

    def compute_train_targets(self, x0: torch.Tensor):
        t, r = self.sample_t_r(x0.shape[0], x0.device, x0.dtype)
        x_t, aux = self.build_bridge_state(x0, t)
        v_target = aux["noise"] - x0
        return {
            "x_t": x_t,
            "t": t,
            "r": r,
            "v_target": v_target,
        }

    def single_step_generate(self, x_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor, u_theta: torch.Tensor):
        delta = (t - r).view(-1, 1, 1, 1)
        return x_t - delta * u_theta

    def residual_loss(self, r_hat: torch.Tensor, r_gt: torch.Tensor):
        return F.mse_loss(r_hat, r_gt)
