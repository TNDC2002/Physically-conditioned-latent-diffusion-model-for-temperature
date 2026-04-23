import torch
import torch.nn.functional as F


class MeanFlowCore:
    """Core MeanFlow utilities reused by training and inference."""

    def __init__(
        self,
        flow_ratio: float = 0.75,
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

    def meanflow_teacher(self, v_target: torch.Tensor, t: torch.Tensor, r: torch.Tensor, dudt: torch.Tensor):
        delta = (t - r).view(-1, 1, 1, 1)
        return v_target - delta * dudt

    def compute_teacher_error(
        self,
        backbone_model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        v_target: torch.Tensor,
        create_graph: bool,
        return_details: bool = False,
    ):
        # MeanFlow: u, dudt = jvp(fn, (z, r, t), (v, 0, 1))
        u_theta, dudt = torch.autograd.functional.jvp(
            backbone_model,
            (x_t, r, t),
            (v_target, torch.zeros_like(r), torch.ones_like(t)),
            create_graph=create_graph,
        )
        u_tgt = self.meanflow_teacher(v_target, t, r, dudt)
        error = u_theta - u_tgt.detach()
        if return_details:
            return error, u_theta, u_tgt.detach()
        return error

    def adaptive_l2_loss(self, error: torch.Tensor, gamma: float = 0.5, c: float = 1e-3):
        # Follow upstream MeanFlow robust weighting:
        # loss = stopgrad(w) * ||error||^2, w = 1 / (||error||^2 + c)^(1-gamma)
        delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
        p = 1.0 - gamma
        w = 1.0 / (delta_sq + c).pow(p)
        return (w.detach() * delta_sq).mean()

    def residual_loss(self, r_hat: torch.Tensor, r_gt: torch.Tensor):
        return F.mse_loss(r_hat, r_gt)
