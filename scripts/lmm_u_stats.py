from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@dataclass
class RunningMoments:
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0

    def update(self, tensor: torch.Tensor) -> None:
        t = tensor.detach().to(dtype=torch.float64)
        self.count += int(t.numel())
        self.sum += float(t.sum().item())
        self.sum_sq += float((t * t).sum().item())

    def mean_std(self) -> Tuple[float, float]:
        if self.count == 0:
            return float("nan"), float("nan")
        mean = self.sum / self.count
        var = max((self.sum_sq / self.count) - (mean * mean), 0.0)
        return mean, var ** 0.5


def _compute_u_pair(model, batch) -> Tuple[torch.Tensor, torch.Tensor]:
    latent_target, context_dict = model.build_latent_and_context(batch)
    train_targets = model.meanflow_core.compute_train_targets(latent_target)
    x_t = train_targets["x_t"]
    t = train_targets["t"]
    r = train_targets["r"]
    v_target = train_targets["v_target"]

    def backbone(x_state, time_r, time_t):
        return model.mf_unet(x_state, time_t, time_r, context=context_dict)

    _, u_pred, u_tgt = model.meanflow_core.compute_teacher_error(
        backbone_model=backbone,
        x_t=x_t,
        t=t,
        r=r,
        v_target=v_target,
        create_graph=False,
        return_details=True,
    )
    return u_pred, u_tgt


def _accumulate_stats(model, loader: Iterable, split_name: str) -> Dict[str, Tuple[float, float]]:
    pred_stats = RunningMoments()
    tgt_stats = RunningMoments()

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            batch = tuple(
                x.to(model.device, non_blocking=True) if torch.is_tensor(x) else x for x in batch
            )
        u_pred, u_tgt = _compute_u_pair(model, batch)
        pred_stats.update(u_pred)
        tgt_stats.update(u_tgt)

    pred_mean, pred_std = pred_stats.mean_std()
    tgt_mean, tgt_std = tgt_stats.mean_std()
    return {
        f"{split_name}/u_pred": (pred_mean, pred_std),
        f"{split_name}/u_tgt": (tgt_mean, tgt_std),
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("ckpt_path must be provided. Example: ckpt_path=/path/to/last.ckpt")

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    datamodule.setup(stage=None)
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    report = {}
    report.update(_accumulate_stats(model, val_loader, "val"))
    report.update(_accumulate_stats(model, test_loader, "test"))

    print("=== u_pred/u_tgt mean-std report ===")
    print(f"ckpt_path: {ckpt_path}")
    for key in ("val/u_pred", "val/u_tgt", "test/u_pred", "test/u_tgt"):
        mean, std = report[key]
        print(f"{key}: mean={mean:.10f}, std={std:.10f}")


if __name__ == "__main__":
    main()
