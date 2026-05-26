"""Val-loop MeanFlow loss probe (float32 + float64) — used by meanflow_loss_deviation_probe.ipynb."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.inference_mode()
def mf_loss_on_batch_f32_and_f64(
    model,
    batch: Tuple[torch.Tensor, ...],
) -> Tuple[float, float, Dict[str, float]]:
    """One forward (f32, training path); f64 loss from ``error.double()`` only."""
    z0, context = model.build_latent_and_context(batch)
    core = model.meanflow_core
    train_targets = core.compute_train_targets(z0)
    x_t = train_targets["x_t"]
    t = train_targets["t"]
    r = train_targets["r"]
    v_target = train_targets["v_target"]

    def backbone(x_state, time_r, time_t):
        return model.mf_unet(x_state, time_t, time_r, context=context)

    error, u_pred, u_tgt = core.compute_teacher_error(
        backbone_model=backbone,
        x_t=x_t,
        t=t,
        r=r,
        v_target=v_target,
        create_graph=False,
        return_details=True,
    )
    mf_f32 = float(core.adaptive_l2_loss(error).detach().cpu())
    mf_f64 = float(core.adaptive_l2_loss(error.double()).detach().cpu())
    loss_mid = torch.sum((error**2).reshape(error.shape[0], -1), dim=-1)
    extras = {
        "loss_mid_mean": float(loss_mid.mean().cpu()),
        "error_l2": float(torch.sqrt(torch.mean(error**2)).cpu()),
    }
    return mf_f32, mf_f64, extras


def run_val_mf_loss_probe(
    model,
    dataloader: DataLoader,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Scan val loader; return per-batch f32/f64 losses and summary stats."""
    model.eval()
    mf_f32: List[float] = []
    mf_f64: List[float] = []
    dev_f32: List[float] = []
    dev_f64: List[float] = []

    it = tqdm(dataloader, desc="val mf_loss", disable=not show_progress)
    for bi, batch in enumerate(it):
        if max_batches is not None and bi >= max_batches:
            break
        batch = tuple(
            b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b
            for b in batch
        )
        v32, v64, _ = mf_loss_on_batch_f32_and_f64(model, batch)
        mf_f32.append(v32)
        mf_f64.append(v64)
        dev_f32.append(v32 - 1.0)
        dev_f64.append(v64 - 1.0)

    a32 = np.asarray(mf_f32, dtype=np.float64)
    a64 = np.asarray(mf_f64, dtype=np.float64)
    d32 = np.asarray(dev_f32, dtype=np.float64)
    d64 = np.asarray(dev_f64, dtype=np.float64)

    def _stats(x: np.ndarray, mf_raw: np.ndarray) -> Dict[str, float]:
        if x.size == 0:
            return {}
        return {
            "n": int(x.size),
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "range": float(x.max() - x.min()),
            "abs_min": float(np.abs(x).min()),
            "abs_max": float(np.abs(x).max()),
            "abs_mean": float(np.abs(x).mean()),
            "p01": float(np.percentile(x, 1)),
            "p50": float(np.percentile(x, 50)),
            "p99": float(np.percentile(x, 99)),
            "frac_abs_lt_1e-4": float(np.mean(np.abs(x) < 1e-4)),
            "frac_abs_lt_1e-6": float(np.mean(np.abs(x) < 1e-6)),
            "frac_abs_lt_1e-10": float(np.mean(np.abs(x) < 1e-10)),
            "frac_exactly_zero_dev": float(np.mean(x == 0.0)),
            "num_mf_eq_1": int(np.sum(mf_raw == 1.0)),
        }

    summary = {
        "mf_loss_f32": _stats(a32, a32),
        "mf_loss_f64": _stats(a64, a64),
        "mf_minus_1_f32": _stats(d32, a32),
        "mf_minus_1_f64": _stats(d64, a64),
        "max_abs_diff_mf_f64_minus_f32": float(np.max(np.abs(a64 - a32))) if a32.size else None,
    }

    return {
        "mf_f32": mf_f32,
        "mf_f64": mf_f64,
        "dev_f32": dev_f32,
        "dev_f64": dev_f64,
        "summary": summary,
    }


def save_probe_results(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")
