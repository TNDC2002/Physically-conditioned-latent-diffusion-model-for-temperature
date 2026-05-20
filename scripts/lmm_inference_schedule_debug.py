#!/usr/bin/env python3
"""Debug LMM multi-step inference schedule (t, r) on one test sample.

Prints go to stdout with flush so Slurm ``.out`` logs capture them (unlike
``jupyter nbconvert``, which strips cell output).

Example (repo root, GPU node)::

    export LDM_DATA_ROOT=$PWD/LDM-downscaling/full_Dataset/
    export LMM_CKPT=./logs/train/runs/.../checkpoints/last.ckpt
    ./.venv/bin/python scripts/lmm_inference_schedule_debug.py --n-steps 3 --schedule uniform

    ./.venv/bin/python scripts/lmm_inference_schedule_debug.py --n-steps 3 --schedule inv_ni

Slurm::

    LMM_CKPT=... bash scripts/submit_lmm_schedule_debug.sh
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lightning as L
import pyrootutils
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

_REPO = Path(__file__).resolve().parents[1]
pyrootutils.setup_root(str(_REPO), indicator=".project-root", pythonpath=True)

from utils.inference_utils import (  # noqa: E402
    lmm_inv_ni_tr_schedule,
    lmm_predict_final_n_steps_inv_ni,
    lmm_predict_final_n_steps_uniform,
    lmm_uniform_tr_schedule,
)

_SCHEDULES = {
    "uniform": (lmm_uniform_tr_schedule, lmm_predict_final_n_steps_uniform),
    "inv_ni": (lmm_inv_ni_tr_schedule, lmm_predict_final_n_steps_inv_ni),
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def print_schedule(
    schedule: str, n_steps: int, device: torch.device, dtype: torch.dtype
) -> None:
    build_pairs, _ = _SCHEDULES[schedule]
    pairs = build_pairs(n_steps, device, dtype)
    _log(f"=== MeanFlow schedule={schedule!r} (n_steps={n_steps}) ===")
    _log("step | t_i (start) | r_i (end) | delta=t-r")
    for i, (t_val, r_val) in enumerate(pairs):
        _log(f"  {i:3d} | {t_val:.6f}    | {r_val:.6f}  | {t_val - r_val:.6f}")
    _log("=== end schedule ===")


def _resolve_data_dir(arg: str | None) -> str:
    raw = (arg or os.environ.get("LDM_DATA_ROOT") or str(_REPO / "LDM-downscaling" / "full_Dataset")).strip()
    if not raw.endswith(os.sep):
        raw = raw + os.sep
    p = Path(raw)
    if not p.is_dir():
        raise FileNotFoundError(f"Data directory not found: {p}")
    norm = p / "normalization_data.pkl"
    if not norm.is_file():
        raise FileNotFoundError(f"Missing {norm}")
    return raw


def _load_lmm(ckpt_path: Path, device: torch.device):
    os.environ.setdefault("PROJECT_ROOT", str(_REPO))
    config_dir = str(_REPO / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="train", overrides=["experiment=downscaling_LMM_res_2mT"])
    model = instantiate(cfg.model).to(device)
    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
        pde = _REPO / "pretrained_models" / "lmm_pde_loss_model_checkpoint.ckpt"
        if pde.is_file():
            partial = torch.load(pde, map_location=device, weights_only=False)["state_dict"]
            merged = state.copy()
            merged.update(partial)
            state = merged
            _log(f"Merged PDE partial weights from {pde}")
        model.load_state_dict(state, strict=False)
        _log(f"Loaded checkpoint: {ckpt_path}")
    else:
        _log(f"WARNING: checkpoint not found ({ckpt_path}); using random init weights")
    model.eval()
    return model


def _one_test_batch(data_dir: str, sample_index: int | None, seed: int):
    from src.data.downscaling_datamodule import DownscalingDataModule

    target_vars = {
        "high_res": ["2mT"],
        "low_res": [
            "2mT", "PMSL", "U10", "V10", "dp2mT", "SST", "SNDPT", "TP",
            "SSRadIn", "Q850", "T850", "U850", "V850", "W850",
        ],
    }
    root = Path(data_dir)
    static_vars = {
        "dtm_tif_file": str(root / "static_var/dtm_2km_domain_trim_EPSG3035.tif"),
        "lc_tif_file": str(root / "static_var/land_cover_classes_2km_domain_trim_EPSG3035.tif"),
        "lat_tif_file": str(root / "static_var/lat_2km_domain_trim_EPSG3035.tif"),
    }
    dm = DownscalingDataModule(
        data_dir=data_dir,
        target_vars=target_vars,
        batch_size=1,
        num_workers=0,
        nn_lowres=False,
        static_vars=static_vars,
        metadata_file_name="metadata.csv",
    )
    dm.setup("test")
    n = len(dm.data_test)
    if n < 1:
        raise RuntimeError("Test split is empty")
    if sample_index is None:
        g = torch.Generator()
        g.manual_seed(seed)
        sample_index = int(torch.randint(0, n, (1,), generator=g).item())
    if not (0 <= sample_index < n):
        raise IndexError(f"sample_index={sample_index} out of range [0, {n})")
    batch = dm.data_test[sample_index]
    _log(f"Test set size={n}; using sample_index={sample_index}")
    low_res, y_hr, static, ts = batch
    return low_res.unsqueeze(0), y_hr.unsqueeze(0), static.unsqueeze(0), int(ts), sample_index, n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LMM schedule debug on one test sample")
    parser.add_argument("--n-steps", type=int, default=3)
    parser.add_argument(
        "--schedule",
        choices=list(_SCHEDULES),
        default="uniform",
        help="uniform: linspace(1,0); inv_ni: r_i=1/(n_steps*i)",
    )
    parser.add_argument("--ckpt", type=str, default=os.environ.get("LMM_CKPT", ""))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    if args.n_steps < 1:
        parser.error("--n-steps must be >= 1")

    L.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)
    _log(f"device={device} schedule={args.schedule} n_steps={args.n_steps} seed={args.seed}")

    data_dir = _resolve_data_dir(args.data_dir)
    _log(f"data_dir={data_dir}")

    low_res, y_hr, static, ts, idx, n_test = _one_test_batch(data_dir, args.sample_index, args.seed)
    _log(f"timestamp_ns={ts} (test index {idx}/{n_test})")
    _log(f"tensor shapes: low_res={tuple(low_res.shape)} y_hr={tuple(y_hr.shape)} static={tuple(static.shape)}")

    print_schedule(args.schedule, args.n_steps, device, low_res.dtype)

    ckpt = Path(args.ckpt) if args.ckpt else Path()
    if not args.ckpt:
        ckpt = _REPO / "logs/train/runs/2026-04-24_18-27-35/checkpoints/last.ckpt"
    model = _load_lmm(ckpt, device)

    low_res = low_res.to(device)
    y_hr = y_hr.to(device)
    static = static.to(device)

    _, predict_fn = _SCHEDULES[args.schedule]
    _log("Running multi-step inference...")
    y_hat = predict_fn(model, low_res, static, y_hr, args.n_steps)
    mae = (y_hat - y_hr).abs().mean().item()
    _log(f"Done. y_hat shape={tuple(y_hat.shape)} MAE vs y_hr={mae:.6f} (normalized space)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
