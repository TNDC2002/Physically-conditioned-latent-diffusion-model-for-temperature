#!/usr/bin/env python3
"""LMM smoke: LDM vs LMM latent parity, one-batch metrics, optional micro-train (plan L-K).

**Designed for GPU (e.g. one MIG slice).** On CPU-only hosts this script **exits immediately**
unless ``LMM_SMOKE_ALLOW_CPU=1`` (slow; not recommended).

**MIG cluster (recommended)**::

    sbatch configs/experiment/Submitscript_LMM_MIG.sh

**Interactive GPU** (login node with GPU or local CUDA)::

    export PROJECT_ROOT=$PWD
    ./.venv/bin/python scripts/lmm_smoke_suite.py paths.data_dir=$PROJECT_ROOT/LDM-downscaling/full_Dataset/

Env:

- ``LMM_SMOKE_DATA_DIR`` — overrides default ``paths.data_dir`` (also set by the MIG submit script).
- ``LMM_SMOKE_SKIP_TRAIN=1`` — parity + metrics only (faster).
- ``LMM_SMOKE_MAX_STEPS`` — cap Lightning steps for micro-train (submit script sets 10).
- ``LMM_SMOKE_QUIET=1`` — less console noise from Hydra/Lightning.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pyrootutils
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

_REPO = Path(__file__).resolve().parents[1]
pyrootutils.setup_root(str(_REPO), indicator=".project-root", pythonpath=True)

from lightning import seed_everything  # noqa: E402

from src import utils  # noqa: E402
from src.models.latent_residual_inputs import (  # noqa: E402
    build_latent_target_and_context_dict,
    context_dict_shapes_match,
    context_dict_structure_equal,
)
from src.train import train  # noqa: E402


def _patch_paths_for_compose(cfg, run_name: str = "lmm_smoke_compose") -> None:
    """``compose()`` has no Hydra job; replace ``hydra:runtime`` interpolations under ``paths``."""
    with open_dict(cfg.paths):
        cfg.paths.output_dir = str(_REPO / "logs" / run_name)
        cfg.paths.work_dir = str(_REPO)
    OmegaConf.resolve(cfg)


def _default_overrides() -> list[str]:
    data_dir = os.environ.get("LMM_SMOKE_DATA_DIR", str(_REPO / "LDM-downscaling"))
    return [
        "experiment=downscaling_LMM_res_2mT_smoke",
        "model=lmm",
        f"paths.pretrained_models_dir={_REPO}/pretrained_models/",
        f"paths.data_dir={data_dir}",
    ]


def _device_arg() -> torch.device:
    if torch.cuda.is_available() and os.environ.get("LMM_SMOKE_FORCE_CPU") != "1":
        return torch.device("cuda", 0)
    return torch.device("cpu")


def _gpu_only_or_bail() -> bool:
    """Return True if we should run heavy checks. False = write stub report and exit."""
    if torch.cuda.is_available() and os.environ.get("LMM_SMOKE_FORCE_CPU") != "1":
        return True
    if os.environ.get("LMM_SMOKE_ALLOW_CPU") == "1":
        print(
            "[lmm_smoke_suite] LMM_SMOKE_ALLOW_CPU=1: running on CPU (very slow). "
            "Prefer: sbatch configs/experiment/Submitscript_LMM_MIG.sh"
        )
        return True
    print(
        "[lmm_smoke_suite] No CUDA device — skipping parity, metrics, and training.\n"
        "  Run on GPU:  sbatch configs/experiment/Submitscript_LMM_MIG.sh\n"
        "  Or locally:  use a CUDA machine, then re-run this script.\n"
        "  Force (slow): LMM_SMOKE_ALLOW_CPU=1"
    )
    return False


def run_latent_and_context_parity(overrides: list[str], dev: torch.device) -> dict:
    """Compare LMM vs LDM latent ``z_R`` (same VAE weights); context keys/shapes; optional tensor parity."""
    pm = _REPO / "pretrained_models"
    ckpt_fix = [
        f"model.autoencoder.unet_regr.ckpt_path={pm}/UNET_2mT.ckpt",
        f"model.ae_load_state_file={pm}/VAE_residual_2mT.ckpt",
    ]
    shared = [o for o in overrides if not o.startswith("experiment=") and not o.startswith("model=")] + ckpt_fix
    cfg_dir = str(_REPO / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg_lmm = compose(
            config_name="train",
            overrides=shared + ["experiment=downscaling_LMM_res_2mT_smoke", "model=lmm"],
        )
        cfg_ldm = compose(
            config_name="train",
            overrides=shared + ["experiment=downscaling_LDM_res_2mT", "model=ldm"],
        )
    _patch_paths_for_compose(cfg_lmm, "lmm_smoke_lmm")
    _patch_paths_for_compose(cfg_ldm, "lmm_smoke_ldm")

    lmm = instantiate(cfg_lmm.model).to(dev)
    ldm = instantiate(cfg_ldm.model).to(dev)
    lmm.eval()
    ldm.eval()

    dm = instantiate(cfg_lmm.data)
    dm.prepare_data()
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    batch = tuple(b.to(dev) if isinstance(b, torch.Tensor) else b for b in batch)

    z_lmm, c_lmm = build_latent_target_and_context_dict(
        lmm.autoencoder, lmm.context_encoder, lmm.conditional, batch
    )
    z_ldm, c_ldm = build_latent_target_and_context_dict(
        ldm.autoencoder, ldm.context_encoder, ldm.conditional, batch
    )

    ok_z = torch.allclose(z_lmm, z_ldm, rtol=0.0, atol=1e-5)
    ok_shapes, shape_msg = context_dict_shapes_match(c_lmm, c_ldm)
    ok_ctx, ctx_msg = context_dict_structure_equal(c_lmm, c_ldm, rtol=0.0, atol=1e-5)

    return {
        "device": str(dev),
        "latent_allclose": bool(ok_z),
        "latent_max_abs": float((z_lmm - z_ldm).abs().max().item()) if z_lmm.numel() else 0.0,
        "context_shapes_match": bool(ok_shapes),
        "context_shapes_msg": shape_msg,
        "context_values_match_ldm_lmm": bool(ok_ctx),
        "context_values_msg": ctx_msg,
        "latent_shape": tuple(z_lmm.shape),
    }


def run_one_batch_metrics(overrides: list[str], dev: torch.device) -> dict:
    """Timing + MAE for prior vs LMM ``predict_final``; one LDM vs LMM ``shared_step`` timing pair."""
    cfg_dir = str(_REPO / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="train", overrides=overrides)
    _patch_paths_for_compose(cfg, "lmm_smoke_metrics")

    lmm = instantiate(cfg.model).to(dev)
    lmm.eval()
    dm = instantiate(cfg.data)
    dm.prepare_data()
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    x, y, z, _ts = tuple(b.to(dev) if isinstance(b, torch.Tensor) else b for b in batch)

    merged = lmm.autoencoder.nn_lr_and_merge_with_static(x, z)
    y_up = lmm.autoencoder.unet(merged)
    with torch.no_grad():
        t0 = time.perf_counter()
        y_hat = lmm.predict_final(x, z, y)
        t_lmm = time.perf_counter() - t0

    mae_prior = float((y_up - y).abs().mean().item())
    mae_lmm = float((y_hat - y).abs().mean().item())

    pm = _REPO / "pretrained_models"
    ld_over = [
        o
        for o in overrides
        if not o.startswith("experiment=") and not o.startswith("model=")
    ] + [
        "experiment=downscaling_LDM_res_2mT",
        "model=ldm",
        f"model.autoencoder.unet_regr.ckpt_path={pm}/UNET_2mT.ckpt",
        f"model.ae_load_state_file={pm}/VAE_residual_2mT.ckpt",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg_ldm = compose(config_name="train", overrides=ld_over)
    _patch_paths_for_compose(cfg_ldm, "lmm_smoke_ldm_timing")
    ldm = instantiate(cfg_ldm.model).to(dev)
    ldm.train()
    t0 = time.perf_counter()
    _ = ldm.shared_step(batch)
    t_ldm_step = time.perf_counter() - t0
    lmm.train()
    t0 = time.perf_counter()
    _ = lmm.shared_step(batch, create_graph=False)
    t_lmm_step = time.perf_counter() - t0

    return {
        "device": str(dev),
        "predict_final_seconds": t_lmm,
        "mae_prior_up_vs_y": mae_prior,
        "mae_lmm_final_vs_y": mae_lmm,
        "batch_spatial": tuple(y.shape[-2:]),
        "ldm_shared_step_seconds": t_ldm_step,
        "lmm_shared_step_seconds": t_lmm_step,
    }


def run_micro_train(overrides: list[str]) -> dict:
    cfg_dir = str(_REPO / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="train", overrides=overrides)
    _patch_paths_for_compose(cfg, "lmm_smoke_train")
    ms = os.environ.get("LMM_SMOKE_MAX_STEPS")
    if ms is not None:
        with open_dict(cfg.trainer):
            cfg.trainer.max_steps = int(ms)
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    if os.environ.get("LMM_SMOKE_QUIET") == "1" and "extras" in cfg:
        with open_dict(cfg.extras):
            cfg.extras.print_config = False
    utils.extras(cfg)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
    t0 = time.perf_counter()
    metric_dict, _ = train(cfg)
    elapsed = time.perf_counter() - t0

    def _scalarize(d):
        out = {}
        for k, v in d.items():
            try:
                if torch.is_tensor(v):
                    out[k] = float(v.detach().float().mean().cpu())
                else:
                    out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = str(v)
        return out

    return {"train_wall_seconds": elapsed, "last_metrics": _scalarize(dict(metric_dict))}


def main(argv: list[str]) -> int:
    os.environ.setdefault("PROJECT_ROOT", str(_REPO))
    overrides = _default_overrides() + (list(argv[1:]) if len(argv) > 1 else [])

    report_lines: list[str] = ["# LMM smoke suite report", f"overrides: {overrides}", ""]

    if not _gpu_only_or_bail():
        report_lines.append("status=skipped_no_cuda")
        out_path = _REPO / "logs" / "lmm_smoke_report.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"\n[lmm_smoke_suite] Wrote stub {out_path}")
        return 0

    dev = _device_arg()

    print(f"[lmm_smoke_suite] Latent + context parity (LMM vs LDM) on {dev}...")
    p = run_latent_and_context_parity(overrides, dev)
    for k, v in p.items():
        line = f"parity.{k}={v}"
        print(line)
        report_lines.append(line)
    if not p["latent_allclose"]:
        print("[lmm_smoke_suite] WARNING: latent tensors differ (check VAE paths / strictness).")
    if not p["context_shapes_match"]:
        print(f"[lmm_smoke_suite] WARNING: context shape mismatch: {p['context_shapes_msg']}")
    if not p["context_values_match_ldm_lmm"]:
        print(
            f"[lmm_smoke_suite] NOTE: context tensor values differ (expected if context_encoder "
            f"not loaded from same ckpt): {p['context_values_msg']}"
        )

    print("\n[lmm_smoke_suite] One-batch metrics...")
    m = run_one_batch_metrics(overrides, dev)
    for k, v in m.items():
        line = f"metrics.{k}={v}"
        print(line)
        report_lines.append(line)

    if os.environ.get("LMM_SMOKE_SKIP_TRAIN") == "1":
        print("\n[lmm_smoke_suite] Skipping micro-train (LMM_SMOKE_SKIP_TRAIN=1).")
    else:
        print("\n[lmm_smoke_suite] Micro-train (Lightning, GPU)...")
        tr = run_micro_train(overrides)
        for k, v in tr.items():
            line = f"train.{k}={v}"
            print(line)
            report_lines.append(line)

    out_path = _REPO / "logs" / "lmm_smoke_report.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[lmm_smoke_suite] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
