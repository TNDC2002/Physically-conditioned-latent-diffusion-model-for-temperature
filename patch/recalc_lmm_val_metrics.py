#!/usr/bin/env python3
"""Recalculate LMM validation metrics and write them to TensorBoard.

Example:
  .venv/bin/python patch/recalc_lmm_val_metrics.py \
    --run-dir logs/train/runs/2026-04-22_05-45-02 \
    --ckpt-path logs/train/runs/2026-04-20_13-30-10/checkpoints/last.ckpt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import pyrootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _discover_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    all_ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return all_ckpts[0] if all_ckpts else None


def _discover_all_checkpoints(run_dir: Path) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    # Prefer epoch checkpoints; skip "last.ckpt" so each point is a distinct epoch.
    epoch_ckpts = sorted(ckpt_dir.glob("*epoch*.ckpt"), key=lambda p: p.stat().st_mtime)
    if epoch_ckpts:
        return epoch_ckpts
    # Fallback: all checkpoints except last.
    return sorted([p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"], key=lambda p: p.stat().st_mtime)


def _epoch_from_checkpoint(ckpt_path: Path) -> int:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", None)
    if epoch is None:
        return 0
    return int(epoch)


def _load_cfg(run_dir: Path) -> DictConfig:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Run config not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


def _resolve_cfg(cfg: DictConfig, run_dir: Path) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    root_dir = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PROJECT_ROOT", str(root_dir))
    cfg.paths.output_dir = str(run_dir)
    cfg.paths.work_dir = str(root_dir)
    return cfg


def recalc_validation_metrics(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    cfg = _resolve_cfg(_load_cfg(run_dir), run_dir)

    if args.ckpt_path:
        ckpt_paths = [Path(args.ckpt_path).resolve()]
    else:
        ckpt_paths = _discover_all_checkpoints(run_dir)
        if not ckpt_paths:
            latest = _discover_checkpoint(run_dir)
            if latest is not None:
                ckpt_paths = [latest]

    if not ckpt_paths:
        raise FileNotFoundError(
            "No checkpoint found for validation recalculation.\n"
            f"- looked in: {run_dir / 'checkpoints'}\n"
            "Provide one explicitly with --ckpt-path."
        )
    for p in ckpt_paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    tb_dir = run_dir / "tensorboard_recalc" / args.version
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    trainer_kwargs: dict[str, Any] = {
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": True,
        "accelerator": args.accelerator or cfg.trainer.accelerator,
        "devices": args.devices if args.devices is not None else cfg.trainer.devices,
    }
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches

    print(f"TensorBoard dir : {tb_dir}")
    print(f"Found {len(ckpt_paths)} checkpoint(s) for recalculation.")

    for ckpt_path in ckpt_paths:
        trainer = Trainer(**trainer_kwargs)
        metrics = trainer.validate(model=model, datamodule=datamodule, ckpt_path=str(ckpt_path), verbose=True)
        if not metrics:
            continue
        epoch_idx = _epoch_from_checkpoint(ckpt_path)
        metric_map = metrics[0]
        print(f"\nCheckpoint: {ckpt_path}")
        print(f"Epoch idx : {epoch_idx}")
        for k, v in metric_map.items():
            v_float = float(v)
            # epoch-level logging as requested.
            writer.add_scalar(k, v_float, global_step=epoch_idx)
            print(f"  {k}: {v_float}")

    writer.flush()
    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recalculate LMM validation metrics and log to TensorBoard.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to training run directory (must contain .hydra/config.yaml).",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Optional single checkpoint path. If omitted, script recalculates all epoch checkpoints in <run-dir>/checkpoints.",
    )
    parser.add_argument(
        "--version",
        default="version_0",
        help="TensorBoard version folder name under <run-dir>/tensorboard_recalc.",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        help="Override accelerator (e.g., gpu, cpu).",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Override devices count (e.g., 1).",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=None,
        help="Optional Lightning limit_val_batches override (e.g., 0.25 or 10).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    recalc_validation_metrics(parse_args())
