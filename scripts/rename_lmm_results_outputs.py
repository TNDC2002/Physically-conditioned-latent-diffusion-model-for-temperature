#!/usr/bin/env python3
"""
Rename LMM model keys in saved inference pickles to the human-readable convention and refresh timing plots.

**Target names** (see ``utils/lmm_naming.py`` and ``notebooks/models_inference.ipynb``)::

    LMM_PDE_res_055_1     # filename contains ``epoch_055`` (etc.), 1 MeanFlow step
    LMM_PDE_res_last_3   # no epoch in filename (e.g. ``last.ckpt``), 3 steps

**Sources migrated**

1. Long keys from an older notebook layout::

       LMM_PDE_res_best_LMM_res_2mT_epoch_055_steps_1

2. Legacy index keys (short-lived)::

       LMM_PDE_resw0s1

   These are remapped using **sorted** checkpoint stems parsed from any long-form keys still
   present in the same ``model`` column (same rule as the old script). If the file only
   contains ``LMM_PDE_resw*`` keys and no long slugs, remap is skipped for those rows
   (restore from ``.pkl.bak`` or re-run inference).

Updates under ``--outputs-dir`` when present::

    Our_results_trained_models_<target>.pkl   — column ``model``
    inference_timing_trained_models_<target>.pkl — column ``model``

``LMM_PDE_res_steps_<n>`` (single-checkpoint) is unchanged.

Pickles may reference ``torch.Tensor``; use the same environment as inference (torch + pandas + matplotlib).

Examples::

    python scripts/rename_lmm_results_outputs.py --dry-run
    python scripts/rename_lmm_results_outputs.py --target-vars 2mT UV
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.lmm_naming import lmm_checkpoint_tag, lmm_multickpt_model_key  # noqa: E402

# Long-form: full checkpoint stem before ``_steps_<n>``.
_LONG_LMM = re.compile(r"^LMM_PDE_res_(.+)_steps_(\d+)$")
# Legacy index form from a previous notebook revision.
_LEGACY_W = re.compile(r"^LMM_PDE_resw(\d+)s(\d+)$")
# Intermediate naming: ``LMM_PDE_res_best_055_1`` → ``LMM_PDE_res_055_1``.
_LEGACY_BEST_PREFIX = re.compile(r"^LMM_PDE_res_best_(\d+)_(\d+)$")


def _sorted_slugs_from_models(models: list[str]) -> list[str]:
    found: set[str] = set()
    for m in models:
        mo = _LONG_LMM.match(str(m))
        if mo:
            found.add(mo.group(1))
    return sorted(found)


def build_rename_map(models: list[str]) -> dict[str, str]:
    """Map old LMM keys -> ``LMM_PDE_res_<epoch|last>_<n>``."""
    out: dict[str, str] = {}
    slugs = _sorted_slugs_from_models(models)
    tags = [lmm_checkpoint_tag(f"_/{s}.ckpt") for s in slugs]

    for ms in models:
        ms = str(ms)
        if ms == "LMM_PDE_res" or ms.startswith("LMM_PDE_res_steps_"):
            continue

        mb = _LEGACY_BEST_PREFIX.match(ms)
        if mb:
            new = f"LMM_PDE_res_{mb.group(1)}_{mb.group(2)}"
            if new != ms:
                out[ms] = new
            continue

        mo = _LONG_LMM.match(ms)
        if mo:
            slug, n = mo.group(1), mo.group(2)
            new = lmm_multickpt_model_key(f"_/{slug}.ckpt", int(n))
            if new != ms:
                out[ms] = new
            continue

        mw = _LEGACY_W.match(ms)
        if mw and slugs:
            widx, n = int(mw.group(1)), mw.group(2)
            if widx < len(tags):
                new = f"LMM_PDE_res_{tags[widx]}_{n}"
                if new != ms:
                    out[ms] = new

    return out


def apply_rename(df: pd.DataFrame, col: str, mapping: dict[str, str]) -> tuple[pd.DataFrame, int]:
    if not mapping or col not in df.columns:
        return df, 0
    before = df[col].astype(str)
    after = before.replace(mapping)
    n = int((before != after).sum())
    out = df.copy()
    out[col] = after
    return out, n


def replot_inference_timing(models: list[str], seconds: list[float], title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.bar(models, seconds)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Inference Time [s]")
    ax.tick_params(axis="x", rotation=35)
    for i, v in enumerate(seconds):
        ax.text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=9)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def process_target(
    outputs_dir: Path,
    target_var: str,
    *,
    dry_run: bool,
    backup: bool,
    do_plot: bool,
) -> int:
    results_path = outputs_dir / f"Our_results_trained_models_{target_var}.pkl"
    timing_path = outputs_dir / f"inference_timing_trained_models_{target_var}.pkl"
    plot_path = outputs_dir / f"inference_timing_trained_models_{target_var}.png"

    changed = 0
    if not results_path.is_file():
        print(f"[skip] missing {results_path}", file=sys.stderr)
        return 0

    try:
        results_df = pd.read_pickle(results_path)
    except ModuleNotFoundError as exc:
        print(
            f"[{target_var}] cannot load {results_path}: {exc}\n"
            "  Use the project Python env (with torch installed).",
            file=sys.stderr,
        )
        return 0

    try:
        timing_df = pd.read_pickle(timing_path) if timing_path.is_file() else None
    except ModuleNotFoundError as exc:
        print(f"[{target_var}] cannot load {timing_path}: {exc}", file=sys.stderr)
        timing_df = None

    models_list = results_df["model"].astype(str).unique().tolist()
    mapping = build_rename_map(models_list)
    legacy_w = [m for m in models_list if _LEGACY_W.match(str(m))]
    if legacy_w and not _sorted_slugs_from_models(models_list):
        print(
            f"[{target_var}] warning: found {len(legacy_w)} ``LMM_PDE_resw*`` keys but no long "
            "``*_steps_*`` slugs in this file; cannot infer checkpoint tags. Restore ``.pkl.bak`` "
            "or re-run inference.",
            file=sys.stderr,
        )

    if not mapping:
        print(f"[{target_var}] no LMM keys to rename.")
    else:
        print(f"[{target_var}] rename map ({len(mapping)} entries):")
        for old, new in sorted(mapping.items(), key=lambda x: x[0]):
            print(f"  {old} -> {new}")

    if dry_run:
        print(f"[{target_var}] dry-run: not writing files.")
        return 0

    if mapping:
        if backup:
            shutil.copy2(results_path, results_path.with_suffix(".pkl.bak"))
            if timing_df is not None:
                shutil.copy2(timing_path, timing_path.with_suffix(".pkl.bak"))

        results_df, n1 = apply_rename(results_df, "model", mapping)
        changed += n1
        results_df.to_pickle(results_path)
        print(f"[{target_var}] wrote {results_path} ({n1} rows relabelled).")

        if timing_df is not None:
            timing_df, n2 = apply_rename(timing_df, "model", mapping)
            changed += n2
            timing_df.to_pickle(timing_path)
            print(f"[{target_var}] wrote {timing_path} ({n2} rows relabelled).")

    if do_plot and timing_df is not None:
        if "timing_scope" in timing_df.columns:
            inf = timing_df[timing_df["timing_scope"] == "model_inference"]
        else:
            inf = timing_df
        inf = inf.sort_values("inference_seconds", ascending=False).reset_index(drop=True)
        replot_inference_timing(
            inf["model"].tolist(),
            inf["inference_seconds"].tolist(),
            f"Model Inference Time ({target_var})",
            plot_path,
        )
        print(f"[{target_var}] saved timing plot {plot_path}")
    elif do_plot and timing_df is None:
        print(f"[{target_var}] skip plot: missing {timing_path}", file=sys.stderr)

    return changed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing Our_results_*.pkl and inference_timing_*.pkl",
    )
    p.add_argument(
        "--target-vars",
        nargs="+",
        default=["2mT", "UV"],
        help="Target variable suffixes in filenames",
    )
    p.add_argument("--dry-run", action="store_true", help="Print mapping only; do not write files")
    p.add_argument("--no-backup", action="store_true", help="Do not write .pkl.bak copies before overwrite")
    p.add_argument("--no-plot", action="store_true", help="Skip regenerating inference_timing_*.png")

    args = p.parse_args()
    outputs_dir = args.outputs_dir.resolve()
    if not outputs_dir.is_dir():
        print(f"Outputs directory not found: {outputs_dir}", file=sys.stderr)
        return 1

    total = 0
    for tv in args.target_vars:
        total += process_target(
            outputs_dir,
            tv,
            dry_run=args.dry_run,
            backup=not args.no_backup,
            do_plot=not args.no_plot,
        )
    print(f"Done. Rows touched (model column): {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
