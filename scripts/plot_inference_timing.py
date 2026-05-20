#!/usr/bin/env python3
"""
Plot inference timing from ``inference_timing_trained_models_<target>.pkl`` (see ``notebooks/models_inference.ipynb``).

The pickle is a pandas DataFrame (``all_timing_df``) with columns such as ``model``, ``inference_seconds``,
``seconds_per_timestep``, ``timing_scope`` (``model_inference`` vs ``baseline_processing``), and ``target_var``.

**Choose models**

1. Edit ``SELECTED_MODELS`` below (list of strings, or ``None`` for all model runs). Bars are ordered by **inference time (slowest first)**, same as the notebook.
2. Optionally set ``MODEL_DISPLAY_NAMES`` to the same length: bar labels on the plot (pickle ``model`` keys stay in ``SELECTED_MODELS``).
3. Or pass ``--models`` on the command line (overrides ``SELECTED_MODELS``; labels are the pickle names).

Examples::

    python scripts/plot_inference_timing.py \\
        --pickle outputs/inference_timing_trained_models_2mT.pkl

    python scripts/plot_inference_timing.py \\
        --pickle outputs/inference_timing_trained_models_2mT.pkl \\
        --models UNET GAN LDM_PDE_res

    python scripts/plot_inference_timing.py \\
        --pickle outputs/inference_timing_trained_models_2mT.pkl \\
        --models LMM_PDE_res_last_1,LMM_PDE_res_016_3 \\
        --output outputs/my_timing_subset.png

    # By default the figure is written next to the pickle (``…2mT.pkl`` -> ``…2mT.png``).
    # Use ``--no-save`` to skip writing; use ``--no-show`` to skip ``plt.show()`` (e.g. batch jobs).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Optional: set models here when you prefer editing lists to CLI flags.
# Use SELECTED_MODELS=None to plot every row in the chosen timing scope (unless --models is set).
#
# MODEL_DISPLAY_NAMES: same length as SELECTED_MODELS; strings shown on the x-axis.
# Use None to show the pickle ``model`` column as the label.
#
# Example:
#   SELECTED_MODELS = ["UNET", "GAN", "LDM_res", "LDM_PDE_res", "LMM_PDE_res_016_1"]
#   MODEL_DISPLAY_NAMES = ["UNET", "GAN", "LDM_res", "LDM_PDE_res", "LMM_PDE_res"]
# ---------------------------------------------------------------------------
SELECTED_MODELS: list[str] | None = None
MODEL_DISPLAY_NAMES: list[str] | None = None

SELECTED_MODELS = ["UNET", "GAN", "LDM_res", "LDM_PDE_res", "LMM_PDE_res_016_1"]
MODEL_DISPLAY_NAMES = ["UNET", "GAN", "LDM_res", "LDM_PDE_res", "LMM_PDE_res"]


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_models_arg(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    out: list[str] = []
    for part in raw:
        for token in part.replace(",", " ").split():
            t = token.strip()
            if t:
                out.append(t)
    return out or None


def _filter_timing_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all" or "timing_scope" not in df.columns:
        return df
    return df[df["timing_scope"].astype(str) == scope].copy()


def _sort_by_inference_time(df: pd.DataFrame) -> pd.DataFrame:
    """Match ``notebooks/models_inference.ipynb``: slowest model leftmost (descending seconds)."""
    return df.sort_values("inference_seconds", ascending=False).reset_index(drop=True)


def plot_timing_bars(
    models: list[str],
    seconds: list[float],
    title: str,
    *,
    out_path: Path | None,
    do_show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.bar(models, seconds)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Inference Time [s]")
    ax.tick_params(axis="x", rotation=35)
    for i, v in enumerate(seconds):
        ax.text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=9)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {out_path}")
    if do_show and os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--pickle",
        type=Path,
        default=_REPO_ROOT / "outputs" / "inference_timing_trained_models_2mT.pkl",
        help="Path to inference_timing_trained_models_*.pkl",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Subset of model names (overrides SELECTED_MODELS). Bar labels match pickle names; file MODEL_DISPLAY_NAMES is ignored.",
    )
    p.add_argument(
        "--timing-scope",
        choices=("model_inference", "baseline_processing", "all"),
        default="model_inference",
        help="Row filter (matches notebook bar chart default: model_inference only).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="PNG path (dpi=180, bbox_inches=tight). Default: same path as --pickle with .png suffix.",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write a PNG (only useful with a working display and default show behavior).",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Skip plt.show() (recommended for batch/SSH without a display).",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title (default: inferred from target_var in the table).",
    )
    args = p.parse_args()

    pkl = args.pickle.expanduser().resolve()
    if not pkl.is_file():
        print(f"Pickle not found: {pkl}", file=sys.stderr)
        return 1

    df = pd.read_pickle(pkl)
    if "model" not in df.columns or "inference_seconds" not in df.columns:
        print("Expected columns 'model' and 'inference_seconds' in pickle.", file=sys.stderr)
        return 1

    df = _filter_timing_scope(df, args.timing_scope)
    if df.empty:
        print(f"No rows after timing_scope={args.timing_scope!r}.", file=sys.stderr)
        return 1

    cli_models = _parse_models_arg(args.models)
    want = cli_models if cli_models is not None else SELECTED_MODELS

    if cli_models is None:
        if MODEL_DISPLAY_NAMES is not None and SELECTED_MODELS is None:
            print(
                "MODEL_DISPLAY_NAMES is set but SELECTED_MODELS is None; set both lists (same length), "
                "or set MODEL_DISPLAY_NAMES to None.",
                file=sys.stderr,
            )
            return 1
        if (
            MODEL_DISPLAY_NAMES is not None
            and SELECTED_MODELS is not None
            and len(MODEL_DISPLAY_NAMES) != len(SELECTED_MODELS)
        ):
            print(
                f"MODEL_DISPLAY_NAMES length ({len(MODEL_DISPLAY_NAMES)}) must match "
                f"SELECTED_MODELS ({len(SELECTED_MODELS)}).",
                file=sys.stderr,
            )
            return 1

    if want is not None:
        missing = [m for m in want if m not in set(df["model"].astype(str))]
        if missing:
            print(f"Warning: not found in filtered table (skipped): {missing}", file=sys.stderr)
        df = df[df["model"].astype(str).isin(want)]
        if df.empty:
            print("No rows left after --models / SELECTED_MODELS filter.", file=sys.stderr)
            return 1

    df = _sort_by_inference_time(df)

    target_var = str(df["target_var"].iloc[0]) if "target_var" in df.columns else "?"
    title = args.title or f"Model Inference Time ({target_var})"
    if args.timing_scope != "model_inference":
        title = f"{title} [{args.timing_scope}]"

    model_keys = df["model"].astype(str).tolist()
    if cli_models is not None or MODEL_DISPLAY_NAMES is None:
        bar_labels = model_keys
    else:
        label_map = dict(zip(SELECTED_MODELS or [], MODEL_DISPLAY_NAMES or []))
        bar_labels = [label_map.get(m, m) for m in model_keys]

    if args.no_save:
        out_path: Path | None = None
    elif args.output is not None:
        out_path = args.output.expanduser().resolve()
    else:
        out_path = pkl.with_suffix(".png")

    plot_timing_bars(
        bar_labels,
        df["inference_seconds"].astype(float).tolist(),
        title,
        out_path=out_path,
        do_show=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
