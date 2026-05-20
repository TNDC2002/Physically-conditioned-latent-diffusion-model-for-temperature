"""Load a small slice of inference results without re-reading the full pickle every time.

The monolithic ``Our_results_trained_models_*.pkl`` (multi-GB) cannot be partially
unpickled. Workflow:

1. Run ``load_results_subset(...)`` once — reads the full pickle **once**, writes a
   compact cache (numpy fields + parquet metadata + JSON manifest).
2. Later notebook runs load only the cache (~tens of MiB for a few models/times).

Also writes/reads a lightweight ``*.index.json`` (models + timestamps, no tensors).
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

TimeLike = Union[str, pd.Timestamp]


def _as_timestamp(ts: TimeLike) -> pd.Timestamp:
    return pd.to_datetime(ts)


def _normalize_times(times: Sequence[TimeLike]) -> List[str]:
    return sorted({_as_timestamp(t).isoformat() for t in times})


def _normalize_models(models: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(models))


def subset_config_hash(
    *,
    source_path: Path,
    models: Sequence[str],
    times: Sequence[TimeLike],
    target_var: str,
    plot_var: str,
) -> str:
    payload = {
        "source": str(source_path.resolve()),
        "source_mtime": source_path.stat().st_mtime if source_path.is_file() else None,
        "models": _normalize_models(models),
        "times": _normalize_times(times),
        "target_var": target_var,
        "plot_var": plot_var,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def index_path_for(source_pkl: Path) -> Path:
    return source_pkl.with_suffix(".index.json")


def cache_dir_for(source_pkl: Path, config_hash: str) -> Path:
    return source_pkl.parent / f"{source_pkl.stem}_subset_{config_hash}"


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "values"):
        value = value.values
    arr = np.asarray(value, dtype=np.float32).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D field, got shape {arr.shape}")
    return arr


def load_results_index(source_pkl: Path) -> Optional[Dict[str, Any]]:
    path = index_path_for(source_pkl)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_results_index(source_pkl: Path, df: pd.DataFrame) -> Path:
    meta = df.drop(columns=["spat_distr"], errors="ignore")
    index = {
        "source_pickle": str(source_pkl.resolve()),
        "source_size_bytes": source_pkl.stat().st_size,
        "models": sorted(meta["model"].unique().tolist()),
        "time_steps": sorted(pd.to_datetime(meta["time_step"]).astype(str).unique().tolist()),
        "target_vars": sorted(meta["target_var"].unique().tolist()),
        "variables": sorted(meta["variable"].unique().tolist()),
        "n_rows": int(len(meta)),
    }
    path = index_path_for(source_pkl)
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return path


def _load_full_pickle(source_pkl: Path) -> pd.DataFrame:
    if not source_pkl.is_file():
        raise FileNotFoundError(f"Results file not found: {source_pkl}")
    size_mb = source_pkl.stat().st_size / (1024 ** 2)
    print(f"Reading full pickle ({size_mb:.1f} MiB): {source_pkl}")
    try:
        df = pd.read_pickle(source_pkl)
    except Exception as exc:
        if "truncated" in str(exc).lower():
            raise RuntimeError(
                f"Pickle at {source_pkl} is incomplete ({size_mb:.1f} MiB). "
                "Wait for OneDrive sync or regenerate via models_inference.ipynb."
            ) from exc
        raise
    df["time_step"] = pd.to_datetime(df["time_step"])
    return df


def _filter_results_df(
    df: pd.DataFrame,
    *,
    models: Sequence[str],
    times: Sequence[TimeLike],
    target_var: str,
    plot_var: str,
) -> pd.DataFrame:
    models = _normalize_models(models)
    times = [_as_timestamp(t) for t in times]
    sub = df[
        (df["target_var"] == target_var)
        & (df["variable"] == plot_var)
        & (df["model"].isin(models))
        & (df["time_step"].isin(times))
    ].copy()
    if sub.empty:
        raise ValueError(
            "No rows matched the requested models/times. "
            f"models={models}, times={[str(t) for t in times]}"
        )
    missing_models = set(models) - set(sub["model"].unique())
    missing_times = {t.isoformat() for t in times} - {
        pd.Timestamp(t).isoformat() for t in sub["time_step"].unique()
    }
    if missing_models:
        present = sorted(sub["model"].unique()) if not sub.empty else []
        raise ValueError(
            f"Missing models in pickle: {sorted(missing_models)}. "
            f"Present for target={target_var!r}, variable={plot_var!r}: {present}"
        )
    if missing_times:
        raise ValueError(f"Missing timestamps in pickle: {sorted(missing_times)}")
    return sub


def _write_subset_cache(sub: pd.DataFrame, cache_dir: Path, manifest: Dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    meta_rows = []
    for i, row in sub.iterrows():
        key = (
            f"{row['model']}__{pd.Timestamp(row['time_step']).strftime('%Y%m%d_%H%M%S')}"
        )
        arrays[key] = _tensor_to_numpy(row["spat_distr"])
        meta_rows.append(
            {
                "array_key": key,
                "input_var": row.get("input_var"),
                "target_var": row["target_var"],
                "model": row["model"],
                "variable": row["variable"],
                "min": float(row["min"]),
                "max": float(row["max"]),
                "time_step": pd.Timestamp(row["time_step"]).isoformat(),
            }
        )
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(cache_dir / "meta.csv", index=False)
    np.savez_compressed(cache_dir / "fields.npz", **arrays)
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _read_subset_cache(cache_dir: Path) -> pd.DataFrame:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    meta_df = pd.read_csv(cache_dir / "meta.csv")
    arrays = np.load(cache_dir / "fields.npz")
    spat = [arrays[k] for k in meta_df["array_key"]]
    out = meta_df.copy()
    out["time_step"] = pd.to_datetime(out["time_step"])
    out["spat_distr"] = spat
    return out, manifest


def build_subset_cache(
    source_pkl: Path,
    *,
    models: Sequence[str],
    times: Sequence[TimeLike],
    target_var: str = "2mT",
    plot_var: str = "2mT",
    also_write_index: bool = True,
) -> Tuple[pd.DataFrame, Path]:
    """One-time heavy pass: full pickle -> compact per-subset cache directory."""
    df = _load_full_pickle(source_pkl)
    if also_write_index:
        save_results_index(source_pkl, df)
        print(f"Wrote index: {index_path_for(source_pkl)}")
    sub = _filter_results_df(
        df, models=models, times=times, target_var=target_var, plot_var=plot_var
    )
    del df

    cfg_hash = subset_config_hash(
        source_path=source_pkl,
        models=models,
        times=times,
        target_var=target_var,
        plot_var=plot_var,
    )
    cache_dir = cache_dir_for(source_pkl, cfg_hash)
    manifest = {
        "config_hash": cfg_hash,
        "source_pickle": str(source_pkl.resolve()),
        "source_mtime": source_pkl.stat().st_mtime,
        "target_var": target_var,
        "plot_var": plot_var,
        "models": _normalize_models(models),
        "times": _normalize_times(times),
        "n_rows": int(len(sub)),
    }
    _write_subset_cache(sub, cache_dir, manifest)
    size_mb = sum(f.stat().st_size for f in cache_dir.iterdir()) / (1024 ** 2)
    print(f"Wrote subset cache ({size_mb:.1f} MiB, {len(sub)} rows): {cache_dir}")
    sub = sub.copy()
    sub["spat_distr"] = [_tensor_to_numpy(v) for v in sub["spat_distr"]]
    return sub, cache_dir


def load_results_subset(
    source_pkl: Path,
    *,
    models: Sequence[str],
    times: Sequence[TimeLike],
    target_var: str = "2mT",
    plot_var: str = "2mT",
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load only selected models/times — from cache if available, else build cache once."""
    cfg_hash = subset_config_hash(
        source_path=source_pkl,
        models=models,
        times=times,
        target_var=target_var,
        plot_var=plot_var,
    )
    cache_dir = cache_dir_for(source_pkl, cfg_hash)
    manifest_path = cache_dir / "manifest.json"

    if (
        not force_rebuild
        and manifest_path.is_file()
        and (cache_dir / "fields.npz").is_file()
        and (cache_dir / "meta.csv").is_file()
    ):
        out, manifest = _read_subset_cache(cache_dir)
        if source_pkl.is_file() and manifest.get("source_mtime") != source_pkl.stat().st_mtime:
            print("Source pickle changed on disk — rebuilding subset cache.")
            force_rebuild = True
        else:
            size_mb = sum(f.stat().st_size for f in cache_dir.iterdir()) / (1024 ** 2)
            print(f"Loaded subset cache ({size_mb:.1f} MiB, {len(out)} rows): {cache_dir}")
            return out

    out, _ = build_subset_cache(
        source_pkl,
        models=models,
        times=times,
        target_var=target_var,
        plot_var=plot_var,
    )
    return out


def fig_snapshots_color_limits(df_ts: pd.DataFrame, variable: str = "2mT") -> Tuple[float, float]:
    """Match ``utils.plotting_utils.show_snapshots`` 2mT vmin/vmax for one timestamp."""
    sub = df_ts[df_ts["variable"] == variable]
    if sub.empty:
        raise ValueError(f"No rows for variable={variable!r}")
    min_value = float(sub["min"].min())
    max_value = float(sub["max"].max())
    if min_value < 0 and max_value > 0:
        bound = max(abs(max_value), abs(min_value))
        min_value, max_value = -bound, bound
    return min_value, max_value


def fig_snapshots_zoom_slice() -> Tuple[slice, slice]:
    """Pixel slices matching Fig_snapshots zoom panel (x/y EPSG:3035 limits)."""
    ys = np.arange(2697000, 1354000 - 1, -2000)
    xs = np.arange(3911000, 5062000, 2000)
    y_lo, y_hi = 1748000, 2070000
    x_lo, x_hi = 4150000, 4450000
    row_idx = np.where((ys >= y_lo) & (ys <= y_hi))[0]
    col_idx = np.where((xs >= x_lo) & (xs <= x_hi))[0]
    if len(row_idx) == 0 or len(col_idx) == 0:
        raise ValueError("Could not map Fig_snapshots zoom limits to grid indices.")
    return slice(int(row_idx[0]), int(row_idx[-1]) + 1), slice(int(col_idx[0]), int(col_idx[-1]) + 1)
