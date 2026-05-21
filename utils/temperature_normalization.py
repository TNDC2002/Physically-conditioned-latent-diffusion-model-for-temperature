"""CMCC / ERA5 temperature normalization (same ``normalization_data.pkl`` as LDM & LMM training)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np

# COSMO-CLM high-res fields in the downscaling pipeline use the ``CMCC`` statistics.
COSMO_SOURCE = "CMCC"
ERA5_SOURCE = "ERA5"


def default_normalization_pickle(start: Path | None = None) -> Path:
    """Resolve ``normalization_data.pkl`` under ``LDM-downscaling/full_Dataset``."""
    if start is None:
        start = Path(__file__).resolve().parents[1]
    candidates = [
        start / "LDM-downscaling" / "full_Dataset" / "normalization_data.pkl",
        start / "LDM-downscaling" / "normalization_data.pkl",
        start / "full_Dataset" / "normalization_data.pkl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "normalization_data.pkl not found. Tried:\n  " + "\n  ".join(str(p) for p in candidates)
    )


def load_norm_stats(path: Path | str | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load ``norm_values`` dict (keys ``CMCC``, ``ERA5``; each has ``mean``/``std`` per variable)."""
    if path is None:
        path = default_normalization_pickle(repo_root)
    with open(path, "rb") as f:
        return pickle.load(f)


def _stats(
    norm_values: Mapping[str, Any],
    variable: str,
    source: str,
) -> tuple[float, float]:
    try:
        mean = float(norm_values[source]["mean"][variable])
        std = float(norm_values[source]["std"][variable])
    except KeyError as exc:
        raise KeyError(f"Missing norm stats for source={source!r}, variable={variable!r}") from exc
    if std <= 0:
        raise ValueError(f"Non-positive std for {source}/{variable}: {std}")
    return mean, std


def normalize_temperature(
    field: np.ndarray,
    variable: str = "2mT",
    source: str = COSMO_SOURCE,
    norm_values: Mapping[str, Any] | None = None,
    *,
    norm_pickle: Path | str | None = None,
    repo_root: Path | None = None,
) -> np.ndarray:
    """``(T - mean) / std`` — same scaling as ``DownscalingDataset`` / inference batches."""
    if norm_values is None:
        norm_values = load_norm_stats(norm_pickle, repo_root=repo_root)
    mean, std = _stats(norm_values, variable, source)
    return (np.asarray(field, dtype=float) - mean) / std


def denormalize_temperature(
    field_norm: np.ndarray,
    variable: str = "2mT",
    source: str = COSMO_SOURCE,
    norm_values: Mapping[str, Any] | None = None,
    *,
    norm_pickle: Path | str | None = None,
    repo_root: Path | None = None,
) -> np.ndarray:
    """``T_norm * std + mean`` — inverse of :func:`normalize_temperature`."""
    if norm_values is None:
        norm_values = load_norm_stats(norm_pickle, repo_root=repo_root)
    mean, std = _stats(norm_values, variable, source)
    return np.asarray(field_norm, dtype=float) * std + mean


def norm_stats_summary(
    norm_values: Mapping[str, Any],
    variable: str = "2mT",
    source: str = COSMO_SOURCE,
) -> MutableMapping[str, float]:
    mean, std = _stats(norm_values, variable, source)
    return {"source": source, "variable": variable, "mean": mean, "std": std}
