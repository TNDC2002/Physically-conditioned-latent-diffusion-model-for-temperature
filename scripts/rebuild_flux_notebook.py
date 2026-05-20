"""Regenerate notebooks/test_anisotropic_flux_loss.ipynb without bloated outputs."""
import json
from pathlib import Path


def cell_md(src: str):
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in src.split("\n")]}


def cell_code(src: str):
    lines = src.split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + ("\n" if i < len(lines) - 1 else "") for i, ln in enumerate(lines)],
    }


HELPERS = r'''
def as_array(value) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'values'):
        value = value.values
    arr = np.asarray(value, dtype=float).squeeze()
    if arr.ndim != 2:
        raise ValueError(f'Expected 2-D field, got shape {arr.shape}')
    return arr


def get_temperature_field(df, model: str, ts) -> np.ndarray:
    ts = pd.to_datetime(ts)
    row = df[(df['model'] == model) & (df['time_step'] == ts)]
    if row.empty:
        raise KeyError(f'No row for model={model!r}, time={ts}')
    return as_array(row.iloc[0]['spat_distr'])


def compute_gradients(T, dx=1.0, dy=1.0):
    T = np.asarray(T, dtype=float)
    H, W = T.shape
    dTdx = np.zeros_like(T)
    dTdy = np.zeros_like(T)
    if W > 2:
        dTdx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2.0 * dx)
    if H > 2:
        dTdy[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * dy)
    if W > 1:
        dTdx[:, 0] = (T[:, 1] - T[:, 0]) / dx
        dTdx[:, -1] = (T[:, -1] - T[:, -2]) / dx
    if H > 1:
        dTdy[0, :] = (T[1, :] - T[0, :]) / dy
        dTdy[-1, :] = (T[-1, :] - T[-2, :]) / dy
    return dTdx, dTdy


def compute_q_field(T, dx=1.0, dy=1.0, eps=1e-6):
    dTdx, dTdy = compute_gradients(T, dx=dx, dy=dy)
    grad_sq = dTdx ** 2 + dTdy ** 2
    qx = -grad_sq * dTdx
    qy = -grad_sq * dTdy
    q_mag = np.sqrt(qx ** 2 + qy ** 2 + eps)
    return qx, qy, q_mag, dTdx, dTdy


def flux_mse(q_pred_x, q_pred_y, q_gt_x, q_gt_y) -> float:
    return float(np.mean((q_pred_x - q_gt_x) ** 2 + (q_pred_y - q_gt_y) ** 2))


def validate_times(df, times):
    available_times = set(pd.to_datetime(df['time_step']))
    missing = [ts for ts in times if pd.to_datetime(ts) not in available_times]
    if missing:
        raise ValueError(f'Missing timestamps: {missing}')


def temperature_limits_for_timestamp(df, ts, variable=PLOT_VAR):
    if USE_FIG_SNAPSHOTS_STYLE:
        ts_df = df[pd.to_datetime(df['time_step']) == pd.to_datetime(ts)]
        return fig_snapshots_color_limits(ts_df, variable=variable)
    arrays = [as_array(v) for v in df[pd.to_datetime(df['time_step']) == pd.to_datetime(ts)]['spat_distr']]
    stacked = np.stack(arrays)
    return float(np.nanpercentile(stacked, 1)), float(np.nanpercentile(stacked, 99))
'''

PLOT_FUNCS = r'''
def plot_q_panel(T, qx, qy, ax_T, ax_qmag=None, title=None, stride=40, region_slice=None,
                 T_vmin=None, T_vmax=None, q_key_scale=None, cmap_T=CMAP_T):
    if region_slice is not None:
        T, qx, qy = T[region_slice], qx[region_slice], qy[region_slice]
    if T_vmin is None or T_vmax is None:
        T_vmin, T_vmax = float(np.nanmin(T)), float(np.nanmax(T))
    ax_T.imshow(T, origin='upper', cmap=cmap_T, vmin=T_vmin, vmax=T_vmax)
    H, W = T.shape
    yy, xx = np.mgrid[0:H:stride, 0:W:stride]
    q_plot = ax_T.quiver(xx, yy, qx[::stride, ::stride], -qy[::stride, ::stride],
                         color='w', angles='xy', scale_units='xy', width=0.003, alpha=0.9)
    if q_key_scale is None:
        q_key_scale = np.nanpercentile(np.sqrt(qx ** 2 + qy ** 2), 95)
    if q_key_scale > 0:
        ax_T.quiverkey(q_plot, 0.88, 0.08, q_key_scale, f'{q_key_scale:.2e}',
                       labelpos='S', coordinates='axes', color='w', fontproperties={'size': 9})
    if title:
        ax_T.set_title(title, fontsize=10)
    ax_T.set_xticks([])
    ax_T.set_yticks([])
    if ax_qmag is not None:
        q_mag = np.sqrt(qx ** 2 + qy ** 2)
        ax_qmag.imshow(q_mag, origin='upper', cmap='magma',
                       vmin=np.nanpercentile(q_mag, 2), vmax=np.nanpercentile(q_mag, 98))
        ax_qmag.set_xticks([])
        ax_qmag.set_yticks([])


def plot_flux_snapshot(df, ts, models, gt_model=GT_MODEL, stride_full=50, stride_zoom=10,
                       zoom_slice=None, show_q_magnitude=True, save_path=None):
    ts = pd.to_datetime(ts)
    n_models, n_cols = len(models), 3 if show_q_magnitude else 2
    fig, axes = plt.subplots(n_models, n_cols, figsize=(4.2 * n_cols, 3.4 * n_models),
                             squeeze=False, constrained_layout=True)
    fig.suptitle(f'Anisotropic flux q — {ts}', fontsize=13)

    T_vmin, T_vmax = temperature_limits_for_timestamp(df, ts)
    if gt_model is not None:
        T_gt = get_temperature_field(df, gt_model, ts)
        q_gt_x, q_gt_y, _, _, _ = compute_q_field(T_gt, dx=dx, dy=dy, eps=grad_eps)
    else:
        q_gt_x = q_gt_y = None

    q_probe = get_temperature_field(df, models[0], ts)
    _qx, _qy, _, _, _ = compute_q_field(q_probe, dx=dx, dy=dy, eps=grad_eps)
    q_key_scale = np.nanpercentile(np.sqrt(_qx ** 2 + _qy ** 2), 95)

    for row, model in enumerate(models):
        T = get_temperature_field(df, model, ts)
        qx, qy, _, _, _ = compute_q_field(T, dx=dx, dy=dy, eps=grad_eps)
        if gt_model is not None and model != gt_model:
            label = f'{model}\nMSE={flux_mse(qx, qy, q_gt_x, q_gt_y):.3e}'
        else:
            label = model
        plot_q_panel(T, qx, qy, axes[row, 0], title=label, stride=stride_full,
                     T_vmin=T_vmin, T_vmax=T_vmax, q_key_scale=q_key_scale)
        plot_q_panel(T, qx, qy, axes[row, 1], ax_qmag=axes[row, 2] if show_q_magnitude else None,
                     stride=stride_zoom, region_slice=zoom_slice,
                     T_vmin=T_vmin, T_vmax=T_vmax, q_key_scale=q_key_scale)
        axes[row, 0].set_ylabel(model, fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, axes
'''

cells = [
    cell_md(
        "# Test: anisotropic flux loss\n\n"
        "Loads **only** selected models × timestamps via `utils/results_subset_loader.py` "
        "(not the full 6+ GB pickle every run).\n\n"
        "**Fig_snapshots style:** `coolwarm` + saved `min`/`max` per timestamp; zoom matches Fig_snapshots panel 2."
    ),
    cell_code(
        """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

NOTEBOOK_DIR = Path.cwd()
if NOTEBOOK_DIR.name != 'notebooks':
    NOTEBOOK_DIR = Path('notebooks') if Path('notebooks').exists() else NOTEBOOK_DIR
sys.path.insert(0, str((NOTEBOOK_DIR / '..').resolve()))

from utils.results_subset_loader import (
    fig_snapshots_color_limits,
    fig_snapshots_zoom_slice,
    index_path_for,
    load_results_index,
    load_results_subset,
)

RESULTS_FILE = (NOTEBOOK_DIR / '../outputs/Our_results_trained_models_2mT.pkl').resolve()
OUTPUT_DIR = (NOTEBOOK_DIR / '../outputs').resolve()
TARGET_VAR = '2mT'
PLOT_VAR = '2mT'
GT_MODEL = None

FORCE_REBUILD_SUBSET = False

index = load_results_index(RESULTS_FILE)
if index is not None:
    available_models = index['models']
    available_times = [pd.Timestamp(t) for t in index['time_steps']]
    print(f'Index: {index_path_for(RESULTS_FILE)} ({index["n_rows"]} rows in source)')
else:
    available_models, available_times = [], []
    print('No index yet — built on first subset-cache extraction.')

if GT_MODEL is not None and GT_MODEL not in available_models:
    print(f'Warning: GT_MODEL={GT_MODEL!r} missing; using None')
    GT_MODEL = None

if RESULTS_FILE.is_file():
    print(f'Source: {RESULTS_FILE} ({RESULTS_FILE.stat().st_size / 2**30:.2f} GiB)')"""
    ),
    cell_md("## Models, timestamps, Fig_snapshots styling"),
    cell_code(
        """SKIP_MODELS = set()

model_order = ['ERA5', 'COSMO-CLM']
exclude_models = []

time_slices = [
    '2014-04-24 02:00:00',
    '2014-12-28 03:00:00',
    '2016-05-02 04:00:00',
    '2006-05-14 10:00:00',
    '2019-09-02 02:00:00',
]
selected_times = pd.to_datetime(time_slices)

USE_FIG_SNAPSHOTS_STYLE = True
CMAP_T = 'coolwarm' if USE_FIG_SNAPSHOTS_STYLE else 'viridis'
dx, dy, grad_eps = 1.0, 1.0, 1e-6
quiver_stride_full = 50
quiver_stride_zoom = 10
zoom_slice = fig_snapshots_zoom_slice() if USE_FIG_SNAPSHOTS_STYLE else (slice(180, 340), slice(180, 340))

def resolve_models(available, model_order=None, skip=None, exclude=None):
    skip, exclude = set(skip or []), set(exclude or [])
    models = (
        [m for m in model_order if m not in exclude and m not in skip]
        if model_order is not None
        else [m for m in sorted(available) if m not in skip and m not in exclude]
    )
    missing = [m for m in models if m not in set(available)]
    if missing:
        raise ValueError(f'Missing models: {missing}')
    if not models:
        raise ValueError('No models to plot.')
    return models

display(pd.DataFrame([
    {'model': m, 'action': ('plot' if m in set(resolve_models(available_models, model_order, SKIP_MODELS, exclude_models)) else 'skip')}
    for m in available_models
]).sort_values(['action', 'model']))"""
    ),
    cell_md("## Load subset cache\n\nFirst run: one full-pickle read (torch required). Later runs: fast."),
    cell_code(
        """models_to_load = resolve_models(available_models, model_order, SKIP_MODELS, exclude_models)
if GT_MODEL is not None and GT_MODEL not in models_to_load:
    models_to_load = [GT_MODEL] + models_to_load

plot_df = load_results_subset(
    RESULTS_FILE,
    models=models_to_load,
    times=selected_times,
    target_var=TARGET_VAR,
    plot_var=PLOT_VAR,
    force_rebuild=FORCE_REBUILD_SUBSET,
)
models_to_plot = resolve_models(plot_df['model'].unique(), model_order, SKIP_MODELS, exclude_models)
print(f'Ready: {len(plot_df)} rows; plot models: {models_to_plot}')"""
    ),
    cell_code(HELPERS.strip()),
    cell_md("## Flux MSE vs ground truth"),
    cell_code(
        """if GT_MODEL is None:
    print('Skipping flux MSE (GT_MODEL is None).')
else:
    validate_times(plot_df, selected_times)
    models_for_metrics = [m for m in models_to_plot if m != GT_MODEL]
    mse_rows = []
    for ts in selected_times:
        T_gt = get_temperature_field(plot_df, GT_MODEL, ts)
        q_gt_x, q_gt_y, _, _, _ = compute_q_field(T_gt, dx=dx, dy=dy, eps=grad_eps)
        for model in models_for_metrics:
            T_pred = get_temperature_field(plot_df, model, ts)
            qx, qy, _, _, _ = compute_q_field(T_pred, dx=dx, dy=dy, eps=grad_eps)
            mse_rows.append({'time_step': ts, 'model': model, 'flux_mse': flux_mse(qx, qy, q_gt_x, q_gt_y)})
    flux_mse_df = pd.DataFrame(mse_rows)
    display(flux_mse_df.pivot(index='model', columns='time_step', values='flux_mse'))
    display(flux_mse_df.groupby('model')['flux_mse'].mean().sort_values().to_frame('mean_flux_mse'))"""
    ),
    cell_md("## Plot q vectors (coolwarm + quiver)"),
    cell_code(PLOT_FUNCS.strip()),
    cell_code(
        """for ts in selected_times:
    fig, _ = plot_flux_snapshot(
        plot_df, ts, models=models_to_plot,
        stride_full=quiver_stride_full, stride_zoom=quiver_stride_zoom, zoom_slice=zoom_slice,
        # save_path=OUTPUT_DIR / f'flux_q_{pd.Timestamp(ts):%Y%m%d_%H%M}.png',
    )
    plt.show()"""
    ),
    cell_md("## Optional: PyTorch loss snippet"),
    cell_code(
        """import torch
import torch.nn.functional as F

def anisotropic_flux_mse_loss(T_pred, T_gt, dx=1.0, dy=1.0):
    # Same q = -||grad T||^2 grad T as numpy path above
    def grad(T):
        if T.dim() == 4:
            T = T[:, 0]
        B, H, W = T.shape
        dTdx = torch.zeros_like(T)
        dTdy = torch.zeros_like(T)
        if W > 2:
            dTdx[:, :, 1:-1] = (T[:, :, 2:] - T[:, :, :-2]) / (2 * dx)
        if H > 2:
            dTdy[:, 1:-1, :] = (T[:, 2:, :] - T[:, :-2, :]) / (2 * dy)
        return dTdx, dTdy
    dTdx, dTdy = grad(T_pred)
    g2 = dTdx ** 2 + dTdy ** 2
    qx, qy = -g2 * dTdx, -g2 * dTdy
    dTdx_g, dTdy_g = grad(T_gt)
    g2g = dTdx_g ** 2 + dTdy_g ** 2
    qx_g, qy_g = -g2g * dTdx_g, -g2g * dTdy_g
    return F.mse_loss(qx, qx_g) + F.mse_loss(qy, qy_g)"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "test_anisotropic_flux_loss.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size} bytes)")
