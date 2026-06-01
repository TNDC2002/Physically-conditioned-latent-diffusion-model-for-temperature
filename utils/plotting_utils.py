import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import xarray as xr
import pandas as pd
import numpy as np
import torch
from pysteps.visualization.spectral import plot_spectrum1d
import seaborn as sns


def from_torchtensor_to_xarray(torch_tensor, target_grid, coords_name='lat_lon'):
    numpy_array = torch_tensor
    numpy_array = numpy_array.detach().cpu()
    numpy_array = numpy_array.numpy()
    target_lats = target_grid.coords['y'].values
    target_lons = target_grid.coords['x'].values
    if coords_name == 'lat_lon':
        ds = xr.DataArray(numpy_array, coords={'lat': target_lats, 'lon': target_lons}, dims=['lat','lon'])
    elif coords_name == 'y_x':
        ds = xr.DataArray(numpy_array, coords={'y': target_lats, 'x': target_lons}, dims=['y','x'])
    else:
        print('Un-recognized string for coords_name')
    return ds


def quadratic_regrid_era5_xr_to_high_res(
    era5_on_low_grid: xr.DataArray, target_grid_high_res: xr.DataArray
) -> xr.DataArray:
    """Upsample an ERA5 low-res field to the high-res grid using scipy **quadratic** splines.

    xarray only allows ``method in {'linear', 'nearest'}`` for simultaneous
    multi-dimensional ``interp_like`` / ``interp`` over all axes at once; calling
    ``interp_like(..., method='quadratic')`` on ``(y, x)`` therefore raises
    ``ValueError`` on all supported xarray versions. To keep ``method='quadratic'``
    (research baseline), interpolate **along y**, then **along x** — each step
    is a valid 1-D quadratic pass (tensor-product style on a rectilinear grid).

    ``target_grid_high_res`` is only used for its ``y`` and ``x`` coordinates
    (e.g. ``get_target_grid('high')``); its data values are ignored.
    """
    hr_y = target_grid_high_res.coords["y"]
    hr_x = target_grid_high_res.coords["x"]
    scipy_kw = {"fill_value": "extrapolate"}
    along_y = era5_on_low_grid.interp(
        y=hr_y,
        method="quadratic",
        assume_sorted=False,
        kwargs=scipy_kw,
    )
    return along_y.interp(
        x=hr_x,
        method="quadratic",
        assume_sorted=False,
        kwargs=scipy_kw,
    )


def linear_regrid_era5_xr_to_high_res(
    era5_on_low_grid: xr.DataArray, target_grid_high_res: xr.DataArray
) -> xr.DataArray:
    """Upsample ERA5 low-res to high-res with xarray **linear** ND ``interp_like``.

    ``linear`` is supported on ``(y, x)`` simultaneously; use alongside
    :func:`quadratic_regrid_era5_xr_to_high_res` to compare baselines.
    """
    return era5_on_low_grid.interp_like(
        target_grid_high_res,
        method="linear",
        assume_sorted=False,
        kwargs={"fill_value": "extrapolate"},
    )


def show_snapshots(spat_dist_df: pd.DataFrame, target_res: str, output_dir: str, main_title: str = None, borders_file: str = None):
    # Set up the target grid
    target_grid = get_target_grid(target_res=target_res)

    # Load borders files if available
    if borders_file:
        gdf_bn = gpd.read_file(borders_file)

    # Count available target vars and models
    my_target_variables = spat_dist_df['target_var'].unique()
    my_models = spat_dist_df['model'].unique()

    # Set up figure
    rig_max = len(my_models)
    col_max = len(my_target_variables)*2
    fig = plt.figure(figsize=(4*col_max,5*int(rig_max)), constrained_layout=True)
    if len(my_target_variables)>1:
        subfigs = fig.subfigures(nrows=1, ncols=len(my_target_variables))
        axs = []
        for i in range(len(my_target_variables)):
            axs.append(subfigs[i].subplots(nrows=rig_max, ncols=2, sharex='col'))
    else:
        axs = [fig.subplots(nrows=rig_max, ncols=2, sharex='col')]
    if main_title is not None: 
        fig.suptitle(main_title, fontsize=22)  
    labels = {'2mT': '[K]',
              'WS10': '[m/s]'} 

    # Loop over target variables (columns)
    for col_sup, target_var in enumerate(my_target_variables):
        spat_dist_tv = spat_dist_df[(spat_dist_df['target_var'] == target_var)]
        if target_var == '2mT':
            var ='2mT'
            cmap = 'coolwarm'
            min_value = min(spat_dist_tv[(spat_dist_tv['variable'] == var)]['min'])
            max_value = max(spat_dist_tv[(spat_dist_tv['variable'] == var)]['max'])
        elif target_var == 'UV':
            var='WS10'
            cmap = 'jet'
            min_value = min(spat_dist_tv[(spat_dist_tv['variable'] == var)]['min'])
            max_value = 16 # max(spat_dist_df_filter[(spat_dist_df_filter['variable'] == 'U10')]['max'])
        # Loop over zoom-in columns
        for col in range(0,2):
            # Set up lims for different zoom-ins
            if col%2 == 0:
                x_lim = [target_grid.coords['x'].min().values,target_grid.coords['x'].max().values]
                y_lim = [target_grid.coords['y'].min().values,target_grid.coords['y'].max().values]
            else:
                x_lim = [4150000,4450000]
                y_lim = [1748000,2070000]                           
            # Loop over models
            for sim_row in range(0, rig_max):
                sim = my_models[sim_row]
                axs[col_sup][sim_row, col].set_xlim(x_lim)
                axs[col_sup][sim_row, col].set_ylim(y_lim)
                if var == '2mT':
                    if min_value < 0 and max_value>0:
                        max_value = max(abs(max_value), abs(min_value))
                        min_value = -max(abs(max_value), abs(min_value))
                    plot_tensor = spat_dist_tv[(spat_dist_tv['variable'] == var) & (spat_dist_tv['model'] == sim)]['spat_distr'].values
                    map = from_torchtensor_to_xarray(plot_tensor[0], target_grid)
                    we = map.plot.imshow(ax= axs[col_sup][sim_row, col], robust=True, add_colorbar=False, x='lon', y='lat', cmap=cmap, vmin = min_value, vmax=max_value) #                     
                else:
                    if col%2 == 0:
                        pick_stride = 50
                    else:
                        pick_stride = 10
                    plot_tensor_U10 = spat_dist_tv[(spat_dist_tv['variable'] == 'U10') & (spat_dist_tv['model'] == sim)]['spat_distr'].values
                    plot_tensor_V10 = spat_dist_tv[(spat_dist_tv['variable'] == 'V10') & (spat_dist_tv['model'] == sim)]['spat_distr'].values
                    plot_tensor_WS10 = spat_dist_tv[(spat_dist_tv['variable'] == 'WS10') & (spat_dist_tv['model'] == sim)]['spat_distr'].values
                    mapU10 = from_torchtensor_to_xarray(plot_tensor_U10[0], target_grid)
                    mapV10 = from_torchtensor_to_xarray(plot_tensor_V10[0], target_grid)
                    mapWS10 = from_torchtensor_to_xarray(plot_tensor_WS10[0], target_grid)
                    map = xr.merge([mapU10.rename('U10'), mapV10.rename('V10'), mapWS10.rename('WS10')], compat='no_conflicts', join='outer', combine_attrs='override')
                    map['U10'] = map['U10'].transpose()
                    map['V10'] = map['V10'].transpose()
                    map['WS10'] = map['WS10'].transpose()
                    we = map['WS10'].plot.imshow(ax= axs[col_sup][sim_row, col], robust=True, add_colorbar=False, x='lon', y='lat', cmap=cmap, alpha = .8, vmin = min_value, vmax=max_value)
                    we0 = map.thin(pick_stride).plot.quiver(ax= axs[col_sup][sim_row, col], u='U10', v='V10', x='lon', y='lat', scale=200, add_guide = False)
                    # Vector options declaration
                    veclenght = 8
                    maxstr = '%3.1f m/s' % veclenght
                    plt.quiverkey(we0,0.9,0.07,veclenght,maxstr,labelpos='S', coordinates='axes', fontproperties= {'size':13}).set_zorder(11)
                    rect = patches.Rectangle((x_lim[1]-(x_lim[1]-x_lim[0])/5, y_lim[0]), (x_lim[1]-x_lim[0])/5, (y_lim[1]-y_lim[0])/11, linestyle='-', linewidth=2,edgecolor='w', facecolor='w')
                    axs[col_sup][sim_row, col].add_patch(rect)
                # Add borders to the plot
                if borders_file:
                    gdf_bn.plot(ax= axs[col_sup][sim_row, col], color="black")
                # Remove axes ticks and labels
                axs[col_sup][sim_row, col].get_xaxis().set_visible(False)
                if col_sup == 0 and col == 0:
                    axs[col_sup][sim_row, col].yaxis.set_tick_params(labelleft=False, left=False)
                    axs[col_sup][sim_row, col].set_ylabel(spat_dist_tv[(spat_dist_tv['target_var'] == target_var) & (spat_dist_tv['model'] == sim)]['model'].values[0], fontsize=20)
                else:
                    axs[col_sup][sim_row, col].get_yaxis().set_visible(False)

        # Plot color bars
        if len(my_target_variables)>1:
            cbar = subfigs[col_sup].colorbar(we, ax=axs[col_sup], location='bottom')
        else:
            cbar = fig.colorbar(we, ax=axs[col_sup], location='bottom')
        cbar.set_label(var + ' ' + labels[var], fontsize=20)
        cbar.ax.tick_params(labelsize=18)

    # Plot and save to file
    filename = 'Fig_snapshots_' + str(main_title) + '.jpg'
    plt.savefig(output_dir + filename, bbox_inches='tight')
    plt.show()
    plt.close()


def _target_res_for_field(field: np.ndarray) -> str:
    for res in ("high", "low"):
        coords = get_target_coords(res)
        if field.shape == (len(coords["y"]), len(coords["x"])):
            return res
    raise ValueError(f"Field shape {field.shape} does not match high/low target grids.")


def _vectors_for_quiver_display(
    u: np.ndarray,
    v: np.ndarray,
    boost: float | None,
    *,
    use_lens: bool = True,
    key_length: float = 8.0,
    magnitude_gamma: float = 0.45,
    ref_percentile: float = 50.0,
    min_arrow_frac: float = 0.15,
    max_arrow_frac: float = 1.0,
    eps: float = 1e-20,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Map raw vectors to quiver arrows (plot only; physics unchanged).

    - **use_lens** ``False``: pass **raw** ``u, v`` (matches histogram |q|; weak arrows may vanish).
    - **use_lens** ``True`` (default): display "lens" — direction kept, magnitude remapped:
    - **magnitude_gamma** in (0, 1]: compresses dynamic range (``0.45`` default).
      ``1.0`` = linear scaling from median ref; smaller γ = more equal arrow lengths.
    - **min/max_arrow_frac**: floor weak arrows (no dots), cap extreme arrows.
    - **boost**: if set, linear ``u*boost`` (ignores gamma; use for manual tuning).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    mag = np.hypot(u, v)
    u_hat = np.zeros_like(u)
    v_hat = np.zeros_like(v)
    mask = mag > eps
    u_hat[mask] = u[mask] / mag[mask]
    v_hat[mask] = v[mask] / mag[mask]

    if not use_lens:
        ref = float(np.nanpercentile(mag[mask], ref_percentile)) if np.any(mask) else 1.0
        if ref <= 0:
            ref = 1.0
        tag = f"raw, ref=p{ref_percentile:g}"
        return u, v, tag, ref

    if boost is not None:
        mag_disp = mag * boost
        tag = f"linear ×{boost:.2e}"
    else:
        ref = float(np.nanpercentile(mag[mask], ref_percentile)) if np.any(mask) else 1.0
        if ref <= 0:
            ref = 1.0
        mag_norm = np.where(mask, mag / ref, 0.0)
        gamma = float(np.clip(magnitude_gamma, 0.05, 1.0))
        mag_disp = key_length * np.power(mag_norm, gamma)
        tag = f"γ={gamma:g}, ref=p{ref_percentile:g}"

    lo = min_arrow_frac * key_length
    hi = max_arrow_frac * key_length
    mag_disp = np.clip(mag_disp, lo, hi)
    return u_hat * mag_disp, v_hat * mag_disp, tag, key_length


_N_Q_PANELS = 3
_FLUX_PANEL_META = [
    ("A", "Flux q (raw qx, qy)"),
    ("B", "Flux q (raw qx, qy)"),
    ("G", "Flux q (raw qx, qy)"),
    ("C", "Grad T (raw dTdx, dTdy)"),
    ("D", "Grad T (raw dTdx, dTdy)"),
    ("E", "trace(J) + raw grad T (recovered J)"),
    ("F", "trace(J) + raw grad T (recovered J)"),
]

# Default deep-zoom window (EPSG:3035 m): centre of WS10 zoom box, ~150 km span
_DEFAULT_Q_DEEP_ZOOM_X = (4_225_000.0, 4_375_000.0)
_DEFAULT_Q_DEEP_ZOOM_Y = (1_828_500.0, 1_989_500.0)
_DEFAULT_Q_DEEP_ZOOM_STRIDE = 3


def _flux_panel_label(panel: int) -> tuple[str, str, str]:
    letter, name = _FLUX_PANEL_META[panel]
    if panel == 0 or panel in (3, 5):
        region = "domain"
    elif panel == 2:
        region = "deep zoom"
    else:
        region = "zoom"
    return letter, name, region


def _flux_panel_kind(panel: int) -> str:
    if panel < _N_Q_PANELS:
        return "q"
    if panel < 5:
        return "grad"
    return "j"


def _annotate_flux_panel(ax, panel: int) -> None:
    letter, name, region = _flux_panel_label(panel)
    ax.set_title(f"{letter}. {name} — {region}", fontsize=11, fontweight="bold", loc="left")
    ax.text(
        0.02,
        0.97,
        letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", alpha=0.65, pad=3, edgecolor="none"),
        zorder=12,
    )


def _numpy_to_field_map(fields: dict[str, np.ndarray], target_grid, t_var: str = "2mT") -> xr.Dataset:
    das = []
    for name, arr in fields.items():
        da = from_torchtensor_to_xarray(torch.from_numpy(np.asarray(arr, dtype=float)), target_grid)
        das.append(da.rename(t_var if name == "T" else name))
    field_map = xr.merge(das, compat="no_conflicts", join="outer", combine_attrs="override")
    for v in field_map.data_vars:
        field_map[v] = field_map[v].transpose()
    return field_map


def _panel_limits(
    panel: int,
    target_grid,
    *,
    q_deep_zoom_x: tuple[float, float] = _DEFAULT_Q_DEEP_ZOOM_X,
    q_deep_zoom_y: tuple[float, float] = _DEFAULT_Q_DEEP_ZOOM_Y,
    q_deep_zoom_stride: int = _DEFAULT_Q_DEEP_ZOOM_STRIDE,
):
    if panel == 2:
        x_lim = [q_deep_zoom_x[0], q_deep_zoom_x[1]]
        y_lim = [q_deep_zoom_y[0], q_deep_zoom_y[1]]
        pick_stride = max(1, int(q_deep_zoom_stride))
    elif panel in (1, 4, 6):
        x_lim = [4_150_000, 4_450_000]
        y_lim = [1_748_000, 2_070_000]
        pick_stride = 10
    else:
        x_lim = [target_grid.coords["x"].min().values, target_grid.coords["x"].max().values]
        y_lim = [target_grid.coords["y"].min().values, target_grid.coords["y"].max().values]
        pick_stride = 50
    return x_lim, y_lim, pick_stride


def _expand_geo_limits(
    x_lim,
    y_lim,
    *,
    lon_mult: float = 1.0,
    lat_mult: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Widen the map window around its centre (EPSG m). ``2`` = 2× span on that axis."""
    lon_mult = max(1e-6, float(lon_mult))
    lat_mult = max(1e-6, float(lat_mult))
    x0, x1 = float(x_lim[0]), float(x_lim[1])
    y0, y1 = float(y_lim[0]), float(y_lim[1])
    x_c, y_c = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    x_half = abs(x1 - x0) / 2.0 * lon_mult
    y_half = abs(y1 - y0) / 2.0 * lat_mult
    return [x_c - x_half, x_c + x_half], [y_c - y_half, y_c + y_half]


def _crop_field_map_to_geo_box(
    field_map: xr.Dataset, x_lim, y_lim
) -> xr.Dataset:
    """Select grid cells inside ``x_lim`` / ``y_lim`` (EPSG m on lon/lat coords)."""
    x0, x1 = sorted((float(x_lim[0]), float(x_lim[1])))
    y0, y1 = sorted((float(y_lim[0]), float(y_lim[1])))
    lon = field_map["lon"]
    lat = field_map["lat"]
    return field_map.isel(
        lon=(lon >= x0) & (lon <= x1),
        lat=(lat >= y0) & (lat <= y1),
    )


def _crop_mask_to_geo_box(
    mask: np.ndarray, field_map: xr.Dataset, x_lim, y_lim
) -> np.ndarray:
    """Crop a 2-D mask (lat, lon) to the same geo box as ``_crop_field_map_to_geo_box``."""
    da = xr.DataArray(
        np.asarray(mask, dtype=bool),
        coords={"lat": field_map["lat"], "lon": field_map["lon"]},
        dims=["lat", "lon"],
    )
    return (
        _crop_field_map_to_geo_box(da.to_dataset(name="_m"), x_lim, y_lim)["_m"]
        .values
    )


def _thin_field_map(
    field_map: xr.Dataset, stride_lat: int, stride_lon: int | None = None
) -> xr.Dataset:
    stride_lon = stride_lat if stride_lon is None else stride_lon
    stride_lat = max(1, int(stride_lat))
    stride_lon = max(1, int(stride_lon))
    if stride_lat == stride_lon:
        return field_map.thin(stride_lat)
    return field_map.thin({"lat": stride_lat, "lon": stride_lon})


def _thin_bool_mask(
    skip_mask: np.ndarray,
    field_map: xr.Dataset,
    stride_lat: int,
    stride_lon: int | None = None,
) -> np.ndarray:
    stride_lon = stride_lat if stride_lon is None else stride_lon
    skip_da = xr.DataArray(
        skip_mask.astype(bool),
        coords={"lat": field_map["lat"], "lon": field_map["lon"]},
        dims=["lat", "lon"],
    )
    if stride_lat == stride_lon:
        return skip_da.thin(stride_lat).values
    return skip_da.thin({"lat": stride_lat, "lon": stride_lon}).values


def _paper_figsize_from_geo_extent(
    x_lim,
    y_lim,
    height_in: float = 5.0,
) -> tuple[float, float]:
    """Figure (width, height) in inches matching the zoom box aspect (EPSG m)."""
    x_span = abs(float(x_lim[1]) - float(x_lim[0]))
    y_span = abs(float(y_lim[1]) - float(y_lim[0]))
    if y_span <= 0:
        y_span = 1.0
    width_in = max(float(height_in) * (x_span / y_span), float(height_in) * 0.5)
    return (width_in, float(height_in))


# Base paper font sizes (pt) at reference min(fig_w, fig_h) = 5 in — scaled in _paper_font_sizes.
_PAPER_FONT_REF_IN = 5.0


def _paper_font_sizes(fig_w: float, fig_h: float) -> dict[str, float]:
    """Scale header, colorbar, and legend fonts with figure size."""
    fig_w, fig_h = float(fig_w), float(fig_h)
    scale = min(fig_w, fig_h) / _PAPER_FONT_REF_IN
    scale = float(np.clip(scale, 0.85, 4.0))
    # Header uses √(area) so wide/tall figures (e.g. 10×20 in) get a larger title.
    header_scale = float(np.sqrt(fig_w * fig_h)) / _PAPER_FONT_REF_IN
    header_scale = float(np.clip(header_scale, 0.85, 4.0))
    return {
        "header": 9.0 * header_scale,
        "cbar_label": 8.0 * scale,
        "cbar_ticks": 7.0 * scale,
        "legend": 7.0 * scale,
    }


def _style_q_panel_ax(
    ax, x_lim, y_lim, gdf_bn, borders_file, sim, col: int, *, ylabel: str | None = None
):
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    if borders_file:
        gdf_bn.plot(ax=ax, color="black")
    ax.get_xaxis().set_visible(False)
    if col == 0 and ylabel:
        ax.yaxis.set_tick_params(labelleft=False, left=False)
        ax.set_ylabel(ylabel, fontsize=16 if len(ylabel) < 20 else 14)
    else:
        ax.get_yaxis().set_visible(False)


# WS10 / 7bf7bad: quiver ``scale=200`` only — do not use scale_units='xy' on lon/lat in metres.
_QUIVER_SCALE = 200


def _normalize_q_mask_mode(
    mode: str,
    *,
    skip_quantile_pct: float = 0.0,
    abs_threshold: float | None = None,
) -> str:
    """Return ``none``, ``quantile``, or ``absolute``."""
    mode = (mode or "none").strip().lower()
    if mode == "none" and skip_quantile_pct > 0:
        mode = "quantile"
    if mode not in ("none", "quantile", "absolute"):
        raise ValueError(f"q_mask_mode must be 'none', 'quantile', or 'absolute', got {mode!r}")
    if mode == "quantile" and skip_quantile_pct <= 0:
        raise ValueError("quantile masking requires skip_quantile_pct > 0")
    if mode == "absolute":
        if abs_threshold is None:
            raise ValueError("absolute masking requires q_mask_abs_threshold")
        if abs_threshold < 0:
            raise ValueError(f"q_mask_abs_threshold must be >= 0, got {abs_threshold}")
    return mode


def _q_mag_and_mask(
    qx: np.ndarray,
    qy: np.ndarray,
    *,
    mode: str = "none",
    skip_quantile_pct: float = 0.0,
    abs_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None, str]:
    """Return (q_mag, skip_mask, threshold, mode). Mask where |q| <= threshold."""
    mode = _normalize_q_mask_mode(
        mode, skip_quantile_pct=skip_quantile_pct, abs_threshold=abs_threshold
    )
    q_mag = np.hypot(np.asarray(qx, dtype=float), np.asarray(qy, dtype=float))
    if mode == "none":
        return q_mag, np.zeros(q_mag.shape, dtype=bool), None, mode
    if mode == "absolute":
        thr = float(abs_threshold)
    else:
        thr = float(np.percentile(q_mag, skip_quantile_pct))
    return q_mag, q_mag <= thr, thr, mode


def summarize_q_mask_thresholds(
    flux_by_model: dict[str, dict[str, np.ndarray]],
    *,
    mode: str = "none",
    skip_quantile_pct: float = 0.0,
    abs_threshold: float | None = None,
    time_label: str | None = None,
    training_at_qmag_min: float | None = None,
) -> pd.DataFrame:
    """Log |q| cutoff used for masking (pixels with |q| <= threshold are filtered out).

    Table columns document what enters the mask:

    - **|q| used for mask**: ``hypot(qx, qy)`` from the same ``T`` as this row's flux dict.
    - **Plot rule**: hide arrow / purple dot where ``|q| <= |q|_filter_threshold``.
    - **Training (LMM)**: ``TemperatureFieldLosses`` keeps pixels with
      ``|q_gt| > at_qmag_min`` on **normalized** ``T_hr`` (COSMO-CLM z-score) only;
      ``q_pred`` does not affect the mask. Plot matches training only when the row's
      ``T`` is that normalized GT and ``abs_threshold == at_qmag_min`` (strict ``>`` kept).
    """
    mode = _normalize_q_mask_mode(
        mode, skip_quantile_pct=skip_quantile_pct, abs_threshold=abs_threshold
    )
    if mode == "absolute":
        plot_rule = (
            f"plot: skip |q| <= {abs_threshold:g}; "
            "training: keep |q_gt| > at_qmag_min (GT normalized T_hr only)"
        )
    elif mode == "quantile":
        plot_rule = (
            f"plot: skip |q| <= p{skip_quantile_pct:g} of this row's |q|; "
            "training: keep |q_gt| >= batch quantile if at_qmag_quantile set"
        )
    else:
        plot_rule = "no mask (all pixels plotted; training may still use at_qmag_min)"

    rows = []
    for model, flux in flux_by_model.items():
        q_mag, skip, thr, _ = _q_mag_and_mask(
            flux["qx"],
            flux["qy"],
            mode=mode,
            skip_quantile_pct=skip_quantile_pct,
            abs_threshold=abs_threshold,
        )
        keep = ~skip
        matches_training = (
            mode == "absolute"
            and training_at_qmag_min is not None
            and abs_threshold is not None
            and float(abs_threshold) == float(training_at_qmag_min)
            and "z-score" in str(model)
        )
        rows.append(
            {
                "time": time_label,
                "model": model,
                "|q| used for mask": "hypot(qx, qy) from this row's T (see model name for Kelvin vs z-score)",
                "mask_mode": mode,
                "plot mask rule": plot_rule,
                "matches training at_qmag_min?": matches_training,
                "training at_qmag_min ref": training_at_qmag_min,
                "skip_quantile_%": skip_quantile_pct if mode == "quantile" else np.nan,
                "abs_threshold_input": abs_threshold if mode == "absolute" else np.nan,
                "|q|_filter_threshold": thr,
                "n_pixels_masked": int(skip.sum()),
                "frac_masked": float(skip.mean()),
                "max_|q|_among_masked": float(np.max(q_mag[skip])) if skip.any() else np.nan,
                "min_|q|_among_kept": float(np.min(q_mag[keep])) if keep.any() else np.nan,
                "median_|q|_field": float(np.median(q_mag)),
                "max_|q|_field": float(np.max(q_mag)),
            }
        )

    df = pd.DataFrame(rows)
    if mode == "absolute":
        hdr = f"=== q mask |q| thresholds (mask |q| <= {abs_threshold:g}) ==="
    elif mode == "quantile":
        hdr = f"=== q mask |q| thresholds (mask |q| <= p{skip_quantile_pct:g}) ==="
    else:
        hdr = "=== q mask: disabled ==="
    if time_label:
        hdr += f"  [{time_label}]"
    print(hdr)
    if df.empty:
        print("(no models)")
        return df
    for _, r in df.iterrows():
        if mode == "none":
            print(f"  {r['model']}: masking disabled")
            continue
        print(
            f"  {r['model']}: filter |q| <= {r['|q|_filter_threshold']:.6e}  "
            f"({r['n_pixels_masked']} px, {100 * r['frac_masked']:.1f}% of grid)  "
            f"| masked max={r['max_|q|_among_masked']:.6e}, kept min={r['min_|q|_among_kept']:.6e}, "
            f"field median={r['median_|q|_field']:.6e}, max={r['max_|q|_field']:.6e}"
        )
    return df


def apply_q_magnitude_mask(
    qx: np.ndarray,
    qy: np.ndarray,
    *,
    mode: str = "none",
    skip_quantile_pct: float = 0.0,
    abs_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None, str]:
    """Zero-out masked vectors (NaN) for quiver; return skip_mask and effective threshold."""
    q_mag, skip, thr, mode = _q_mag_and_mask(
        qx,
        qy,
        mode=mode,
        skip_quantile_pct=skip_quantile_pct,
        abs_threshold=abs_threshold,
    )
    if mode == "none":
        return qx, qy, skip, None, mode
    qx_out = np.asarray(qx, dtype=float).copy()
    qy_out = np.asarray(qy, dtype=float).copy()
    qx_out[skip] = np.nan
    qy_out[skip] = np.nan
    return qx_out, qy_out, skip, thr, mode


def _plot_q_mask_skip_dots(
    ax,
    field_map: xr.Dataset,
    skip_mask: np.ndarray,
    stride_lat: int,
    stride_lon: int | None = None,
):
    """Purple markers at grid points where |q| was below the skip quantile."""
    if skip_mask is None or not np.any(skip_mask):
        return
    thinned = _thin_field_map(field_map, stride_lat, stride_lon)
    skip_thin = _thin_bool_mask(skip_mask, field_map, stride_lat, stride_lon)
    if not np.any(skip_thin):
        return
    lon_2d, lat_2d = np.meshgrid(thinned["lon"].values, thinned["lat"].values)
    ax.scatter(
        lon_2d[skip_thin],
        lat_2d[skip_thin],
        s=12,
        c="purple",
        alpha=0.85,
        linewidths=0,
        zorder=10,
    )


def _format_q_mask_caption(
    *,
    q_mask_mode: str = "none",
    q_mask_skip_quantile_pct: float = 0.0,
    q_mask_abs_threshold: float | None = None,
) -> str:
    """One-line caption for paper figures (masking only)."""
    mode = _normalize_q_mask_mode(
        q_mask_mode,
        skip_quantile_pct=q_mask_skip_quantile_pct,
        abs_threshold=q_mask_abs_threshold,
    )
    if mode == "none":
        return "q mask: none"
    if mode == "quantile":
        return f"q mask: |q| ≤ p{q_mask_skip_quantile_pct:g} (masked, no arrow)"
    return f"q mask: |q| ≤ {q_mask_abs_threshold:g} (masked, no arrow)"


def _plot_jet_T_quiver_panel_paper(
    ax,
    field_map: xr.Dataset,
    t_var: str,
    *,
    t_vmin: float,
    t_vmax: float,
    u_var: str,
    v_var: str,
    stride_lat: int,
    stride_lon: int | None = None,
    quiver_color: str | None = None,
    q_skip_mask: np.ndarray | None = None,
    x_lim=None,
    y_lim=None,
    legend_fontsize: float = 7.0,
):
    """Deep-zoom paper panel: temperature + quiver, no quiverkey or inset box."""
    mappable = field_map[t_var].plot.imshow(
        ax=ax,
        robust=True,
        add_colorbar=False,
        x="lon",
        y="lat",
        cmap="jet",
        alpha=0.8,
        vmin=t_vmin,
        vmax=t_vmax,
    )
    quiver_kw = dict(
        ax=ax,
        u=u_var,
        v=v_var,
        x="lon",
        y="lat",
        scale=_QUIVER_SCALE,
        add_guide=False,
    )
    if quiver_color is not None:
        quiver_kw["color"] = quiver_color
    _thin_field_map(field_map, stride_lat, stride_lon).plot.quiver(**quiver_kw)
    if q_skip_mask is not None:
        _plot_q_mask_skip_dots(ax, field_map, q_skip_mask, stride_lat, stride_lon)
    if x_lim is not None and y_lim is not None:
        _finalize_paper_map_ax(ax, x_lim, y_lim)
    _add_paper_flux_vector_note(
        ax,
        show_mask_dots=q_skip_mask is not None,
        fontsize=legend_fontsize,
    )
    return mappable


def _add_paper_flux_vector_note(ax, *, show_mask_dots: bool, fontsize: float = 7.0) -> None:
    """In-panel legend (lower right): purple dot + arrow samples."""
    ms = max(5.0, fontsize * 0.75)
    handles: list[Line2D] = []
    labels: list[str] = []
    if show_mask_dots:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markerfacecolor="purple",
                markeredgecolor="purple",
                alpha=0.85,
                markersize=ms,
            )
        )
        labels.append("Filtered |q|")
    handles.append(
        Line2D(
            [0, 1],
            [0, 0],
            color="black",
            linewidth=1.6,
            linestyle="-",
            marker=">",
            markersize=ms,
            markevery=[1],
            solid_capstyle="butt",
        )
    )
    labels.append("Retained q")
    leg = ax.legend(
        handles,
        labels,
        loc="lower right",
        fontsize=fontsize,
        framealpha=0.88,
        edgecolor="0.45",
        fancybox=False,
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=2.0,
        handletextpad=0.55,
    )
    leg.set_zorder(15)


def _style_paper_deep_zoom_ax(ax, x_lim, y_lim, gdf_bn, borders_file):
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if borders_file and gdf_bn is not None:
        gdf_bn.plot(ax=ax, color="black", linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _finalize_paper_map_ax(ax, x_lim, y_lim) -> None:
    """Restore geo limits + equal aspect after xarray plots (avoids square stretch)."""
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_jet_T_quiver_panel(
    ax,
    field_map: xr.Dataset,
    t_var: str,
    *,
    t_vmin: float,
    t_vmax: float,
    u_var: str,
    v_var: str,
    stride_lat: int,
    stride_lon: int | None = None,
    quiver_ref: float,
    quiver_label: str,
    x_lim,
    y_lim,
    quiver_color: str | None = None,
    q_skip_mask: np.ndarray | None = None,
):
    """WS10-style panel: ``jet`` 2mT background + vector overlay."""
    mappable = field_map[t_var].plot.imshow(
        ax=ax,
        robust=True,
        add_colorbar=False,
        x="lon",
        y="lat",
        cmap="jet",
        alpha=0.8,
        vmin=t_vmin,
        vmax=t_vmax,
    )
    quiver_kw = dict(
        ax=ax,
        u=u_var,
        v=v_var,
        x="lon",
        y="lat",
        scale=_QUIVER_SCALE,
        add_guide=False,
    )
    if quiver_color is not None:
        quiver_kw["color"] = quiver_color
    q_plot = _thin_field_map(field_map, stride_lat, stride_lon).plot.quiver(**quiver_kw)
    _add_ws10_quiverkey(q_plot, quiver_ref, quiver_label)
    if q_skip_mask is not None:
        _plot_q_mask_skip_dots(ax, field_map, q_skip_mask, stride_lat, stride_lon)
    _add_ws10_key_box(ax, x_lim, y_lim)
    return mappable


def _add_ws10_quiverkey(quiver_plot, ref_len: float, label: str):
    plt.quiverkey(
        quiver_plot,
        0.9,
        0.07,
        ref_len,
        label,
        labelpos="S",
        coordinates="axes",
        fontproperties={"size": 13},
    ).set_zorder(11)


def _add_ws10_key_box(ax, x_lim, y_lim):
    rect = patches.Rectangle(
        (x_lim[1] - (x_lim[1] - x_lim[0]) / 5, y_lim[0]),
        (x_lim[1] - x_lim[0]) / 5,
        (y_lim[1] - y_lim[0]) / 11,
        linestyle="-",
        linewidth=2,
        edgecolor="w",
        facecolor="w",
    )
    ax.add_patch(rect)


def _plot_jet_T_Jtrace_panel(
    ax,
    field_map: xr.Dataset,
    t_var: str,
    *,
    t_vmin: float,
    t_vmax: float,
    j_tr_vmin: float,
    j_tr_vmax: float,
    x_lim,
    y_lim,
):
    """``jet`` 2mT background + ``plasma`` trace(J) overlay (recovered J, not real K)."""
    mappable = field_map[t_var].plot.imshow(
        ax=ax,
        robust=True,
        add_colorbar=False,
        x="lon",
        y="lat",
        cmap="jet",
        alpha=0.8,
        vmin=t_vmin,
        vmax=t_vmax,
    )
    field_map["J_trace"].plot.imshow(
        ax=ax,
        add_colorbar=False,
        x="lon",
        y="lat",
        cmap="plasma",
        alpha=0.65,
        vmin=j_tr_vmin,
        vmax=j_tr_vmax,
    )
    _add_ws10_key_box(ax, x_lim, y_lim)
    return mappable


_FLUX_FIELD_KEYS = ("T", "dTdx", "dTdy", "J_trace", "qx", "qy")


def _validate_flux_by_model(flux_by_model: dict[str, dict[str, np.ndarray]], models) -> None:
    for model in models:
        if model not in flux_by_model:
            raise KeyError(f"flux_by_model missing model={model!r}")
        missing = [k for k in _FLUX_FIELD_KEYS if k not in flux_by_model[model]]
        if missing:
            raise KeyError(f"flux_by_model[{model!r}] missing keys: {missing}")


def show_q_snapshots(
    spat_dist_df: pd.DataFrame,
    flux_by_model: dict[str, dict[str, np.ndarray]],
    *,
    variable: str = "2mT",
    output_dir: str | None = None,
    main_title: str | None = None,
    borders_file: str | None = None,
    vector_display_boost: float | None = None,
    q_display_boost: float | None = None,
    grad_display_boost: float | None = None,
    quiver_key_length: float = 8.0,
    use_quiver_lens: bool = True,
    display_magnitude_gamma: float = 0.45,
    display_min_arrow_frac: float = 0.15,
    display_max_arrow_frac: float = 1.0,
    q_mask_mode: str = "none",
    q_mask_skip_quantile_pct: float = 0.0,
    q_mask_abs_threshold: float | None = None,
    q_deep_zoom_x: tuple[float, float] | None = None,
    q_deep_zoom_y: tuple[float, float] | None = None,
    q_deep_zoom_stride: int | None = None,
    show_grad_j_panels: bool = True,
    temperature_unit_label: str | None = None,
    q_panel_indices: tuple[int, ...] | None = None,
    paper_style: bool = False,
    paper_lon_extent_mult: float = 1.0,
    paper_lat_extent_mult: float = 1.0,
    paper_fig_width: float | None = None,
    paper_fig_height: float = 5.0,
    output_basename: str | None = None,
    dpi: int = 150,
):
    """Plot precomputed flux fields (7 panels per model, 3×3 WS10 layout).

    Set ``show_grad_j_panels=False`` for **q-only** (one row: domain | zoom | deep zoom).

    Set ``q_panel_indices=(2,)`` with ``paper_style=True`` for a single deep-zoom panel per model
    (third column; same rendering as the working A|B|C row, minimal text).

    Paper deep-zoom (panel index 2): ``paper_lon_extent_mult`` / ``paper_lat_extent_mult``
    widen the **map window** and **figure size** only. Quiver uses the same ``q_deep_zoom_stride``
    as panel C on the **cropped** viewport so vectors / grid cells in view stay fixed (~1/s²).

    Row 1: flux **q** — domain | zoom | **deep zoom** (panel G, denser quiver).

    Physics (∇T, recovered **J**, q) must be computed in the notebook; pass
    ``flux_by_model[model]`` with keys ``T``, ``dTdx``, ``dTdy``, ``J_trace``, ``qx``, ``qy``.
    ``J_trace`` = trace(J) = |∇T|² (scalar summary of recovered J; not real conductive K).

    Quiver display: set ``use_quiver_lens=False`` to plot **raw** ``qx/qy`` (consistent with |q|
    histograms). ``use_quiver_lens=True`` (default) applies power-law compression (γ) plus
    min/max arrow floors so weak and extreme vectors remain visible.
    Set ``display_magnitude_gamma=1`` for linear lens; manual ``*_display_boost`` overrides lens.

    **q masking** (purple dots = masked, no arrow):

    - ``q_mask_mode='quantile'`` + ``q_mask_skip_quantile_pct`` (e.g. ``50``): mask |q| at or
      below the per-slice percentile.
    - ``q_mask_mode='absolute'`` + ``q_mask_abs_threshold`` (e.g. ``1e-9``): mask all pixels with
      |q| <= that value.
    - ``q_mask_mode='none'``: no mask. Legacy: ``q_mask_skip_quantile_pct > 0`` alone still
      enables quantile mode.
    """
    spat_dist_df = spat_dist_df[spat_dist_df["variable"] == variable].copy()
    if spat_dist_df.empty:
        raise ValueError(f"No rows for variable={variable!r}")

    gdf_bn = gpd.read_file(borders_file) if borders_file else None

    my_models = spat_dist_df["model"].unique()
    _validate_flux_by_model(flux_by_model, my_models)
    rig_max = len(my_models)
    var = variable
    t_unit = temperature_unit_label if temperature_unit_label is not None else "[K]"
    labels = {"2mT": t_unit}
    min_value = min(spat_dist_df["min"])
    max_value = max(spat_dist_df["max"])

    q_deep_x = q_deep_zoom_x if q_deep_zoom_x is not None else _DEFAULT_Q_DEEP_ZOOM_X
    q_deep_y = q_deep_zoom_y if q_deep_zoom_y is not None else _DEFAULT_Q_DEEP_ZOOM_Y
    q_deep_stride = (
        q_deep_zoom_stride if q_deep_zoom_stride is not None else _DEFAULT_Q_DEEP_ZOOM_STRIDE
    )

    if q_panel_indices is not None:
        if show_grad_j_panels:
            raise ValueError("q_panel_indices requires show_grad_j_panels=False")
        panels_to_plot = tuple(q_panel_indices)
        ncols = len(panels_to_plot)
        nrows_per_model = 1
        n_panels = _N_Q_PANELS
        n_slots_per_model = ncols
    else:
        panels_to_plot = None
        n_panels = 7 if show_grad_j_panels else _N_Q_PANELS
        ncols = 3
        nrows_per_model = 3 if show_grad_j_panels else 1
        n_slots_per_model = nrows_per_model * ncols

    nrows = rig_max * nrows_per_model
    row_h = 5.9 if paper_style else 5.0
    if paper_style and panels_to_plot == (2,) and rig_max == 1:
        if paper_fig_width is not None:
            fig_w = float(paper_fig_width)
            fig_h = float(paper_fig_height) * float(paper_lat_extent_mult)
        else:
            _probe_grid = get_target_grid(
                target_res=_target_res_for_field(flux_by_model[my_models[0]]["T"])
            )
            _px, _py, _ = _panel_limits(
                2,
                _probe_grid,
                q_deep_zoom_x=q_deep_x,
                q_deep_zoom_y=q_deep_y,
                q_deep_zoom_stride=q_deep_stride,
            )
            _px, _py = _expand_geo_limits(
                _px,
                _py,
                lon_mult=paper_lon_extent_mult,
                lat_mult=paper_lat_extent_mult,
            )
            fig_h = float(paper_fig_height) * float(paper_lat_extent_mult)
            fig_w, fig_h = _paper_figsize_from_geo_extent(_px, _py, height_in=fig_h)
    else:
        fig_w, fig_h = 5 * ncols, row_h * nrows
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=not paper_style,
    )
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = np.atleast_2d(axs)
    elif ncols == 1:
        axs = axs.reshape(-1, 1)

    paper_fonts = _paper_font_sizes(fig_w, fig_h) if paper_style else None

    title_fs = 14 if paper_style else 22
    if main_title is not None and not paper_style:
        fig.suptitle(main_title, fontsize=title_fs, y=1.02)

    q_mask_mode_resolved = _normalize_q_mask_mode(
        q_mask_mode,
        skip_quantile_pct=q_mask_skip_quantile_pct,
        abs_threshold=q_mask_abs_threshold,
    )
    paper_mask_caption = (
        _format_q_mask_caption(
            q_mask_mode=q_mask_mode,
            q_mask_skip_quantile_pct=q_mask_skip_quantile_pct,
            q_mask_abs_threshold=q_mask_abs_threshold,
        )
        if paper_style
        else None
    )
    if not paper_style:
        if use_quiver_lens:
            arrow_note = (
                f"Arrows: display lens ON (γ={display_magnitude_gamma:g}, "
                f"min={display_min_arrow_frac:g}×key, max={display_max_arrow_frac:g}×key) — "
                "direction kept; lengths remapped for visibility"
            )
        else:
            arrow_note = (
                "Arrows: display lens OFF — raw qx/qy (true relative magnitudes; may hide weak vectors)"
            )
        if q_mask_mode_resolved == "quantile":
            arrow_note += (
                f" | q mask: |q| <= p{q_mask_skip_quantile_pct:g} per slice (purple dots = masked)"
            )
        elif q_mask_mode_resolved == "absolute":
            arrow_note += (
                f" | q mask: |q| <= {q_mask_abs_threshold:g} (purple dots = masked)"
            )
        fig.text(0.5, 0.005, arrow_note, ha="center", fontsize=10, style="italic")

    im_mag = None
    for sim_row in range(rig_max):
        sim = my_models[sim_row]
        flux = flux_by_model[sim]
        target_res = _target_res_for_field(flux["T"])
        target_grid = get_target_grid(target_res=target_res)

        q_boost = q_display_boost if q_display_boost is not None else vector_display_boost
        g_boost = grad_display_boost if grad_display_boost is not None else vector_display_boost
        disp_kw = dict(
            use_lens=use_quiver_lens,
            key_length=quiver_key_length,
            magnitude_gamma=display_magnitude_gamma,
            min_arrow_frac=display_min_arrow_frac,
            max_arrow_frac=display_max_arrow_frac,
        )
        q_raw_max = float(np.nanmax(np.hypot(flux["qx"], flux["qy"])))
        if (
            not paper_style
            and use_quiver_lens
            and q_raw_max > 0
            and q_raw_max < 1e-4
        ):
            print(
                f"Warning [{sim}]: max |q|={q_raw_max:.3e} — lens remaps arrow lengths "
                f"(γ={display_magnitude_gamma:g}, min_frac={display_min_arrow_frac:g}); "
                "they are not proportional to physical |q|. Set use_quiver_lens=False to verify."
            )
        qx_plot, qy_plot, q_skip_mask, _q_skip_thr, _q_mask_mode_used = apply_q_magnitude_mask(
            flux["qx"],
            flux["qy"],
            mode=q_mask_mode,
            skip_quantile_pct=q_mask_skip_quantile_pct,
            abs_threshold=q_mask_abs_threshold,
        )
        qx_d, qy_d, q_tag, q_key = _vectors_for_quiver_display(
            qx_plot, qy_plot, q_boost, **disp_kw
        )
        field_q = _numpy_to_field_map(
            {"T": flux["T"], "qx": qx_d, "qy": qy_d}, target_grid, t_var=var
        )
        if show_grad_j_panels:
            gx_d, gy_d, g_tag, g_key = _vectors_for_quiver_display(
                flux["dTdx"], flux["dTdy"], g_boost, **disp_kw
            )
            field_grad = _numpy_to_field_map(
                {"T": flux["T"], "qx": gx_d, "qy": gy_d}, target_grid, t_var=var
            )
            field_j_grad = _numpy_to_field_map(
                {"T": flux["T"], "dTdx": gx_d, "dTdy": gy_d, "J_trace": flux["J_trace"]},
                target_grid,
                t_var=var,
            )
            j_tr_vmax = float(np.nanpercentile(flux["J_trace"], 99)) or 1.0

        panel_loop = panels_to_plot if panels_to_plot is not None else range(n_slots_per_model)
        for slot_idx, panel in enumerate(panel_loop):
            if panels_to_plot is not None:
                grid_row = sim_row
                col = slot_idx
            else:
                grid_row = sim_row * nrows_per_model + panel // ncols
                col = panel % ncols
            ax = axs[grid_row, col]
            if panels_to_plot is None and panel >= n_panels:
                ax.set_visible(False)
                continue

            x_lim, y_lim, pick_stride = _panel_limits(
                panel,
                target_grid,
                q_deep_zoom_x=q_deep_x,
                q_deep_zoom_y=q_deep_y,
                q_deep_zoom_stride=q_deep_stride,
            )
            if panel == 2 and paper_style and (
                paper_lon_extent_mult != 1.0 or paper_lat_extent_mult != 1.0
            ):
                x_lim, y_lim = _expand_geo_limits(
                    x_lim,
                    y_lim,
                    lon_mult=paper_lon_extent_mult,
                    lat_mult=paper_lat_extent_mult,
                )
            stride_lat = stride_lon = pick_stride
            q_mask_plot = (
                q_skip_mask if _q_mask_mode_used != "none" else None
            )
            field_q_on_ax = field_q
            q_mask_on_ax = q_mask_plot
            if panel == 2:
                field_q_on_ax = _crop_field_map_to_geo_box(field_q, x_lim, y_lim)
                if q_mask_plot is not None:
                    q_mask_on_ax = _crop_mask_to_geo_box(
                        q_mask_plot, field_q, x_lim, y_lim
                    )
            if paper_style:
                _style_paper_deep_zoom_ax(ax, x_lim, y_lim, gdf_bn, borders_file)
            else:
                if panel // ncols == 0 or not show_grad_j_panels:
                    row_ylabel = sim
                else:
                    row_ylabel = "grad T & J" if panel // ncols == 1 else ""
                _style_q_panel_ax(
                    ax,
                    x_lim,
                    y_lim,
                    gdf_bn,
                    borders_file,
                    sim,
                    col,
                    ylabel=row_ylabel if col == 0 else None,
                )
                _annotate_flux_panel(ax, panel)

            kind = _flux_panel_kind(panel)
            if kind == "q":
                if paper_style:
                    im_mag = _plot_jet_T_quiver_panel_paper(
                        ax,
                        field_q_on_ax,
                        var,
                        t_vmin=min_value,
                        t_vmax=max_value,
                        u_var="qx",
                        v_var="qy",
                        stride_lat=stride_lat,
                        stride_lon=stride_lon,
                        q_skip_mask=q_mask_on_ax,
                        x_lim=x_lim,
                        y_lim=y_lim,
                        legend_fontsize=(
                            paper_fonts["legend"] if paper_fonts is not None else 7.0
                        ),
                    )
                else:
                    im_mag = _plot_jet_T_quiver_panel(
                        ax,
                        field_q_on_ax,
                        var,
                        t_vmin=min_value,
                        t_vmax=max_value,
                        u_var="qx",
                        v_var="qy",
                        stride_lat=stride_lat,
                        stride_lon=stride_lon,
                        quiver_ref=q_key,
                        quiver_label=f"q {q_tag}",
                        x_lim=x_lim,
                        y_lim=y_lim,
                        q_skip_mask=q_mask_on_ax,
                    )
            elif show_grad_j_panels and kind == "grad":
                im_mag = _plot_jet_T_quiver_panel(
                    ax,
                    field_grad,
                    var,
                    t_vmin=min_value,
                    t_vmax=max_value,
                    u_var="qx",
                    v_var="qy",
                    stride_lat=stride_lat,
                    stride_lon=stride_lon,
                    quiver_ref=g_key,
                    quiver_label=f"grad T {g_tag}",
                    x_lim=x_lim,
                    y_lim=y_lim,
                )
            elif show_grad_j_panels:
                im_mag = _plot_jet_T_Jtrace_panel(
                    ax,
                    field_j_grad,
                    var,
                    t_vmin=min_value,
                    t_vmax=max_value,
                    j_tr_vmin=0.0,
                    j_tr_vmax=j_tr_vmax,
                    x_lim=x_lim,
                    y_lim=y_lim,
                )
                q_k = _thin_field_map(field_j_grad, stride_lat, stride_lon).plot.quiver(
                    ax=ax,
                    u="dTdx",
                    v="dTdy",
                    x="lon",
                    y="lat",
                    scale=_QUIVER_SCALE,
                    add_guide=False,
                    color="w",
                    alpha=0.9,
                )
                _add_ws10_quiverkey(q_k, g_key, f"grad T {g_tag}")
                ax.text(
                    0.02,
                    0.88,
                    f"trace(J) max (p99)={j_tr_vmax:.2e}",
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.5, pad=2),
                )

    if im_mag is not None:
        if paper_style:
            fig.subplots_adjust(top=0.90, bottom=0.14, left=0.02, right=0.98)
            header_parts = []
            if paper_mask_caption:
                header_parts.append(paper_mask_caption)
            if main_title:
                header_parts.append(str(main_title))
            if header_parts:
                fig.suptitle(
                    " - ".join(header_parts),
                    fontsize=paper_fonts["header"],
                    y=0.98,
                )
            cbar = fig.colorbar(
                im_mag, ax=axs, location="bottom", shrink=0.48, pad=0.02, aspect=28
            )
            cbar.set_label(
                var + " " + labels.get(var, ""),
                fontsize=paper_fonts["cbar_label"],
                labelpad=3,
            )
            cbar.ax.tick_params(
                labelsize=paper_fonts["cbar_ticks"], length=2, width=0.6
            )
        else:
            cbar = fig.colorbar(
                im_mag, ax=axs, location="bottom", shrink=0.5, pad=0.06
            )
            cbar.set_label(var + " " + labels.get(var, ""), fontsize=20)
            cbar.ax.tick_params(labelsize=18)

    if output_dir is not None:
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if output_basename:
            stem = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in str(output_basename)
            )
            path = out / f"{stem}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved {path}")
        else:
            filename = "Fig_q_snapshots_" + str(main_title) + ".jpg"
            fig.savefig(out / filename, bbox_inches="tight")

    plt.show()
    plt.close(fig)
    return fig


def show_q_deep_zoom_paper(
    spat_dist_df: pd.DataFrame,
    flux_by_model: dict[str, dict[str, np.ndarray]],
    *,
    time_label: str,
    variable: str = "2mT",
    borders_file: str | None = None,
    output_dir: str | None = None,
    output_basename: str | None = None,
    dpi: int = 300,
    vector_display_boost: float | None = None,
    q_display_boost: float | None = None,
    use_quiver_lens: bool = True,
    display_magnitude_gamma: float = 0.45,
    display_min_arrow_frac: float = 0.15,
    display_max_arrow_frac: float = 1.0,
    q_mask_mode: str = "none",
    q_mask_skip_quantile_pct: float = 0.0,
    q_mask_abs_threshold: float | None = None,
    q_deep_zoom_x: tuple[float, float] | None = None,
    q_deep_zoom_y: tuple[float, float] | None = None,
    q_deep_zoom_stride: int | None = None,
    paper_lon_extent_mult: float = 3.0,
    paper_lat_extent_mult: float = 2.0,
    paper_fig_width: float | None = None,
    paper_fig_height: float = 5.0,
    temperature_unit_label: str | None = None,
    show_mask_dots: bool = True,
):
    """Paper deep-zoom only — thin wrapper around ``show_q_snapshots`` (panel index 2)."""
    return show_q_snapshots(
        spat_dist_df,
        flux_by_model,
        variable=variable,
        main_title=str(time_label),
        borders_file=borders_file,
        output_dir=output_dir,
        output_basename=output_basename,
        dpi=dpi,
        vector_display_boost=vector_display_boost,
        q_display_boost=q_display_boost,
        use_quiver_lens=use_quiver_lens,
        display_magnitude_gamma=display_magnitude_gamma,
        display_min_arrow_frac=display_min_arrow_frac,
        display_max_arrow_frac=display_max_arrow_frac,
        q_mask_mode=q_mask_mode,
        q_mask_skip_quantile_pct=q_mask_skip_quantile_pct,
        q_mask_abs_threshold=q_mask_abs_threshold,
        q_deep_zoom_x=q_deep_zoom_x,
        q_deep_zoom_y=q_deep_zoom_y,
        q_deep_zoom_stride=q_deep_zoom_stride,
        show_grad_j_panels=False,
        temperature_unit_label=temperature_unit_label,
        q_panel_indices=(2,),
        paper_style=True,
        paper_lon_extent_mult=paper_lon_extent_mult,
        paper_lat_extent_mult=paper_lat_extent_mult,
        paper_fig_width=paper_fig_width,
        paper_fig_height=paper_fig_height,
    )


def show_grad_K_debug_snapshots(*args, **kwargs):
    """Alias for ``show_q_snapshots`` (grad T / trace(J) on the second row)."""
    return show_q_snapshots(*args, **kwargs)


def get_target_grid(target_res: str):
    coords = get_target_coords(target_res)
    if target_res == 'high':
        target_grid = xr.DataArray(coords=coords, dims=['y','x']) 
    elif target_res == 'low':
        target_grid = xr.DataArray(coords=coords, dims=['y','x']) 
    return target_grid

def get_target_coords(target_res: str):
    if target_res == 'high':
        coords = {'y': range(2698000-1000, 1354000, -2000), 'x': range(3910000+1000, 5062000, 2000)} 
    elif target_res == 'low':
        coords = {'y': range(2698000-8000, 1354000, -16000),'x': range(3910000+8000, 5062000, 16000)} 
    return coords

def show_spatial_errors(spat_err_df: pd.DataFrame, target_res, output_dir: str, main_title: str = None,
                        minmax_flag: bool = False, borders_file: str = None):
    # Set up the target grid
    target_grid = get_target_grid(target_res=target_res)

    # Load borders files if available
    if borders_file:
        gdf_bn = gpd.read_file(borders_file)

    # Count available target vars and models
    my_target_variables = spat_err_df['variable'].unique()
    my_models = spat_err_df['model'].unique()

    # Set up figure
    rig_max = len(my_target_variables)
    col_max = len(my_models)
    fig, axs = plt.subplots(rig_max, col_max, figsize=(7*col_max,8*int(rig_max)), sharey=True, sharex=True, constrained_layout=True)
    if main_title is not None: 
        fig.suptitle(main_title, fontsize=22) 
    labels = {'WS': 'Magnitude Diff. [m/s]',
              '2mT': 'Magnitude Diff. [K]'}    
    cmap_vars = {'2mT':'coolwarm',
                 'WS': 'RdBu_r'}
    # Loop over variables
    for row,tv in enumerate(my_target_variables):
        spat_err_df_tv = spat_err_df[(spat_err_df['variable'] == tv)]
        cmap = cmap_vars[tv]
        # min_value = min(spat_err_df_tv[(spat_err_df_tv['variable'] == var)]['min'])
        # max_value = max(spat_err_df_tv[(spat_err_df_tv['variable'] == var)]['max'])
        # if min_value < 0 and max_value>0:
        #     max_value = max(abs(max_value), abs(min_value))
        #     min_value = -max(abs(max_value), abs(min_value))
        min_value =-1.5
        max_value = 1.5
        # Loop over models
        for col, mod in enumerate(my_models):
            axs[row, col].set_xlim([target_grid.coords['x'].min().values,target_grid.coords['x'].max().values])
            axs[row, col].set_ylim([target_grid.coords['y'].min().values,target_grid.coords['y'].max().values])
            plot_tensor = spat_err_df_tv[spat_err_df_tv['model'] == mod]['spat_distr'].values
            map = from_torchtensor_to_xarray(plot_tensor[0], target_grid)
            we = map.plot.imshow(ax=axs[row, col], robust=True, add_colorbar=False, cmap=cmap, vmin = min_value, vmax=max_value)
            # Add borders to the plot
            gdf_bn.plot(ax=axs[row, col], color="black")
            # Remove axes ticks and labels
            axs[row, col].get_xaxis().set_visible(False)
            axs[row, col].get_yaxis().set_visible(False)
            # Add min-max info
            if minmax_flag == True:
                mod_min = np.round(spat_err_df_tv[spat_err_df_tv['model'] == mod]['min'].values[0],2)
                mod_max = np.round(spat_err_df_tv[spat_err_df_tv['model'] == mod]['max'].values[0],2)
                tit_add_on = ' [' + str(mod_min) +','+ str(mod_max) + ']'
                tit_all = spat_err_df_tv[spat_err_df_tv['model'] == mod]['model'].values[0] + tit_add_on
                axs[row, col].set_title(tit_all, fontsize=20) 
            else:
                axs[row, col].set_title(spat_err_df_tv[spat_err_df_tv['model'] == mod]['model'].values[0], fontsize=20) 
        # Plot colorbars
        cbar = fig.colorbar(we, ax=axs[row,:].ravel().tolist())
        cbar.set_label(tv + ' ' + labels[tv], fontsize=20)
        cbar.ax.tick_params(labelsize=18)
    # Plot and save to file
    filename = 'Fig_spatial_distrib_errors.jpg'
    plt.savefig(output_dir + filename, bbox_inches='tight')
    plt.show()
    plt.close()

def show_power_spectra(spectra_df, output_dir: str):
    # Set up resources
    wavelength_ticks = [300,100,50,20,10,5,4]
    vline_indexes = [5, 67, 150, 268]
    vlines_labels = ['(a)', '(b)', '(c)', '(d)']
    vlines_minmax = {'2mT': {'min': [35, 3-3, -12,-15-3]},
                    'WS': {'min': [28, -5, -16, -22]}}
    histtype_list = ['stepfilled','step','step','step','step','step','step','step','step']
    vline_length = {'2mT': 13,
                    'WS': 15}
    ylim = {'2mT': [None,50],
            'WS': [-25,45]}
    y_units = {'WS': '[m/s]',
            '2mT': '[C]'} 
    x_units = {'WS': 'km',
            '2mT': 'km'} 
    title = {'WS': 'RAPSD Kinetic Energy',
            '2mT': 'RAPSD 2-m Temperature'}
    tv_name = {'WS': 'KE',
            '2mT': '2mT'}
    model_to_color = {'GAN': 'g',
                        'UNET': 'b',
                        'Quadratic Interp.': 'r',
                        'Linear Interp.': '#17becf',
                        'COSMO-CLM': 'k',
                        'LDM_res': 'orange',
                        'LDM_PDE_res': 'blue',
                        'LMM_PDE_res': '#9467bd'}
    # Count available target vars and models
    my_models = spectra_df['model'].unique()
    my_variables = spectra_df['variable'].unique()
    # Set up figure
    row_max = len(vline_indexes) + 1
    col_max = len(my_variables)
    height_ratio = [1] * row_max
    height_ratio[0] = 3
    hight_fig = 1.5 * sum(height_ratio)
    fig, axs = plt.subplots(row_max, col_max, sharey=False, sharex=False, constrained_layout=True,
                            gridspec_kw = {'height_ratios':height_ratio}, figsize=(9,hight_fig))
    # Loop over variables
    for col,tv in enumerate(my_variables):
        # Loop over zooms
        for row in range(row_max):
            # Loop over models
            for mod_idx,mod in enumerate(my_models):
                col_i = model_to_color[mod]
                sp_i = spectra_df[(spectra_df['variable'] == tv) & (spectra_df['model'] == mod)].reset_index(drop=True)
                ax = axs[row,col]
                if mod == 'COSMO-CLM':
                    lw=2.5
                    face_col = 'gray'
                else:
                    lw=1
                    face_col=col_i
                if row == 0:
                    plot_spectrum1d(sp_i['fft_freq'][0], sp_i['spectra'][0].mean(axis=0), x_units=x_units[tv], y_units=y_units[tv],
                                                            color = col_i, wavelength_ticks=wavelength_ticks, lw=lw, label=mod, ax=ax)
                    legend = ax.legend()
                    tv_title = title[tv] 
                    ax.set_title(tv_title)
                    ax.set_xlim(10*np.log10(0.003),10*np.log10(0.26))
                    ax.set_ylim(top=ylim[tv][1], bottom=ylim[tv][0])
                    for vline_idx, vline_lab, vline_min in zip(vline_indexes, vlines_labels, vlines_minmax[sp_i['variable'][0]]['min']):
                        ax.vlines(10 * np.log10(sp_i['fft_freq'][0][vline_idx]), vline_min, vline_min+vline_length[tv], color='gray', linestyle='-', lw=1)
                        ax.annotate(vline_lab, (10 * np.log10(sp_i['fft_freq'][0][vline_idx]), vline_min+vline_length[tv]), fontsize=10, color='gray')
                else:
                    ax.hist(10*np.log10(sp_i['spectra'][0][:, vline_indexes[row-1]]+1e-8), bins=40, label=mod,
                            color = face_col, histtype=histtype_list[mod_idx], edgecolor= col_i)
                    tv_title = f'({chr(97+row-1)}) {tv_name[tv]} Frequency Distribution @{(1/sp_i["fft_freq"][0][vline_indexes[row-1]]):.0f} km'
                    ax.set_title(tv_title, fontsize=8)
                    ax.set_ylim(bottom = 0, top=730)
                    ax.set_ylabel("Count", fontsize=9)
                    if row==row_max-1:
                        power_units = rf"$10log_{{ 10 }}(\frac{{ {y_units[tv]}^2 }}{{ {x_units[tv]} }})$"
                        ax.set_xlabel(f"Power {power_units}", fontsize=9)
                ax.grid()
    # Plot and save to file
    filename = 'Fig_power_spectra.jpg'
    plt.savefig(output_dir + filename, bbox_inches='tight')
    plt.show()
    plt.close()    

def show_freq_distrib(freq_df, output_dir: str):
    # Set up resources
    nr_zoom = 2
    xlim = {'2mT': [10,27],
            'WS': [-1,11]}
    ylim = {'2mT': [18,18.6],
            'WS': [14.5,17.5]}

    units = {'WS': 'm/s',
            '2mT': '$^\circ$C'} 
    title = {'WS': '10-m Wind Speed',
            '2mT': '2-m Temperature'}
    model_to_color = {'GAN': 'g',
                        'UNET': 'b',
                        'Quadratic Interp.': 'r',
                        'Linear Interp.': '#17becf',
                        'COSMO-CLM': 'k',
                        'LDM_res': 'orange',
                        'LDM_PDE_res': 'blue',
                        'LMM_PDE_res': '#9467bd'}
    # Count available target vars and models
    my_models = freq_df['model'].unique()
    my_variables = freq_df['variable'].unique()
    # Set up figure
    row_max = nr_zoom
    col_max = len(my_variables)
    height_ratio = [1] * row_max
    height_ratio[0] = 2
    hight_fig = 2.25 * sum(height_ratio)
    fig, axs = plt.subplots(row_max, col_max, sharey=False, sharex=False, constrained_layout=True,
                            gridspec_kw = {'height_ratios':height_ratio}, figsize=(9,hight_fig))
    # Loop over variables
    for col,tv in enumerate(my_variables):
        # Loop over rows
        for row in range(row_max):
            # Loop over models
            for mod in my_models:
                col_i = model_to_color[mod]
                freq_i = freq_df[(freq_df['variable'] == tv) & (freq_df['model'] == mod)]
                ax = axs[row,col]
                if mod == 'COSMO-CLM':
                    lw=2.5
                else:
                    lw=1
                ax.plot(freq_i['x_s'], np.log(freq_i['freq_distr'].astype(np.float32)), color=col_i, linewidth=lw, label=mod)
                ax.set_xlabel(f"[{units[tv]}]")
                ax.set_ylabel(f"Log(freq distrib)")
                if row == 0:
                    legend = ax.legend()
                    title_prefix = ''
                    # Create a Rectangle patch
                    rect = patches.Rectangle((xlim[tv][0], ylim[tv][0]), xlim[tv][1]-xlim[tv][0], ylim[tv][1]-ylim[tv][0],
                                            linestyle='--', linewidth=2,edgecolor='gray', facecolor='none')
                    ax.add_patch(rect)
                else:
                    title_prefix = 'Zoom-in: '
                    ax.set_xlim(xlim[tv][0],xlim[tv][1])
                    ax.set_ylim(bottom=ylim[tv][0], top=ylim[tv][1])
                ax.set_title(title_prefix + title[tv])
                ax.grid()
    # Plot and save to file
    filename = 'Fig_freq_distrib.jpg'
    plt.savefig(output_dir + filename, bbox_inches='tight')
    plt.show()
    plt.close()

def show_metrics(metrics, output_dir):
    # Set up plotting resources
    box_palette = {
        'Quadratic Interp.': 'r',
        'Linear Interp.': '#17becf',
        'UNET': 'b',
        'GAN': 'g',
        'VAE_res': 'pink',
        'LDM_res': 'orange',
        'LDM_PDE_res': 'blue',
        'LMM_PDE_res': '#9467bd',
    }
    y_ref = [0,0,1,1,0,0,1,1]
    # Plot boxplots
    sns.set_theme(font_scale=1.5, style="whitegrid")
    g = sns.catplot(data=metrics, kind='box', x="model", y="value", col="metric", row='var', hue='model', native_scale=True, sharey=False, margin_titles=True, palette=box_palette, showmeans=True,
                    meanprops={'marker':'v','markerfacecolor':'w','markeredgecolor':'black','markersize':'8'})
    for i,ax in enumerate(g.axes.flat):
        for label in ax.get_xticklabels():
            label.set_rotation(90)
        ax.axline((0, y_ref[i]), slope=0, linestyle='--', color='gray', linewidth=3)
        ax.set(xlabel=None)
    g._legend.remove()
    # Save to file
    g.savefig(output_dir + 'Fig_metrics.jpg')
