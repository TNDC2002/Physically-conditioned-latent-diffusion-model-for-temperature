import geopandas as gpd
from matplotlib import pyplot as plt
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


def _q_mask_skip_from_quantile(qx: np.ndarray, qy: np.ndarray, skip_quantile_pct: float):
    """Mask bottom ``skip_quantile_pct`` of |q| per field (per time slice / model)."""
    if skip_quantile_pct <= 0:
        return qx, qy, np.zeros(np.shape(qx), dtype=bool), None
    q_mag = np.hypot(np.asarray(qx, float), np.asarray(qy, float))
    thr = float(np.percentile(q_mag, skip_quantile_pct))
    skip = q_mag <= thr
    qx_out = np.asarray(qx, dtype=float).copy()
    qy_out = np.asarray(qy, dtype=float).copy()
    qx_out[skip] = np.nan
    qy_out[skip] = np.nan
    return qx_out, qy_out, skip, thr


def _plot_q_mask_skip_dots(ax, field_map: xr.Dataset, skip_mask: np.ndarray, pick_stride: int):
    """Purple markers at grid points where |q| was below the skip quantile."""
    if skip_mask is None or not np.any(skip_mask):
        return
    skip_da = xr.DataArray(
        skip_mask.astype(bool),
        coords={"lat": field_map["lat"], "lon": field_map["lon"]},
        dims=["lat", "lon"],
    )
    thinned = field_map.thin(pick_stride)
    skip_thin = skip_da.thin(pick_stride).values
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


def _plot_jet_T_quiver_panel(
    ax,
    field_map: xr.Dataset,
    t_var: str,
    *,
    t_vmin: float,
    t_vmax: float,
    u_var: str,
    v_var: str,
    pick_stride: int,
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
    q_plot = field_map.thin(pick_stride).plot.quiver(**quiver_kw)
    _add_ws10_quiverkey(q_plot, quiver_ref, quiver_label)
    if q_skip_mask is not None:
        _plot_q_mask_skip_dots(ax, field_map, q_skip_mask, pick_stride)
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
    q_mask_skip_quantile_pct: float = 0.0,
    q_deep_zoom_x: tuple[float, float] | None = None,
    q_deep_zoom_y: tuple[float, float] | None = None,
    q_deep_zoom_stride: int | None = None,
):
    """Plot precomputed flux fields (7 panels per model, 3×3 WS10 layout).

    Row 1: flux **q** — domain | zoom | **deep zoom** (panel G, denser quiver).

    Physics (∇T, recovered **J**, q) must be computed in the notebook; pass
    ``flux_by_model[model]`` with keys ``T``, ``dTdx``, ``dTdy``, ``J_trace``, ``qx``, ``qy``.
    ``J_trace`` = trace(J) = |∇T|² (scalar summary of recovered J; not real conductive K).

    Quiver display: set ``use_quiver_lens=False`` to plot **raw** ``qx/qy`` (consistent with |q|
    histograms). ``use_quiver_lens=True`` (default) applies power-law compression (γ) plus
    min/max arrow floors so weak and extreme vectors remain visible.
    Set ``display_magnitude_gamma=1`` for linear lens; manual ``*_display_boost`` overrides lens.

    ``q_mask_skip_quantile_pct`` (e.g. ``50``): per model / time slice, hide the bottom
    that fraction of |q| (arrows omitted); skipped grid points are drawn as purple dots.
    ``0`` disables masking.
    """
    spat_dist_df = spat_dist_df[spat_dist_df["variable"] == variable].copy()
    if spat_dist_df.empty:
        raise ValueError(f"No rows for variable={variable!r}")

    gdf_bn = gpd.read_file(borders_file) if borders_file else None

    my_models = spat_dist_df["model"].unique()
    _validate_flux_by_model(flux_by_model, my_models)
    rig_max = len(my_models)
    var = variable
    labels = {"2mT": "[K]"}
    min_value = min(spat_dist_df["min"])
    max_value = max(spat_dist_df["max"])

    q_deep_x = q_deep_zoom_x if q_deep_zoom_x is not None else _DEFAULT_Q_DEEP_ZOOM_X
    q_deep_y = q_deep_zoom_y if q_deep_zoom_y is not None else _DEFAULT_Q_DEEP_ZOOM_Y
    q_deep_stride = (
        q_deep_zoom_stride if q_deep_zoom_stride is not None else _DEFAULT_Q_DEEP_ZOOM_STRIDE
    )

    n_panels = 7
    ncols = 3
    nrows_per_model = 3
    n_slots_per_model = nrows_per_model * ncols
    nrows = rig_max * nrows_per_model
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), constrained_layout=True
    )
    if nrows == 1:
        axs = np.array([axs])
    if main_title is not None:
        fig.suptitle(main_title, fontsize=22, y=1.02)
    if use_quiver_lens:
        arrow_note = (
            f"Arrows: display lens ON (γ={display_magnitude_gamma:g}, "
            f"min={display_min_arrow_frac:g}×key, max={display_max_arrow_frac:g}×key) — "
            "direction kept; lengths remapped for visibility"
        )
    else:
        arrow_note = "Arrows: display lens OFF — raw qx/qy (true relative magnitudes; may hide weak vectors)"
    if q_mask_skip_quantile_pct > 0:
        arrow_note += (
            f" | q mask: skip bottom {q_mask_skip_quantile_pct:g}% |q| per slice "
            "(purple dots = masked)"
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
        if use_quiver_lens and q_raw_max > 0 and q_raw_max < 1e-4:
            print(
                f"Warning [{sim}]: max |q|={q_raw_max:.3e} — lens remaps arrow lengths "
                f"(γ={display_magnitude_gamma:g}, min_frac={display_min_arrow_frac:g}); "
                "they are not proportional to physical |q|. Set use_quiver_lens=False to verify."
            )
        qx_plot, qy_plot, q_skip_mask, q_skip_thr = _q_mask_skip_from_quantile(
            flux["qx"], flux["qy"], q_mask_skip_quantile_pct
        )
        if q_mask_skip_quantile_pct > 0:
            n_skip = int(np.sum(q_skip_mask))
            print(
                f"[{sim}] q mask: skip bottom {q_mask_skip_quantile_pct:g}% |q| "
                f"(|q| <= {q_skip_thr:.3e}) -> {n_skip}/{q_skip_mask.size} pixels "
                f"({100.0 * n_skip / q_skip_mask.size:.1f}%)"
            )
        qx_d, qy_d, q_tag, q_key = _vectors_for_quiver_display(
            qx_plot, qy_plot, q_boost, **disp_kw
        )
        gx_d, gy_d, g_tag, g_key = _vectors_for_quiver_display(
            flux["dTdx"], flux["dTdy"], g_boost, **disp_kw
        )

        field_q = _numpy_to_field_map(
            {"T": flux["T"], "qx": qx_d, "qy": qy_d}, target_grid, t_var=var
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

        for panel in range(n_slots_per_model):
            grid_row = sim_row * nrows_per_model + panel // ncols
            col = panel % ncols
            ax = axs[grid_row, col]
            if panel >= n_panels:
                ax.set_visible(False)
                continue

            x_lim, y_lim, pick_stride = _panel_limits(
                panel,
                target_grid,
                q_deep_zoom_x=q_deep_x,
                q_deep_zoom_y=q_deep_y,
                q_deep_zoom_stride=q_deep_stride,
            )
            row_ylabel = sim if panel // ncols == 0 else ("grad T & J" if panel // ncols == 1 else "")
            _style_q_panel_ax(
                ax, x_lim, y_lim, gdf_bn, borders_file, sim, col, ylabel=row_ylabel if col == 0 else None
            )
            _annotate_flux_panel(ax, panel)

            kind = _flux_panel_kind(panel)
            if kind == "q":
                im_mag = _plot_jet_T_quiver_panel(
                    ax,
                    field_q,
                    var,
                    t_vmin=min_value,
                    t_vmax=max_value,
                    u_var="qx",
                    v_var="qy",
                    pick_stride=pick_stride,
                    quiver_ref=q_key,
                    quiver_label=f"q {q_tag}",
                    x_lim=x_lim,
                    y_lim=y_lim,
                    q_skip_mask=q_skip_mask if q_mask_skip_quantile_pct > 0 else None,
                )
            elif kind == "grad":
                im_mag = _plot_jet_T_quiver_panel(
                    ax,
                    field_grad,
                    var,
                    t_vmin=min_value,
                    t_vmax=max_value,
                    u_var="qx",
                    v_var="qy",
                    pick_stride=pick_stride,
                    quiver_ref=g_key,
                    quiver_label=f"grad T {g_tag}",
                    x_lim=x_lim,
                    y_lim=y_lim,
                )
            else:
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
                q_k = field_j_grad.thin(pick_stride).plot.quiver(
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
        cbar = fig.colorbar(im_mag, ax=axs, location="bottom", shrink=0.5, pad=0.06)
        cbar.set_label(var + " " + labels.get(var, ""), fontsize=20)
        cbar.ax.tick_params(labelsize=18)

    if output_dir is not None:
        filename = "Fig_q_snapshots_" + str(main_title) + ".jpg"
        plt.savefig(output_dir + filename, bbox_inches="tight")
    plt.show()
    plt.close()
    return fig


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
