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


def _as_numpy_field(value) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "values"):
        value = value.values
    return np.asarray(value, dtype=float).squeeze()


def _compute_q_field_numpy(T: np.ndarray, dx: float = 1.0, dy: float = 1.0, eps: float = 1e-6):
    """q = -||grad T||^2 grad T (same definition as flux-loss notebook)."""
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
    grad_sq = dTdx ** 2 + dTdy ** 2
    qx = -grad_sq * dTdx
    qy = -grad_sq * dTdy
    q_mag = np.sqrt(qx ** 2 + qy ** 2 + eps)
    return qx, qy, q_mag


def target_res_for_field(field: np.ndarray) -> str:
    for res in ("high", "low"):
        coords = get_target_coords(res)
        if field.shape == (len(coords["y"]), len(coords["x"])):
            return res
    raise ValueError(
        f"Field shape {field.shape} does not match high/low target grids "
        f"(high={(len(get_target_coords('high')['y']), len(get_target_coords('high')['x']))}, "
        f"low={(len(get_target_coords('low')['y']), len(get_target_coords('low')['x']))})."
    )


def field_to_latlon_da(field: np.ndarray, target_res: str) -> xr.DataArray:
    grid = get_target_grid(target_res=target_res)
    lats = grid.coords["y"].values
    lons = grid.coords["x"].values
    field = _as_numpy_field(field)
    if field.shape != (len(lats), len(lons)):
        raise ValueError(f"Shape {field.shape} != grid {(len(lats), len(lons))} for {target_res=}")
    return xr.DataArray(field, coords={"lat": lats, "lon": lons}, dims=["lat", "lon"])


def show_temperature_flux_snapshots(
    spat_dist_df: pd.DataFrame,
    *,
    variable: str = "2mT",
    output_dir: str | None = None,
    main_title: str | None = None,
    borders_file: str | None = None,
    quiver_scale: float = 200.0,
    pick_stride_full: int = 50,
    pick_stride_zoom: int = 10,
    zoom_x_lim: tuple[float, float] = (4150000, 4450000),
    zoom_y_lim: tuple[float, float] = (1748000, 2070000),
):
    """Fig_snapshots **WS10 / UV layout** (2 columns: full | zoom): one panel each with
    **2mT coolwarm** background + **q** quiver overlay.

    Same setup as wind (``WS10`` imshow + ``U10``/``V10`` quiver, stride 50/10, scale 200,
    quiverkey box, borders) but background uses **2mT [K]** ``min``/``max`` — not ``|q|`` jet.
    Arrow length encodes ``|q|``.
    """
    spat_dist_df = spat_dist_df[spat_dist_df["variable"] == variable].copy()
    if spat_dist_df.empty:
        raise ValueError(f"No rows for variable={variable!r}")

    my_models = list(spat_dist_df["model"].unique())
    min_value = float(spat_dist_df["min"].min())
    max_value = float(spat_dist_df["max"].max())
    if min_value < 0 and max_value > 0:
        bound = max(abs(min_value), abs(max_value))
        min_value, max_value = -bound, bound

    gdf_bn = gpd.read_file(borders_file) if borders_file else None
    n_rows = len(my_models)
    fig, axs = plt.subplots(nrows=n_rows, ncols=2, figsize=(8, 5 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axs = np.array([axs])
    if main_title is not None:
        fig.suptitle(str(main_title), fontsize=22)

    ref_quiver_len = None
    for sim_row, sim in enumerate(my_models):
        row = spat_dist_df[spat_dist_df["model"] == sim].iloc[0]
        T = _as_numpy_field(row["spat_distr"])
        target_res = target_res_for_field(T)
        qx, qy, q_mag = _compute_q_field_numpy(T)
        map_t = field_to_latlon_da(T, target_res)
        map_qx = field_to_latlon_da(qx, target_res)
        map_qy = field_to_latlon_da(qy, target_res)
        field_map = xr.merge(
            [map_t.rename("T"), map_qx.rename("qx"), map_qy.rename("qy")],
            compat="no_conflicts",
            join="outer",
            combine_attrs="override",
        )
        field_map["qx"] = field_map["qx"].transpose()
        field_map["qy"] = field_map["qy"].transpose()
        field_map["T"] = field_map["T"].transpose()

        target_grid = get_target_grid(target_res=target_res)
        if target_res == "low":
            stride_full = max(2, pick_stride_full // 8)
            stride_zoom = max(1, pick_stride_zoom // 2)
        else:
            stride_full, stride_zoom = pick_stride_full, pick_stride_zoom

        for col in range(2):
            ax = axs[sim_row, col]
            if col == 0:
                x_lim = [target_grid.coords["x"].min().values, target_grid.coords["x"].max().values]
                y_lim = [target_grid.coords["y"].min().values, target_grid.coords["y"].max().values]
                pick_stride = stride_full
            else:
                x_lim = list(zoom_x_lim)
                y_lim = list(zoom_y_lim)
                pick_stride = stride_zoom
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

            we = field_map["T"].plot.imshow(
                ax=ax,
                robust=True,
                add_colorbar=False,
                x="lon",
                y="lat",
                cmap="coolwarm",
                alpha=0.85,
                vmin=min_value,
                vmax=max_value,
            )
            we0 = field_map.thin(pick_stride).plot.quiver(
                ax=ax,
                u="qx",
                v="qy",
                x="lon",
                y="lat",
                scale=quiver_scale,
                add_guide=False,
            )
            if ref_quiver_len is None:
                ref_quiver_len = float(
                    np.nanpercentile(
                        np.sqrt(field_map["qx"].values ** 2 + field_map["qy"].values ** 2),
                        95,
                    )
                )
            vec_len = ref_quiver_len if ref_quiver_len > 0 else 1.0
            maxstr = f"{vec_len:.2e}"
            plt.quiverkey(
                we0,
                0.9,
                0.07,
                vec_len,
                maxstr,
                labelpos="S",
                coordinates="axes",
                fontproperties={"size": 13},
            ).set_zorder(11)
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
            if gdf_bn is not None:
                gdf_bn.plot(ax=ax, color="black")
            ax.get_xaxis().set_visible(False)
            if col == 0:
                ax.yaxis.set_tick_params(labelleft=False, left=False)
                ax.set_ylabel(str(sim), fontsize=20)
            else:
                ax.get_yaxis().set_visible(False)

    cbar = fig.colorbar(we, ax=axs, location="bottom")
    cbar.set_label(f"{variable} [K]", fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    if output_dir is not None:
        filename = "Fig_temp_flux_" + str(main_title) + ".jpg"
        plt.savefig(output_dir + filename, bbox_inches="tight")
    plt.show()
    plt.close()
    return fig


def _symmetric_2mT_limits(min_v: float, max_v: float) -> tuple[float, float]:
    if min_v < 0 and max_v > 0:
        bound = max(abs(min_v), abs(max_v))
        return -bound, bound
    return min_v, max_v


def _style_flux_map_axes(ax, x_lim, y_lim, gdf_bn, ylabel: str | None = None, show_ylabel: bool = False):
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if gdf_bn is not None:
        gdf_bn.plot(ax=ax, color="black")
    ax.get_xaxis().set_visible(False)
    if show_ylabel and ylabel:
        ax.set_ylabel(ylabel, fontsize=20)
        ax.yaxis.set_tick_params(labelleft=False, left=False)
    else:
        ax.get_yaxis().set_visible(False)


def _plot_2mT_background(
    ax,
    field_map: xr.Dataset,
    vmin: float,
    vmax: float,
    x_lim,
    y_lim,
    gdf_bn,
    *,
    pick_stride: int | None = None,
    quiver_scale: float = 200.0,
    ylabel: str | None = None,
    show_ylabel: bool = False,
):
    """Coolwarm 2mT background; optional black q quiver (arrow length encodes |q|)."""
    mappable = field_map["T"].plot.imshow(
        ax=ax,
        x="lon",
        y="lat",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        robust=True,
        alpha=0.85 if pick_stride is not None else 1.0,
    )
    _style_flux_map_axes(ax, x_lim, y_lim, gdf_bn, ylabel=ylabel, show_ylabel=show_ylabel)
    if pick_stride is None:
        return mappable
    thin = field_map.thin(pick_stride)
    q_plot = thin.plot.quiver(
        ax=ax,
        u="qx",
        v="qy",
        x="lon",
        y="lat",
        scale=quiver_scale,
        add_guide=False,
        color="k",
    )
    ref_len = float(
        np.nanpercentile(np.sqrt(field_map["qx"].values ** 2 + field_map["qy"].values ** 2), 95)
    )
    if ref_len <= 0:
        ref_len = 1.0
    plt.quiverkey(
        q_plot,
        0.9,
        0.07,
        ref_len,
        f"{ref_len:.2e}",
        labelpos="S",
        coordinates="axes",
        fontproperties={"size": 11},
    ).set_zorder(11)
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
    return mappable


def show_temperature_flux_four_panel(
    spat_dist_df: pd.DataFrame,
    *,
    variable: str = "2mT",
    output_dir: str | None = None,
    main_title: str | None = None,
    borders_file: str | None = None,
    quiver_scale: float = 200.0,
    pick_stride_full: int = 50,
    pick_stride_zoom: int = 10,
    zoom_x_lim: tuple[float, float] = (4150000, 4450000),
    zoom_y_lim: tuple[float, float] = (1748000, 2070000),
):
    """Four columns per model — **two separate 2mT [K] colorbars**.

    | Col | Content |
    |-----|---------|
    | 1–2 | 2mT only (full \\| zoom), coolwarm |
    | 3–4 | 2mT coolwarm background + **q** quiver (WS10 layout); flux from arrows, not ``|q|`` colors |
    """
    spat_dist_df = spat_dist_df[spat_dist_df["variable"] == variable].copy()
    if spat_dist_df.empty:
        raise ValueError(f"No rows for variable={variable!r}")

    my_models = list(spat_dist_df["model"].unique())
    gdf_bn = gpd.read_file(borders_file) if borders_file else None
    n_rows = len(my_models)
    fig, axs = plt.subplots(nrows=n_rows, ncols=4, figsize=(16, 5 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axs = np.array([axs])
    if main_title is not None:
        fig.suptitle(str(main_title), fontsize=22)

    im_T = None
    im_Tq = None
    for sim_row, sim in enumerate(my_models):
        row = spat_dist_df[spat_dist_df["model"] == sim].iloc[0]
        T = _as_numpy_field(row["spat_distr"])
        target_res = target_res_for_field(T)
        qx, qy, _ = _compute_q_field_numpy(T)
        map_t = field_to_latlon_da(T, target_res)
        map_qx = field_to_latlon_da(qx, target_res)
        map_qy = field_to_latlon_da(qy, target_res)
        field_map = xr.merge(
            [map_t.rename("T"), map_qx.rename("qx"), map_qy.rename("qy")],
            compat="no_conflicts",
            join="outer",
            combine_attrs="override",
        )
        for v in ("T", "qx", "qy"):
            field_map[v] = field_map[v].transpose()

        t_vmin, t_vmax = _symmetric_2mT_limits(float(row["min"]), float(row["max"]))
        grid = get_target_grid(target_res=target_res)
        x_full = [grid.coords["x"].min().values, grid.coords["x"].max().values]
        y_full = [grid.coords["y"].min().values, grid.coords["y"].max().values]
        x_zoom, y_zoom = list(zoom_x_lim), list(zoom_y_lim)

        if target_res == "low":
            s_full = max(2, pick_stride_full // 8)
            s_zoom = max(1, pick_stride_zoom // 2)
        else:
            s_full, s_zoom = pick_stride_full, pick_stride_zoom

        im_T = _plot_2mT_background(
            axs[sim_row, 0], field_map, t_vmin, t_vmax, x_full, y_full, gdf_bn,
            ylabel=str(sim), show_ylabel=True,
        )
        _plot_2mT_background(axs[sim_row, 1], field_map, t_vmin, t_vmax, x_zoom, y_zoom, gdf_bn)
        im_Tq = _plot_2mT_background(
            axs[sim_row, 2], field_map, t_vmin, t_vmax, x_full, y_full, gdf_bn,
            pick_stride=s_full, quiver_scale=quiver_scale,
        )
        _plot_2mT_background(
            axs[sim_row, 3], field_map, t_vmin, t_vmax, x_zoom, y_zoom, gdf_bn,
            pick_stride=s_zoom, quiver_scale=quiver_scale,
        )

    if im_T is not None:
        cbar_t = fig.colorbar(im_T, ax=axs[:, :2].ravel().tolist(), location="bottom", shrink=0.55, pad=0.02)
        cbar_t.set_label(f"{variable} [K]", fontsize=20)
        cbar_t.ax.tick_params(labelsize=18)
    if im_Tq is not None:
        cbar_tq = fig.colorbar(im_Tq, ax=axs[:, 2:].ravel().tolist(), location="bottom", shrink=0.55, pad=0.14)
        cbar_tq.set_label(f"{variable} [K]", fontsize=20)
        cbar_tq.ax.tick_params(labelsize=18)

    if output_dir is not None:
        plt.savefig(output_dir + f"Fig_temp_flux_{main_title}.jpg", bbox_inches="tight")
    plt.show()
    plt.close()
    return fig


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
