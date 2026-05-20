import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "notebooks" / "test_anisotropic_flux_loss.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))

nb["cells"][9]["source"] = [
    "## Plot q vectors (WS10-style: 2 columns)\n",
    "\n",
    "Same **layout as WS10 wind** in `Fig_snapshots` / `show_snapshots`: **2 columns** (full domain | zoom), "
    "each panel = **scalar map + quiver** on lon/lat.\n",
    "\n",
    "Difference from wind: background is **2mT [K]** (`coolwarm`, saved `min`/`max`), not WS10 jet. "
    "**q** arrows on top; length shows |q|.\n",
]

nb["cells"][10]["source"] = """from utils.plotting_utils import show_temperature_flux_snapshots

for ts in selected_times:
    ts_df = plot_df[pd.to_datetime(plot_df['time_step']) == pd.to_datetime(ts)]
    show_temperature_flux_snapshots(
        ts_df,
        variable=PLOT_VAR,
        main_title=str(ts),
        borders_file=str(BORDERS_FILE) if BORDERS_FILE else None,
        output_dir=None,
        quiver_scale=200,
        pick_stride_full=quiver_stride_full,
        pick_stride_zoom=quiver_stride_zoom,
        zoom_x_lim=tuple(FIG_ZOOM_XLIM),
        zoom_y_lim=tuple(FIG_ZOOM_YLIM),
    )
""".splitlines(keepends=True)

nb["cells"][11]["source"] = [
    "# To save JPGs like Fig_snapshots: set output_dir=str(OUTPUT_DIR) + '/' in the cell above.\n",
]
nb["cells"][11]["outputs"] = []

p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"patched {p}")
