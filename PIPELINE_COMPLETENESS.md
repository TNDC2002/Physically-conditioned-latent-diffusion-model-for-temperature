# Pipeline completeness report

This file summarizes what you need to run the full pipeline and what was missing (and fixed or still required). The doc was not fully trusted; checks were done by tracing configs, imports, and scripts.

---

## What you need to run the pipeline

1. **Dataset** – Sample (~45 GB) or full (~330 GB)  
   - Scripts: `data/download_sample_dataset.sh` or `data/download_full_dataset.sh`  
   - **Caveat (sample):** The sample script uses `https://localhost:8080/...` (Zenodo via SSH reverse proxy). To download without a proxy, use direct Zenodo URLs and adjust the script (e.g. `https://zenodo.org/records/12934521/files/sample_data.zip?download=1`).

2. **Pretrained checkpoints** (~15 GB)  
   - Script: `pretrained_models/download_pretrained_models.sh`  
   - Provides: `UNET_2mT.ckpt`, `VAE_residual_2mT.ckpt`, `LDM_residual_2mT.ckpt`, etc., plus `pde_loss_model_checkpoint.ckpt`.

3. **Environment**  
   - `pip install -r requirements.txt`  
   - Set **`PROJECT_ROOT`** to the repo root (e.g. `export PROJECT_ROOT=/path/to/Physically-conditioned-latent-diffusion-model-for-temperature`). The README does not mention this; `configs/paths/default.yaml` requires it.

4. **Missing code (critical)**  
   - The repo does **not** contain the `src/data/` package. Training and the inference notebook expect:
     - `src.data.downscaling_datamodule.DownscalingDataModule`
     - `src.data.components.downscaling_dataset.DownscalingDataset`
   - These exist in the **DiffScaler** codebase. You must get them from [DiffScaler](https://github.com/DSIP-FBK/DiffScaler) (copy the files below into this repo). Without these, `train.py` and `notebooks/models_inference.ipynb` cannot run.

---

## Why DiffScaler? Which files exactly?

**How we know it’s from DiffScaler**

1. **README**  
   - It states: *"This work builds heavily on the [DiffScaler](https://github.com/DSIP-FBK/DiffScaler) codebase."*

2. **`train.log` in this repo**  
   - The log is from runs on another machine. The stack traces use absolute paths under `/usr/project/xtmp/par55/DiffScaler/`, for example:
     - `File ".../DiffScaler/src/data/downscaling_datamodule.py", line 71, in setup`
     - `File ".../DiffScaler/src/data/components/downscaling_dataset.py", line 22, in __init__`
   - So the code was run from a checkout named “DiffScaler” that contained `src/data/downscaling_datamodule.py` and `src/data/components/downscaling_dataset.py`. The failure was about missing `metadata.csv`, not missing Python files—so those modules were present there.

3. **Imports in this repo**  
   - `notebooks/models_inference.ipynb` and `notebooks/metric_computation.ipynb` do:
     - `from src.data.downscaling_datamodule import DownscalingDataModule`
     - `from src.data.components.downscaling_dataset import DownscalingDataset`
   - Hydra config uses `_target_: src.data.downscaling_datamodule.DownscalingDataModule`. So the required module/class names and package layout match DiffScaler’s `src/data/` tree.

**Exact files to take from DiffScaler**

From the [DiffScaler GitHub](https://github.com/DSIP-FBK/DiffScaler) (branch `main`), the `src/data/` tree is:

| Path in DiffScaler | Purpose |
|--------------------|--------|
| `src/data/downscaling_datamodule.py` | `DownscalingDataModule` (Lightning DataModule, creates train/val/test datasets) |
| `src/data/components/downscaling_dataset.py` | `DownscalingDataset` (PyTorch Dataset; used in datamodule and in notebooks) |

**Steps**

1. Clone or download DiffScaler:  
   `git clone https://github.com/DSIP-FBK/DiffScaler.git`
2. Copy into this repo (from this repo’s root):
   - `DiffScaler/src/data/downscaling_datamodule.py` → `src/data/downscaling_datamodule.py`
   - `DiffScaler/src/data/components/downscaling_dataset.py` → `src/data/components/downscaling_dataset.py`
3. Ensure package structure: create `src/data/__init__.py` and `src/data/components/__init__.py` (empty files) if they don’t exist, so `src.data` and `src.data.components` are importable.

**Optional**

- DiffScaler’s `src/data/` tree (from GitHub API) only lists those two `.py` files and the `components` folder; there are no other files under `src/data` in the repo. If you copy the whole `src/data/` folder, you get exactly those two modules. If any `__init__.py` are missing in DiffScaler, add them as above.

---

## Data layout expected by the code

After downloading and (for sample) symlinking, the code expects something like:

- `$PROJECT_ROOT/data/` (or `data_dir` from config)
  - `static_var/dtm_2km_domain_trim_EPSG3035.tif`
  - `static_var/land_cover_classes_2km_domain_trim_EPSG3035.tif`
  - `static_var/lat_2km_domain_trim_EPSG3035.tif`
  - `normalization_data.pkl`
  - `plotting_resources/borders_downscaling_domain_3035.geojson`
  - `metadata.csv` (train/val; used by `DownscalingDataModule` / `DownscalingDataset`)
  - `metadata_test_paper_sample.csv` (test; used by the inference notebook)
  - Sample/full netCDF or HDF5 data (e.g. under `sample_dataset/` or similar), as referenced by the metadata files.

The sample download script creates `data/sample_dataset` → symlink to the extracted sample data; the datamodule (from DiffScaler) will expect metadata and paths consistent with that layout.

---

## Fixes applied in this repo

1. **`pretrained_models/download_pretrained_models.sh`**  
   - `unzip pretrained_models` → `unzip -o pretrained_models.zip`.  
   - PDE checkpoint: it is a single `.ckpt` file; download saved as `pde_loss_model_checkpoint.ckpt` and removed the incorrect `unzip`/`.zip` handling.

2. **`configs/data/downscaling.yaml`**  
   - Added so Hydra can load the data config (previously `configs/data/` was empty and `train.yaml` references `data: downscaling.yaml`).

3. **`configs/experiment/downscaling_LDM_res_2mT.yaml` and `downscaling_LDM_res_UV.yaml`**  
   - `ckpt_path` was a hardcoded path on another machine. It now uses `ckpt_path: ${paths.pretrained_models_dir}LDM_residual_2mT.ckpt` (and the UV variant) so that after downloading pretrained models, training uses the local `pretrained_models/` dir.

---

## Optional / notebook-only

- **`pretrained_models/outputs/`**  
  - Notebooks like `Fig_snapshots.ipynb`, `Fig_metrics.ipynb`, `metric_computation.ipynb` expect precomputed outputs under `pretrained_models/outputs/` (e.g. `results_trained_models_2mT.pkl`, `metrics_trained_models.pkl`). These are not downloaded by the pretrained script; you generate them by running inference/metrics, or they may be provided elsewhere.

---

## Summary

| Item | Status |
|------|--------|
| Dataset (sample or full) | Scripts present; sample uses proxy – may need URL/config change for direct Zenodo |
| Pretrained checkpoints | Script fixed; run from `pretrained_models/` |
| `PROJECT_ROOT` | Required; set before training/inference |
| `configs/data/downscaling.yaml` | Added |
| `ckpt_path` in LDM experiments | Switched to local `paths.pretrained_models_dir` |
| **`src/data/` (datamodule + dataset)** | **Missing; get from DiffScaler** |

So: **downloading the dataset and pretrained checkpoints is necessary but not sufficient.** You also need to add the **`src/data/`** package from DiffScaler and set **`PROJECT_ROOT`** to run the full pipeline.
