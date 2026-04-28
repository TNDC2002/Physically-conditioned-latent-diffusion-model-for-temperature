# LDM_res: Latent Diffusion Model for Meteorological Downscaling

<div align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
  <a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
  <a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
</div>

## Overview

**LDM_res** implements a physics‑conditioned Latent Diffusion Model (LDM) for statistical downscaling of meteorological fields.  Given coarse ERA5 reanalysis inputs and static high‑resolution features, LDM_res reconstructs fine‑scale 2 m temperature with improved physical consistency via a novel PDE‑based regularization.

This repository provides:
- Training and inference code for the LDM_res model and baselines.
- Utilities to download sample/full datasets and pretrained checkpoints.
- Configuration templates for distributed training and single‑GPU workflows.
- Notebooks for evaluation metrics and result visualization, including physical‑loss computation.

## Pipeline block diagrams (`ae_flag: residual`)

**How to read these figures:** **A** is the legacy **LDM_PDE + UNet** stack. **B** is the **current codebase fork** (pixel-space MeanFlow on `r_gt`): it was added when the implementation drifted from the paper’s **latent** LDM Stage‑2 sketch — it is **not** the long-term target. The **target** pipeline is **LMM (Latent Meanflow Model)**: same latent path as **A** (encode `z_R`, decoder, PDE, fusion), with **only** the diffusion denoiser block swapped for MeanFlow on **`z_R`**. Block graph and build plan: **[LMM_PIPELINE_PLAN.md](./docs/LMM_PIPELINE_PLAN.md)** (Figure C).

### A — `LDM_PDE + UNet` (legacy)

```
Stage 0
  [x]───┐
  [z]───┴──►[merge HR grid]──►[UNET_regr]──►Ŷ_up
  [y]───────────────────────────────────►(loss vs y)


Stage 1
  [x][z]──►[merge]──►[UNET_regr]──►Ŷ_up───┐
  [y]────────────────────────────────────┴──►(R=y−Ŷ_up)──►[Encoder_S1]──┐
                                                                          ├──►[Decoder_S1]──►R̂
                                     R ──────────────────────────────────┘
                                     loss: |R−R̂| + KL(latent)


Stage 2
  [x][y][z]──►[preprocess: UNET_regr]──►R──►[Encoder_S1]──►z_R
  [z][x]──────────────────────────────────────►[encoder_ctx]──►context

  z_R + ε + t──►[q_sample]──►z_{R,t}───┐
  context (dict: encoder_ctx, T_c=x)───┴──►[denoiser]──►L_diff

  (if temp_pde_coef|temp_energy_coef > 0, pde_mode=temp)
  [denoiser]──►[Decoder_S1]──►field──►[PDE / energy]──►+ λ·L_phys  (added to L_diff)


Infer
  noise──►[sampler + denoiser + context(z,x)]──►ẑ_R──►[Decoder_S1]──►R̂
  [x][z]──►[merge+UNET_regr]──►Ŷ_up──►(+)──►Ŷ
```

### B — **Current** `MeanFlow_PDE + UNet` fork (pixel `r_gt`; interim implementation)

This diagram matches **`HrResidualMeanFlowLitModule`** today (`experiment=downscaling_MEANFLOW_res_2mT`). It is **not** the paper-aligned “replace the **latent** denoiser only” target; that target is **LMM** in [LMM_PIPELINE_PLAN.md](./docs/LMM_PIPELINE_PLAN.md).

Relative to **A Stage 2**, this fork **drops** the residual **latent** path (`Encoder_S1` → `q_sample` → denoiser → `Decoder_S1` on `z_R`) for the correction and instead runs **MeanFlow in pixel space on `r_gt = y − Ŷ_up`**, while reusing the frozen **UNET_regr** prior and a similar **context(z, x)** style.  
Optional static-only Stage‑1 for `context_encoder[0]` (`ae_mode: static_ctx`, `experiment=downscaling_VAE_static_2mT`) is orthogonal to the residual VAE in **A Stage 1** (on **R**, not Y).

```
Stage 0                          (same as A)
  [x][z]──►[merge]──►[UNET_regr]──►Ŷ_up
  [y]──────────────────────────────►(loss vs y)


Stage 1 — LDM branch (A only)    Stage 1 — MeanFlow-side static AE (optional, design)
  [x][y][z]──►R──►[VAE on R]       [z]──►[VAE / AE on Z only]──►ckpt → context_encoder[0]
       (experiment: VAE_res)     (experiment: VAE_static_ctx)


Stage 2 — LDM (A)                Stage 2 — MeanFlow (replaces latent denoise+decode for r)
  … z_R, q, denoiser, Decoder…   [x][y][z]──►[merge+UNET_regr]──►Ŷ_up───┐
                                   [y]───────────────────────────────────┴──► r_gt
                                   [z][x]──►[encoder_ctx]──►context
                                   r_gt──►[MeanFlowCore]──►(x_t,t,r,v_tgt)
                                   x_t,t,r,context──►[MFUNet]──► loss


Infer (MeanFlow)
  [x][z]──►[merge+UNET_regr]──►Ŷ_up───┐
  [z][x]──►[encoder_ctx]──►context──►[MFUNet one-step]──►r̂──┴──►(+)──►Ŷ

                    ┌─────┐
                    │ PDE │
                    └─────┘
```

**Implemented `experiment=downscaling_MEANFLOW_res_2mT` (verify):**  
`HrResidualMeanFlowLitModule` only **forward-uses** `legacy_autoencoder.nn_lr_and_merge_with_static` + **`unet`** (frozen); it does **not** run legacy **encode/decode** on `r`.  
`configs/model/meanflow_residual.yaml` sets `legacy_autoencoder.ae_mode: static_ctx` (so **`preprocess_batch` would be Z-only**, not R — that path is for other tooling, not this training step).  
`stage1_encoder_ckpt` in the experiment YAML currently points at **`VAE_residual_2mT.ckpt`** → loads into **`context_encoder.autoencoder[0]`** with `strict=False`. To match the **Z-only Stage‑1** idea above, that path should instead use the checkpoint from **`downscaling_VAE_static_2mT`** (naming like `VAE_static_ctx_2mT.ckpt`).

## Acknowledgments

This work builds heavily on the [DiffScaler](https://github.com/DSIP-FBK/DiffScaler) codebase.  We extend and adapt implementations by **Elena Tomasi**, **Gabriele Franch**, and **Marco Cristoforetti**—many thanks for their foundational contributions.

## Installation

```bash
# After cloning the project, breate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data & Model Download

### Sample Dataset (~45 GB)
```bash
cd data
bash download_sample_dataset.sh
```

### Full Dataset (~330 GB)
```bash
cd data
bash download_full_dataset.sh
# Alternative stable download if issues arise:
bash Better_download_full_dataset.sh
```

### Pretrained Checkpoints (~15 GB)
```bash
cd pretrained_models
bash download_pretrained_models.sh

# Once PDE loss model is downloaded you can run inference in notebooks/models_inference.ipynb
```

## Configuration

- **Parallel training:** YAML specs in `configs/trainer/` enable multi‑node/GPU jobs via PyTorch Lightning.
- **Experiment configs:** see `configs/experiment/downscaling_LDM_res_2mT.yaml`, etc.
  - Define dataset paths, model hyperparameters, logging, and checkpointing.
  - **NEW** fields:
    ```yaml
    trainable_parts:
      - "denoiser.output_blocks"
      - "autoencoder.decoder"
      - "denoiser.middle_block"
    ```
    to restrict training to submodules (compute‑limited mode).
  - All PDE‑loss weightings (`lambda_PDE`, etc.) are specified in these files.

## Training

### Full‑scale LDM (100 GB VRAM required)
```bash
python3 src/train.py experiment=downscaling_LDM_res_2mT
```
- Requires GPUs with ≥ 100 GB VRAM for end‑to‑end LDM fine‑tuning.
- Leverages distributed/multi‑GPU parallelism via `configs/trainer/*.yaml`.

### Compute‑Limited Mode (Single‑GPU)
- Edit `configs/experiment/downscaling_LDM_res_2mT.yaml`:
  - Set `trainable_parts` as shown above.
- Launch on a single GPU with:
```bash
python3 src/train.py experiment=downscaling_LDM_res_2mT
```
- Note: performance may degrade when training only decoder and selected blocks.

### SLURM Submission
- A template submission script is provided at `configs/experiment/Submitscript.sh`.
- **Edit file paths** to match your cluster environment before use.

> **Important:** We do **not** modify the UV predictor, UNET, or GAN architectures—only LDM components.

## Inference & Evaluation

- **GPU requirement:** ≥ 16 GB VRAM for single‑frame inference.
- Notebooks in `notebooks/` guide:
  - `models_inference.ipynb` for loading pretrained models and running inference.
  - `Fig_snapshots.ipynb` and others to reproduce visualizations and metrics.
- Metric computation and inference code has been extended to incorporate the PDE‑based physical loss—see `src/models/ldm_module.py` for implementation details.


## Further Resources

For detailed usage patterns, advanced configurations, and troubleshooting tips, please refer to the original DiffScaler README and documentation.

```bash
srun --jobid= <id> --pty watch -n 1 nvidia-smi
tensorboard --logdir logs/train/runs
tail -n 1 -f /home/chuongtnd/git-repo/Physically-conditioned-latent-diffusion-model-for-temperature/slurm_logs/LMM_res_2mT-11647.out
```