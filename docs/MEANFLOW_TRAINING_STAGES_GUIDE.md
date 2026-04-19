# MeanFlow 3-Stage Training Guide

This guide explains how to run the **current** pixel-residual MeanFlow fork (README pipeline **B**, `HrResidualMeanFlowLitModule`) in three stages while keeping the legacy `LDM_PDE + UNET` flow unchanged. For the **target** **Latent Meanflow Model (LMM)** — same **latent** Stage‑2 as LDM, **only** the denoiser swapped for MeanFlow on `z_R` — see **[LMM_PIPELINE_PLAN.md](./LMM_PIPELINE_PLAN.md)**.

## Scope and Goal

- New flow target: replace LDM residual generation (`Denoiser + Decoder`) with MeanFlow residual generation.
- Keep legacy flow available: do not edit or remove existing LDM experiments/configs.
- Stage order is strict:
  1. Stage-1 static encoder training
  2. Stage-2 UNET upscaler training
  3. Stage-3 MeanFlow residual training

## Before You Start

- Work from repository root:
  - `/home/chuongtnd/git-repo/Physically-conditioned-latent-diffusion-model-for-temperature`
- Ensure Python environment is active (same environment used for existing training runs).
- Confirm data is accessible and identical across all stages.
- Record the output checkpoint path after each stage; next stage depends on it.

---

## Stage-1: Train Static Encoder (Context Encoder Source)

Purpose: train encoder on static HR input only, so it provides stable static context features.

### 1) Prepare a Stage-1 experiment config

Use a Stage-1 config dedicated to static context training. If you do not already have one, create a config from current AE setup and set:

- AE route to static-only:
  - `model.ae_mode: static_ctx`
  - `model.use_for: meanflow_context_only`
- Make sure dataset batch still returns `(low_res, high_res, static, [time])`, because static tensor is needed by `static_ctx` mode.
- Keep legacy AE/residual configs untouched.

### 2) Run Stage-1 training

Example command pattern:

```bash
python src/train.py experiment=<your_stage1_static_experiment>
```

### 3) Validate Stage-1 output

- Confirm training/validation losses are finite.
- Save best or last checkpoint path, called:
  - `STAGE1_ENCODER_CKPT=<path_to_stage1_encoder_ckpt>`

---

## Stage-2: Train UNET Upscaler (Legacy-Compatible)

Purpose: produce `y_up` prior that MeanFlow will refine via residual learning.

### 1) Use current UNET experiment

Current baseline experiment:

- `configs/experiment/downscaling_UNET_2mT.yaml`

### 2) Run Stage-2 training

```bash
python src/train.py experiment=downscaling_UNET_2mT
```

### 3) Validate Stage-2 output

- Confirm inference quality is reasonable for prior field.
- Save checkpoint path:
  - `STAGE2_UNET_CKPT=<path_to_unet_ckpt>`

---

## Stage-3: Train MeanFlow Residual Branch

Purpose: train MeanFlow model to predict residual `R_hat` and combine with Stage-2 output:

- `R_gt = y_hr - y_up`
- `y_final = y_up + R_hat`

### 1) Use MeanFlow experiment config

Current experiment:

- `configs/experiment/downscaling_MEANFLOW_res_2mT.yaml`

Current model config:

- `configs/model/meanflow_residual.yaml`

### 2) Wire checkpoints into Stage-3 config

In `configs/experiment/downscaling_MEANFLOW_res_2mT.yaml`, set:

- `model.legacy_autoencoder.unet_regr.ckpt_path: ${STAGE2_UNET_CKPT}`
- `model.stage1_encoder_ckpt: ${STAGE1_ENCODER_CKPT}`

### 3) Confirm default freeze behavior

Current default is:

- `freeze_stage1_encoder: false`

So Stage-1 encoder is **not frozen by default** in MeanFlow flow unless you set this to `true` explicitly.

### 4) Run Stage-3 training

```bash
python src/train.py experiment=downscaling_MEANFLOW_res_2mT
```

### 5) Stage-3 sanity checks

- Loss decreases on short run/tiny overfit.
- Outputs are finite (no NaN/Inf).
- Tensor shapes match between `R_gt`, `R_hat`, and `y_up`.

---

## Inference with New MeanFlow Flow

Use `model_type: meanflow-residual` in your inference entrypoint to activate the new branch added in `utils/inference_utils.py`.

- Legacy `model_type: ldm` remains unchanged.
- New branch computes final output via:
  - Stage-2 upscaler prior
  - one-step MeanFlow residual generation
  - post-fusion `y_final = y_up + R_hat`

---

## Recommended Run Order Checklist

- [ ] Stage-1 static encoder trained and checkpoint saved
- [ ] Stage-2 UNET trained and checkpoint saved
- [ ] Stage-3 config updated with both checkpoint paths
- [ ] Stage-3 MeanFlow training runs without shape/runtime errors
- [ ] MeanFlow inference tested with `model_type=meanflow-residual`
- [ ] Legacy LDM inference smoke test still passes (`model_type=ldm`)

---

## Troubleshooting

- If Stage-3 fails in context building:
  - verify static tensor exists in batch and `ae_mode: static_ctx` is set where needed.
- If outputs are unstable:
  - reduce batch size, verify checkpoint compatibility, and confirm input normalization consistency.
- If Stage-2 checkpoint load fails:
  - check `out_ch` and variable target alignment (`2mT` vs `UV`).
