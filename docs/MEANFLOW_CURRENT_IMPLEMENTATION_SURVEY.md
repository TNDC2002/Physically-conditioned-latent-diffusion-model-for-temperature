# Survey: current MeanFlow pipeline vs `MEANFLOW_INTEGRATION_PLAN.md`

**Purpose:** inventory the **implemented** MeanFlow fork (README pipeline **B**), compare it to the **written plan** in [MEANFLOW_INTEGRATION_PLAN.md](./MEANFLOW_INTEGRATION_PLAN.md), and list every related file. Use this before cleanup/refinement toward **LMM** ([LMM_PIPELINE_PLAN.md](./LMM_PIPELINE_PLAN.md)).

**Important:** the plan’s early “replace Denoiser + Decoder” wording describes a **pixel-space residual** MeanFlow path. The **LMM** target is different: **latent** `z_R` like LDM, **only** the denoiser block swapped. This survey describes **what exists today**, not LMM.

---

## 1. Sketched model (plan) vs actual implementation

### 1.1 What the integration plan assumed (Stages 1–3, §3 + Phase D)

| Plan element | Stated in `MEANFLOW_INTEGRATION_PLAN.md` | **Actual in code** |
|----------------|------------------------------------------|----------------------|
| Stage-3 target | \(R_{gt} = y_{hr} - \hat{y}_{up}\), MeanFlow estimates **\(R\)** in same space, \(\hat{y}_{final} = \hat{y}_{up} + \hat{R}\) | **Matches:** `r_gt = y_hr - y_up`, loss on bridge of `r_gt`, infer `y_up + r_hat`. |
| Conditioning | Stage-1 **static-Z VAE encoder** reused to build context before MeanFlow | **Matches in forward, with caveats:** `build_context` → `AFNOConditionerNetCascade.forward` → for input 0, **`self.autoencoder[0].encode(static)[0]`** (see [`conditioner.py`](../src/models/components/ldm/conditioner.py) `process_input`). So the static encoder is **not** config-only — it runs every step. Caveats: wrong `stage1_encoder_ckpt`, `strict=False`, and `train_autoenc: true` (see **§3**). |
| Replace “Denoiser + Decoder” | Single MeanFlow block instead of latent denoise + decode of **residual** | **Interpretation mismatch with LDM paper:** code **never** runs `Encoder_S1` / `Decoder_S1` on **\(R\)** for Stage-3. MeanFlow state **`x_t`** has shape of **pixel** `r_gt` (`in_channels: 1` in `meanflow_residual.yaml`), not latent `z_R`. So the plan’s **wording** aligns with **pixel** \(R\); it does **not** match “LDM denoiser only” / **LMM** (latent). |
| Normalize \(R_{gt}\) | Phase D §3: “Normalize \(R_{gt}\) with chosen residual normalizer” | **Not implemented** — `r_gt` used raw. |
| JVP teacher + adaptive L2 | §1, Phase D | **Implemented:** `MeanFlowCore.compute_teacher_error` + `adaptive_l2_loss`. |
| One-step inference | Phase E | **Implemented:** `generate_residual_one_step` in `meanflow_infer.py`. |
| Numerical safeguards (clamp, NaN) | Phase E acceptance | **Not implemented** in `meanflow_infer.py`. |
| Inference logging (`pred_residual_norm`, etc.) | Phase F | **Not implemented** in `utils/inference_utils.py`. |
| PDE | “No PDE changes” / legacy untouched | **Matches:** no PDE in `HrResidualMeanFlowLitModule`; LDM PDE unchanged. |
| Stage-1 static encoder ckpt into context | Phase C / 6.1 | **Partially implemented:** `stage1_encoder_ckpt` loads into `context_encoder.autoencoder[0]` with `strict=False`. **Experiment** often points to **residual** VAE ckpt, not necessarily static-Z Stage-1 (README already flags this). |
| `ae_mode: static_ctx` on MeanFlow-only AE | Phase C, config contract | **Present** on both `legacy_autoencoder` and first cascade AE in `meanflow_residual.yaml`. **Prior bundle** (`legacy_autoencoder`) only uses **merge + `unet()`** in forward — `static_ctx` there is **misleading** for Stage-3 training (does not drive `shared_step` encode path). |
| Unit tests | Phase A: shapes, finite, seed | **`tests/test_meanflow_core.py`** exercises `MeanFlowCore` only — **no** test for `HrResidualMeanFlowLitModule` or Hydra instantiate. |
| CFG inside MeanFlow | §1.5 | **Not implemented** (no CFG knobs wired in module). |

### 1.2 Refined sketch of **current** implementation (what code actually is)

This is the **pixel residual MeanFlow** fork (README **B**), not LMM:

```
batch (low_res, y_hr, static [, ts])
    → no_grad: y_up = legacy_autoencoder.nn_lr_and_merge_with_static(low_res, static)
               → legacy_autoencoder.unet(merged)
    → r_gt = y_hr - y_up                    # shape like y_hr, C=1 (config)
    → z_ctx = context_encoder([(static, 0s), (low_res, 0s)])
    → MeanFlowCore.compute_train_targets(r_gt)  # bridge on r_gt
    → MFUNet(x_t, t, r, context=z_ctx) + JVP teacher → adaptive_l2_loss

Inference: predict_final → y_up + generate_residual_one_step (rand x_t, t=1, r=0)
```

**Takeaway:** the repo **fully implements** the plan’s **Stage-3 pixel residual + JVP + one-step** story. It **does not** implement latent `z_R` training, residual **VAE encode/decode** around MeanFlow, or several **nice-to-have** plan items (normalize `R`, inference logging, infer safeguards, CFG).

---

## 2. File inventory (current MeanFlow pipeline)

### 2.1 Core Python (MeanFlow-specific)

| File | Role |
|------|------|
| [`src/models/meanflow_module.py`](../src/models/meanflow_module.py) | `HrResidualMeanFlowLitModule`: prior `y_up`, `r_gt`, context, train loss, `predict_residual` / `predict_final`. |
| [`src/models/components/meanflow/meanflow_core.py`](../src/models/components/meanflow/meanflow_core.py) | `MeanFlowCore`: `sample_t_r`, `build_bridge_state`, `compute_train_targets`, `single_step_generate`, JVP `compute_teacher_error`, `adaptive_l2_loss`. |
| [`src/models/components/meanflow/__init__.py`](../src/models/components/meanflow/__init__.py) | Exports `MeanFlowCore`. |
| [`src/models/components/ldm/denoiser/mf_unet.py`](../src/models/components/ldm/denoiser/mf_unet.py) | `MFUNet`: dual `(t, r)` embeddings + UNet body; `forward(x_t, t, r, context=None)`. |
| [`src/models/components/ldm/denoiser/meanflow_infer.py`](../src/models/components/ldm/denoiser/meanflow_infer.py) | `generate_residual_one_step` (stateless, `torch.no_grad()`). |
| [`src/models/components/ldm/denoiser/__init__.py`](../src/models/components/ldm/denoiser/__init__.py) | Re-exports `MFUNet` (with `UNetModel`, `DDIMSampler`, `LitEma`). |
| [`src/models/components/ldm/conditioner.py`](../src/models/components/ldm/conditioner.py) | **`AFNOConditionerNetBase`**: each branch calls `autoencoder[i].encode(x[i])[0]` then `proj` + AFNO analysis — this is where **static `z`** enters the **context** path for Flow B slot 0. |
| [`configs/experiment/downscaling_VAE_static_2mT.yaml`](../configs/experiment/downscaling_VAE_static_2mT.yaml) (and `_MIG`) | Intended **Stage-1** train of VAE on HR **static** only; weights should feed `stage1_encoder_ckpt` for Flow B alignment. |

### 2.2 Configs (Hydra)

| File | Role |
|------|------|
| [`configs/model/meanflow_residual.yaml`](../configs/model/meanflow_residual.yaml) | Instantiates `HrResidualMeanFlowLitModule`, `MFUNet` (1 ch), `MeanFlowCore`, `legacy_autoencoder`, `context_encoder`, `stage1_encoder_ckpt` / freeze flags. |
| [`configs/experiment/downscaling_MEANFLOW_res_2mT.yaml`](../configs/experiment/downscaling_MEANFLOW_res_2mT.yaml) | Overrides `model.legacy_autoencoder.unet_regr.ckpt_path`, `model.stage1_encoder_ckpt` (often **absolute paths**). |

### 2.3 Inference integration

| File | Role |
|------|------|
| [`utils/inference_utils.py`](../utils/inference_utils.py) | Branch `model_type == 'meanflow-residual'` → `model.predict_final(...)`. |

### 2.4 Tests

| File | Role |
|------|------|
| [`tests/test_meanflow_core.py`](../tests/test_meanflow_core.py) | Unit tests for `MeanFlowCore` only (`B,1,8,8`-style tensors). |

### 2.5 Documentation (MeanFlow / stages / README B)

| File | Role |
|------|------|
| [`MEANFLOW_INTEGRATION_PLAN.md`](./MEANFLOW_INTEGRATION_PLAN.md) | Original phased plan + file contract (written for **pixel** residual MeanFlow). |
| [`MEANFLOW_TRAINING_STAGES_GUIDE.md`](./MEANFLOW_TRAINING_STAGES_GUIDE.md) | How to run static AE / UNET / MeanFlow stages; now states it matches **current** fork **B**. |
| [`README.md`](../README.md) | Pipeline **B** diagram = **current** implementation. |
| [`LMM_PIPELINE_PLAN.md`](./LMM_PIPELINE_PLAN.md) | **Target** LMM (latent); contrasts with **B**. |

### 2.6 Upstream reference (not imported by `src/` at runtime)

| Path | Role |
|------|------|
| [`MeanFlow/meanflow.py`](../MeanFlow/meanflow.py) | Upstream reference; logic was adapted into `meanflow_core.py`. |
| [`MeanFlow/models/dit.py`](../MeanFlow/models/dit.py) | Time-interface reference per plan; **not** used in training path. |
| Other files under `MeanFlow/` | Standalone submodule / vendor tree. |

### 2.7 Shared code touched for MeanFlow (not MeanFlow-only)

| File | MeanFlow-related change |
|------|-------------------------|
| [`src/models/ae_module.py`](../src/models/ae_module.py) | `ae_mode: static_ctx` branch in `preprocess_batch` (two-item batch, `smt,smt` return) — supports **static AE training** and conditioner configs; **legacy LDM** routing unchanged when mode unset. |

### 2.8 Not found / minimal linkage

- **No** `downscaling_MEANFLOW_res_UV.yaml` (plan mentioned UV optional).
- **No** `grep` hits for `meanflow` in `notebooks/*.ipynb` in this repo snapshot (notebooks may still load models via generic paths).
- **`src/train.py`**: generic Hydra instantiate — no MeanFlow-specific logic; experiment yaml selects model.

---

## 3. Flow B — static-Z VAE, context, and where the “paper gap” actually bites

Your Flow B story in one line: **Stage 1** = train VAE on **static `z`** only → **Stage 2** = reuse that **encoder** so **context** encodes `z` (and LR branch) → **MeanFlow** replaces **denoiser+decoder on `R`**, generating **pixel `R̂`** from context; **UNet prior** unchanged.

### 3.1 What the code **does** match

- **Context uses the first cascade `AutoencoderKL` on raw `static`:** `conditioner.py` runs `encode(static)[0]` for branch 0, not `legacy_autoencoder` and not `preprocess_batch` on a residual.
- **`stage1_encoder_ckpt`:** `HrResidualMeanFlowLitModule.__init__` loads into **`context_encoder.autoencoder[0]`** only — the correct **object** for “reuse Stage-1 static encoder.”
- **MeanFlow on `r_gt` + UNet prior:** as in §1.2.

### 3.2 Logic bugs / paper–code gaps (hitlist — **not** “yaml only”)

| Issue | Why it matters |
|--------|----------------|
| **`stage1_encoder_ckpt` often = residual VAE ckpt** | Weights trained for **\(R\)** VAE, loaded with `strict=False` into **static-Z encoder** architecture → silent skip/mismatch; context is **not** the Stage-1 static-Z representation you think. |
| **`train_autoenc: true`** in [`meanflow_residual.yaml`](../configs/model/meanflow_residual.yaml) | `AFNOConditionerNetBase` does `autoencoder[i].requires_grad_(train_autoenc)` — the “Stage-1” encoder **keeps training** during MeanFlow unless you set `train_autoenc: false` and/or **`freeze_stage1_encoder: true`** in the Lightning module. That can violate “freeze encoder after Stage 1.” |
| **`freeze_stage1_encoder: false` by default** | Same: first-slot encoder is **not** frozen after ckpt load unless you flip the flag. |
| **`strict=False` on `load_state_dict`** | Hides shape / key mismatches between Stage-1 file and `context_encoder.autoencoder[0]`. |
| **`legacy_autoencoder.ae_mode: static_ctx`** | **Different module instance** from `context_encoder.autoencoder[0]`. `HrResidualMeanFlowLitModule` **never** calls `encode`/`decode` on `legacy_autoencoder` for context — only **merge + `unet`**. So `static_ctx` on **`legacy_autoencoder`** is **misleading / dead for the Flow B story** and confuses audits (looks like “static VAE lives here” but it does not). |
| **Architecture alignment** | First-slot encoder (`in_dim: 18`, `ch_mult: 3`, …) must match **static tensor channel count** and the **same** architecture used when you trained `downscaling_VAE_static_2mT`; otherwise even a correct ckpt path can be wrong. |

### 3.3 What is **not** missing in code (clarification)

- The **edited VAE / `static_ctx`** path for **context** is **active in forward** via `conditioner.py`, not “config only.”
- The **bug** is mostly **wiring and defaults** (wrong ckpt, trainable encoder, misleading duplicate AE on `legacy_autoencoder`), not “MeanFlow forgot to call an encoder.”

---

## 4. Config-driven tensor shapes (current)

From `meanflow_residual.yaml`:

- **`MFUNet`:** `in_channels` / `out_channels`: **1** (single-channel residual field).
- **Prior `unet_regr`:** `in_ch: 32`, `out_ch: 1` (matches LDM-style merged LR+static input → HR prior).
- **`context_encoder`:** `embed_dim_out: 256`, AFNO cascade with static + LR branches.

For **LMM**, expect **`MFUNet` channels = latent width** (e.g. 32), not 1 — see [LMM_PIPELINE_PLAN.md](./LMM_PIPELINE_PLAN.md).

---

## 5. Implications for “cleanup / refine → LMM”

| Current asset | Toward LMM |
|----------------|------------|
| `MeanFlowCore` | **Reuse** on latent tensors (same API; `x0` becomes `z_R`-shaped). |
| `MFUNet` | **Reuse class**, **change** `in_channels` / `out_channels` / spatial size to match VAE latent. |
| `meanflow_infer.py` | **Replace or parallel** helper: one-step in **latent**, then decode (mirror LDM infer). |
| `HrResidualMeanFlowLitModule` | **Do not extend** as LMM; add **new** `LightningModule` (or heavily forked file) for latent path; keep or deprecate pixel module per project policy. |
| `meanflow_residual.yaml` / `downscaling_MEANFLOW_res_2mT.yaml` | **Keep** for reproducibility of fork **B**; add **`lmm.yaml`** + new experiment for LMM. |
| `inference_utils.py` | Add **`model_type`** for LMM; keep `meanflow-residual` during transition. |
| `ae_module.py` `static_ctx` | Still useful for **static Stage-1 training** and for **context_encoder** slot-0 `AutoencoderKL` training path; **decouple** misleading `static_ctx` on **`legacy_autoencoder`** in `meanflow_residual.yaml` (see **§3**). |
| `conditioner.py` + `context_encoder` yaml | **Fix** ckpt / `train_autoenc` / `freeze_stage1_encoder` so Flow B matches “train on `z`, reuse encoder, then MeanFlow” (see **§3**). |

---

## 6. Summary table

| Topic | Status |
|-------|--------|
| Parallel module (no overwrite of `LatentDiffusion`) | **Done** |
| Pixel `r_gt` MeanFlow + JVP + adaptive L2 | **Done** |
| Frozen prior UNet inside `legacy_autoencoder` | **Done** |
| Context via `AFNOConditionerNetCascade` | **Done** (`encode(static)` on slot 0 — see **§3**) |
| Flow B **static-Z VAE story** (correct Stage-1 ckpt + frozen encoder + no misleading `legacy_autoencoder` flags) | **Often wrong in default yaml / experiment** — see **§3** |
| Residual **latent** encode/decode + MeanFlow on `z_R` (**LMM**) | **Not done** |
| Plan: normalize `R_gt` | **Not done** |
| Plan: inference logging / safeguards / CFG | **Not done** |
| Plan: integration tests for full module | **Not done** |

This survey is the baseline for deciding **what to delete, deprecate, or refactor** when you implement LMM alongside fork **B**.

**After LMM exists:** follow **[LMM_PIPELINE_PLAN.md](./LMM_PIPELINE_PLAN.md) § Pipeline B — retirement, merge options, and anti-“trash” hygiene** (Phases **R-1–R-5**). For **B-remove / B-merge**, the plan requires **deleting** B-only files from the tree (**no** in-repo `legacy/` rename shadow copies; use git history). **Before** treating Flow B as “done,” close or document the static-Z VAE / context items in **§3** above and the **Flow B static-Z VAE / context** table in the LMM plan. **Per-phase diagrams:** [`LMM_PHASE_IMPLEMENTATION_LOG.md`](./LMM_PHASE_IMPLEMENTATION_LOG.md).
