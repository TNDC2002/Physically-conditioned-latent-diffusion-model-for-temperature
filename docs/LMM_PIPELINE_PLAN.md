# Latent Meanflow Model (LMM) — pipeline plan

This document fixes a documentation mistake: **[README.md](../README.md)** (README **pipeline B**) describes the **current, messy implementation** (pixel-space MeanFlow on `r_gt`, `HrResidualMeanFlowLitModule`) that grew from the **gap between the paper’s LDM sketch and the codebase**. It is **not** the engineering target.

**TARGET (this file):** build a **new** pipeline — **Latent Meanflow Model (LMM)** — that **nearly duplicates** legacy **`LDM_PDE + UNet`**: same Stages 0–1, same residual preprocessing, **`Encoder_S1` → `z_R`**, same **context**, **`Decoder_S1`**, optional **PDE / energy** on the decoded field, same fusion at inference. The **only** deliberate swap is replacing the **diffusion denoiser block** (`q_sample` + discrete-time `UNetModel` training / DDIM sampling) with a **MeanFlow** objective and sampler on **latent `z_R`** (continuous `t`, `r`, `MFUNet`, JVP teacher, one-step or chosen sampler).

**Separation:** implement LMM in **new** modules + Hydra configs + inference `model_type`; **do not** overwrite `LatentDiffusion` / `ldm.yaml` / `model_type == 'ldm'`.

### Three pipelines in parallel (default strategy) — not an in-place “fix” of B

Yes: as written, **LMM is a third runnable pipeline**, alongside:

| # | Pipeline | Train / infer hook (today) |
|---|------------|------------------------------|
| 1 | **A — LDM** | `LatentDiffusion`, `experiment=downscaling_LDM_res_*`, `model_type == 'ldm'` |
| 2 | **B — pixel MeanFlow** | `HrResidualMeanFlowLitModule`, `experiment=downscaling_MEANFLOW_res_*`, `model_type == 'meanflow-residual'` |
| 3 | **C — LMM (new)** | new module + `experiment=downscaling_LMM_res_*`, new `model_type` (e.g. `'lmm'`) |

**Why a third path first:** same reason as the original MeanFlow integration plan — **no overwrite**, easy A/B metrics, and **B** stays reproducible for anything that already depends on it (notebooks, checkpoints, papers).

**This is not a commitment to three forever.** After LMM is validated, project policy can:

- **Retire B:** **delete** B-only modules and configs (see § *Pipeline B — retirement*; **no** `legacy/` folder for B — use git history); update README figure B; *or*
- **Keep B** as a lightweight baseline (pixel residual ablation); *or*
- **Replace B in place** (reuse filenames / one `model_type`) — *higher* risk of breaking existing runs; only if you explicitly choose that refactor.

The plan does **not** require editing **`HrResidualMeanFlowLitModule`** to become LMM; LMM is a **separate** implementation. Turning B into LMM inside the same class would blur pipelines and is **out of scope** for the written phases unless you change policy.

Phased implementation detail (**What** / **How** / **Acceptance checks**, plus **L.1** file contract and checklist) is in **§ Detailed work breakdown — LMM** below, written in the same style as [MEANFLOW_INTEGRATION_PLAN.md](./MEANFLOW_INTEGRATION_PLAN.md) §5.

---

## Requirements — phase block diagram log (mandatory)

While LMM (and later pipeline **B** retirement) is rolled out:

1. **After every phase** in **§ Detailed work breakdown — LMM** (**L-A … L-K**) and every **§ Pipeline B** cleanup phase (**R-1 … R-5**) that touches implementation, append one dated section to **[`LMM_PHASE_IMPLEMENTATION_LOG.md`](./LMM_PHASE_IMPLEMENTATION_LOG.md)**.
2. Each log entry **must** include:
   - **Block graph** (ASCII) of what is **actually implemented** after that phase — whole pipeline or **only the subgraph** this phase changed; use the same box/arrow style as Figures A–C in this plan; label **NEW** / **CHANGED** / **FROZEN** as appropriate.
   - **Placement note:** does the new/changed block connect where Figure **C** expects it?
   - **Gap note:** list any **missing** items vs that phase’s **Acceptance checks** here, or **mistakes** found when drawing the graph (drawing often exposes missing edges).
3. **Do not** replace or erase older log sections — append only, so history remains reviewable.

**Rationale:** you can see **what landed**, **whether it sits in the right layer**, and **if something was forgotten**, without re-reading the whole codebase each phase.

---

## Figure legend

| Figure | Meaning |
|--------|---------|
| **A** | **Legacy** — `LDM_PDE + UNet` (`LatentDiffusion`); unchanged by LMM work. |
| **B** | **Current mess** — what [README.md](../README.md) labels pipeline B today: pixel MeanFlow on `r_gt` (`HrResidualMeanFlowLitModule`); **not** the LMM target. |
| **C** | **TARGET — LMM** — same latent stack as A; **denoiser region** replaced by MeanFlow on `z_R`. |

---

## Figure A — Legacy `LDM_PDE + UNet` (reference)

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
  context (dict: encoder_ctx, T_c=x)───┴──►[denoiser: UNetModel]──►L_diff

  (if temp_pde_coef|temp_energy_coef > 0, pde_mode=temp)
  [denoiser out]──►[Decoder_S1]──►field──►[PDE / energy]──►+ λ·L_phys


Infer
  noise──►[DDIM + denoiser + context]──►ẑ_R──►[Decoder_S1]──►R̂
  [x][z]──►[merge+UNET_regr]──►Ŷ_up──►(+)──►Ŷ
```

**Code:** `src/models/ldm_module.py`, `configs/model/ldm.yaml`, `model_type == 'ldm'`.

---

## Figure B — **Current** implementation (README pipeline B; interim / messy)

This is **`HrResidualMeanFlowLitModule`**: MeanFlow runs on **pixel** `r_gt = y −Ŷ_up`, **not** on `z_R`. There is **no** `Encoder_S1` / `Decoder_S1` on the residual in this path — it diverges from the paper-style **latent** LDM diagram.

```
Stage 2 (as implemented today — NOT the LMM target)
  [x][y][z]──►[merge+UNET_regr]──►Ŷ_up───┐
  [y]───────────────────────────────────┴──► r_gt  (pixel)
  [z][x]──►[encoder_ctx]──►context

  r_gt──►[MeanFlowCore]──►(x_t,t,r,v)──►[MFUNet]──► L_MF

Infer
  …──►[one-step MFUNet]──►r̂──►(+)──►Ŷ_up──►Ŷ
```

**Code:** `src/models/meanflow_module.py`, `configs/model/meanflow_residual.yaml`, `model_type == 'meanflow-residual'`.

---

## Figure C — **TARGET — LMM** (`LDM_PDE` layout, denoiser = MeanFlow on **z_R**)

Same boxes as **A** except the **shaded region**: remove `q_sample` + discrete `UNetModel` loop; insert **MeanFlow** on latent state (bridge on `z_R`, `MFUNet(z-like, t, r, context)`). Decoder, PDE, and inference fusion **stay** as in A.

```
Stage 0, Stage 1  ═══════════════════════════════════  identical to Figure A


Stage 2 — LMM (TARGET)
  [x][y][z]──►[preprocess: UNET_regr]──►R──►[Encoder_S1]──►z_R
  [z][x]──────────────────────────────────────►[encoder_ctx]──►context

  ┌── REPLACE THIS REGION ONLY (conceptual) ─────────────────────────────────┐
  │  Was:  z_R + ε + t ──► [q_sample] ──► z_{R,t} ──► [UNetModel denoiser]      │
  │  Now:  z_R ──► [MeanFlowCore] ──► x_t, t, r, v_tgt  (continuous times)     │
  │        x_t, t, r, context ──► [MFUNet] ──► JVP teacher loss  (train)      │
  │        (inference: chosen MeanFlow sampler / one-step ──► ẑ_R  not DDIM)   │
  └────────────────────────────────────────────────────────────────────────────┘

  (if temp_pde_coef|temp_energy_coef > 0, pde_mode=temp)
  [latent prediction used for PDE surrogate]──►[Decoder_S1]──►field──►[PDE / energy]──►+ λ·L_phys
       └── same *attachment idea* as LDM: decode then physics on field; exact tensor
           (denoiser out vs one-step ẑ) must match one locked choice in implementation.


Infer (LMM — TARGET)
  [init / MeanFlow sampler + MFUNet + context]──►ẑ_R──►[Decoder_S1]──►R̂
  [x][z]──►[merge+UNET_regr]──►Ŷ_up──►(+)──►Ŷ
```

**Implementation sketch (new, parallel to legacy):**

- New `LightningModule` (e.g. `LatentMeanFlowLitModule` / `LMM`) that **reuses** the same batch layout and `shared_step` **data preparation** as `LatentDiffusion` (encode `R` → `z_R`, build `context_dict` with `T_c`, etc.).
- **`MFUNet`:** `in_channels` / `out_channels` = **latent width** (e.g. 32), not 1 — reuse [`MFUNet`](../src/models/components/ldm/denoiser/mf_unet.py) with LDM-shaped tensors.
- **[`MeanFlowCore`](../src/models/components/meanflow/meanflow_core.py):** unchanged algorithm; operates on `z_R`-shaped tensors.
- **New** Hydra: `configs/model/lmm.yaml`, `configs/experiment/downscaling_LMM_res_2mT.yaml`.
- **Inference:** new `model_type` (e.g. `'lmm'`) in [`utils/inference_utils.py`](../utils/inference_utils.py): encode `R` → latent → MeanFlow generate `ẑ_R` → decode → fusion, mirroring `'ldm'` but swapping the sampler block.

**Code reuse from B:** `MeanFlowCore`, `MFUNet` interface, possibly shared PDE helpers with `ldm_module.py`. **Do not** treat pixel `HrResidualMeanFlowLitModule` as the LMM implementation; it is **Figure B**, to be superseded or deprecated.

---

## What went wrong (why README B exists)

The paper’s Stage 2 is **latent**: `z_R`, denoiser, decoder. An early integration approximated “replace denoiser” as “run MeanFlow on **pixel residual** `r_gt`,” which is **simpler to wire** but **not** the same graph as LDM. README B documents **that** fork. **LMM (Figure C)** is the correction: MeanFlow **on the same latent** the LDM denoiser would have seen.

---

## Detailed work breakdown — LMM (mirror style of [MEANFLOW_INTEGRATION_PLAN.md](./MEANFLOW_INTEGRATION_PLAN.md) §5)

Each phase uses the same subsection pattern as that plan: **What** / **How** / **Acceptance checks**.

Phases are labeled **L-A … L-K** (“L” = latent / LMM) to avoid confusion with the existing MeanFlow plan’s **A–H** (pixel fork).

**Global acceptance (every phase L-A … L-K):** append one section to [`LMM_PHASE_IMPLEMENTATION_LOG.md`](./LMM_PHASE_IMPLEMENTATION_LOG.md) per **Requirements — phase block diagram log** above before closing the phase.

---

### Phase L-A — Lock scope, notation, and non-goals

**What**

- Freeze engineering decisions so LMM does not creep into unrelated refactors.

**How**

1. **In scope:** new `LightningModule`, new Hydra `model` + `experiment`, new `model_type` for inference; **reuse** `MeanFlowCore`, `MFUNet` **class**, frozen residual **VAE** (`AutoencoderKL`) + same `preprocess_batch` / `encode` / `decode` contract as `LatentDiffusion` when `ae_flag == 'residual'`; same `context_encoder` + `context_dict` pattern (including `T_c` for temperature PDE).
2. **Explicit swap only:** remove use of `register_schedule`, `q_sample`, integer `t`, and `UNetModel` **denoiser** training path for LMM; replace with `MeanFlowCore` + `MFUNet` on **tensor shaped like `latent_target`** (mean of VAE encode of residual, same as LDM `shared_step`).
3. **Out of scope for v1:** changing Stage-0 UNET architecture or training script; changing PDE **formula** (only **which decoded tensor** is passed in must be locked — see L-F); merging LMM into `LatentDiffusion` class body (prefer **separate** module file).
4. **Notation:** reuse integration plan disambiguation: **\(x_t\)** = MeanFlow bridge state (here: latent space, same rank as `z_R`); **context** = conditioner output / dict as in LDM; not pixel `r_gt`.

**Acceptance checks**

- Written one-page “PDE decode source” decision (L-F) agreed before coding L-E.
- No edits to `LatentDiffusion` forward contract for `'ldm'` checkpoints.

---

### Phase L-B — Latent `MFUNet` and `MeanFlowCore` readiness

**What**

- Ensure the **existing** MeanFlow building blocks operate on **latent** tensors `(B, C, H', W')` with `C` equal to LDM denoiser channels (e.g. **32** from [`configs/model/ldm.yaml`](../configs/model/ldm.yaml)), not `1` like [`meanflow_residual.yaml`](../configs/model/meanflow_residual.yaml).

**How**

1. Add Hydra-instanced `MFUNet` for LMM with `in_channels` / `out_channels` / `context_ch` **matching** the LDM denoiser + conditioner (copy numeric defaults from `ldm.yaml` denoiser block, then set `time_scale`, `separate_r_mlp` per ablation appetite — same guidance as integration plan §6.4).
2. Run a **forward-only** script or unit test: random `x_t` with latent shape + random `t,r` + dummy `context` → output shape **equals** `x_t`; enable grad, run a dummy backward through `MFUNet` only.
3. Run **`MeanFlowCore.compute_train_targets`** on a random `z_0` tensor of latent shape; assert keys `x_t, t, r, v_target` shapes; run **`compute_teacher_error`** with a trivial `backbone_model` closure on CPU with small spatial size to confirm JVP works at latent resolution (memory check).

**Acceptance checks**

- JVP path completes on at least one GPU with crop size reduced if needed (document max resolution in config comments).
- `MeanFlowCore` requires **no** fork for latent vs pixel (same code as today).

---

### Phase L-C — New `LightningModule` skeleton (`lmm_module.py` or agreed name)

**What**

- A **parallel** training module that owns: frozen `autoencoder`, optional `context_encoder`, `meanflow_core`, `mf_unet`, PDE hyperparameters mirroring `LatentDiffusion.__init__` (`pde_lambda`, `pde_mode`, `temp_pde_coef`, `temp_energy_coef`), optimizer hyperparameters (`lr`, `lr_warmup`, `loss_type` if reused), optional `trainable_parts`, optional **`use_ema`** targeting **`mf_unet`** only (not the old `denoiser` name).

**How**

1. New file e.g. [`src/models/lmm_module.py`](../src/models/lmm_module.py) (name TBD) — **do not** subclass `LatentDiffusion` unless it reduces duplication without importing `register_schedule`.
2. Constructor: instantiate the **same** YAML targets as LDM for `autoencoder` and `context_encoder` where possible (Hydra defaults list or explicit duplication in `lmm.yaml`).
3. **Do not** call `register_schedule` / `q_sample` / `p_losses` from LDM; optionally **import** small helpers from `ldm_module` only if extracted without side effects (prefer copy-paste minimal slice first, refactor to shared util in a follow-up).
4. Load VAE / optional partial weights exactly as LDM (`ae_load_state_file`) from config.
5. Log at rank 0 once: trainable parameter count for `mf_unet` vs frozen AE.

**Acceptance checks**

- `python -c "import hydra; ... instantiate lmm"` succeeds from repo root.
- `named_modules()` shows `autoencoder` frozen and `mf_unet` trainable (unless `trainable_parts` restricts).

---

### Phase L-D — `shared_step` data path parity with LDM

**What**

- Bit-for-bit **same** construction of `latent_target` and `context_dict` as [`LatentDiffusion.shared_step`](../src/models/ldm_module.py) for the same batch layout `(x, y, z, ts)`.

**How**

1. Copy the logic path: `ae_flag == 'residual'` → `preprocess_batch` → `encode` → mean only for latent target (same as LDM today); build `context_dict = {"T_c": x}` then merge `context_encoder` outputs.
2. Add an optional **assertion mode** (config flag): in dev runs, compare `latent_target` and `context_dict` keys/shapes between LDM and LMM on one fixed batch (CPU/GPU) — numbers may differ only if RNG in encode; use `encode(..., sample_posterior=False)` path if you need deterministic parity check.
3. Document batch length 3 vs 4 same as LDM.

**Acceptance checks**

- Shapes of `latent_target` match `gen_shape` used in LDM inference utilities for the same `loaded_data`.
- `context_dict` always contains `T_c` when temperature PDE is enabled, same as LDM.

---

### Phase L-E — MeanFlow **training** loss in latent (replace `p_losses`)

**What**

- Single training loss from MeanFlow JVP teacher + optional adaptive L2; **no** `MSE(denoiser_out, target_v)` from diffusion.

**How**

1. Let `z_0 = latent_target` (or explicitly sampled posterior if you choose to match stochastic LDM — **lock** mean vs sample in L-A).
2. `train_targets = meanflow_core.compute_train_targets(z_0)` → `x_t, t, r, v_target`.
3. Build `def backbone(x, r, t): return mf_unet(x, t, r, context=cond_tensor_or_dict)` consistent with [`MeanFlowCore.compute_teacher_error`](../src/models/components/meanflow/meanflow_core.py) argument order `(x_t, r, t)` (mirror working closure in [`meanflow_module.py`](../src/models/meanflow_module.py)).
4. `loss = meanflow_core.adaptive_l2_loss(meanflow_core.compute_teacher_error(..., create_graph=True))` in `training_step`; `create_graph=False` in `validation_step` if memory constrained.
5. **Optional:** add plain `l2` on velocity as config switch (integration plan allows MSE optional).

**Acceptance checks**

- Loss finite on one real batch; backward does not touch frozen `autoencoder` weights.
- Tiny overfit (1–2 batches): loss trend down when only `mf_unet` is trainable.

---

### Phase L-F — PDE / energy parity with LDM

**What**

- Same physics terms as `LatentDiffusion.p_losses` when `pde_mode == "temp"` (and UV branch if you support it), applied to a **decoded temperature field**.

**How**

1. **Design lock (required):** choose which latent tensor is decoded for PDE during training, e.g. (a) `decode(u_theta)` mid-step — usually wrong; (b) `decode(z_hat)` from **one-step** `single_step_generate` from current bridge — closer to “current estimate”; (c) **only** at validation: PDE on final sampled latent — cheaper. Document chosen option in `lmm.yaml` comments.
2. Implement by **reusing** methods on the new module (copy from `ldm_module.py` or extract `temperature_pde_loss` / `temperature_energy_loss` to a shared module **without** changing LDM behavior in the same PR, if possible split PR: extract helper + LDM call site unchanged).
3. Match coefficient names: `temp_pde_coef`, `temp_energy_coef` for parity with experiment yaml.

**Acceptance checks**

- With coefs set to zero, LMM numerically ignores PDE branch (same as LDM).
- With coefs > 0, loss includes finite physics term; gradient flows only into allowed tensors per design lock.

---

### Phase L-G — EMA, `trainable_parts`, optimizers

**What**

- Training ergonomics aligned with LDM for fair comparison.

**How**

1. If `use_ema`: `LitEma(self.mf_unet)` and `on_train_batch_end` update; validation can mirror `validation_step` double pass (`loss` vs `loss_ema`) like `LatentDiffusion`.
2. Reuse `configure_optimizers` pattern from `ldm_module.py` (AdamW + `ReduceLROnPlateau` on `val/loss` or `val/loss_ema`).
3. Support `trainable_parts` listing substrings of `mf_unet` if compute-limited (same idea as LDM experiment yaml comments in README).

**Acceptance checks**

- EMA weights differ from raw weights after one epoch when enabled.
- Scheduler monitors an existing logged key.

---

### Phase L-H — Latent **inference** (replace DDIM block only)

**What**

- Produce `ẑ_R` without `DDIMSampler` / integer diffusion; then **reuse** existing decode + residual fusion identical to `'ldm'` branch in [`utils/inference_utils.py`](../utils/inference_utils.py).

**How**

1. Add stateless helper e.g. `generate_latent_one_step(mf_unet, meanflow_core, context, shape, device, dtype)` in new file **`src/models/components/ldm/denoiser/lmm_infer.py`** or extend [`meanflow_infer.py`](../src/models/components/ldm/denoiser/meanflow_infer.py) with clearly named latent variant — avoid breaking pixel `generate_residual_one_step` signature.
2. Document inference policy: same `(x_t, t, r)` init strategy as pixel helper today vs improved sampler (config-driven).
3. Wire `predict`-style method on LMM module: encode residual → `ẑ_R` → decode → add `unet(merge(x,z))` when `ae_flag == 'residual'`.

**Acceptance checks**

- Output spatial shape matches LDM inference for same inputs.
- No import of `DDIMSampler` in LMM inference path.

---

### Phase L-I — Hydra configs and run recipes

**What**

- Clean entrypoints analogous to integration plan **Phase G**.

**How**

1. Add [`configs/model/lmm.yaml`](../configs/model/lmm.yaml): `_target_` → new LMM class; `mf_unet` with latent channels; `meanflow_core`; `autoencoder` + `context_encoder` blocks aligned with `ldm.yaml` (can use YAML anchors or explicit duplicate — prefer readable duplicate first).
2. Add [`configs/experiment/downscaling_LMM_res_2mT.yaml`](../configs/experiment/downscaling_LMM_res_2mT.yaml): same `data` / `trainer` defaults family as `downscaling_LDM_res_2mT.yaml`; set `ae_load_state_file`, UNET checkpoint on autoencoder, PDE coefs; **`load_optimizer_state`** policy same as LDM experiments.
3. Prefer `${paths.pretrained_models_dir}` over machine-local paths (per [`PIPELINE_COMPLETENESS.md`](../PIPELINE_COMPLETENESS.md) guidance).

**Acceptance checks**

- `python src/train.py experiment=downscaling_LMM_res_2mT` starts on a machine with data + ckpts configured.
- Resume from checkpoint works.

---

### Phase L-J — Inference API integration

**What**

- New branch in inference helper without touching `'ldm'` tensor math.

**How**

1. In `utils/inference_utils.py`, add `elif model_type == 'lmm':` (or agreed string): same preprocessing as LDM up to latent, then call LMM module’s `predict_final` or explicit encode → latent infer → decode → fusion.
2. Optional logging: `pred_latent_norm`, `pred_residual_pixel_norm`, `pred_final_norm` (integration plan Phase F spirit).

**Acceptance checks**

- Side-by-side: same random seed and batch → LDM vs LMM forward **shapes** logged; values differ (models differ) but no crashes.

---

### Phase L-K — Validation and comparison protocol

**What**

- Same spirit as integration plan **Phase H**: evidence LMM is correct and comparable.

**How**

1. **Sanity:** shape / finite checks on `z_0`, `x_t`, `ẑ_R`, decoded `R̂`, `ŷ_final`.
2. **Tiny overfit:** 1–2 batches, LMM loss decreases.
3. **Baselines:** prior-only `ŷ_up`, legacy LDM (`experiment=downscaling_LDM_res_2mT`), LMM — report residual MAE/RMSE and final-field metrics consistent with project notebooks.
4. **Ablations:** `time_scale` ∈ `{1, 100, 1000}` for `MFUNet` (integration plan §6.4); optional `flow_ratio` / `time_dist` from `MeanFlowCore`.
5. **Regression:** legacy LDM train 1 step + infer 1 batch unchanged on CI or manual smoke.

**Acceptance checks**

- Table of metrics for at least one variable (e.g. 2mT).
- Documented runtime per sample LDM vs LMM.

---

### L.1 File-by-file implementation contract (LMM, handoff-ready)

| # | File | Must |
|---|------|------|
| 1 | `src/models/lmm_module.py` (new) | Own LMM `LightningModule`: frozen AE, `mf_unet`, `meanflow_core`, context + PDE flags; `shared_step` parity with LDM data path; MeanFlow loss in latent; optional EMA on `mf_unet`. |
| 2 | `src/models/components/ldm/denoiser/mf_unet.py` | **Reuse**; LMM config sets latent `in_channels` / `out_channels`. |
| 3 | `src/models/components/meanflow/meanflow_core.py` | **Reuse** unchanged. |
| 4 | `src/models/components/ldm/denoiser/lmm_infer.py` (new) *or* extend `meanflow_infer.py` | Stateless latent one-step (or sampler) → `ẑ_R`; do not break pixel `generate_residual_one_step`. |
| 5 | `configs/model/lmm.yaml` (new) | Full Hydra wiring for LMM + `MFUNet` latent + meanflow hyperparameters. |
| 6 | `configs/experiment/downscaling_LMM_res_2mT.yaml` (new) | Overrides paths, PDE coefs, tags, checkpoint filenames. |
| 7 | `utils/inference_utils.py` | New `model_type`; `'ldm'` branch **unchanged**. |
| 8 | `tests/test_lmm_*.py` (new, optional split) | Instantiate / shape tests; optional parity assert flag for `latent_target` vs LDM. |
| 9 | `src/models/ldm_module.py` | **No behavior change** in same PR unless extracting shared PDE helper with identical outputs (prefer follow-up PR). |

---

## Deliverables checklist (LMM) — maps to phases above

- [ ] **[`LMM_PHASE_IMPLEMENTATION_LOG.md`](./LMM_PHASE_IMPLEMENTATION_LOG.md)** exists and stays updated (**Requirements — phase block diagram log** + **R-*** entries).
- [ ] **L-A** Scope doc + PDE decode-source decision recorded in yaml comments.
- [ ] **L-B** Latent `MFUNet` forward + JVP smoke on latent-shaped tensors.
- [ ] **L-C** New `LightningModule` file + successful Hydra instantiate.
- [ ] **L-D** `shared_step` data path matches LDM (shapes / `context_dict`).
- [ ] **L-E** MeanFlow training loss runs; tiny overfit passes.
- [ ] **L-F** PDE/energy branch matches LDM policy when enabled.
- [ ] **L-G** EMA / optim / `trainable_parts` aligned with project patterns.
- [ ] **L-H** Latent inference helper + module `predict_*` path.
- [ ] **L-I** `lmm.yaml` + `downscaling_LMM_res_2mT.yaml` + train/resume smoke.
- [ ] **L-J** `inference_utils.py` new `model_type`; LDM branch untouched.
- [ ] **L-K** Metrics table + ablation notes; legacy LDM smoke.
- [ ] **Policy** Record **Pipeline B disposition** (§ *Pipeline B — retirement…*) in `README` + this file + optional `docs/adr/`.

---

## Pipeline B — retirement, merge options, and anti-“trash” hygiene

Until now the docs emphasized **not overwriting** A/B, but **did not** spell out how to avoid **permanent** duplication and dead code. This section fixes that.

### Risk

If LMM ships as a **third** path **without** a dated decision and a **cleanup phase**, the repo can accumulate: two MeanFlow training entrypoints, two inference `model_type`s for “non-LDM MeanFlow”, duplicate docs, and configs that confuse new contributors.

### Record the decision (required once LMM exists)

Write explicitly (pick one primary strategy; others become “not chosen”):

| Strategy | Meaning |
|----------|---------|
| **B-freeze** | Keep **B** code and configs **unchanged** but mark **deprecated** in README + yaml comments; default new work uses **LMM** only. |
| **B-remove** | After gates pass, **delete** every **B-only** file and code path (see **R-3** table). **Do not** move B to a `legacy/` folder — **delete** from tree; git retains history. Keep only **shared** artifacts (`MeanFlowCore`, `MFUNet`, shared tests, `conditioner.py`, `ae_module.py`, etc.). |
| **B-merge** | Same end state as **B-remove** for B-only files: migrate any still-needed **strings** (logging, one-liner docs) into LMM, then **delete** the B module/configs/infer branch. **No** rename-to-`legacy_*` or shadow copies in-repo. |

Store the choice in: this file’s checklist **Policy** row, [README.md](../README.md) pipeline section, and optionally a one-page `docs/adr/` note.

### Preconditions before **B-remove** or **B-merge**

Do **not** delete B-only files until:

1. **L-K** complete: LMM metrics at least match agreed acceptance vs LDM or prior for your use case.
2. **No unique dependency:** search repo for `meanflow-residual`, `HrResidualMeanFlowLitModule`, `meanflow_residual`, `downscaling_MEANFLOW_res`, `MEANFLOW_res` — notebooks, SLURM scripts, external docs updated or archived.
3. **Checkpoint policy:** list published checkpoints that are **B-only**; archive or re-export if still needed for reproducibility.

### Flow B static-Z VAE / context (paper gap) — **must** be on the hitlist

This was under-specified next to “delete `meanflow_module`.” Flow B’s intended story is: **Stage-1 VAE on static `z` → reuse encoder in context → MeanFlow generates `R̂`**. The **biggest logic risk** is not “missing `encode` calls” — **`conditioner.py` already calls `encode(static)` on `context_encoder.autoencoder[0]`** every forward. The gap is **defaults and wiring**:

| Item | Action before calling Flow B “clean” or deleting B |
|------|-----------------------------------------------------|
| `stage1_encoder_ckpt` | Must point to weights from **[`downscaling_VAE_static_2mT`](../configs/experiment/downscaling_VAE_static_2mT.yaml)** (or equivalent), **not** residual VAE; prefer **`strict=True`** once shapes verified, or mandatory **load logging** of missing/unexpected keys. |
| `train_autoenc` | If Stage-1 encoder should stay frozen during MeanFlow, set **`train_autoenc: false`** in `meanflow_residual.yaml` (otherwise AFNO base **unfreezes** branch encoders). |
| `freeze_stage1_encoder` | Set **`true`** in experiment yaml when policy is “reuse frozen encoder.” |
| `legacy_autoencoder.ae_mode: static_ctx` | **Remove** from prior bundle if unused (it does **not** implement the static-Z context path — that is **`context_encoder.autoencoder[0]`** only). |

Full analysis: [MEANFLOW_CURRENT_IMPLEMENTATION_SURVEY.md](./MEANFLOW_CURRENT_IMPLEMENTATION_SURVEY.md) **§3**.

### Phased cleanup checklist (execute after strategy is chosen)

**Phase R-1 — Inventory (read-only)**

- Run ripgrep for strings above; paste results into issue/ADR appendix.
- Confirm [MEANFLOW_CURRENT_IMPLEMENTATION_SURVEY.md](./MEANFLOW_CURRENT_IMPLEMENTATION_SURVEY.md) §2 file list is still complete; review **§3** (static-Z VAE / context) for any **B-freeze** “known broken” caveats to document in README.
- Append **R-1** section to [`LMM_PHASE_IMPLEMENTATION_LOG.md`](./LMM_PHASE_IMPLEMENTATION_LOG.md) (current repo graph + “B still present” snapshot).

**Phase R-2 — Deprecation (low risk)**

- Add `DeprecationWarning` or log-on-import in `meanflow_module.py` **if** strategy is **B-freeze** or prelude to removal.
- README: label diagram **B** as deprecated with link to LMM; point training commands to LMM experiment.
- Append **R-2** entry to [`LMM_PHASE_IMPLEMENTATION_LOG.md`](./LMM_PHASE_IMPLEMENTATION_LOG.md).

**Phase R-3 — Delete B-only implementation (irreversible path)**

When strategy is **B-remove** or **B-merge**, **delete** the following from the working tree if they are **not** shared by LDM or LMM (confirm with ripgrep before merge). **Do not** rename to `legacy_*`, **do not** add a `legacy/` subtree for these — deletion only; archival runs rely on **git tags / old commits**.

| Artifact | Action |
|----------|--------|
| [`src/models/meanflow_module.py`](../src/models/meanflow_module.py) | **Delete** file (B-only `LightningModule`). |
| [`configs/model/meanflow_residual.yaml`](../configs/model/meanflow_residual.yaml) | **Delete** file (B-only Hydra model). |
| [`configs/experiment/downscaling_MEANFLOW_res_2mT.yaml`](../configs/experiment/downscaling_MEANFLOW_res_2mT.yaml) | **Delete** file (and any B-only MEANFLOW experiment variants). |
| [`utils/inference_utils.py`](../utils/inference_utils.py) | **Delete** the `elif model_type == 'meanflow-residual':` branch entirely. |
| [MEANFLOW_TRAINING_STAGES_GUIDE.md](./MEANFLOW_TRAINING_STAGES_GUIDE.md) | **Delete** file if fully superseded by an LMM training doc; **or** **edit** in place to remove **only** B Stage-3 pixel content and point Stage-3 to LMM — do **not** keep a duplicate “legacy” copy under another name. |
| B-only tests (if any) | **Delete**. |
| [`LMM_PHASE_IMPLEMENTATION_LOG.md`](./LMM_PHASE_IMPLEMENTATION_LOG.md) | **Append** an **R-3** section: block graph of repo **after** B-only deletions (do **not** delete this log file; only append). |

**Keep — do not delete** (shared across pipelines): `MeanFlowCore`, `MFUNet`, [`MeanFlow/`](../MeanFlow/) vendor tree (optional), [`meanflow_infer.py`](../src/models/components/ldm/denoiser/meanflow_infer.py) **if** LMM still imports pixel helper during transition (otherwise fold latent helper only and delete unused symbol), [`tests/test_meanflow_core.py`](../tests/test_meanflow_core.py), [`conditioner.py`](../src/models/components/ldm/conditioner.py), [`ae_module.py`](../src/models/ae_module.py), **[`configs/experiment/downscaling_VAE_static_2mT.yaml`](../configs/experiment/downscaling_VAE_static_2mT.yaml)** (Stage-1 static AE — not B-only).

**Phase R-4 — Doc sweep**

- [MEANFLOW_INTEGRATION_PLAN.md](./MEANFLOW_INTEGRATION_PLAN.md): banner that Stage-3 **pixel** path is superseded if **B-remove**.
- [LMM_PIPELINE_PLAN.md](./LMM_PIPELINE_PLAN.md): remove “three pipelines” default language if you collapsed to two; update Figure legend.
- Append **R-4** entry to [`LMM_PHASE_IMPLEMENTATION_LOG.md`](./LMM_PHASE_IMPLEMENTATION_LOG.md).

**Phase R-5 — CI / Hydra**

- Remove `experiment=downscaling_MEANFLOW_res_2mT` from example commands in README if B removed.
- Ensure `python src/train.py --help` or documented default experiment list does not reference dead configs.
- Append **R-5** entry to [`LMM_PHASE_IMPLEMENTATION_LOG.md`](./LMM_PHASE_IMPLEMENTATION_LOG.md).

### “Merge” B into LMM (same number of pipelines, not three) — optional design

If you want **two** pipelines only (**LDM** + **LMM**), **merge** means:

- Reuse **one** `model_type` string (e.g. only `'lmm'`) and **one** experiment naming pattern for “non-LDM MeanFlow”.
- Migrate any **unique** behavior from B (e.g. logging hooks) into LMM, then execute **R-3** on B files.

This is **not** “edit `HrResidualMeanFlowLitModule` in place to use latents” without a rename and config migration — that confuses checkpoint compatibility; prefer **new LMM module** then **delete** B-only files in one PR after parity (**no** `legacy/` directory for B — **delete** from tree per **R-3**).

---

## Document map

| Read first | Purpose |
|------------|---------|
| **This file** (`LMM_PIPELINE_PLAN.md`) | **TARGET:** LMM = LDM with MeanFlow denoiser on `z_R`. |
| [`LMM_PHASE_IMPLEMENTATION_LOG.md`](./LMM_PHASE_IMPLEMENTATION_LOG.md) | **Mandatory** per-phase ASCII block graphs + gap notes (**L-A…L-K**, **R-1…R-5**). |
| [MEANFLOW_CURRENT_IMPLEMENTATION_SURVEY.md](./MEANFLOW_CURRENT_IMPLEMENTATION_SURVEY.md) | **Current code** vs `MEANFLOW_INTEGRATION_PLAN.md` + full file list (fork **B**). |
| [README.md](../README.md) | **A** = legacy LDM; **B** = **current** pixel MeanFlow fork (mess). |
| [MEANFLOW_INTEGRATION_PLAN.md](./MEANFLOW_INTEGRATION_PLAN.md) | Original phased plan; **Core direction** updated to distinguish implemented fork vs LMM. |

---

## Summary

| Pipeline | Role |
|----------|------|
| **LDM_PDE + UNet (A)** | Legacy; frozen API and behavior. |
| **README B** | **Current** pixel MeanFlow experiment; not the science target for “replace denoiser.” |
| **LMM (C)** | **Target:** duplicate latent LDM_PDE stack; **only** swap denoiser region for MeanFlow on **`z_R`**; new codepath, unplugged from legacy. |
