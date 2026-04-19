# MeanFlow integration plan (locked scope, UNet backbone)

**Engineering target (Latent Meanflow Model, LMM):** see [LMM_PIPELINE_PLAN.md](./LMM_PIPELINE_PLAN.md) — duplicate the **latent** `LDM_PDE + UNet` pipeline and **replace only the denoiser** with MeanFlow on `z_R`. [README.md](../README.md), **pipeline B**, documents the **current** pixel-space fork (`HrResidualMeanFlowLitModule`), not that target.

This plan defines the implementation work to add a new MeanFlow-based residual branch while preserving the existing pipeline structure. **Inventory of what was actually built vs this document:** [MEANFLOW_CURRENT_IMPLEMENTATION_SURVEY.md](./MEANFLOW_CURRENT_IMPLEMENTATION_SURVEY.md).

Core direction (as **implemented** in this repo — README pipeline **B**):
- Keep the existing **UNet upscaler block** as-is.
- Add a **new MeanFlow flow in parallel** with legacy LDM flow (no overwrite).
- Legacy **LDM_PDE + UNET** train and inference paths must remain available and unchanged.
- Keep **UNet** (not DiT) as the MeanFlow backbone.
- Scope of the **implemented** fork: replace **LDM’s latent residual denoiser + decode of \(z_R\)** (for the correction) with a **single MeanFlow generator acting on pixel-space** \(R_{gt}=y-\hat{y}_{up}\) (see `HrResidualMeanFlowLitModule`). That is **not** the same as “only swap the latent denoiser while keeping `Encoder_S1`/`Decoder_S1` on \(R\)” — that latter goal is **LMM** in [LMM_PIPELINE_PLAN.md](./LMM_PIPELINE_PLAN.md). A detailed inventory vs this document is in [MEANFLOW_CURRENT_IMPLEMENTATION_SURVEY.md](./MEANFLOW_CURRENT_IMPLEMENTATION_SURVEY.md).

---

## 1. Locked decisions from your clarification

1. **JVP teacher objective (training) + single-step equation (inference)**  
   Train with MeanFlow teacher target:
   \[
     u,\frac{du}{dt}=\mathrm{jvp}(f,(z,r,t),(v,0,1)),\quad
     u_{tgt}=v-(t-r)\frac{du}{dt}
   \]
   and optimize `metric(u - stopgrad(u_tgt))`.  
   Use one-step equation to produce \(\hat{R}\) at inference/evaluation time.

2. **Stage-1 decoder usage**  
   The static autoencoder decoder is used only in Stage 1 to verify reconstruction quality, then not used in later stages.

3. **Residual target statement**  
   Stage 3 objective: encoder + MeanFlow estimate \(R\) close to \(R_{gt}\), then combine that \(R\) with UNet-upscaler output for final result.

4. **Scope guard**  
   Follow current implementation boundaries: do not change UNet upscaler behavior. Add MeanFlow as a separate path; do not remove or break legacy LDM path.
   Do not modify PDE implementation in this integration.

5. **Conditioning / CFG guard**  
   MeanFlow must generate \(R\) from condition \(Z\). If CFG is needed, keep it inside the MeanFlow residual generation logic only. Do not alter external fusion stages.

---

## 2. Notation (disambiguation)

To avoid confusion:
- \(Z_{ctx}\): context latent from static encoder + LR fusion (condition for UNet).
- \(x_t\): MeanFlow state at time \(t\) (this is the `z` variable name in the MeanFlow example code).
- \(R\): predicted residual.
- \(R_{gt}\): ground-truth residual.
- \(\hat{y}_{up}\): output of pretrained upscaler UNet (Stage 2).

The previous draft mixed `Z` and `z`; this section is canonical for the rest of the document.

---

## 3. Pipeline definition (three stages)

### Stage 1: static autoencoder pretraining
- Train encoder \(\mathcal{F}\) and decoder \(\mathcal{G}\) on high-resolution static data only.
- Goal: obtain a reliable static encoder representation \(Z\) for conditioning.
- After Stage 1:
  - Keep \(\mathcal{F}\) for later stages.
  - \(\mathcal{G}\) is not used again except optional diagnostics.

### Stage 2: UNet upscaler pretraining (unchanged)
- Keep current upscaler workflow and checkpoints.
- This block provides prior prediction \(\hat{y}_{up}\).
- No architectural or training changes in this integration.

### Stage 3: MeanFlow residual branch
- Compute residual target:
  \[
    R_{gt} = y_{hr} - \hat{y}_{up}
  \]
- Build condition \(Z_{ctx}\) exactly as you described:
  1. encode high-res static input with Stage-1 encoder;
  2. concatenate/fuse that latent with LR input in the same style as current implementation;
  3. pass resulting \(Z_{ctx}\) as condition for MeanFlow backbone.
- MeanFlow core computes \(x_t\), \(t\), and \(r\) internally.
- UNet backbone receives \((x_t, t, r, Z_{ctx})\) and predicts average velocity \(U_\theta\).
- Use one-step equation to produce residual \(R\) from \(U_\theta\).
- MeanFlow training loss in this scoped integration follows the JVP teacher objective above (e.g. adaptive-L2/MSE metric on `u - stopgrad(u_tgt)`), with no PDE changes.
- Final prediction:
  \[
    \hat{y}_{final} = \hat{y}_{up} + \hat{R}
  \]

---

## 4. Mapping to current repository

Current files to **extend without breaking legacy behavior**:
- `utils/inference_utils.py` (add new `model_type` branch only)
- config registry/defaults where needed to add new experiment entry

New files preferred for MeanFlow path:
- `src/models/meanflow_module.py` (parallel LightningModule)
- `src/models/components/meanflow/meanflow_core.py`
- `src/models/components/ldm/denoiser/mf_unet.py`
- `src/models/components/ldm/denoiser/meanflow_infer.py`
- `configs/model/meanflow_residual.yaml`
- `configs/experiment/downscaling_MEANFLOW_res_*.yaml`

MeanFlow reference files:
- `MeanFlow/meanflow.py` (loss + sampling logic)
- `MeanFlow/models/dit.py` (time interface pattern `(x,t,r,...)`)

Important adaptation:
- Reuse MeanFlow algorithmic logic, not DiT architecture.
- Preserve project conditioning style (context features), not ImageNet class-only conditioning.

---

## 5. Detailed work breakdown (what + how)

### Phase A — Add MeanFlow core utilities

**What**
- Introduce MeanFlow utility class for training/inference equation handling.

**How**
1. Create folder `src/models/components/meanflow/`.
2. **Yes, reuse the existing implementation**:
   - copy `MeanFlow/meanflow.py` into `src/models/components/meanflow/meanflow_core.py` as the starting point;
   - then adapt function signatures and naming to this repository.
3. Port only needed parts from copied file:
   - time pair sampling (`t`, `r`)
   - bridge construction
   - residual objective utilities
   - optional CFG internals (kept local to MeanFlow path)
4. Remove any class-label/ImageNet-only assumptions from the copied file unless explicitly needed.
5. Keep public API explicit:
   - `compute_train_targets(...)`
   - `single_step_generate(...)` for Stage-3 equation
6. Add minimal unit test file `tests/test_meanflow_core.py`:
   - shape checks
   - finite value checks
   - deterministic behavior with fixed seed

Acceptance checks:
- Functions run on CPU and GPU.
- No dependency on DiT-specific modules.

---

### Phase B — Build UNet backbone with MeanFlow interface

**What**
- Implement a UNet that accepts `(x, t, r, context)` and predicts residual velocity/state used by MeanFlow equation.

**How**
1. Add new module `src/models/components/ldm/denoiser/mf_unet.py`.
2. Start from existing `UNetModel` internals:
   - retain spatial blocks and context/cross-attention path.
3. Add second time input path:
   - embed `t` and `r` separately
   - **recommended v1 scaling:** `t_embed_in = t * time_scale`, `r_embed_in = r * time_scale` with `time_scale=1000` default
   - pass each through sinusoidal projection + trainable MLP path
   - fuse embeddings (sum first for v1; optional concat+mlp later)
4. Forward signature:
   - `forward(self, x_t, t, r, context=None)`
5. `context` is \(Z_{ctx}\) only (no change to outside fusion flow).
6. Keep output shape equal to input residual shape.

Implementation note on embedding:
- The scalar embedding pipeline is usually split into:
  - fixed sinusoidal projection (`timestep_embedding`-style),
  - trainable MLP (`time_embed`-style).
- This works for continuous `t,r` in `(0,1)`; discrete integer timesteps are not required.
- Scaling remains useful because fixed sin/cos features may be too smooth with raw `(0,1)` inputs.

Acceptance checks:
- Forward pass works with current conditioner output.
- Parameter count and memory usage logged.
- JVP/autograd path runs without graph errors.

---

### Phase C — Stage-1 static VAE/AE refactor (required before MeanFlow training)

**What**
- Ensure the encoder used for `Z_ctx` is trained on **high-resolution static data only**, not on high-resolution target/result residuals.

**How**
1. Add or update a dedicated Stage-1 config (recommended new file), e.g.:
   - `configs/experiment/downscaling_VAE_static_*.yaml`
2. Dataset wiring for Stage 1:
   - input tensor = HR static,
   - reconstruction target = HR static (same tensor family).
3. **Explicit AE data-routing change (required):**
   - add a new AE mode/flag for new MeanFlow flow only (example: `ae_mode: static_ctx`);
   - in this mode:
     - `x_ae = static_hr`
     - `y_ae = static_hr`
   - keep legacy mode unchanged (`ae_mode: residual` keeps current behavior).
4. If reusing `AutoencoderKL`, ensure Stage-1 run path does **not** use residual-target-specific behavior:
   - no residual-target preprocessing,
   - no dependency on target/result fields,
   - no coupling to Stage-2 upscaler outputs.
5. Add new-flow-only config keys so routing is explicit and non-breaking:
   - `model.autoencoder.ae_mode: static_ctx`
   - `model.autoencoder.use_for: meanflow_context_only`
6. Save and register Stage-1 encoder checkpoint path for Stage-3 loading.
7. In Stage-3 MeanFlow module:
   - load Stage-1 encoder weights,
   - set encoder to frozen by default (recommended v1),
   - use it only to produce `Z_ctx`.

Acceptance checks:
- Stage-1 training reconstructs HR static (not HR result).
- Stage-3 can load Stage-1 encoder checkpoint without shape mismatch.
- No call path in Stage-3 uses Stage-1 decoder.
- Legacy AE/LDM preprocessing outputs remain identical when `ae_mode` is not `static_ctx`.

---

### Phase D — Add Stage-3 Lightning module (new, isolated)

**What**
- Implement a new training module for MeanFlow residual learning, separate from `LatentDiffusion`.

**How**
1. Create `src/models/meanflow_module.py` (name can be `HrResidualMeanFlowLitModule`).
2. In `shared_step`:
   - run frozen Stage-2 UNet to get \(\hat{y}_{up}\) (same call path as today)
   - compute \(R_{gt} = y_{hr} - \hat{y}_{up}\)
   - build condition \(Z_{ctx}\): static encoder output + LR concat/fusion as current style
3. Normalize \(R_{gt}\) with chosen residual normalizer.
4. Let MeanFlow core generate internal \(x_t\), \(t\), \(r\), then call `mf_unet(x_t, t, r, context=Z_ctx)`.
5. Compute JVP teacher target:
   - `u, dudt = jvp(fn, (x_t, r, t), (v_target, 0, 1))`
   - `u_tgt = v_target - (t-r) * dudt`
6. Train loss as MeanFlow teacher objective:
   - `loss = metric(u - stopgrad(u_tgt))` (adaptive-L2 by default, MSE optional)
7. Keep one-step \(\hat{R}\) generation helper for inference/evaluation path.
7. Keep optimizer/scheduler style close to current LDM settings for stable comparison.
8. Keep Stage-2 UNet frozen; no grad updates outside Stage-3 path.

Acceptance checks:
- Training step runs end-to-end on one batch.
- Loss decreases in tiny-overfit test.
- Stage-2 UNet weights unchanged after epoch.
- No PDE code path is modified for either new or legacy flow.

---

### Phase E — Implement single-step inference equation

**What**
- Replace LDM/DDIM residual generation with one-step MeanFlow residual generation.

**How**
1. Add helper in `src/models/components/ldm/denoiser/meanflow_infer.py`.
2. Implement one-step residual generation function:
   - input: condition \(Z_{ctx}\), MeanFlow-generated \(x_t,t,r\), velocity from UNet
   - output: generated \(\hat{R}\)
3. Keep function stateless and easy to call from inference utilities.
4. Add numerical safeguards:
   - clamp or norm guard if needed
   - NaN/Inf check with fallback logging

Acceptance checks:
- One-step function returns tensor matching \(R_{gt}\) shape.
- Inference path no longer depends on DDIM for this mode.

---

### Phase F — Wire into existing inference pipeline

**What**
- Add a new model type branch without breaking old LDM behavior.

**How**
1. Update `utils/inference_utils.py`:
   - add `model_type == 'meanflow-residual'` branch
   - compute \(\hat{R}\) via Phase-D helper
   - combine with upscaler output to produce final field
2. Keep existing `'ldm'` branch untouched for baseline reproducibility.
3. Add explicit logging keys:
   - `pred_residual_norm`
   - `pred_final_norm`
   - optional CFG flag used/not used

Acceptance checks:
- Both old LDM and new MeanFlow branches run from same script.
- Fusion happens after Stage-3 residual generation only.

---

### Phase G — Hydra configs and run recipes

**What**
- Add clean config entrypoints for Stage-3 MeanFlow experiments.

**How**
1. Create `configs/model/meanflow_residual.yaml`:
   - `denoiser: mf_unet`
   - `meanflow` settings (including optional CFG knobs)
2. Create `configs/experiment/downscaling_MEANFLOW_res_2mT.yaml` (and UV if needed):
   - reuse dataset and trainer defaults from LDM experiment
   - load Stage-1 encoder checkpoint
   - load Stage-2 UNet checkpoint (frozen)
3. Add inference config switch for `model_type: meanflow-residual`.

Acceptance checks:
- `python src/train.py experiment=...` starts without manual edits.
- Checkpoint save/load works for resumed training.

---

### Phase H — Validation and comparison protocol

**What**
- Verify correctness and compare against unchanged baselines.

**How**
1. Sanity checks:
   - shape alignment for \(R_{gt}\), \(\hat{R}\), \(\hat{y}_{up}\), \(\hat{y}_{final}\)
   - finite output checks
2. Short overfit:
   - 1–2 batches, confirm loss trend
3. Full validation:
   - compare against prior-only (`\hat{y}_{up}`) and existing LDM baseline
4. Report:
   - residual MAE/RMSE
   - final MAE/RMSE
   - runtime per sample (new branch vs LDM branch)
5. Time-embedding ablation:
   - test `time_scale` in `{1, 100, 1000}`
   - compare early loss descent, stability (NaN/grad spikes), and validation RMSE/MAE

Acceptance checks:
- Reproducible metrics table for at least one target variable.

---

## 6.1 File-by-file implementation contract (handoff-ready)

Use this as an execution contract for another coding agent.

1. `src/models/components/meanflow/meanflow_core.py`
- Must provide:
  - `sample_t_r(batch_size, device, config)` -> `(t, r)` with shape `(B,)`
  - `build_bridge_state(x0, t, noise=None)` -> `(x_t, aux)` where `x_t` shape = `x0`
  - `compute_train_targets(...)` -> tensors required to compute Stage-3 loss
  - `single_step_generate(x_t, t, r, u_theta)` -> `r_hat`
- Must not import DiT code.
- Must not alter behavior outside MeanFlow branch.

2. `src/models/components/ldm/denoiser/mf_unet.py`
- Must expose class `MFUNet` (or similarly named) with:
  - `forward(x_t, t, r, context=None) -> u_theta`
- Keep conditioning style compatible with current context flow.
- Output shape must match `x_t`.

3. `src/models/meanflow_module.py`
- New Lightning module for Stage-3 training only.
- `shared_step(batch)` must:
  - run frozen Stage-2 upscaler
  - compute `R_gt = y_hr - y_up`
  - build `Z_ctx` from static encoder + LR fusion
  - generate `x_t, t, r` via meanflow_core
  - build `v_target` from bridge state (`noise - R_gt`)
  - call `u_theta = mf_unet(x_t, t, r, context=Z_ctx)`
  - compute `dudt` with JVP using tangent `(v_target, 0, 1)`
  - compute `u_tgt = v_target - (t-r)*dudt`
  - return teacher loss `metric(u_theta - stopgrad(u_tgt))`

3b. `src/models/ae_module.py` (or the active AE preprocessing location)
- Add an explicit new mode for static-only encoding used by MeanFlow flow:
  - `ae_mode == "static_ctx"` routes `x_ae,y_ae` to HR static.
- Keep default/legacy routing unchanged.
- Add runtime logging/assertion of active mode to prevent silent mis-routing.

4. `src/models/components/ldm/denoiser/meanflow_infer.py`
- Must provide one-step residual generation helper for inference.
- Stateless function(s), no training-only side effects.

5. `utils/inference_utils.py`
- Add branch: `model_type == "meanflow-residual"`.
- Use new helper to compute `R_hat`, then:
  - `y_final = y_up + R_hat`
- Keep old `'ldm'` path unchanged.
- Explicit non-regression requirement: existing `'ldm'` inference outputs must be unchanged when using old checkpoints/configs.

6. `configs/model/meanflow_residual.yaml`
- Model wiring for `meanflow_module` + `MFUNet` + MeanFlow config block.
- Must explicitly set AE mode for new flow (e.g. `ae_mode: static_ctx`).

7. `configs/experiment/downscaling_MEANFLOW_res_2mT.yaml`
- Stage-3 run config reusing current data/trainer conventions.
- Load pretrained Stage-1 encoder and Stage-2 upscaler checkpoints.
- Must not override or mutate legacy `ldm.yaml` defaults.

---

## 6.2 Ordered task list for another agent

1. Copy/adapt `MeanFlow/meanflow.py` into repository path in Phase A.
2. Implement `MFUNet` with `(x_t, t, r, context)` interface.
3. Implement AE static-only routing mode (`ae_mode: static_ctx`) without touching legacy mode behavior.
4. Implement new Stage-3 Lightning module.
5. Add one-step inference helper.
6. Integrate new inference branch.
7. Add Hydra configs.
8. Run a tiny-overfit sanity run.
9. Run one validation pass and report metrics.
10. Run legacy LDM non-regression smoke test (train/infer entrypoint still works).

Definition of done:
- Train command starts and completes at least 1 epoch.
- Inference command runs on at least one batch.
- Output tensor and metric logs are valid.
- Legacy LDM_PDE+UNET command path still runs without code changes to user workflow.

---

## 6.3 Suggested handoff prompt (copy/paste)

```text
Implement MeanFlow residual branch according to docs/MEANFLOW_INTEGRATION_PLAN.md.
Hard constraints:
1) Keep Stage-2 UNet upscaler unchanged and frozen.
2) Add MeanFlow as a new parallel flow; do NOT remove/overwrite legacy LDM_PDE+UNET.
3) Use UNet backbone, not DiT.
4) Use notation/flow: Z_ctx (condition) is separate from x_t (MeanFlow state).
5) MeanFlow core computes x_t,t,r internally; UNet predicts average velocity; residual is generated by one-step equation.
6) Keep old LDM path working.
7) Do not modify PDE implementation; keep PDE behavior only in legacy LDM flow.
8) Add explicit AE routing mode for new flow (`ae_mode: static_ctx`) so VAE sees HR static->HR static only in new flow; legacy AE route must remain unchanged.

Execute tasks in order:
- Add src/models/components/meanflow/meanflow_core.py by adapting MeanFlow/meanflow.py
- Add src/models/components/ldm/denoiser/mf_unet.py
- Add non-breaking AE mode for static-only context encoding (new-flow only)
- Add src/models/meanflow_module.py
- Add src/models/components/ldm/denoiser/meanflow_infer.py
- Update utils/inference_utils.py with model_type='meanflow-residual'
- Add configs/model/meanflow_residual.yaml
- Add configs/experiment/downscaling_MEANFLOW_res_2mT.yaml

Then run sanity checks:
- shape/finite checks
- tiny overfit
- one validation pass
- legacy LDM inference smoke test
Return: changed files, commands run, and final metrics summary.
```

---

## 6.4 Time embedding specification (t, r)

This section is strict implementation guidance for the next coding agent.

### Fixed vs learnable parts
- **Fixed:** sinusoidal scalar projection (`timestep_embedding`-style).
- **Learnable:** MLP after projection (`time_embed`-style).

Implication:
- The model still adapts to your time definition during training through learnable MLP + UNet weights.

### Continuous-time compatibility
- MeanFlow uses continuous `t,r` in `(0,1)`.
- Sin/cos projection accepts real-valued inputs directly.
- No discretization to `(1,2,3,...)` is required.

### Required v1 implementation
1. Keep current UNet embedding mechanism.
2. Add a second embedding path for `r`.
3. Add config key `time_scale` (default `1000`).
4. Compute:
   - `emb_t = TimeMLP(Sinusoid(t * time_scale))`
   - `emb_r = TimeMLP_or_R_TimeMLP(Sinusoid(r * time_scale))`
   - `emb = emb_t + emb_r`
5. Feed `emb` everywhere the current UNet expects time embedding.

### v1/v2 options
- **v1 (recommended):** share one MLP for `t` and `r`.
- **v2 (optional):** separate MLPs for `t` and `r` if capacity is insufficient.

### Why scaling is still recommended
- Without scaling, fixed sinusoidal features over `(0,1)` may have low frequency diversity.
- MLP can adapt but often trains slower from a weaker basis.
- Keep `time_scale` as an ablation hyperparameter in Hydra.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Single-step generation unstable at start | add normalization + warmup + gradient clipping |
| JVP/autograd memory overhead | begin with smaller crop/batch and enable AMP carefully |
| Condition misuse (outside scope) | keep all CFG/condition operations inside MeanFlow branch only |
| Regression in legacy LDM path | maintain separate config + inference branch and run both in CI/manual tests |
| Weak time conditioning signal | expose `time_scale` in config; run ablation (`1`, `100`, `1000`) |
| Scope creep into PDE path | keep PDE code untouched and validate legacy PDE flow still runs |

---

## 8. Deliverables checklist

- [ ] `meanflow_core.py` added and tested
- [ ] `mf_unet.py` with `(x,t,r,context)` forward
- [ ] explicit `t,r` embedding in `mf_unet.py` with configurable `time_scale`
- [ ] new Stage-3 Lightning module for residual MeanFlow
- [ ] MeanFlow JVP teacher loss implemented (`u - stopgrad(u_tgt)`), with one-step `R_hat` kept for inference
- [ ] single-step MeanFlow inference helper
- [ ] `inference_utils.py` new branch (`meanflow-residual`) with post-stage fusion
- [ ] Hydra model + experiment configs
- [ ] ablation results for `time_scale in {1,100,1000}`
- [ ] baseline comparison report (prior-only vs LDM vs MeanFlow)
- [ ] non-regression check result for legacy LDM_PDE + UNET train/infer
- [ ] brief README section with train/infer commands

---

## 9. Out of scope for this integration

- Changing Stage-2 UNet upscaler architecture or training recipe.
- Changing PDE implementation or PDE training logic.
- Migrating entire project to DiT.
- Refactoring unrelated GAN/VAE/UNET experiment pipelines.

---

Reference algorithm source: `./MeanFlow` (upstream: `https://github.com/haidog-yaqub/MeanFlow.git`).
