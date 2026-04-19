# LMM implementation — phase block diagrams (living log)

**Requirement (see [LMM_PIPELINE_PLAN.md](./LMM_PIPELINE_PLAN.md)):** after **each** implementation phase (**L-A … L-K** for LMM build, and **R-1 … R-5** when retiring pipeline B), append a **new dated section** below. Each section **must** include:

1. **Phase id** (e.g. `L-D`, `R-3`) and **date** / PR link.
2. **ASCII block graph** of the **currently implemented** pipeline (or the **delta** this phase added), in the same style as Figure A/B/C in the LMM plan (boxes, arrows). Mark **NEW** / **CHANGED** blocks explicitly.
3. **Placement check:** one short paragraph — does this block sit in the right place vs Figure C?
4. **Gap check:** bullet list — anything **missing** vs that phase’s **Acceptance checks** in the plan? Anything **extra** or wrong?
5. **Git checkpoint:** once this section (including the block graph) is complete, **commit** all changes for that phase to git with a message that **starts with the phase id** (e.g. `LMM phase L-D: <short title>` or `LMM R-3: <short title>`). One phase → one dedicated commit (no unrelated edits in the same commit). That yields a **revertible, bisectable** history if a later phase mis-implements something.

Older phases stay in this file for audit trail; do not delete past sections.

---

## Template (copy for each new phase)

```markdown
## Phase <L-X or R-X> — <title> — YYYY-MM-DD (<PR or branch>)

### Block graph (as implemented after this phase)

\`\`\`
(paste ASCII diagram here)
\`\`\`

### Placement vs [LMM_PIPELINE_PLAN.md](./LMM_PIPELINE_PLAN.md) Figure C

(text)

### Gap check vs plan acceptance

- [ ] …

### Git checkpoint

After saving this section, commit with a message that includes the phase id, e.g. `LMM phase L-X: <title>`. Optional: note short hash here after commit: `<abc1234>`.
```

---

## Log entries

## Phase L-B … L-J (initial runnable LMM) — 2026-04-19 (implementation batch)

### Block graph (as implemented after this phase)

```
Stage 0–1  ═══════════════════════════════════  FROZEN — same Hydra blocks as LDM (UNET on autoencoder, etc.)

Stage 2 — LMM [NEW: LatentMeanFlowLitModule]
  [x][y][z]──►[preprocess_batch]──►R──►[Encoder_S1]──►z_R
  [z][x]──►[context_encoder]──►context (dict: T_c + cascade)

  z_R──►[MeanFlowCore: bridge + (t,r)]──►(x_t,t,r,v_tgt)
  x_t,t,r,context──►[MFUNet 32×32×H'×W']──►JVP teacher──►train/val loss
  (optional) temp PDE/energy: decode(single_step_generate(x_t,t,r,uθ))──►T_f vs T_c

Infer [NEW: model_type == 'lmm']
  [low_res][y_hr][static]──►encode R──►generate_latent_one_step──►decode R̂──►+ ŷ_up──►ŷ
```

### Placement vs [LMM_PIPELINE_PLAN.md](./LMM_PIPELINE_PLAN.md) Figure C

MeanFlow sits on **latent `z_R`** with the same `preprocess_batch` → `encode` mean path and **`context_dict` (`T_c` + conditioner output)** as `LatentDiffusion.shared_step`. PDE attachment uses the **L-F lock** documented in `configs/model/lmm.yaml` (one-step latent estimate, then decode). Pixel fork **B** is untouched.

### Gap check vs plan acceptance

- [x] New module `src/models/lmm_module.py`, `lmm_infer.py`, `temperature_field_losses.py`, `configs/model/lmm.yaml`, `configs/experiment/downscaling_LMM_res_2mT.yaml`, `inference_utils` `lmm` branch, tests `tests/test_lmm_shapes.py`.
- [ ] **L-K:** metrics table / runtime LDM vs LMM on real data not produced in-repo.
- [ ] **Train smoke:** `python src/train.py experiment=downscaling_LMM_res_2mT` not executed here (needs checkpoints + dataset paths).
- [ ] **Parity assert:** optional LDM vs LMM `latent_target` dev flag from plan **L-D** not added.

### Git checkpoint

Git checkpoint: message `LMM phase L-J: initial LMM pipeline (module, configs, inference, tests)` — record `git rev-parse --short HEAD` here after your local commit if this file was merged without the hash line amended in.
