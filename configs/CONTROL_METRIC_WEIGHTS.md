# `control_metric_weights` policy

## Rule

| Location | `control_metric_weights` |
|----------|--------------------------|
| `src/models/lmm_module.py` | all **0.0** (code defaults) |
| `configs/model/*.yaml` | all **0.0** |
| `.hydra/config.yaml` | all **0.0** (generated snapshot; do not put experiment weights here) |
| **`configs/experiment/*.yaml` only** | may set **non-zero** keys |

Hydra merges `configs/model/lmm.yaml` then your `experiment=...` file. With base all zeros, **only keys listed in the experiment YAML** affect `val/control_score`.

## Experiment files (non-zero allowed here)

| File | Non-zero keys |
|------|----------------|
| `downscaling_LMM_res_2mT_pretrain.yaml` | `mf_minus_1`, `at_mag_pure`, `at_dir_pure` |
| `downscaling_LMM_res_2mT.yaml` | `rmse`, `temp_pde_pure`, `at_mag_pure`, `at_dir_pure` |

No `control_metric_weights` block (inherits all 0 until you add one):

- `downscaling_LMM_res_2mT_MIG.yaml`
- `downscaling_LMM_res_2mT_smoke.yaml`

## MeanFlow monitoring (TensorBoard)

Logged: `val/mf_minus_1`, `val/mf_minus_1_x1e8`, `val/phys_loss` (not flat `mf_loss` / `loss_total_*` or EMA duplicates).

## Verify before a long run

```bash
python src/train.py experiment=downscaling_LMM_res_2mT_pretrain --cfg job | grep -A20 control_metric_weights
```
