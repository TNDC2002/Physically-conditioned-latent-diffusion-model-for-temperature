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

## MeanFlow monitoring

Use `mf_minus_1` in **experiment** YAML for checkpoint/LR control (not `loss` / `mf_loss_f64` near 1.0). TensorBoard still logs `val/mf_loss_f64`, `val/mf_minus_1`, etc. from code.

## Verify before a long run

```bash
python src/train.py experiment=downscaling_LMM_res_2mT_pretrain --cfg job | grep -A20 control_metric_weights
```
