"""Shape / instantiate smoke tests for LMM (latent MeanFlow)."""

from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.models.temperature_field_losses import TemperatureFieldLosses


def test_temperature_field_losses_pde_scalar():
    phys = TemperatureFieldLosses()
    T_f = torch.randn(2, 1, 16, 16)
    T_c = torch.randn(2, 1, 16, 16)
    loss = phys.temperature_pde_loss(T_f, T_c, num_supercells=2)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_anisotropic_transport_loss():
    phys = TemperatureFieldLosses()
    R_hat = torch.randn(2, 1, 32, 32)
    T_hr = torch.randn(2, 1, 32, 32)
    out = phys.anisotropic_transport_loss(
        R_hat, T_hr, dx=2000.0, dy=-2000.0, lambda_mag=1.0, lambda_dir=0.5, direction_kind="cosine"
    )
    assert set(out.keys()) == {"L_mag", "L_dir", "L_total"}
    for v in out.values():
        assert v.ndim == 0
        assert torch.isfinite(v)
    assert torch.allclose(out["L_total"], 1.0 * out["L_mag"] + 0.5 * out["L_dir"])


def test_hydra_instantiate_lmm_model_config():
    """Compose only ``model/lmm.yaml`` so ``train.yaml`` / ``paths`` (PROJECT_ROOT) are not required."""
    repo = Path(__file__).resolve().parents[1]
    config_dir = str(repo / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="model/lmm")
    OmegaConf.resolve(cfg)
    model = hydra.utils.instantiate(cfg.model)
    assert type(model).__name__ == "LatentMeanFlowLitModule"
