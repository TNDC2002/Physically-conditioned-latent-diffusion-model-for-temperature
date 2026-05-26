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
    base_keys = {"L_mag", "L_dir", "L_dir_cosine", "L_dir_unit_mse", "L_total"}
    assert set(out.keys()) == base_keys
    for v in out.values():
        assert v.ndim == 0
        assert torch.isfinite(v)
    assert torch.allclose(out["L_dir"], out["L_dir_cosine"])
    assert torch.allclose(out["L_total"], 1.0 * out["L_mag"] + 0.5 * out["L_dir"])


def test_anisotropic_transport_loss_qmag_mask():
    phys = TemperatureFieldLosses()
    torch.manual_seed(0)
    T_hr = torch.randn(1, 1, 16, 16)
    R_hat = torch.randn(1, 1, 16, 16)
    unmasked = phys.anisotropic_transport_loss(R_hat, T_hr, qmag_quantile=None)
    masked = phys.anisotropic_transport_loss(R_hat, T_hr, qmag_quantile=0.5)
    assert "at_mask_frac" in masked
    assert 0.0 < masked["at_mask_frac"].item() <= 1.0
    assert not torch.allclose(unmasked["L_mag"], masked["L_mag"])


def test_anisotropic_transport_loss_qmag_min_mask():
    phys = TemperatureFieldLosses()
    T_hr = torch.zeros(1, 1, 8, 8)
    T_hr[0, 0, 2:6, 2:6] = 10.0
    R_hat = torch.randn(1, 1, 8, 8)
    masked = phys.anisotropic_transport_loss(R_hat, T_hr, qmag_min=1.0e-12)
    assert "at_mask_frac" in masked
    assert 0.0 < masked["at_mask_frac"].item() < 1.0


def test_hydra_instantiate_lmm_model_config():
    """Compose only ``model/lmm.yaml`` so ``train.yaml`` / ``paths`` (PROJECT_ROOT) are not required."""
    repo = Path(__file__).resolve().parents[1]
    config_dir = str(repo / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="model/lmm")
    OmegaConf.resolve(cfg)
    model = hydra.utils.instantiate(cfg.model)
    assert type(model).__name__ == "LatentMeanFlowLitModule"
