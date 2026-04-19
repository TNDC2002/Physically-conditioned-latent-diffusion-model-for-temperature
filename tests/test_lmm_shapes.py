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


def test_hydra_instantiate_lmm_model_config():
    """Compose only ``model/lmm.yaml`` so ``train.yaml`` / ``paths`` (PROJECT_ROOT) are not required."""
    repo = Path(__file__).resolve().parents[1]
    config_dir = str(repo / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="model/lmm")
    OmegaConf.resolve(cfg)
    model = hydra.utils.instantiate(cfg.model)
    assert type(model).__name__ == "LatentMeanFlowLitModule"
