import torch

from src.models.components.meanflow.meanflow_core import MeanFlowCore


def test_sample_t_r_shapes_and_order():
    core = MeanFlowCore()
    t, r = core.sample_t_r(batch_size=8, device=torch.device("cpu"))
    assert t.shape == (8,)
    assert r.shape == (8,)
    assert torch.all(t >= r)


def test_build_bridge_state_shapes_and_finite():
    core = MeanFlowCore()
    x0 = torch.randn(4, 1, 8, 8)
    t = torch.rand(4)
    x_t, aux = core.build_bridge_state(x0, t)
    assert x_t.shape == x0.shape
    assert aux["noise"].shape == x0.shape
    assert torch.isfinite(x_t).all()


def test_single_step_generate_shape():
    core = MeanFlowCore()
    x_t = torch.randn(2, 1, 8, 8)
    t = torch.ones(2)
    r = torch.zeros(2)
    u = torch.randn_like(x_t)
    r_hat = core.single_step_generate(x_t, t, r, u)
    assert r_hat.shape == x_t.shape
