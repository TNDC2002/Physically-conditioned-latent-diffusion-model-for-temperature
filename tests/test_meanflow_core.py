import torch

from src.models.components.meanflow.meanflow_core import MeanFlowCore
from src.models.components.meanflow.meanflow_paper_core import MeanFlowPaperCore


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


def test_compute_train_targets_contains_velocity_target():
    core = MeanFlowCore()
    x0 = torch.randn(3, 1, 8, 8)
    out = core.compute_train_targets(x0)
    assert out["x_t"].shape == x0.shape
    assert out["v_target"].shape == x0.shape
    assert out["t"].shape == (3,)
    assert out["r"].shape == (3,)


def test_meanflow_teacher_and_adaptive_l2_are_finite():
    core = MeanFlowCore()
    v_target = torch.randn(2, 1, 8, 8)
    dudt = torch.randn_like(v_target)
    t = torch.ones(2)
    r = torch.zeros(2)
    u_tgt = core.meanflow_teacher(v_target, t, r, dudt)
    assert u_tgt.shape == v_target.shape
    loss = core.adaptive_l2_loss(torch.randn_like(v_target))
    assert torch.isfinite(loss)


def test_paper_core_matches_public_api_smoke():
    core = MeanFlowPaperCore()
    t, r = core.sample_t_r(batch_size=4, device=torch.device("cpu"))
    assert t.shape == (4,) and r.shape == (4,)
    assert torch.all(t >= r)
    x0 = torch.randn(2, 1, 8, 8)
    out = core.compute_train_targets(x0)
    assert out["v_target"].shape == x0.shape
    loss = core.adaptive_l2_loss(torch.randn_like(x0))
    assert torch.isfinite(loss)


def test_paper_core_teacher_jvp_finite():
    core = MeanFlowPaperCore()

    def backbone(z, cur_r, cur_t):
        return z * 0.1 + cur_r.view(-1, 1, 1, 1) + cur_t.view(-1, 1, 1, 1)

    x_t = torch.randn(2, 1, 8, 8, requires_grad=True)
    t = torch.rand(2, requires_grad=True)
    r = torch.rand(2, requires_grad=True)
    v = torch.randn_like(x_t)
    err = core.compute_teacher_error(backbone, x_t, t, r, v, create_graph=False)
    assert err.shape == x_t.shape
    assert torch.isfinite(err).all()
