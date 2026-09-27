import math

import pytest
import torch
from torch import nn

from snr_grad.batch_control import BatchController, BatchProbe, NoiseScale, _nuclear_statistics, probe_batch


def _loss(model, batch):
    return model(batch).mean()


def test_per_example_scaling_and_gradient_preservation():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    model.weight.grad = torch.tensor([[17.]])
    batch = torch.tensor([[0.], [2.], [4.], [6.]])
    p = probe_batch(model, _loss, batch, splits=4)
    # Unbiased variance of [0, 2, 4, 6] = 20/3, mean gradient = 3.
    assert p.euclidean.noise == pytest.approx(20 / 3)
    assert p.euclidean.signal == pytest.approx(9 - (20 / 3) / 4)
    assert p.euclidean.scale == pytest.approx((20 / 3) / (9 - 5 / 3))
    assert p.muon.scale == pytest.approx(20 / 27)
    assert model.weight.grad.item() == 17
    # With only two contiguous chunks the estimate is noisy; b=2 restores
    # the per-example scale of the *estimated* chunk-mean variance.
    q = probe_batch(model, _loss, batch, splits=2)
    assert q.euclidean.noise == pytest.approx(16)


def test_frozen_adam_preconditioner_changes_geometry():
    model = nn.Linear(2, 1, bias=False)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    weight = model.weight
    opt.state[weight]["exp_avg_sq"] = torch.tensor([[1., 100.]]) * (1 - .999)
    opt.state[weight]["step"] = 1
    batch = torch.tensor([[1., 2.], [2., 2.], [3., 2.], [4., 2.]])
    p = probe_batch(model, _loss, batch, splits=4, optimizer=opt)
    assert p.adamw.scale != pytest.approx(p.euclidean.scale)
    assert opt.state[weight]["exp_avg_sq"][0, 1].item() == pytest.approx(.1)


def test_muon_dual_norm_is_rotation_invariant_and_fallback_separate():
    model = nn.Linear(2, 2)
    x = torch.tensor([[2., 0.], [1., 1.], [0., 0.], [1., -1.]])
    p = probe_batch(model, _loss, x, splits=4)
    assert p.muon is not None and p.muon_fallback is not None
    assert p.muon.noise > 0
    assert math.isfinite(p.muon_fallback.noise)
    assert p.muon.signal > 0


def test_muon_noise_uses_row_covariance_for_rank_two_residuals():
    mean = torch.diag(torch.tensor([2., 1.]))
    residuals = [torch.diag(torch.tensor(pair)) for pair in
                 ((1., 0.), (-1., 0.), (0., 1.), (0., -1.))]
    signal, noise = _nuclear_statistics([mean + r for r in residuals], b=1)
    # C_row = diag(2/3, 2/3), so tr(sqrt(C_row))**2 = 8/3.
    assert signal.item() == pytest.approx(9.)
    assert noise.item() == pytest.approx(8 / 3)


def _probe(scale, seconds=0.):
    n = NoiseScale(1., scale, scale)
    return BatchProbe(8, 2, seconds, n, n, n, None)


def test_controller_warmup_dwell_missing_and_one_rung():
    c = BatchController([8, 16, 32], initial=8, ema=0., warmup=1, dwell=2)
    c.observe(_probe(100.))
    assert c.recommend().reason == "warmup"
    c.observe(_probe(100.))
    assert c.recommend().batch_size == 16
    assert c.recommend().batch_size == 16
    c.observe(None)
    assert c.recommend().reason == "no_probe"
    assert c.recommend().batch_size == 16
    c.observe(_probe(100.))
    assert c.recommend().batch_size == 32
    c.observe(_probe(float("inf")))
    assert c.recommend().reason == "unstable_probe"
    assert c.recommend().batch_size == 32


def test_probe_cost_guard_and_invalid_splits():
    c = BatchController([8, 16], initial=8, warmup=0)
    c.observe(_probe(100., seconds=2.), step_seconds=1.)
    assert c.recommend().reason == "expensive_probe"
    with pytest.raises(ValueError, match="equal"):
        probe_batch(nn.Linear(1, 1), _loss, torch.ones(5, 1), splits=2)


def test_noise_dominated_probe_does_not_force_batch_change():
    model = nn.Linear(1, 1, bias=False)
    p = probe_batch(model, _loss, torch.tensor([[-1.], [1.]]), splits=2)
    assert p.euclidean.signal == 0
    assert math.isinf(p.euclidean.scale)
    controller = BatchController([2, 4], initial=2, warmup=0)
    controller.observe(p)
    assert controller.recommend().reason == "unstable_probe"
    assert controller.recommend().batch_size == 2
