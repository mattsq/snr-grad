import math

import pytest
import torch
from torch import nn

from snr_grad.batch_control import (BatchController, BatchProbe, CostAwareBatchController,
                                    GradientNoiseAccumulator, NoiseScale, StepTimeModel,
                                    _nuclear_statistics, coupled_lr, probe_batch)
from snr_grad.variance import tree_split


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
    assert signal == pytest.approx(9.)
    assert noise == pytest.approx(8 / 3)


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


def _mlp_loss(model, batch):
    x, y = batch
    return nn.functional.mse_loss(model(x), y)


def test_accumulator_matches_probe_and_keeps_the_training_gradient():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x, y = torch.randn(32, 5), torch.randn(32, 3)
    _mlp_loss(model, (x, y)).backward()
    opt.step()
    opt.zero_grad()
    reference = probe_batch(model, _mlp_loss, (x, y), splits=4, optimizer=opt)
    noise = GradientNoiseAccumulator(model.parameters(), 4, optimizer=opt)
    for chunk in tree_split((x, y), 4):
        (_mlp_loss(model, chunk) / 4).backward()
        noise.record()
    probe = noise.finish(32)
    for name in ("euclidean", "adamw", "muon", "muon_fallback"):
        a, b = getattr(probe, name), getattr(reference, name)
        assert a.signal == pytest.approx(b.signal, rel=1e-4)
        assert a.noise == pytest.approx(b.noise, rel=1e-4)
    full = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3))
    full.load_state_dict(model.state_dict())
    _mlp_loss(full, (x, y)).backward()
    for p, q in zip(model.parameters(), full.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)


def test_accumulator_handles_existing_gradients_and_grad_scale():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    model.weight.grad = torch.tensor([[5.]])
    noise = GradientNoiseAccumulator(model.parameters(), 4)
    for value in (0., 2., 4., 6.):
        # A scaled loss, as under GradScaler with scale 8.
        (model(torch.tensor([[value]])).mean() * 8 / 4).backward()
        noise.record(grad_scale=8.)
    probe = noise.finish(4)
    assert probe.euclidean.noise == pytest.approx(20 / 3)
    assert probe.euclidean.signal == pytest.approx(9 - (20 / 3) / 4)
    with pytest.raises(RuntimeError):
        noise.record()
    with pytest.raises(ValueError):
        GradientNoiseAccumulator(model.parameters(), 1)


def test_welford_moments_are_stable_when_signal_dominates():
    grads = [torch.full((3,), 1e4) + torch.tensor([1., -1., 0.]) * s for s in (1., -1., .5, -.5)]
    model = nn.Linear(3, 1, bias=False)
    noise = GradientNoiseAccumulator(model.parameters(), 4, matrix_sensor=False)
    for g in grads:
        model.weight.grad = g.view(1, 3) / 4 + (model.weight.grad if model.weight.grad is not None else 0)
        noise.record()
    stacked = torch.stack(grads).double()
    expected = float(stacked.var(0, unbiased=True).sum())
    assert noise.finish(4).euclidean.noise == pytest.approx(expected, rel=1e-3)


def test_coupled_lr_rules():
    assert coupled_lr(0.1, 64, 16, "linear") == pytest.approx(0.4)
    assert coupled_lr(0.1, 64, 16, "sqrt") == pytest.approx(0.2)
    assert coupled_lr(0.1, 64, 16, "none") == pytest.approx(0.1)
    with pytest.raises(ValueError):
        coupled_lr(0.1, 64, 16, "cube")


def test_step_time_model_interpolates_and_extrapolates():
    times = StepTimeModel({8: 1., 32: 1., 64: 2.}, ema=0.5)
    assert times(16) == pytest.approx(1.)
    assert times(48) == pytest.approx(1.5)
    assert times(128) == pytest.approx(4.)
    times.record(8, 3.)
    assert times(8) == pytest.approx(2.)
    assert StepTimeModel({8: 1.})(16) is None


def _scale_probe(scale):
    n = NoiseScale(1., scale, scale)
    return BatchProbe(8, 2, 0., n, n, n, None)


def _run_controller(controller, scale, decisions=12):
    for _ in range(decisions):
        controller.observe(_scale_probe(scale))
        controller.recommend()
    return controller.current


def test_cost_aware_controller_matches_the_budget():
    sizes = (4, 8, 16, 32, 64)
    # Pure example price: the smallest batch is always the most efficient.
    examples = CostAwareBatchController(sizes, initial=16, example_price=1., time_price=0.,
                                        ema=0., warmup=0, dwell=0)
    assert _run_controller(examples, 50.) == 4
    # Flat accelerator step time: the largest batch wins at a material noise scale.
    flat = StepTimeModel({b: 1. for b in sizes})
    steps = CostAwareBatchController(sizes, initial=4, step_times=flat, ema=0., warmup=0, dwell=0)
    assert _run_controller(steps, 10.) == 64
    # Near-noiseless gradients: the 5% deadband stops a move that gains too little.
    quiet = CostAwareBatchController(sizes, initial=16, step_times=flat, ema=0., warmup=0, dwell=0)
    assert _run_controller(quiet, 1.) == 16
    # Fixed plus per-example time: optimum near sqrt(B_noise * t0 / t1) = 16.
    linear = StepTimeModel({b: 1. + b / 16 for b in sizes})
    mixed = CostAwareBatchController(sizes, initial=4, step_times=linear, ema=0., warmup=0, dwell=0)
    assert _run_controller(mixed, 16.) == 16
    # The same cost with a much larger noise scale asks for a larger batch.
    noisier = CostAwareBatchController(sizes, initial=4, step_times=linear, ema=0., warmup=0, dwell=0)
    assert _run_controller(noisier, 256.) == 64


def test_cost_aware_controller_moves_one_rung_and_needs_known_costs():
    sizes = (4, 8, 16, 32, 64)
    c = CostAwareBatchController(sizes, initial=4, step_times=StepTimeModel({b: 1. for b in sizes}),
                                 ema=0., warmup=0, dwell=0)
    c.observe(_scale_probe(10.))
    assert c.recommend().batch_size == 8
    assert c.recommend().reason == "increase"
    unknown = CostAwareBatchController(sizes, initial=4, step_times=StepTimeModel(), ema=0.,
                                       warmup=0, dwell=0)
    unknown.observe(_scale_probe(10.))
    assert unknown.recommend().reason == "unknown_cost"
    with pytest.raises(ValueError):
        CostAwareBatchController(sizes, initial=4)
