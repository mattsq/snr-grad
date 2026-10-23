"""Tests for the closed-form stability indices (snr_grad.stability_theory)."""

import math

import pytest
import torch
import torch.nn as nn

from snr_grad import (
    ExactQuadraticCurvature,
    LowRankCurvature,
    PreconditionedCurvature,
    adam_edge_ratio,
    adam_edge_threshold,
    adam_memory_forecast,
    adam_reactivation_gain,
    adam_stationary_preconditioner,
    dominant_mode_tail,
    noise_fluctuation_index,
    preconditioned_top_eigenvalue,
    sgd_stochastic_edge,
)


def _simulate_sgd(lam, lr, B, l_star, steps, seeds, gen):
    """Gaussian-design least squares in the eigenbasis: e' = e - lr (S e - noise)."""
    n = lam.numel()
    sq = lam.sqrt()
    e = torch.zeros(seeds, n, dtype=torch.float64)
    late = []
    for t in range(steps):
        z = torch.randn(seeds, B, n, generator=gen, dtype=torch.float64) * sq   # z ~ N(0, H)
        eps = torch.randn(seeds, B, generator=gen, dtype=torch.float64) * math.sqrt(2 * l_star)
        r = (z * e[:, None, :]).sum(-1) + eps                                   # residual z.e + eps
        g = (z * r[..., None]).mean(1)
        e = e - lr * g
        if t > steps // 2:
            late.append(0.5 * (lam * e * e).sum(-1).mean())
    return float(torch.stack(late).mean())


def test_sgd_stationary_excess_matches_closed_form():
    gen = torch.Generator().manual_seed(0)
    lam = torch.tensor([1.0, 0.5, 0.2, 0.1, 0.05, 0.02], dtype=torch.float64)
    B, l_star = 2, 0.5
    for lr in (0.4, 0.8):
        th = sgd_stochastic_edge(lam, lr, B)
        assert th["kappa"] < 1
        sim = _simulate_sgd(lam, lr, B, l_star, 6000, 400, gen)
        assert sim == pytest.approx(l_star * th["excess_over_floor"], rel=0.08)


def test_sgd_stochastic_edge_below_deterministic_edge():
    lam = torch.ones(20, dtype=torch.float64)          # diffuse spectrum, small batch
    th = sgd_stochastic_edge(lam, 0.2, 2)              # kappa = 0.1 * 20 / 1.7 = 1.18
    assert th["deterministic_margin"] > 0              # deterministically stable ...
    assert th["kappa"] > 1                             # ... but the mean loss diverges
    assert th["D_K"] == pytest.approx(20.0)
    gen = torch.Generator().manual_seed(1)
    blown = _simulate_sgd(lam, 0.2, 2, 0.1, 400, 64, gen)
    assert not math.isfinite(blown) or blown > 1e3
    calm = _simulate_sgd(lam, 0.15, 2, 0.1, 2000, 64, gen)  # kappa = 0.845
    assert calm == pytest.approx(0.1 * 0.845 / 0.155, rel=0.25)
    assert sgd_stochastic_edge(lam, 2.0, 2)["kappa"] == float("inf")
    concentrated = torch.tensor([1.0] + [1e-3] * 19, dtype=torch.float64)
    assert sgd_stochastic_edge(concentrated, 0.5, 2)["D_K"] < 1.2


def test_adam_threshold_matches_frozen_heavy_ball():
    beta1 = 0.9
    thr = adam_edge_threshold(beta1)
    assert thr == pytest.approx(38.0)
    for ratio, should_diverge in ((0.95, False), (1.05, True)):
        h, lr = 1.0, ratio * thr
        y, m = 1.0, 0.0
        for _ in range(3000):
            m = beta1 * m + (1 - beta1) * h * y
            y = y - lr * m
        assert (not math.isfinite(y) or abs(y) > 1e3) == should_diverge


def test_preconditioned_eigenvalue_and_ratio():
    gen = torch.Generator().manual_seed(2)
    n = 8
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    h = (q * torch.linspace(3.0, 0.1, n, dtype=torch.float64)) @ q.t()
    p = nn.Parameter(torch.zeros(n, dtype=torch.float64))
    curv = ExactQuadraticCurvature([p], h)
    d = torch.rand(n, generator=gen, dtype=torch.float64) + 0.5
    exact = float(torch.linalg.eigvalsh(d.sqrt()[:, None] * h * d.sqrt()[None, :]).max())
    est = preconditioned_top_eigenvalue(curv, [p], {p: d}, iters=500, tol=1e-12)
    assert est == pytest.approx(exact, rel=1e-6)
    assert adam_edge_ratio(curv, [p], {p: d}, 0.1, 0.9, iters=500, tol=1e-12) == pytest.approx(0.1 * exact / 38.0, rel=1e-6)
    var = {p: torch.full((n,), 4.0, dtype=torch.float64)}
    floor = adam_stationary_preconditioner(var, batch_size=16, eps=0.0)
    assert torch.allclose(floor[p], torch.full((n,), 2.0, dtype=torch.float64))
    mean = {p: torch.full((n,), 0.5, dtype=torch.float64)}
    stat = adam_stationary_preconditioner(var, 16, mean_grad=mean, eps=0.0)
    assert torch.allclose(stat[p], torch.full((n,), 1 / math.sqrt(0.5), dtype=torch.float64))


def test_dominant_mode_tail_consistency():
    B = 4
    # Mean-loss edge of a single mode: E(1 - a chi)^2 = 1  <=>  a = 2 / (1 + 2/B).
    a_mean = 2.0 / (1.0 + 2.0 / B)
    below = dominant_mode_tail(0.9 * a_mean, B, samples=200_000)
    above = dominant_mode_tail(1.1 * a_mean, B, samples=200_000)
    assert below["loss_tail_exponent"] > 1.0 > above["loss_tail_exponent"]
    assert above["lyapunov"] < 0                    # still contracting almost surely
    assert sgd_stochastic_edge([1.0], 0.9 * a_mean, B)["kappa"] < 1 < sgd_stochastic_edge([1.0], 1.1 * a_mean, B)["kappa"]
    assert dominant_mode_tail(3.0, B)["lyapunov"] > 0
    # Larger batches thin the tail at fixed lr * lambda.
    assert dominant_mode_tail(1.2, 64)["loss_tail_exponent"] > dominant_mode_tail(1.2, 4)["loss_tail_exponent"]


def test_noise_fluctuation_index_reduces_to_kappa():
    lam = torch.tensor([1.0, 0.3, 0.1, 0.01], dtype=torch.float64)
    L, lr, B = 0.7, 0.5, 16
    idx = noise_fluctuation_index(lam, 2 * L * lam, lr, B, L)
    th = sgd_stochastic_edge(lam, lr, B)
    # Same feedback sum up to the (1 + 1/B) finite-batch curvature term.
    assert idx["kappa_hat"] == pytest.approx(th["kappa"], rel=0.05)
    assert idx["D_hat"] == pytest.approx(th["D_K"], rel=0.05)
    assert idx["Phi"] == pytest.approx(idx["kappa_hat"] * math.sqrt(2 / idx["D_hat"]))
    slow = noise_fluctuation_index(lam, 2 * L * lam, lr, B, L, horizon=10)
    assert slow["kappa_hat"] < idx["kappa_hat"] and slow["D_hat"] < idx["D_hat"]
    assert noise_fluctuation_index([5.0], [1.0], 1.0, 4, 1.0)["Phi"] == float("inf")


def test_noise_fluctuation_index_rest_term():
    h = torch.tensor([1.0, 0.5], dtype=torch.float64)
    sig = torch.tensor([2.0, 1.0], dtype=torch.float64)
    full = noise_fluctuation_index(torch.cat([h, torch.full((50,), 0.1, dtype=torch.float64)]),
                                   torch.cat([sig, torch.full((50,), 0.02, dtype=torch.float64)]),
                                   0.3, 8, 1.0)
    lumped = noise_fluctuation_index(h, sig, 0.3, 8, 1.0, rest=(0.1, 1.0, 50))
    assert lumped["kappa_hat"] == pytest.approx(full["kappa_hat"])
    assert lumped["D_hat"] == pytest.approx(full["D_hat"])


def test_preconditioned_curvature_low_rank():
    gen = torch.Generator().manual_seed(5)
    n = 12
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    h = (q * torch.logspace(1, -2, n, dtype=torch.float64)) @ q.t()
    p = nn.Parameter(torch.zeros(n, dtype=torch.float64))
    d = torch.rand(n, generator=gen, dtype=torch.float64) + 0.2
    pc = PreconditionedCurvature(ExactQuadraticCurvature([p], h), {p: d})
    dense = d.sqrt()[:, None] * h * d.sqrt()[None, :]
    low = LowRankCurvature.from_proxy(pc, [p], rank=3, power_iters=4)
    assert torch.allclose(low.eigenvalues, torch.linalg.eigvalsh(dense).flip(0)[:3], rtol=1e-6)


def test_adam_memory_forecast():
    f = adam_memory_forecast(0.1, 4.0, 0.999, batch_size=64)
    assert not f["protected"]
    assert f["critical_batch"] == pytest.approx(4.0)
    assert f["onset_steps"] == pytest.approx(2 * math.log(10) / -math.log(0.999))
    assert adam_memory_forecast(0.1, 0.5, 0.999)["onset_steps"] == float("inf")
    assert adam_memory_forecast(1.2, 3.0, 0.99)["onset_steps"] == 0.0


def test_adam_reactivation_gain():
    a = adam_reactivation_gain(0.9, 0.999)
    assert a == pytest.approx(6.57, abs=0.02)
    assert adam_reactivation_gain(0.9, 0.95) == pytest.approx(1.10, abs=0.02)
    # Matches PyTorch Adam on one coordinate after a long silence (bias corrections ~ 1).
    x = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = torch.optim.Adam([x], lr=1.0, betas=(0.9, 0.999), eps=0.0)
    steps = []
    for t in range(20000):
        x.grad = torch.ones(1, dtype=torch.float64) if t < 5000 or t >= 19000 else torch.zeros(1, dtype=torch.float64)
        before = float(x.detach())
        opt.step()
        if t >= 19000:
            steps.append(abs(float(x.detach()) - before))
    expected = adam_reactivation_gain(0.9, 0.999, silence=14000, return_ratio=1.0)
    assert max(steps) == pytest.approx(expected, rel=1e-3)
    # A short silence barely amplifies; the gain falls towards 1.
    assert adam_reactivation_gain(0.9, 0.999, silence=10) < 1.05
