"""Tests for the checkpoint-local stochastic continuation simulator."""

import copy

import pytest
import torch
import torch.nn as nn

from snr_grad import (
    AdaptiveThresholdConfig,
    ActivationPrecondConfig,
    CovarianceFactorNoise,
    DoPr,
    ExactQuadraticCurvature,
    GaussianProbeNoise,
    QuadraticStabilitySimulator,
    ResampledProbeNoise,
    SNRAdamW,
    SpectralSNRMuon,
    StabilityProbe,
    clone_optimizer,
)


def _quadratic(n=6, seed=0):
    gen = torch.Generator().manual_seed(seed)
    evals = torch.logspace(0, -2, n, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    h = (q * evals) @ q.t()
    g0 = torch.randn(n, generator=gen, dtype=torch.float64)
    return h, g0


def _state_tensors(opt):
    out = []
    for group in opt.param_groups:
        for p in group["params"]:
            for k, v in sorted(opt.state[p].items()):
                if torch.is_tensor(v):
                    out.append(v.clone())
    return out


OPTIMIZERS = {
    "adamw": lambda ps: torch.optim.AdamW(ps, lr=1e-2, betas=(0.9, 0.999), weight_decay=0.1),
    "sgd_momentum": lambda ps: torch.optim.SGD(ps, lr=1e-2, momentum=0.9, weight_decay=0.01),
    "snr_adamw": lambda ps: SNRAdamW(ps, lr=1e-2, gate="soft", rho=0.9, weight_decay=0.05),
    "snr_adamw_adaptive": lambda ps: SNRAdamW(
        ps, lr=1e-2, betas=(0.9, 0.95), adaptive_threshold=AdaptiveThresholdConfig(
            mode="target_mean_gate", target_mean_gate=0.5, update_interval=1, warmup_steps=0,
        ),
    ),
    "spectral_muon": lambda ps: SpectralSNRMuon(ps, lr=1e-2),
}


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_state_replay_parity(name):
    """Simulator replay == real optimizer on the same gradient sequence."""
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(4, 3, dtype=torch.float64))
    b = nn.Parameter(torch.randn(3, dtype=torch.float64))
    params = [w, b] if name != "spectral_muon" else [w]
    opt = OPTIMIZERS[name](params)
    for _ in range(4):  # build non-trivial state
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    n = sum(p.numel() for p in params)
    curv = ExactQuadraticCurvature(params, torch.eye(n, dtype=torch.float64))
    sim = QuadraticStabilitySimulator(params, opt, curv, torch.zeros(n, dtype=torch.float64), None)
    gen = torch.Generator().manual_seed(1)
    seq = [torch.randn(n, generator=gen, dtype=torch.float64) for _ in range(6)]

    before_params = [p.detach().clone() for p in params]
    before_state = _state_tensors(opt)
    replayed = sim.replay(seq)
    # The live parameters and optimizer state are untouched by the replay.
    for a, p in zip(before_params, params):
        assert torch.equal(a, p.detach())
    for a, c in zip(before_state, _state_tensors(opt)):
        assert torch.equal(a, c)

    for t, g in enumerate(seq):
        o = 0
        for p in params:
            p.grad = g[o:o + p.numel()].reshape(p.shape).clone()
            o += p.numel()
        opt.step()
        real = torch.cat([p.detach().reshape(-1) for p in params])
        assert torch.allclose(real, replayed[t], atol=1e-12, rtol=1e-10), f"step {t}"


def test_deterministic_sgd_matches_closed_form():
    h, g0 = _quadratic()
    p = nn.Parameter(torch.zeros(6, dtype=torch.float64))
    opt = torch.optim.SGD([p], lr=0.5)
    sim = QuadraticStabilitySimulator([p], opt, ExactQuadraticCurvature([p], h), g0, None, loss0=2.0)
    dist = sim.run(1, 20)
    delta = torch.zeros(6, dtype=torch.float64)
    for _ in range(20):
        delta = delta - 0.5 * (g0 + h @ delta)
    expected = 2.0 + float(g0 @ delta) + 0.5 * float(delta @ h @ delta)
    assert dist.deterministic[-1].item() == pytest.approx(expected, rel=1e-10)
    assert torch.allclose(dist.losses[0], dist.deterministic)  # no noise model


def test_run_leaves_live_state_and_global_rng_unchanged():
    torch.manual_seed(0)
    model = nn.Linear(3, 2).double()
    opt = SNRAdamW(model.parameters(), lr=1e-2, track_stats=True)
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(8, 3, dtype=torch.float64)).pow(2).mean().backward()
        opt.step()
    params = list(model.parameters())
    grads = [{p: torch.randn_like(p) for p in params} for _ in range(8)]
    probe = StabilityProbe(grads, 4)
    n = sum(p.numel() for p in params)
    curv = ExactQuadraticCurvature(params, torch.eye(n, dtype=torch.float64))
    before = [p.detach().clone() for p in params], _state_tensors(opt), torch.get_rng_state()
    sim = QuadraticStabilitySimulator.from_checkpoint(opt, probe, curv, model=model)
    dist = sim.run(3, 15, seed=7)
    assert dist.mean_gate is not None and dist.mean_gate.shape == (3, 15)
    for a, p in zip(before[0], params):
        assert torch.equal(a, p.detach())
    for a, c in zip(before[1], _state_tensors(opt)):
        assert torch.equal(a, c)
    assert torch.equal(before[2], torch.get_rng_state())
    again = sim.run(3, 15, seed=7)
    assert torch.equal(dist.losses, again.losses)  # private, seeded generators


def test_gaussian_probe_noise_covariance():
    gen = torch.Generator().manual_seed(0)
    devs = torch.randn(5, 4, generator=gen, dtype=torch.float64)
    devs = devs - devs.mean(0)
    noise = GaussianProbeNoise(devs, microbatch_size=3, batch_size=12)
    target = 3.0 / 4.0 * devs.t() @ devs / 12.0
    samples = torch.stack([noise(None, 1.0, 0, gen) for _ in range(20000)])
    emp = samples.t() @ samples / samples.shape[0]
    assert torch.allclose(emp, target, atol=0.05 * float(target.abs().max()))
    heavy = GaussianProbeNoise(devs, 3, 12, heavy_tail_dof=5.0)
    hs = torch.stack([heavy(None, 1.0, 0, gen) for _ in range(20000)])
    assert torch.allclose(hs.t() @ hs / hs.shape[0], target, atol=0.1 * float(target.abs().max()))
    loss_prop = GaussianProbeNoise(devs, 3, 12, scaling="loss_proportional")
    torch.manual_seed(0)
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    assert torch.allclose(loss_prop(None, 4.0, 0, g1), 2.0 * noise(None, 1.0, 0, g2))


def test_stability_edge_ranking():
    """Divergence/excursions increase as the learning rate nears 2 / lambda_max."""
    h, g0 = _quadratic(n=8)
    lam = float(torch.linalg.eigvalsh(h).max())
    factor = torch.linalg.cholesky(h + 1e-12 * torch.eye(8, dtype=torch.float64))
    results = {}
    for mult in (0.3, 1.02):
        p = nn.Parameter(torch.zeros(8, dtype=torch.float64))
        opt = torch.optim.SGD([p], lr=mult * 2.0 / lam)
        sim = QuadraticStabilitySimulator(
            [p], opt, ExactQuadraticCurvature([p], h), g0,
            CovarianceFactorNoise(factor, batch_size=4), loss0=10.0,
        )
        results[mult] = sim.run(8, 400, seed=0).summary()
    assert results[0.3]["divergence_fraction"] == 0.0
    assert results[1.02]["divergence_fraction"] > 0.5
    assert results[1.02]["max_rel_excess_median"] > results[0.3]["max_rel_excess_median"]


def test_dopr_simulation_applies_frozen_activation_solve():
    torch.manual_seed(0)
    model = nn.Linear(3, 2, bias=False).double()
    x = torch.randn(16, 3, dtype=torch.float64) * torch.tensor([2.0, 1.0, 0.2], dtype=torch.float64)
    opt = DoPr(torch.optim.SGD(model.parameters(), lr=0.1), model,
               ActivationPrecondConfig(damping=0.1, ema_beta=0.0))
    opt.zero_grad()
    model(x).pow(2).mean().backward()
    opt.step()
    w = model.weight
    sim = QuadraticStabilitySimulator(
        [w], opt, ExactQuadraticCurvature([w], torch.eye(6, dtype=torch.float64)),
        torch.zeros(6, dtype=torch.float64), None,
    )
    g = torch.randn(6, dtype=torch.float64)
    out = sim.replay([g])[0]
    snap = next(iter(opt.ap.covariance_snapshot(source="ema").values()))
    expected = w.detach() - 0.1 * opt.ap._solve_linear(snap.sigma, g.reshape(2, 3))
    assert torch.allclose(out, expected.reshape(-1))


def test_clone_optimizer_is_independent():
    p = nn.Parameter(torch.randn(3))
    opt = torch.optim.Adam([p], lr=0.1)
    p.grad = torch.ones(3)
    opt.step()
    clone, (c,) = clone_optimizer(opt, [p])
    assert c is not p and torch.equal(c, p.detach())
    c.grad = torch.ones(3)
    clone.step()
    assert not torch.equal(c, p.detach())
    assert opt.state[p]["step"].item() == 1
    with pytest.raises(TypeError):
        clone_optimizer(DoPr(torch.optim.SGD([p], lr=0.1), nn.Linear(1, 1)), [p])


def test_check_validity():
    h, g0 = _quadratic(n=4)
    p = nn.Parameter(torch.ones(4, dtype=torch.float64))
    opt = torch.optim.SGD([p], lr=0.1)
    sim = QuadraticStabilitySimulator([p], opt, ExactQuadraticCurvature([p], h), g0, None, loss0=1.0)
    assert sim.check_validity(params_now=[p]) == []
    moved = [p.detach() * 1.5]
    assert any("parameter_drift" in r for r in sim.check_validity(params_now=moved))
    assert any("checkpoint_age" in r for r in sim.check_validity(steps_since_checkpoint=50, max_age=10))
    theta0 = p.detach().clone()
    quad_loss = lambda th: 1.0 + float(g0 @ (th - theta0)) + 0.5 * float((th - theta0) @ h @ (th - theta0))  # noqa: E731
    d = 0.1 * torch.ones(4, dtype=torch.float64)
    assert sim.check_validity(real_loss_fn=quad_loss, fd_direction=d) == []
    cubic = lambda th: quad_loss(th) + 50.0 * float(((th - theta0) ** 3).sum())  # noqa: E731
    assert any("surrogate_fd_error" in r for r in sim.check_validity(real_loss_fn=cubic, fd_direction=d))


def test_resampled_probe_noise_covariance_and_support():
    gen = torch.Generator().manual_seed(0)
    devs = torch.randn(6, 3, generator=gen, dtype=torch.float64) ** 3  # heavy-ish tails
    devs = devs - devs.mean(0)
    target = 6.0 / 5.0 * 4.0 / 16.0 * devs.t() @ devs / 6.0  # Sigma_hat / B, b=4, B=16
    noise = ResampledProbeNoise(devs, microbatch_size=4, batch_size=16)
    assert noise.m == 4
    samples = torch.stack([noise(None, 1.0, 0, gen) for _ in range(40000)])
    emp = samples.t() @ samples / samples.shape[0]
    assert torch.allclose(emp, target, rtol=0.1, atol=0.05 * float(target.abs().max()))
    # B == b: every sample is one rescaled observed deviation.
    single = ResampledProbeNoise(devs, microbatch_size=4, batch_size=4)
    x = single(None, 1.0, 0, gen)
    scaled = devs * (6 / 5) ** 0.5
    assert min(float((x - d).abs().max()) for d in scaled) < 1e-12


def test_from_checkpoint_noise_models():
    p = nn.Parameter(torch.zeros(3, dtype=torch.float64))
    opt = torch.optim.SGD([p], lr=0.1)
    gen = torch.Generator().manual_seed(0)
    probe = StabilityProbe([{p: torch.randn(3, generator=gen, dtype=torch.float64)} for _ in range(5)], 2)
    curv = ExactQuadraticCurvature([p], torch.eye(3, dtype=torch.float64))
    sim = QuadraticStabilitySimulator.from_checkpoint(opt, probe, curv, noise_model="resample")
    assert isinstance(sim.noise, ResampledProbeNoise)
    sim.run(2, 5)
    with pytest.raises(ValueError):
        QuadraticStabilitySimulator.from_checkpoint(opt, probe, curv, noise_model="bogus")
