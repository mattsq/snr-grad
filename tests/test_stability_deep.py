"""Statistical and edge-case tests for the stability sensor and simulator."""

import copy
import math

import pytest
import torch
import torch.nn as nn

from snr_grad import (
    ExactQuadraticCurvature,
    GaussianProbeNoise,
    LowRankCurvature,
    QuadraticStabilitySimulator,
    StabilityProbe,
    aggregate_probe_groups,
    detect_excursions,
    frozen_update_map,
)
from snr_grad.stability import FrozenUpdateMap


def _probe_draws(mu, sigma_sqrt, k, b, gen):
    z = torch.randn(k, b, mu.numel(), generator=gen, dtype=torch.float64)
    return mu + (z @ sigma_sqrt.t()).mean(1)


def _diag_map(p, d):
    return FrozenUpdateMap("diag", True, exact_linear=True, fn=lambda g: {p: g[p] * d})


def test_progress_and_mean_curvature_are_unbiased():
    """Monte Carlo: U-statistics for mu^T P mu and mu^T P H P mu have no 1/K bias."""
    gen = torch.Generator().manual_seed(0)
    n = 5
    mu = torch.tensor([0.3, -0.2, 0.1, 0.0, 0.05], dtype=torch.float64)
    sigma_sqrt = torch.diag(torch.tensor([2.0, 1.0, 1.5, 0.5, 1.0], dtype=torch.float64))
    d = torch.tensor([1.0, 0.5, 2.0, 1.0, 0.3], dtype=torch.float64)
    h = torch.diag(torch.tensor([3.0, 1.0, 0.5, 2.0, 1.0], dtype=torch.float64))
    p = torch.zeros(n, dtype=torch.float64)
    curv = ExactQuadraticCurvature([p], h)
    prog, mc, naive = [], [], []
    for _ in range(3000):
        g = _probe_draws(mu, sigma_sqrt, 4, 2, gen)
        m = StabilityProbe([{p: g[k]} for k in range(4)], 2, update_map=_diag_map(p, d),
                           curvature=curv).measure()
        prog.append(m.directional_progress)
        mc.append(m.mean_curvature_term)
        gbar = g.mean(0)
        naive.append(float(gbar @ (d * gbar)))
    true_prog = float(mu @ (d * mu))
    true_mc = float((d * mu) @ h @ (d * mu))
    se = lambda xs: torch.tensor(xs).std().item() / math.sqrt(len(xs))  # noqa: E731
    assert abs(sum(prog) / len(prog) - true_prog) < 4 * se(prog)
    assert abs(sum(mc) / len(mc) - true_mc) < 4 * se(mc)
    # The naive plug-in is biased upward by tr(P C_micro)/K, far beyond its SE.
    assert sum(naive) / len(naive) - true_prog > 10 * se(naive)


def test_validation_gradient_progress():
    p = torch.zeros(3, dtype=torch.float64)
    grads = [{p: torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)},
             {p: torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)}]
    val = {p: torch.tensor([0.5, 1.0, 0.0], dtype=torch.float64)}
    m = StabilityProbe(grads, 1, validation_gradient=val).measure()
    assert m.directional_progress == pytest.approx(0.5 * 2.0)


def test_log_jackknife_interval_coverage_gaussian():
    gen = torch.Generator().manual_seed(1)
    n = 8
    evals = torch.linspace(0.2, 2.0, n, dtype=torch.float64)
    sigma_sqrt = torch.diag(evals.sqrt())
    h = torch.diag(torch.linspace(1.0, 0.1, n, dtype=torch.float64))
    p = torch.zeros(n, dtype=torch.float64)
    curv = ExactQuadraticCurvature([p], h)
    truth = float((h.diagonal() * evals).sum())
    hits, reps = 0, 400
    for _ in range(reps):
        g = _probe_draws(torch.zeros(n, dtype=torch.float64), sigma_sqrt, 32, 4, gen)
        m = StabilityProbe([{p: g[k]} for k in range(32)], 4, curvature=curv).measure()
        lo, hi = m.risk_trace_ci
        hits += lo <= truth <= hi
    assert 0.90 <= hits / reps <= 0.99
    with pytest.raises(ValueError):
        StabilityProbe([{p: g[0]}, {p: g[1]}], 4, ci_level=1.5)


@pytest.mark.parametrize("nesterov", [False, True])
def test_sgd_momentum_one_step_map(nesterov):
    w = nn.Parameter(torch.randn(4, dtype=torch.float64))
    opt = torch.optim.SGD([w], lr=0.1, momentum=0.9, nesterov=nesterov)
    for _ in range(3):
        w.grad = torch.randn(4, dtype=torch.float64)
        opt.step()
    pmap = frozen_update_map(opt)
    g = torch.randn(4, dtype=torch.float64)
    delta = 1e-3 * torch.randn(4, dtype=torch.float64)
    sd = opt.state_dict()

    def step(grad):
        w2 = nn.Parameter(w.detach().clone())
        o2 = torch.optim.SGD([w2], lr=0.1, momentum=0.9, nesterov=nesterov)
        o2.load_state_dict(copy.deepcopy(sd))
        w2.grad = grad
        o2.step()
        return w2.detach()

    assert torch.allclose(step(g + delta) - step(g), -0.1 * pmap({w: delta})[w], atol=1e-12)


def test_stationary_factor_matches_ema_noise():
    beta1 = 0.9
    p = nn.Parameter(torch.zeros(20000, dtype=torch.float64))
    opt = torch.optim.Adam([p], lr=0.0, betas=(beta1, 0.999))
    p.grad = torch.ones_like(p)
    opt.step()
    stat = frozen_update_map(opt, linearization="stationary")
    factor = float(stat({p: torch.ones_like(p)})[p][0] * (opt.state[p]["exp_avg_sq"][0] / (1 - 0.999 ** 2)).sqrt())
    gen = torch.Generator().manual_seed(0)
    m = torch.zeros(20000, dtype=torch.float64)
    for _ in range(200):
        m = beta1 * m + (1 - beta1) * torch.randn(20000, generator=gen, dtype=torch.float64)
    assert factor == pytest.approx(float(m.std()), rel=0.03)


def test_aggregate_probe_groups():
    gen = torch.Generator().manual_seed(0)
    p = torch.zeros(3, dtype=torch.float64)
    curv = ExactQuadraticCurvature([p], torch.eye(3, dtype=torch.float64))
    metrics = []
    for _ in range(5):
        g = _probe_draws(torch.ones(3, dtype=torch.float64), torch.eye(3, dtype=torch.float64), 8, 2, gen)
        metrics.append(StabilityProbe([{p: g[k]} for k in range(8)], 2, curvature=curv).measure())
    agg = aggregate_probe_groups(metrics)
    vals = torch.tensor([m.risk_trace_per_example for m in metrics])
    assert agg["groups"] == 5
    assert agg["risk_trace_per_example"] == pytest.approx(float(vals.mean()))
    assert agg["risk_trace_per_example_se"] == pytest.approx(float(vals.std() / math.sqrt(5)))


def test_float32_probe():
    p = torch.zeros(4)
    gen = torch.Generator().manual_seed(0)
    grads = [{p: torch.randn(4, generator=gen)} for _ in range(6)]
    m = StabilityProbe(grads, 2, curvature=ExactQuadraticCurvature([p], torch.eye(4))).measure(lr=0.1)
    assert math.isfinite(m.risk_trace_per_example) and m.noise_penalty > 0
    assert m.cost["gradient_bytes"] > 0


def test_detect_excursions_edge_cases():
    r = detect_excursions([1.0, 0.9])
    assert r.num_events == 0 and not r.diverged
    r = detect_excursions([float("nan")] * 10)
    assert r.diverged and r.divergence_step == 0
    x = [1.0 / (1 + t) for t in range(300)]
    x[150] = float("inf")       # transient non-finite value that recovers
    r = detect_excursions(x)
    assert not r.diverged and r.divergence_step == 150


def test_simulator_options_and_low_rank_equivalence():
    gen = torch.Generator().manual_seed(0)
    n = 6
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    h = (q * torch.linspace(2.0, 0.1, n, dtype=torch.float64)) @ q.t()
    p = nn.Parameter(torch.zeros(n, dtype=torch.float64))
    opt = torch.optim.SGD([p], lr=0.3)
    full = ExactQuadraticCurvature([p], h)
    low = LowRankCurvature.from_proxy(full, [p], rank=n, oversample=0)
    g0 = torch.randn(n, generator=gen, dtype=torch.float64)
    devs = torch.randn(5, n, generator=gen, dtype=torch.float64)
    noise = GaussianProbeNoise(devs - devs.mean(0), 2, 4)
    a = QuadraticStabilitySimulator([p], opt, full, g0, noise, loss0=5.0).run(3, 30, seed=4)
    b = QuadraticStabilitySimulator([p], opt, low, g0, noise, loss0=5.0).run(3, 30, seed=4)
    assert torch.allclose(a.losses, b.losses, rtol=1e-8)
    sched = QuadraticStabilitySimulator([p], opt, full, g0, None, loss0=5.0).run(
        1, 10, lr_schedule=lambda t: 0.0, modes=torch.eye(n, dtype=torch.float64)[:, :2])
    assert torch.allclose(sched.losses[0], torch.full((11,), 5.0, dtype=torch.float64))
    assert sched.mode_projections.shape == (1, 11, 2)
    with pytest.raises(ValueError):
        GaussianProbeNoise(devs, 2, 4, heavy_tail_dof=2.0)


def test_low_rank_simulation_projects_onto_subspace():
    """A truncated curvature must not let the surrogate loss fall without bound."""
    gen = torch.Generator().manual_seed(0)
    n = 30
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    h = (q * torch.logspace(0, -4, n, dtype=torch.float64)) @ q.t()
    p = nn.Parameter(torch.zeros(n, dtype=torch.float64))
    opt = torch.optim.SGD([p], lr=0.5)
    low = LowRankCurvature.from_proxy(ExactQuadraticCurvature([p], h), [p], rank=4)
    g0 = torch.randn(n, generator=gen, dtype=torch.float64)
    unprojected = QuadraticStabilitySimulator([p], opt, low, g0, None, loss0=1.0, subspace=None)
    with pytest.warns(RuntimeWarning, match="surrogate loss fell"):
        s_bad = unprojected.run(1, 200).summary()
    assert s_bad["surrogate_below_zero"]
    auto = QuadraticStabilitySimulator([p], opt, low, g0, None, loss0=1.0)
    assert auto.subspace is not None
    s_ok = auto.run(1, 200).summary()
    assert not s_ok["surrogate_below_zero"]
    assert s_ok["deterministic_min_loss"] > 0
    with pytest.raises(ValueError):
        QuadraticStabilitySimulator([p], opt, low, g0, None, subspace="bogus")


def test_fused_low_rank_path_matches_generic():
    gen = torch.Generator().manual_seed(3)
    n = 24
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    h = (q * torch.logspace(1, -3, n, dtype=torch.float64)) @ q.t()
    p = nn.Parameter(torch.randn(n, generator=gen, dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=0.05, weight_decay=0.01)
    p.grad = torch.randn(n, generator=gen, dtype=torch.float64)
    opt.step()
    low = LowRankCurvature.from_proxy(ExactQuadraticCurvature([p], h), [p], rank=5, trace_probes=50)
    assert low.residual > 0
    g0 = torch.randn(n, generator=gen, dtype=torch.float64)
    devs = torch.randn(6, n, generator=gen, dtype=torch.float64)
    noise = GaussianProbeNoise(devs - devs.mean(0), 2, 4, scaling="loss_proportional")
    fused = QuadraticStabilitySimulator([p], opt, low, g0, noise, loss0=3.0)
    assert fused._fused_low_rank(torch.Generator())
    generic = QuadraticStabilitySimulator([p], opt, low, g0, noise, loss0=3.0)
    generic._fused_low_rank = lambda generator: False
    a = fused.run(3, 40, seed=11)
    b = generic.run(3, 40, seed=11)
    assert torch.allclose(a.losses, b.losses, rtol=1e-9, atol=1e-12)
    assert torch.allclose(a.deterministic, b.deterministic, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Review fixes: read-only probing next to a live DoPr, and device placement
# ---------------------------------------------------------------------------

def _dopr_setup():
    from snr_grad import DoPr

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 1))
    opt = DoPr(torch.optim.Adam(model.parameters(), lr=1e-2), model)
    x, y = torch.randn(32, 5), torch.randn(32, 1)
    return model, opt, x, y


def test_diagnostics_do_not_touch_live_dopr_cache():
    from snr_grad import GGNCurvature, collect_microbatch_gradients
    from snr_grad.curvature import collect_activation_snapshot

    def loss_fn(m, chunk):
        return nn.functional.mse_loss(m(chunk[0]), chunk[1])

    results = []
    for probe in (False, True):
        model, opt, x, y = _dopr_setup()
        opt.zero_grad()
        loss_fn(model, (x, y)).backward()
        before = {m: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in acc.items()}
                  for m, acc in opt.ap._accum.items()}
        if probe:
            xp, yp = torch.randn(16, 5) * 3, torch.randn(16, 1)
            collect_microbatch_gradients(model, loss_fn, (xp, yp), num_splits=4)
            ggn = GGNCurvature(model, xp, lambda out: nn.functional.mse_loss(out, yp))
            ggn.matvec({p: torch.ones_like(p) for p in model.parameters()})
            snap = collect_activation_snapshot(model, lambda m: m(xp))
            assert len(snap) == 2          # the snapshot's own hooks still capture
            assert opt.ap._accum.keys() == before.keys()
            for m, acc in opt.ap._accum.items():
                assert acc["count"] == before[m]["count"]
                assert torch.equal(acc["gram"], before[m]["gram"])
        opt.step()
        results.append([p.detach().clone() for p in model.parameters()])
    for a, b in zip(*results):
        assert torch.equal(a, b)


def test_isolated_capture_allows_listed_preconditioner_and_nests():
    from snr_grad import isolated_activation_capture

    model, opt, x, _ = _dopr_setup()
    with isolated_activation_capture():
        model(x)
    assert not opt.ap._accum
    with isolated_activation_capture(opt.ap):
        with isolated_activation_capture():
            model(x)                       # inner block allows nobody
        assert not opt.ap._accum
        model(x)
    assert len(opt.ap._accum) == 2


def test_noise_samples_follow_stored_device():
    from snr_grad import CovarianceFactorNoise, ResampledProbeNoise

    dev = torch.randn(4, 6, device="meta")
    gen = torch.Generator().manual_seed(0)
    for noise in (GaussianProbeNoise(dev, 2, 8), ResampledProbeNoise(dev, 2, 8, sign_flip=True),
                  CovarianceFactorNoise(torch.randn(6, 3, device="meta"), 8)):
        assert noise(None, 1.0, 0, gen).device.type == "meta"


def test_low_rank_build_vectors_on_parameter_device():
    """The private CPU generator's samples must reach the proxy on the parameters' device."""
    from snr_grad import CurvatureProxy
    from snr_grad.curvature import CurvatureMetadata

    class Recorder(CurvatureProxy):
        def __init__(self):
            self.metadata = CurvatureMetadata(name="recorder", psd=True)
            self.devices = []

        def matvec(self, update):
            self.devices += [u.device.type for u in update.values()]
            raise RuntimeError("stop")

    p = nn.Parameter(torch.zeros(6, device="meta"))
    rec = Recorder()
    with pytest.raises(RuntimeError, match="stop"):
        LowRankCurvature.from_proxy(rec, [p], rank=2)
    assert rec.devices == ["meta"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_low_rank_and_noise_on_cuda():
    p = nn.Parameter(torch.zeros(8, device="cuda", dtype=torch.float64))
    h = torch.diag(torch.arange(1.0, 9.0, dtype=torch.float64, device="cuda"))
    low = LowRankCurvature.from_proxy(ExactQuadraticCurvature([p], h), [p], rank=3)
    assert low.eigenvalues.device.type == "cuda"
    assert low.eigenvalues[0].item() == pytest.approx(8.0, rel=1e-6)
    noise = GaussianProbeNoise(torch.randn(4, 8, device="cuda", dtype=torch.float64), 2, 8)
    assert noise(None, 1.0, 0, torch.Generator().manual_seed(0)).device.type == "cuda"
