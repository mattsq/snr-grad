"""Tests for the read-only stability sensor (snr_grad.stability / snr_grad.curvature)."""

import math

import pytest
import torch
import torch.nn as nn

from snr_grad import (
    ActivationCurvature,
    ActivationPrecondConfig,
    ActivationPreconditioner,
    DoPr,
    ExactQuadraticCurvature,
    GGNCurvature,
    HessianCurvature,
    SNRAdamW,
    SpectralSNRMuon,
    StabilityProbe,
    backward_with_microbatch_variance,
    collect_activation_snapshot,
    collect_microbatch_gradients,
    compute_gate,
    detect_excursions,
    exact_noise_geometry,
    frozen_update_map,
    per_example_variance_from_batch_mean,
    spectrum_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_microbatches(sigma_sqrt, mu, k, b, seed=0):
    """K microbatch means of b iid N(mu, Sigma) per-example gradients, as dicts."""
    gen = torch.Generator().manual_seed(seed)
    n = mu.numel()
    p = torch.zeros(n, dtype=torch.float64)
    out = []
    for _ in range(k):
        z = torch.randn(b, n, generator=gen, dtype=torch.float64)
        g = (mu + z @ sigma_sqrt.t()).mean(0)
        out.append({p: g})
    return p, out


def _regression(seed=0, b=32, d_in=5, d_out=3):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(d_in, 8), nn.Tanh(), nn.Linear(8, d_out))
    x = torch.randn(b, d_in)
    y = torch.randn(b, d_out)
    return model, (x, y)


def _mse(model, batch):
    x, y = batch
    return 0.5 * ((model(x) - y) ** 2).sum(1).mean()


def _train_steps(model, opt, batch, n=3):
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        _mse(model, batch).backward()
        opt.step()


# ---------------------------------------------------------------------------
# Exact formulas and scaling conventions
# ---------------------------------------------------------------------------

def test_risk_trace_matches_exact_quadratic():
    n = 12
    evals = 1.0 / torch.arange(1, n + 1, dtype=torch.float64) ** 1.5
    sigma_sqrt = torch.diag(evals.sqrt())
    h = torch.diag(torch.linspace(0.5, 3.0, n, dtype=torch.float64))
    mu = torch.zeros(n, dtype=torch.float64)
    p, grads = _gaussian_microbatches(sigma_sqrt, mu, k=600, b=4)
    curv = ExactQuadraticCurvature([p], h)
    m = StabilityProbe(grads, 4, curvature=curv).measure()
    exact = exact_noise_geometry(h, torch.diag(evals))
    assert m.risk_trace_per_example == pytest.approx(exact["risk_trace"], rel=0.1)
    assert m.trace_sigma == pytest.approx(float(evals.sum()), rel=0.1)
    assert m.concentration_status == "ok"
    assert m.concentration == pytest.approx(exact["concentration"], rel=0.2)
    # Jackknife SE is a sensible order of magnitude.
    assert 0 < m.risk_trace_se < 0.2 * m.risk_trace_per_example


def test_per_example_scale_independent_of_microbatch_size():
    n = 6
    sigma_sqrt = torch.eye(n, dtype=torch.float64) * 0.7
    mu = torch.ones(n, dtype=torch.float64)
    traces = []
    for b in (2, 16):
        _, grads = _gaussian_microbatches(sigma_sqrt, mu, k=800, b=b, seed=b)
        traces.append(StabilityProbe(grads, b).measure().trace_sigma)
    assert traces[0] == pytest.approx(0.49 * n, rel=0.1)
    assert traces[1] == pytest.approx(0.49 * n, rel=0.1)


def test_unbiased_grad_norm_and_gns():
    n = 4
    mu = torch.tensor([1.0, -2.0, 0.5, 0.0], dtype=torch.float64)
    _, grads = _gaussian_microbatches(torch.eye(n, dtype=torch.float64), mu, k=2000, b=2)
    m = StabilityProbe(grads, 2).measure()
    assert m.grad_norm_sq == pytest.approx(float(mu @ mu), rel=0.05)
    assert m.gns_euclidean == pytest.approx(n / float(mu @ mu), rel=0.1)


def test_trace_matches_microbatch_variance_backend_exactly():
    model, batch = _regression(b=24)
    k = 4
    grads = collect_microbatch_gradients(model, _mse, batch, num_splits=k)
    m = StabilityProbe(grads, 24 // k).measure()
    _, gv = backward_with_microbatch_variance(model, _mse, batch, num_splits=k)
    trace = sum(float(per_example_variance_from_batch_mean(s, 24).sum()) for s in gv.values())
    assert m.trace_sigma == pytest.approx(trace, rel=1e-5)


def test_noise_penalty_scaling_and_correlated_data_flag():
    n = 5
    p, grads = _gaussian_microbatches(torch.eye(n, dtype=torch.float64), torch.ones(n, dtype=torch.float64), 50, 4)
    curv = ExactQuadraticCurvature([p], torch.eye(n, dtype=torch.float64))
    probe = StabilityProbe(grads, 4, curvature=curv)
    m8 = probe.measure(lr=0.1, batch_size=8)
    m16 = probe.measure(lr=0.1, batch_size=16)
    assert m8.noise_penalty == pytest.approx(2 * m16.noise_penalty)
    assert m8.noise_penalty == pytest.approx(0.01 * m8.risk_trace_per_example / 16)
    corr = StabilityProbe(grads, 4, curvature=curv, independent=False).measure(lr=0.1, batch_size=8)
    assert corr.noise_penalty is None
    assert any("independence_violated" in f for f in corr.flags)


# ---------------------------------------------------------------------------
# Honest confidence / abstention
# ---------------------------------------------------------------------------

def test_low_k_concentration_abstains():
    n = 50
    p, grads = _gaussian_microbatches(torch.eye(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64), 4, 8)
    curv = ExactQuadraticCurvature([p], torch.eye(n, dtype=torch.float64))
    m = StabilityProbe(grads, 8, curvature=curv).measure()
    assert m.concentration is None
    assert m.concentration_status == "insufficient_samples"
    assert m.concentration_plugin <= 3.0 + 1e-9  # rank <= K-1, not evidence


def test_high_rank_not_resolved():
    n = 200
    p, grads = _gaussian_microbatches(torch.eye(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64), 41, 2)
    curv = ExactQuadraticCurvature([p], torch.eye(n, dtype=torch.float64))
    m = StabilityProbe(grads, 2, curvature=curv).measure()
    assert m.concentration is None
    assert m.concentration_status == "rank_not_resolved"


def test_indefinite_curvature_is_flagged_not_used():
    h = torch.diag(torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64))
    p = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="indefinite"):
        ExactQuadraticCurvature([p], h)
    curv = ExactQuadraticCurvature([p], h, psd_projection="none")
    assert not curv.metadata.psd
    _, grads = _gaussian_microbatches(torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 40, 2)
    grads = [{p: g[next(iter(g))]} for g in grads]
    m = StabilityProbe(grads, 2, curvature=curv).measure()
    assert m.concentration_status == "indefinite_curvature"
    assert "curvature_not_psd" in m.flags
    clipped = ExactQuadraticCurvature([p], h, psd_projection="clip")
    assert clipped.metadata.psd and clipped.metadata.psd_projection == "clip"
    with pytest.raises(ValueError):
        exact_noise_geometry(h, torch.eye(3, dtype=torch.float64))


# ---------------------------------------------------------------------------
# Read-only guarantees
# ---------------------------------------------------------------------------

def _snapshot(model, opt):
    params = [p.detach().clone() for p in model.parameters()]
    grads = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    req = [p.requires_grad for p in model.parameters()]
    state = {
        id(p): {k: (v.clone() if torch.is_tensor(v) else v) for k, v in opt.state[p].items()}
        for p in model.parameters()
    }
    return params, grads, req, state, torch.get_rng_state()


def test_probe_does_not_change_training_state():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 8), nn.Dropout(0.2), nn.Linear(8, 3))
    x, y = torch.randn(32, 5), torch.randn(32, 3)
    opt = SNRAdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    _train_steps(model, opt, (x, y))
    opt.zero_grad(set_to_none=False)
    _mse(model, (x, y)).backward()  # leave a live .grad around
    before = _snapshot(model, opt)
    grads = collect_microbatch_gradients(model, _mse, (x, y), num_splits=4)
    curv = ActivationCurvature.from_model(model, lambda m: m(x))
    probe = StabilityProbe.from_microbatches(grads, 8, opt, curv)
    probe.measure(lr=1e-2, batch_size=32)
    after = _snapshot(model, opt)
    for a, b in zip(before[0], after[0]):
        assert torch.equal(a, b)
    for a, b in zip(before[1], after[1]):
        assert torch.equal(a, b)
    assert before[2] == after[2]
    for pid, st in before[3].items():
        for key, v in st.items():
            w = after[3][pid][key]
            assert torch.equal(v, w) if torch.is_tensor(v) else v == w
    assert torch.equal(before[4], after[4])


def test_collect_microbatch_rejects_unequal_chunks():
    model, batch = _regression(b=10)
    with pytest.raises(ValueError, match="divisible"):
        collect_microbatch_gradients(model, _mse, batch, num_splits=4)


# ---------------------------------------------------------------------------
# Frozen update maps
# ---------------------------------------------------------------------------

def _flat(d, params):
    return torch.cat([d[p].reshape(-1) for p in params])


@pytest.mark.parametrize("make_opt", [
    lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9),
    lambda ps: torch.optim.Adam(ps, lr=1e-2, betas=(0.9, 0.9999)),
    lambda ps: torch.optim.AdamW(ps, lr=1e-2, betas=(0.8, 0.9999), weight_decay=0.1),
])
def test_one_step_map_matches_finite_difference(make_opt):
    import copy

    model, (x, y) = _regression()
    model = model.double()
    batch = (x.double(), y.double())
    opt = make_opt(model.parameters())
    _train_steps(model, opt, batch, n=5)
    params = list(model.parameters())
    opt.zero_grad(set_to_none=True)
    _mse(model, batch).backward()
    g = {p: p.grad.detach().clone() for p in params}
    delta = {p: 1e-5 * torch.randn_like(p) for p in params}
    pmap = frozen_update_map(opt, at_gradient=g)
    assert pmap.exact_linear

    def step_with(grad):
        m2 = copy.deepcopy(model)
        o2 = make_opt(m2.parameters())
        o2.load_state_dict(copy.deepcopy(opt.state_dict()))
        for p2, p in zip(m2.parameters(), params):
            p2.grad = grad[p].clone()
        o2.step()
        return torch.cat([p2.detach().reshape(-1) for p2 in m2.parameters()])

    lr = opt.param_groups[0]["lr"]
    plus = step_with({p: g[p] + delta[p] for p in params})
    base = step_with(g)
    predicted = -lr * _flat(pmap(delta), params)
    rel_err = float((plus - base - predicted).norm() / predicted.norm())
    assert rel_err < 1e-2


def test_snr_adamw_map_gated_vs_ungated():
    model, batch = _regression()
    opt = SNRAdamW(model.parameters(), lr=1e-2, gate="soft")
    _train_steps(model, opt, batch, n=6)
    gated = frozen_update_map(opt, gated=True)
    ungated = frozen_update_map(opt, gated=False)
    assert gated.details["mean_gate"] is not None and 0 <= gated.details["mean_gate"] <= 1
    ones = {p: torch.ones_like(p) for p in model.parameters()}
    g_out, u_out = gated(ones), ungated(ones)
    group = opt.param_groups[0]
    for p in model.parameters():
        st = opt.state[p]
        t = st["step"]
        m_hat = st["exp_avg"] / (1 - group["betas"][0] ** t)
        s_hat = st["exp_grad_var"] / (1 - group["rho"] ** t)
        q = compute_gate(m_hat, s_hat, gate="soft", alpha=1.0, lambda_pop=1.0, gate_eps=group["gate_eps"])
        assert torch.allclose(g_out[p], q * u_out[p])


def test_unsupported_optimizers_are_explicit():
    w = nn.Parameter(torch.randn(4, 4))
    m = frozen_update_map(SpectralSNRMuon([w]))
    assert not m.supported and "nonlinear" in m.reason
    with pytest.raises(RuntimeError):
        m({w: torch.zeros(4, 4)})
    fresh = frozen_update_map(torch.optim.Adam([w]))
    assert not fresh.supported  # no state yet
    probe = StabilityProbe([{w: torch.randn(4, 4)}, {w: torch.randn(4, 4)}], 1, update_map=m)
    out = probe.measure()
    assert out.gns_optimizer is None
    assert any("update_map_unsupported" in f for f in out.flags)


def test_dopr_map_composes_activation_solve():
    torch.manual_seed(0)
    model = nn.Linear(4, 3, bias=False)
    x = torch.randn(16, 4) * torch.tensor([3.0, 1.0, 0.3, 0.1])
    opt = DoPr(torch.optim.Adam(model.parameters(), lr=1e-2), model,
               ActivationPrecondConfig(damping=0.1, ema_beta=0.5))
    for _ in range(3):
        opt.zero_grad()
        (model(x) ** 2).mean().backward()
        opt.step()
    pmap = frozen_update_map(opt)
    assert pmap.supported, pmap.reason
    base = frozen_update_map(opt.base)
    g = {model.weight: torch.randn(3, 4)}
    snap = next(iter(opt.ap.covariance_snapshot(source="ema").values()))
    manual = base({model.weight: opt.ap._solve_linear(snap.sigma, g[model.weight])})
    assert torch.allclose(pmap(g)[model.weight], manual[model.weight])


# ---------------------------------------------------------------------------
# Curvature backends
# ---------------------------------------------------------------------------

def test_activation_proxy_equals_ggn_and_hessian_for_linear_mse():
    torch.manual_seed(1)
    model = nn.Linear(5, 3, bias=False)
    x, y = torch.randn(40, 5), torch.randn(40, 3)
    out_loss = lambda out: 0.5 * ((out - y) ** 2).sum(1).mean()  # noqa: E731
    act = ActivationCurvature.from_model(model, lambda m: m(x))
    ggn = GGNCurvature(model, x, out_loss)
    hess = HessianCurvature(model, x, out_loss)
    u = {model.weight: torch.randn(3, 5)}
    expected = float(((x @ u[model.weight].t()) ** 2).sum(1).mean())
    assert act.quad(u) == pytest.approx(expected, rel=1e-5)
    assert ggn.quad(u) == pytest.approx(expected, rel=1e-5)
    assert hess.quad(u) == pytest.approx(expected, rel=1e-5)
    assert torch.allclose(ggn.matvec(u)[model.weight], act.matvec(u)[model.weight], atol=1e-5)
    assert not hess.metadata.psd and ggn.metadata.psd


def test_activation_proxy_reports_uncovered_params():
    model, (x, _) = _regression()
    act = ActivationCurvature.from_model(model, lambda m: m(x))
    assert act.metadata.covered == 2
    assert sorted(act.metadata.uncovered) == ["0.bias", "2.bias"]


def test_covariance_snapshot_is_read_only():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    x = torch.randn(8, 4)
    ap1 = ActivationPreconditioner(model, ActivationPrecondConfig(distributed=False))
    model.zero_grad()
    model(x).sum().backward()
    snap = ap1.covariance_snapshot()
    (name, s), = snap.items()
    assert s.source == "batch" and s.sample_count == 8 and not s.centered
    assert torch.allclose(s.sigma, x.t() @ x / 8)
    s.sigma.zero_()  # mutating the copy must not affect preconditioning
    assert ap1.step_count == 0
    ap1.precondition_()
    g1 = model.weight.grad.clone()
    ap1.remove_hooks()
    ap2 = ActivationPreconditioner(model, ActivationPrecondConfig(distributed=False))
    model.zero_grad()
    model(x).sum().backward()
    ap2.precondition_()
    assert torch.allclose(model.weight.grad, g1)
    assert ap2.covariance_snapshot() == {}  # batch cache cleared, no EMA


def test_collect_activation_snapshot_leaves_no_hooks():
    model = nn.Linear(3, 2)
    snap = collect_activation_snapshot(model, lambda m: m(torch.randn(5, 3)))
    assert len(snap) == 1
    assert len(model._forward_pre_hooks) == 0
    assert model.weight.grad is None


# ---------------------------------------------------------------------------
# Spectral proxies and excursion detection
# ---------------------------------------------------------------------------

def test_spectrum_metrics_power_law_slope():
    alpha = 0.9
    eigs = torch.arange(1, 401, dtype=torch.float64) ** (-2 * alpha)
    m = spectrum_metrics(eigs)
    assert m["kind"] == "activation_spectrum"
    assert m["powerlaw_slope"] == pytest.approx(-2 * alpha, abs=0.02)
    flat = spectrum_metrics(torch.ones(100, dtype=torch.float64))
    assert flat["participation_ratio"] == pytest.approx(100.0)
    assert m["participation_ratio"] < 5
    assert spectrum_metrics(torch.eye(10), sample_count=4)["status"] == "rank_limited_by_samples"


def test_detect_excursions():
    t = torch.arange(500, dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    smooth = 1.0 / (1 + 0.05 * t) * (1 + 0.01 * torch.randn(500, generator=gen, dtype=torch.float64))
    r = detect_excursions(smooth)
    assert r.num_events == 0 and not r.diverged
    spiky = smooth.clone()
    spiky[200:204] *= 5.0
    spiky[350] *= 3.0
    r = detect_excursions(spiky)
    assert r.num_events == 2
    assert r.event_steps[0] == 200 and r.durations[0] == 4
    assert r.max_excursion > 2.0
    div = smooth.clone()
    div[400:] = float("inf")
    r = detect_excursions(div)
    assert r.diverged and r.divergence_step == 400
