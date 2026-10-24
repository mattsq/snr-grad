"""Tests for the Adam memory guard (snr_grad.guard)."""

import math

import pytest
import torch
import torch.nn as nn

from snr_grad import (
    DoPr,
    GuardedOptimizer,
    MemoryGuard,
    MemoryGuardConfig,
    SNRAdamW,
    SNRMuon,
    adam_reactivation_gain,
    adam_update_overshoot,
    muon_update_gain,
)
from snr_grad.guard import _is_orthogonalized


def _intermittent_grads(steps, dim, seed=0, silence=(150, 1400)):
    """Random gradients with a long silence on half the coordinates, then a return."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for t in range(steps):
        grad = torch.randn(dim, generator=g, dtype=torch.float64)
        if silence[0] <= t < silence[1]:
            grad[: dim // 2] = 0.0
        elif t >= silence[1]:
            grad[: dim // 2] *= 3.0
        out.append(grad)
    return out


def _reference_adam(grads, x0, lr, betas, eps, wd, decoupled, limit, mode):
    """Plain Adam with the guard semantics written out step by step."""
    x = x0.clone()
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)
    b1, b2 = betas
    for t, g in enumerate(grads, start=1):
        if wd and not decoupled:
            g = g + wd * x
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        m_hat, v_hat = m / bc1, v / bc2
        if mode == "refill":
            v_hat = torch.maximum(v_hat, (m_hat.abs() / limit - eps).clamp_min(0) ** 2)
            v = v_hat * bc2
        ratio = (m_hat / (v_hat.sqrt() + eps)).clamp(-limit, limit)
        if wd and decoupled:
            x = x * (1 - lr * wd)
        x = x - lr * ratio
    return x


def _run(opt_factory, grads, x0, config=None):
    x = nn.Parameter(x0.clone())
    opt = GuardedOptimizer(opt_factory([x]), config)
    for g in grads:
        x.grad = g.clone()
        opt.step()
    return x.detach(), opt


def test_disabled_guard_matches_base():
    grads = _intermittent_grads(400, 16)
    x0 = torch.randn(16, dtype=torch.float64)
    guarded, _ = _run(lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1), grads, x0,
                      MemoryGuardConfig(limit=math.inf))
    plain = nn.Parameter(x0.clone())
    opt = torch.optim.AdamW([plain], lr=0.01, weight_decay=0.1)
    for g in grads:
        plain.grad = g.clone()
        opt.step()
    assert torch.allclose(guarded, plain.detach(), atol=1e-14, rtol=0)


@pytest.mark.parametrize("mode", ["cap", "refill"])
@pytest.mark.parametrize("wd,decoupled", [(0.0, True), (0.05, True), (0.05, False)])
def test_matches_reference(mode, wd, decoupled):
    grads = _intermittent_grads(1600, 12, seed=1)
    x0 = torch.randn(12, dtype=torch.float64)
    cls = torch.optim.AdamW if decoupled else torch.optim.Adam
    got, _ = _run(lambda p: cls(p, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd), grads, x0,
                  MemoryGuardConfig(limit=1.0, mode=mode))
    want = _reference_adam(grads, x0, 0.01, (0.9, 0.999), 1e-8, wd, decoupled, 1.0, mode)
    assert torch.allclose(got, want, atol=1e-12, rtol=1e-9)


@pytest.mark.parametrize("mode", ["cap", "refill"])
def test_bounds_reactivation_step(mode):
    """A coordinate silent for long enough steps up to A * lr unguarded, limit * lr guarded."""
    lr = 1.0
    grads = [torch.ones(1, dtype=torch.float64) * (1.0 if t < 500 or t >= 9000 else 0.0) for t in range(9100)]
    steps = {}
    for name, cfg in (("plain", MemoryGuardConfig(limit=math.inf)), ("guarded", MemoryGuardConfig(mode=mode))):
        x = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        opt = GuardedOptimizer(torch.optim.Adam([x], lr=lr, betas=(0.9, 0.999), eps=0.0), cfg)
        largest = 0.0
        for t, g in enumerate(grads):
            before = float(x.detach())
            x.grad = g
            opt.step()
            if t >= 9000:
                largest = max(largest, abs(float(x.detach()) - before))
        steps[name] = largest
    # After 500 steps the memory holds only (1 - 0.999^500) of the unit gradient energy.
    remembered = 1 - 0.999 ** 500
    expected = adam_reactivation_gain(0.9, 0.999, silence=8500, return_ratio=1 / remembered)
    assert steps["plain"] == pytest.approx(expected, rel=1e-3)
    assert steps["plain"] > 5.0
    assert steps["guarded"] == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("impl", ["foreach", "fused"])
def test_multi_tensor_implementations(impl):
    grads = _intermittent_grads(1600, 12, seed=5)
    x0 = torch.randn(12, dtype=torch.float32)
    kw = {impl: True}
    try:
        got, _ = _run(lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.05, **kw),
                      [g.float() for g in grads], x0)
    except RuntimeError as err:          # fused CPU kernels are missing on old torch
        pytest.skip(str(err))
    want = _reference_adam([g.float() for g in grads], x0, 0.01, (0.9, 0.999), 1e-8, 0.05, True, 1.0, "refill")
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)


def test_refill_damps_following_steps_more_than_cap():
    grads = _intermittent_grads(1600, 8, seed=2)
    x0 = torch.zeros(8, dtype=torch.float64)
    _, cap = _run(lambda p: torch.optim.Adam(p, lr=0.01), grads, x0, MemoryGuardConfig(mode="cap", track_stats=True))
    _, refill = _run(lambda p: torch.optim.Adam(p, lr=0.01), grads, x0,
                     MemoryGuardConfig(mode="refill", track_stats=True))
    vc = next(iter(cap.state.values()))["exp_avg_sq"]
    vr = next(iter(refill.state.values()))["exp_avg_sq"]
    assert torch.all(vr >= vc - 1e-18) and bool((vr > vc).any())
    assert refill.guard_stats["max_ratio"] <= cap.guard_stats["max_ratio"] + 1e-12


def test_amsgrad_max_state_is_refilled():
    grads = _intermittent_grads(1600, 8, seed=3)
    x = nn.Parameter(torch.zeros(8, dtype=torch.float64))
    opt = GuardedOptimizer(torch.optim.Adam([x], lr=0.01, amsgrad=True))
    for g in grads:
        x.grad = g.clone()
        opt.step()
    st = opt.state[x]
    assert torch.all(st["max_exp_avg_sq"] >= st["exp_avg_sq"])
    assert max(adam_update_overshoot(opt).values()) <= 1.0 + 1e-9


def test_overshoot_monitor_and_stats():
    grads = _intermittent_grads(1600, 8, seed=4)
    x = nn.Parameter(torch.zeros(8, dtype=torch.float64))
    base = torch.optim.Adam([x], lr=0.01)
    guard = MemoryGuard(base, MemoryGuardConfig(limit=math.inf, track_stats=True))
    largest = 0.0
    for g in grads:
        x.grad = g.clone()
        guard.snapshot_()
        base.step()
        stats = guard.guard_()
        ratio = adam_update_overshoot(base)[x]
        # Without weight decay the step equals lr times the ratio of the largest coordinate.
        assert stats["max_ratio"] == pytest.approx(ratio, rel=1e-9)
        largest = max(largest, ratio)
    assert 1.5 < largest <= adam_reactivation_gain(0.9, 0.999) + 1e-9
    assert guard.last_stats["clipped"] == 0.0
    untracked = MemoryGuard(base)
    untracked.snapshot_()
    base.step()
    assert untracked.guard_() == {} and untracked.last_stats == {}


def test_snr_optimizer_and_dopr_composition():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 1))
    base = SNRAdamW(model.parameters(), lr=0.01, track_stats=True)
    opt = GuardedOptimizer(DoPr(base, model), MemoryGuardConfig(track_stats=True))
    x, y = torch.randn(64, 6), torch.randn(64, 1)
    for _ in range(30):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        before = [p.detach().clone() for p in model.parameters()]
        opt.step()
        for p, b in zip(model.parameters(), before):
            decay = 1 - 0.01 * base.param_groups[0]["weight_decay"]
            assert float((p.detach() - b * decay).abs().max()) <= 0.01 * 1.0 + 1e-7
    assert opt.last_stats is base.last_stats          # base stats still reachable
    assert set(opt.guard_stats) == {"max_ratio", "clipped", "refilled", "max_spectral_ratio", "rescaled"}
    assert len(adam_update_overshoot(opt)) == 4       # unwraps both wrappers


def test_scheduler_closure_and_state_dict():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    base = torch.optim.AdamW(model.parameters(), lr=0.1)
    opt = GuardedOptimizer(base)
    sched = torch.optim.lr_scheduler.StepLR(base, step_size=1, gamma=0.5)
    x, y = torch.randn(16, 4), torch.randn(16, 1)

    def closure():
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        return loss

    for _ in range(3):
        loss = opt.step(closure)
        sched.step()
    assert torch.isfinite(loss)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1 * 0.5 ** 3)
    sd = opt.state_dict()
    other = GuardedOptimizer(torch.optim.AdamW(model.parameters(), lr=0.1))
    other.load_state_dict(sd)
    assert other.param_groups[0]["lr"] == pytest.approx(0.1 * 0.5 ** 3)


def test_invalid_config_and_misuse():
    with pytest.raises(ValueError):
        MemoryGuardConfig(limit=0.0)
    with pytest.raises(ValueError):
        MemoryGuardConfig(mode="clip")
    x = nn.Parameter(torch.zeros(2))
    guard = MemoryGuard(torch.optim.Adam([x], lr=0.1))
    with pytest.raises(RuntimeError):
        guard.guard_()


def test_step_without_gradients_is_a_no_op():
    x = nn.Parameter(torch.ones(3))
    opt = GuardedOptimizer(torch.optim.AdamW([x], lr=0.1))
    opt.zero_grad(set_to_none=True)
    opt.step()                                  # no parameter has a gradient
    assert torch.equal(x.detach(), torch.ones(3))
    x.grad = torch.ones(3)
    opt.step()                                  # and the next real step still works
    assert float(x.detach().max()) < 1.0


# --- Muon-style optimizers: spectral geometry ---------------------------------------

def _muon_grads(steps, shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*shape, generator=g, dtype=torch.float64) for _ in range(steps)]


def _spec(t):
    return float(torch.linalg.matrix_norm(t.double(), ord=2))


def test_muon_update_gain():
    assert muon_update_gain() == pytest.approx(1.2024, abs=1e-4)
    assert muon_update_gain(steps=1) == pytest.approx(1.2024, abs=1e-4)
    assert muon_update_gain((1.5, -0.5, 0.0), steps=5) == pytest.approx(1.0, abs=1e-9)
    # torch Muon's steps never exceed it (up to bfloat16 rounding), and reach it.
    largest = 0.0
    for seed in range(5):
        w = nn.Parameter(torch.zeros(48, 80, dtype=torch.float64))
        opt = torch.optim.Muon([w], lr=1.0, momentum=0.0, nesterov=False)
        w.grad = torch.randn(48, 80, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
        opt.step()
        largest = max(largest, _spec(w.detach()))
    assert 1.1 < largest <= muon_update_gain() * 1.02


@pytest.mark.parametrize("adjust", [None, "original", "match_rms_adamw"])
def test_plain_muon_untouched_by_default(adjust):
    """Default guard leaves torch Muon unchanged; the nominal step uses Muon's lr adjustment."""
    grads = _muon_grads(60, (24, 40))
    x0 = torch.randn(24, 40, dtype=torch.float64)
    runs = {}
    for name in ("plain", "guarded"):
        w = nn.Parameter(x0.clone())
        base = torch.optim.Muon([w], lr=0.02, weight_decay=0.1, adjust_lr_fn=adjust)
        opt = base if name == "plain" else GuardedOptimizer(base, MemoryGuardConfig(track_stats=True))
        ratios = []
        for g in grads:
            w.grad = g.clone()
            opt.step()
            if name == "guarded":
                ratios.append(opt.guard_stats["max_spectral_ratio"])
                assert opt.guard_stats["rescaled"] == 0 and opt.guard_stats["clipped"] == 0
        runs[name] = (w.detach(), ratios)
    assert torch.allclose(runs["guarded"][0], runs["plain"][0], atol=1e-12, rtol=0)
    ratios = runs["guarded"][1]
    assert 1.1 < max(ratios) <= muon_update_gain() * 1.02    # lr_eff matches Muon's own scaling


def test_spectral_cap_enforces_nominal_step():
    """spectral_limit=1 trims Muon's 1.2 overshoot: every step (minus decay) has ||dW||_2 <= lr_eff."""
    grads = _muon_grads(80, (32, 20), seed=1)
    w = nn.Parameter(torch.randn(32, 20, dtype=torch.float64))
    lr, wd = 0.05, 0.1
    base = torch.optim.Muon([w], lr=lr, weight_decay=wd)
    opt = GuardedOptimizer(base, MemoryGuardConfig(spectral_limit=1.0, power_iters=None, track_stats=True))
    nominal = lr * math.sqrt(32 / 20)                    # adjust_lr_fn="original"
    rescaled = 0
    for g in grads:
        before = w.detach().clone()
        w.grad = g.clone()
        opt.step()
        step = _spec(w.detach() - before * (1 - lr * wd)) / nominal
        assert step <= 1.0 + 1e-9                        # exact norm: a strict cap
        rescaled += opt.guard_stats["rescaled"]
    assert rescaled > 40


def test_spectral_cap_bounds_an_overshooting_variant():
    """A Muon-style base whose step can blow up (here: a rank-1 burst) is scaled back exactly."""

    class BurstyMuon(torch.optim.Optimizer):
        def __init__(self, params, lr):
            super().__init__(params, dict(lr=lr, ns_steps=5))

        @torch.no_grad()
        def step(self):
            for group in self.param_groups:
                for p in group["params"]:
                    st = self.state[p]
                    st["t"] = st.get("t", 0) + 1
                    u, _, vh = torch.linalg.svd(p.grad, full_matrices=False)
                    p.sub_(group["lr"] * (u @ vh))
                    if st["t"] % 10 == 0:                      # occasional 8x burst along the top direction
                        p.sub_(group["lr"] * 7.0 * torch.outer(u[:, 0], vh[0]))

    w = nn.Parameter(torch.zeros(12, 16, dtype=torch.float64))
    opt = GuardedOptimizer(BurstyMuon([w], lr=0.1), MemoryGuardConfig(track_stats=True))
    largest_raw = 0.0
    for g in _muon_grads(40, (12, 16), seed=2):
        before = w.detach().clone()
        w.grad = g
        opt.step()
        largest_raw = max(largest_raw, opt.guard_stats["max_spectral_ratio"])
        assert _spec(w.detach() - before) <= 0.1 * 1.25 * (1 + 1e-6)   # power-iteration shortfall
    assert largest_raw == pytest.approx(8.0, rel=1e-6)


def test_snr_muon_mixed_geometry():
    """SNRMuon: matrices are capped in spectral norm, vectors per coordinate; both are refilled."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 1))
    base = SNRMuon(model.parameters(), lr=0.01)
    opt = GuardedOptimizer(base, MemoryGuardConfig(track_stats=True))
    x, y = torch.randn(64, 6), torch.randn(64, 1)
    for _ in range(30):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        before = [p.detach().clone() for p in model.parameters()]
        opt.step()
        for p, b in zip(model.parameters(), before):
            d = p.detach() - b
            if p.ndim == 2:
                assert _spec(d) <= 0.01 * 1.25 + 1e-7
            else:
                assert float(d.abs().max()) <= 0.01 + 1e-7
    assert 0.5 < opt.guard_stats["max_spectral_ratio"] <= 1.0 + 1e-3   # cubic Newton-Schulz: gain 1
    assert opt.guard_stats["max_ratio"] <= 50.0
    assert max(adam_update_overshoot(opt).values()) <= 1.0 + 1e-6      # refill reached the matrices too


def test_geometry_detection():
    w, b = torch.zeros(4, 3), torch.zeros(4)
    conv = torch.zeros(8, 3, 3, 3)
    assert _is_orthogonalized(w, {"ns_steps": 5}, "auto")
    assert _is_orthogonalized(w, {"muon_ns_steps": 5}, "auto")
    assert _is_orthogonalized(conv, {"use_muon": True}, "auto")
    assert not _is_orthogonalized(conv, {"muon_ns_steps": 5}, "auto")   # SNRMuon steps convs per coordinate
    assert not _is_orthogonalized(conv, {"ns_steps": 5}, "auto")
    assert not _is_orthogonalized(w, {"use_muon": False, "ns_steps": 5}, "auto")
    assert not _is_orthogonalized(b, {"ns_steps": 5}, "auto")
    assert not _is_orthogonalized(w, {"betas": (0.9, 0.999)}, "auto")
    assert _is_orthogonalized(w, {}, "spectral") and not _is_orthogonalized(w, {"ns_steps": 5}, "coordinate")
    with pytest.raises(ValueError):
        MemoryGuardConfig(spectral_limit=0.0)
    with pytest.raises(ValueError):
        MemoryGuardConfig(geometry="matrix")
    with pytest.raises(ValueError):
        MemoryGuardConfig(power_iters=0)


def test_muon_with_aux_adam_style_optimizer():
    """One optimizer, Keller Jordan style: ``use_muon`` groups orthogonalize, the others run Adam."""

    class MuonWithAuxAdam(torch.optim.Optimizer):
        def __init__(self, groups):
            for g in groups:
                g.setdefault("weight_decay", 0.0)
            super().__init__(groups, {})

        @torch.no_grad()
        def step(self):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    if group["use_muon"]:
                        buf = st.setdefault("momentum_buffer", torch.zeros_like(p))
                        buf.lerp_(p.grad, 1 - group["momentum"])
                        mat = p.grad.lerp(buf, group["momentum"]).reshape(p.shape[0], -1)
                        u, _, vh = torch.linalg.svd(mat, full_matrices=False)
                        scale = max(1.0, p.grad.size(-2) / p.grad.size(-1)) ** 0.5   # Keller: 1 for square kernels
                        p.add_((u @ vh).reshape(p.shape), alpha=-group["lr"] * scale)
                    else:
                        if not st:
                            st["step"] = 0
                            st["exp_avg"] = torch.zeros_like(p)
                            st["exp_avg_sq"] = torch.zeros_like(p)
                        st["step"] += 1
                        b1, b2 = group["betas"]
                        st["exp_avg"].lerp_(p.grad, 1 - b1)
                        st["exp_avg_sq"].lerp_(p.grad.square(), 1 - b2)
                        m_hat = st["exp_avg"] / (1 - b1 ** st["step"])
                        v_hat = st["exp_avg_sq"] / (1 - b2 ** st["step"])
                        p.add_(m_hat / (v_hat.sqrt() + group["eps"]), alpha=-group["lr"])

    conv = nn.Parameter(torch.randn(16, 2, 2, 2, dtype=torch.float64))   # (16, 8) flattened: rows/cols = 2
    bias = nn.Parameter(torch.zeros(8, dtype=torch.float64))
    opt = GuardedOptimizer(MuonWithAuxAdam([
        dict(params=[conv], use_muon=True, lr=0.02, momentum=0.95, weight_decay=0.01),
        dict(params=[bias], use_muon=False, lr=0.01, betas=(0.9, 0.999), eps=1e-10),
    ]), MemoryGuardConfig(spectral_limit=1.0, track_stats=True))
    grads = _intermittent_grads(1600, 8, seed=7)
    gen = torch.Generator().manual_seed(8)
    for t, g in enumerate(grads):
        before_c, before_b = conv.detach().clone(), bias.detach().clone()
        conv.grad = torch.randn(16, 2, 2, 2, generator=gen, dtype=torch.float64)
        bias.grad = g
        opt.step()
        dc = (conv.detach() - before_c * (1 - 0.02 * 0.01)).reshape(16, -1)
        assert _spec(dc) <= 0.02 * (1 + 1e-6)          # exact SVD steps: power iteration is exact here
        assert float((bias.detach() - before_b).abs().max()) <= 0.01 * (1 + 1e-9)
    assert opt.guard_stats["max_spectral_ratio"] == pytest.approx(1.0, abs=1e-6)   # an exact polar factor
    assert max(adam_update_overshoot(opt).values()) <= 1.0 + 1e-9


def test_stacked_matrices_are_capped_per_slice():
    """A 3-D ``use_muon`` parameter is a stack of matrices (Keller's batched Newton-Schulz)."""

    class StackedMuon(torch.optim.Optimizer):
        def __init__(self, params, lr):
            super().__init__(params, dict(lr=lr, use_muon=True, weight_decay=0.0))

        @torch.no_grad()
        def step(self):
            for group in self.param_groups:
                for p in group["params"]:
                    u, _, vh = torch.linalg.svd(p.grad, full_matrices=False)    # batched polar factor
                    p.sub_(group["lr"] * (u @ vh))
                    p[0].sub_(group["lr"] * 3.0 * (u[0] @ vh[0]))              # slice 0 overshoots 4x

    w = nn.Parameter(torch.zeros(3, 16, 16, dtype=torch.float64))
    opt = GuardedOptimizer(StackedMuon([w], lr=0.1), MemoryGuardConfig(power_iters=None, track_stats=True))
    for g in _muon_grads(5, (3, 16, 16), seed=4):
        before = w.detach().clone()
        w.grad = g
        opt.step()
        steps = [_spec(w.detach()[i] - before[i]) / 0.1 for i in range(3)]
        assert steps[0] == pytest.approx(1.25, rel=1e-9)                # capped
        assert steps[1] == pytest.approx(1.0, rel=1e-9) and steps[2] == pytest.approx(1.0, rel=1e-9)
        assert opt.guard_stats["rescaled"] == 1
        assert opt.guard_stats["max_spectral_ratio"] == pytest.approx(4.0, rel=1e-9)


def test_exact_norm_passes_non_finite_steps_through():
    """With ``power_iters=None`` a NaN gradient gives a NaN step, as in the base, not an SVD error."""
    w = nn.Parameter(torch.randn(8, 6, dtype=torch.float64))
    opt = GuardedOptimizer(torch.optim.Muon([w], lr=0.02), MemoryGuardConfig(power_iters=None, track_stats=True))
    w.grad = torch.full((8, 6), float("nan"), dtype=torch.float64)
    opt.step()
    assert bool(torch.isnan(w.detach()).all())
