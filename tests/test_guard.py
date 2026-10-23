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
    adam_reactivation_gain,
    adam_update_overshoot,
)


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
    assert set(opt.guard_stats) == {"max_ratio", "clipped", "refilled"}
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
