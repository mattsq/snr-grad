"""Tests for adaptive SNR-gate thresholding (snr_grad.adaptive)."""

import copy
import math

import pytest
import torch
import torch.nn as nn

from snr_grad import SNRAdamW, MARSSNRAdamW, AdaptiveThresholdConfig
from snr_grad.adaptive import (
    AdaptiveObservation,
    apply_adaptive_update,
    coerce_adaptive_config,
    lambda_for_target_active_fraction,
    smooth_clamped_update,
)


def _fake_group(cfg, *, lambda_pop=1.0, alpha=1.0, gate="snr", step=1):
    """A minimal optimizer-group-like dict with adaptive state initialised."""
    return {
        "adaptive_threshold": cfg,
        "lambda_pop": lambda_pop,
        "alpha": alpha,
        "gate": gate,
        "_adaptive_state": {
            "step": step,
            "ema_mean_gate": None,
            "ema_active_fraction": None,
            "force_update_countdown": 0,
            "shock_countdown": 0,
        },
    }


def _make_model(in_dim=64, out_dim=16, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(in_dim, out_dim, bias=False)


def _train_step(model, opt, *, batch=16, seed=None, **step_kwargs):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(batch, model.in_features)
    y = torch.randn(batch, model.out_features)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step(**step_kwargs)
    opt.zero_grad(set_to_none=True)
    return loss.item()


# ---------------------------------------------------------------------------
# Config coercion + validation
# ---------------------------------------------------------------------------

class TestConfig:
    def test_coerce_none(self):
        assert coerce_adaptive_config(None) is None

    def test_coerce_dict(self):
        cfg = coerce_adaptive_config({"mode": "target_mean_gate", "target_mean_gate": 0.3})
        assert isinstance(cfg, AdaptiveThresholdConfig)
        assert cfg.mode == "target_mean_gate"
        assert cfg.target_mean_gate == 0.3

    def test_coerce_passthrough(self):
        cfg = AdaptiveThresholdConfig(mode="off")
        assert coerce_adaptive_config(cfg) is cfg

    def test_coerce_bad_type(self):
        with pytest.raises(TypeError):
            coerce_adaptive_config(3.0)

    @pytest.mark.parametrize("kwargs", [
        {"mode": "nonsense"},
        {"target_active_fraction": 0.0},
        {"target_active_fraction": 1.0},
        {"active_gate_threshold": 1.5},
        {"update_interval": 0},
        {"beta": 1.0},
        {"max_log_change": 0.0},
        {"min_lambda_pop": 10.0, "max_lambda_pop": 1.0},
        {"sparse_target_active_fraction": 0.0},
        {"shock_target_active_fraction": 1.0},
        {"shock_fast_beta": 1.0},
        {"shift_detect_threshold": -0.1},
        {"shock_steps": -1},
    ])
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(ValueError):
            AdaptiveThresholdConfig(**kwargs)


# ---------------------------------------------------------------------------
# 1. Mean-gate controller direction
# ---------------------------------------------------------------------------

class TestMeanGateController:
    def test_too_permissive_raises_lambda(self):
        cfg = AdaptiveThresholdConfig(
            mode="target_mean_gate", target_mean_gate=0.2,
            adaptation_lr=0.1, tolerance=0.0, beta=0.0, warmup_steps=0,
        )
        group = _fake_group(cfg, lambda_pop=1.0)
        # observed_mean_gate > target => gate too permissive => lambda should rise
        apply_adaptive_update(group, cfg, AdaptiveObservation(mean_gate=0.6), alpha_value=1.0)
        assert group["lambda_pop"] > 1.0

    def test_too_suppressive_lowers_lambda(self):
        cfg = AdaptiveThresholdConfig(
            mode="target_mean_gate", target_mean_gate=0.2,
            adaptation_lr=0.1, tolerance=0.0, beta=0.0, warmup_steps=0,
        )
        group = _fake_group(cfg, lambda_pop=1.0)
        apply_adaptive_update(group, cfg, AdaptiveObservation(mean_gate=0.05), alpha_value=1.0)
        assert group["lambda_pop"] < 1.0

    def test_within_tolerance_no_change(self):
        cfg = AdaptiveThresholdConfig(
            mode="target_mean_gate", target_mean_gate=0.2,
            adaptation_lr=0.1, tolerance=0.05, beta=0.0, warmup_steps=0,
        )
        group = _fake_group(cfg, lambda_pop=1.0)
        apply_adaptive_update(group, cfg, AdaptiveObservation(mean_gate=0.22), alpha_value=1.0)
        assert group["lambda_pop"] == 1.0

    def test_adapt_alpha_for_soft_gate(self):
        cfg = AdaptiveThresholdConfig(
            mode="target_mean_gate", target_mean_gate=0.2, adapt="alpha",
            adaptation_lr=0.1, tolerance=0.0, beta=0.0, warmup_steps=0,
        )
        group = _fake_group(cfg, gate="soft", alpha=1.0)
        apply_adaptive_update(group, cfg, AdaptiveObservation(mean_gate=0.6), alpha_value=1.0)
        assert group["alpha"] > 1.0

    def test_integration_high_target_lowers_lambda(self):
        # An unreachably-high target keeps the gate "too suppressive", so the
        # controller should drive lambda_pop below its base value.
        model = _make_model()
        opt = SNRAdamW(
            model.parameters(), lr=1e-2, gate="snr", lambda_pop=1.0,
            adaptive_threshold=AdaptiveThresholdConfig(
                mode="target_mean_gate", target_mean_gate=0.95,
                warmup_steps=5, update_interval=5, adaptation_lr=0.2, tolerance=0.0,
            ),
        )
        for i in range(150):
            _train_step(model, opt, seed=i)
        assert opt.param_groups[0]["lambda_pop"] < 1.0

    def test_integration_low_target_raises_lambda(self):
        # A very low target keeps the gate "too permissive", so lambda_pop rises.
        model = _make_model()
        opt = SNRAdamW(
            model.parameters(), lr=1e-2, gate="snr", lambda_pop=1.0,
            adaptive_threshold=AdaptiveThresholdConfig(
                mode="target_mean_gate", target_mean_gate=0.01,
                warmup_steps=5, update_interval=5, adaptation_lr=0.2, tolerance=0.0,
            ),
        )
        for i in range(150):
            _train_step(model, opt, seed=i)
        assert opt.param_groups[0]["lambda_pop"] > 1.0


# ---------------------------------------------------------------------------
# 2. Active-fraction formula
# ---------------------------------------------------------------------------

class TestActiveFractionFormula:
    @pytest.mark.parametrize("p", [0.1, 0.2, 0.5])
    def test_formula_hits_target_fraction(self, p):
        torch.manual_seed(0)
        r = torch.distributions.Exponential(1.0).sample((200_000,))
        q0 = 0.5
        alpha = 1.0
        lam = lambda_for_target_active_fraction(
            r, target_active_fraction=p, active_gate_threshold=q0,
            alpha=alpha, min_lambda=1e-8, max_lambda=1e8,
        )
        q = r / (r + alpha * lam)
        frac = (q >= q0).float().mean().item()
        assert abs(frac - p) < 0.01

    def test_integration_active_fraction_tracks(self):
        model = _make_model(in_dim=128, out_dim=32)
        opt = SNRAdamW(
            model.parameters(), lr=1e-2, gate="snr",
            adaptive_threshold=AdaptiveThresholdConfig(
                mode="target_active_fraction", target_active_fraction=0.3,
                active_gate_threshold=0.5, warmup_steps=5, update_interval=5,
                tolerance=0.0, beta=0.5,
            ),
        )
        for i in range(200):
            _train_step(model, opt, seed=i)
        ema = opt.get_threshold_state()["group_0"]["ema_active_fraction"]
        assert ema is not None
        assert abs(ema - 0.3) < 0.2

    def _active_fraction_group(self, adapt, *, lambda_pop=1.0, alpha=1.0):
        cfg = AdaptiveThresholdConfig(
            mode="target_active_fraction", adapt=adapt,
            target_active_fraction=0.2, active_gate_threshold=0.5,
            warmup_steps=0, update_interval=1, tolerance=0.0, beta=0.0,
            max_log_change=100.0,
        )
        group = _fake_group(cfg, lambda_pop=lambda_pop, alpha=alpha, gate="snr")
        torch.manual_seed(0)
        r = torch.distributions.Exponential(1.0).sample((50_000,))
        obs = AdaptiveObservation(active_fraction=0.9, r_samples=r)
        return cfg, group, obs

    def test_snr_adapt_lambda_pop_only(self):
        cfg, group, obs = self._active_fraction_group("lambda_pop", lambda_pop=1.0, alpha=2.0)
        apply_adaptive_update(group, cfg, obs, alpha_value=2.0)
        assert group["lambda_pop"] != 1.0
        assert group["alpha"] == 2.0  # alpha left untouched

    def test_snr_adapt_alpha_only(self):
        cfg, group, obs = self._active_fraction_group("alpha", lambda_pop=3.0, alpha=1.0)
        apply_adaptive_update(group, cfg, obs, alpha_value=1.0)
        assert group["alpha"] != 1.0
        assert group["lambda_pop"] == 3.0  # lambda_pop left untouched

    def test_snr_adapt_both_hits_product(self):
        cfg, group, obs = self._active_fraction_group("both", lambda_pop=1.0, alpha=1.0)
        # Recover the controller's target product (alpha * lambda) for this r sample.
        q0 = cfg.active_gate_threshold
        r_threshold = torch.quantile(obs.r_samples.float(), 1.0 - cfg.target_active_fraction).item()
        target_scale = r_threshold * (1.0 - q0) / q0
        apply_adaptive_update(group, cfg, obs, alpha_value=1.0)
        assert group["alpha"] != 1.0
        assert group["lambda_pop"] != 1.0
        product = group["alpha"] * group["lambda_pop"]
        assert product == pytest.approx(target_scale, rel=1e-6)

    def test_soft_gate_respects_pinned_alpha(self):
        # soft/hard control the active boundary through alpha; adapt="lambda_pop"
        # must not silently mutate alpha (or lambda_pop, which has no effect here).
        cfg = AdaptiveThresholdConfig(
            mode="target_active_fraction", adapt="lambda_pop",
            target_active_fraction=0.2, warmup_steps=0, update_interval=1,
            tolerance=0.0, beta=0.0,
        )
        group = _fake_group(cfg, lambda_pop=1.0, alpha=1.0, gate="soft")
        torch.manual_seed(0)
        r = torch.distributions.Exponential(1.0).sample((50_000,))
        apply_adaptive_update(group, cfg, AdaptiveObservation(active_fraction=0.9, r_samples=r), 1.0)
        assert group["alpha"] == 1.0
        assert group["lambda_pop"] == 1.0

    def test_soft_gate_adapts_alpha(self):
        cfg = AdaptiveThresholdConfig(
            mode="target_active_fraction", adapt="alpha",
            target_active_fraction=0.2, warmup_steps=0, update_interval=1,
            tolerance=0.0, beta=0.0, max_log_change=100.0,
        )
        group = _fake_group(cfg, lambda_pop=1.0, alpha=1.0, gate="soft")
        torch.manual_seed(0)
        r = torch.distributions.Exponential(1.0).sample((50_000,))
        r_threshold = torch.quantile(r.float(), 0.8).item()
        apply_adaptive_update(group, cfg, AdaptiveObservation(active_fraction=0.9, r_samples=r), 1.0)
        assert group["alpha"] == pytest.approx(r_threshold, rel=1e-6)
        assert group["lambda_pop"] == 1.0


# ---------------------------------------------------------------------------
# shock_then_sparsify mode
# ---------------------------------------------------------------------------

class TestShockThenSparsify:
    def _cfg(self, **kw):
        base = dict(
            mode="shock_then_sparsify", sparse_target_active_fraction=0.05,
            shock_target_active_fraction=0.3, shock_steps=10, shift_detect_threshold=0.1,
            shock_fast_beta=0.0, beta=0.9, warmup_steps=0, update_interval=1, tolerance=0.0,
        )
        base.update(kw)
        return AdaptiveThresholdConfig(**base)

    def _r(self):
        torch.manual_seed(0)
        return torch.distributions.Exponential(1.0).sample((50_000,))

    def test_steady_state_holds_sparse_target(self):
        cfg = self._cfg()
        group = _fake_group(cfg, gate="snr")
        r = self._r()
        for _ in range(30):
            apply_adaptive_update(
                group, cfg,
                AdaptiveObservation(mean_gate=0.05, active_fraction=0.05, r_samples=r), 1.0,
            )
        st = group["_adaptive_state"]
        assert st["shock_countdown"] == 0
        assert st["current_target_active_fraction"] == pytest.approx(0.05)

    def test_shift_raises_target_then_decays(self):
        cfg = self._cfg()
        group = _fake_group(cfg, gate="snr")
        r = self._r()
        # Settle the slow EMA at a low gate.
        for _ in range(20):
            apply_adaptive_update(
                group, cfg,
                AdaptiveObservation(mean_gate=0.05, active_fraction=0.05, r_samples=r), 1.0,
            )
        # A regime shift surfaces as a sudden jump in the realized mean gate.
        apply_adaptive_update(
            group, cfg,
            AdaptiveObservation(mean_gate=0.6, active_fraction=0.6, r_samples=r), 1.0,
        )
        st = group["_adaptive_state"]
        assert st["shock_countdown"] == cfg.shock_steps - 1
        assert st["current_target_active_fraction"] > 0.05

        # Decay back toward sparse as the window elapses.
        targets = []
        for _ in range(cfg.shock_steps + 2):
            apply_adaptive_update(
                group, cfg,
                AdaptiveObservation(mean_gate=0.05, active_fraction=0.05, r_samples=r), 1.0,
            )
            targets.append(group["_adaptive_state"]["current_target_active_fraction"])
        assert all(targets[i] >= targets[i + 1] - 1e-9 for i in range(len(targets) - 1))
        assert targets[-1] == pytest.approx(0.05)
        assert group["_adaptive_state"]["shock_countdown"] == 0

    def test_integration_recovers_after_regime_shift(self):
        torch.manual_seed(0)
        model = nn.Linear(200, 1, bias=False)
        opt = SNRAdamW(
            model.parameters(), lr=5e-2, gate="snr", lambda_pop=1.0,
            adaptive_threshold=AdaptiveThresholdConfig(
                mode="shock_then_sparsify", sparse_target_active_fraction=0.05,
                shock_target_active_fraction=0.25, shock_steps=20,
                shift_detect_threshold=0.04, warmup_steps=20, update_interval=5,
                tolerance=0.0,
            ),
        )
        peak_target = 0.0
        sigma = 1.0
        for step in range(600):
            if step == 300:
                sigma = 5.0
            x = torch.randn(64, 200)
            y = x[:, :5].sum(1) + sigma * torch.randn(64)
            loss = ((model(x).squeeze(-1) - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if 300 <= step <= 420:
                t = opt.get_threshold_state()["group_0"]["target_active_fraction"]
                if t is not None:
                    peak_target = max(peak_target, t)
        # The shock raised the target above the sparse baseline...
        assert peak_target > 0.05
        # ...and it has decayed back to sparse by the end.
        assert opt.get_threshold_state()["group_0"]["target_active_fraction"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 3. Clamp behaviour
# ---------------------------------------------------------------------------

class TestClamps:
    def test_smooth_clamped_update_respects_bounds(self):
        out = smooth_clamped_update(
            old=1.0, proposed=1e9, beta=0.0,
            min_value=1e-3, max_value=10.0, max_log_change=100.0,
        )
        assert out == 10.0
        out = smooth_clamped_update(
            old=1.0, proposed=1e-9, beta=0.0,
            min_value=1e-3, max_value=10.0, max_log_change=100.0,
        )
        assert out == 1e-3

    def test_max_log_change_caps_movement(self):
        out = smooth_clamped_update(
            old=1.0, proposed=1e6, beta=0.0,
            min_value=1e-8, max_value=1e8, max_log_change=0.25,
        )
        assert out == pytest.approx(math.exp(0.25), rel=1e-6)

    def test_lambda_clamped_in_formula(self):
        r = torch.full((1000,), 1000.0)
        lam = lambda_for_target_active_fraction(
            r, target_active_fraction=0.2, active_gate_threshold=0.5,
            alpha=1.0, min_lambda=1e-4, max_lambda=5.0,
        )
        assert lam == 5.0

    def test_mean_gate_lambda_clamped(self):
        cfg = AdaptiveThresholdConfig(
            mode="target_mean_gate", target_mean_gate=0.0,
            adaptation_lr=10.0, tolerance=0.0, beta=0.0, warmup_steps=0,
            min_lambda_pop=1e-3, max_lambda_pop=2.0, max_log_change=100.0,
        )
        group = _fake_group(cfg, lambda_pop=1.0)
        for _ in range(50):
            apply_adaptive_update(group, cfg, AdaptiveObservation(mean_gate=1.0), alpha_value=1.0)
        assert group["lambda_pop"] <= 2.0
        assert group["lambda_pop"] >= 1e-3


# ---------------------------------------------------------------------------
# 4 + 5. Warmup and update-interval behaviour
# ---------------------------------------------------------------------------

class TestSchedule:
    def test_no_change_before_warmup(self):
        model = _make_model()
        opt = SNRAdamW(
            model.parameters(), lr=1e-2, gate="snr", lambda_pop=1.0,
            adaptive_threshold=AdaptiveThresholdConfig(
                mode="target_mean_gate", target_mean_gate=0.9,
                warmup_steps=50, update_interval=1, tolerance=0.0,
            ),
        )
        for i in range(40):
            _train_step(model, opt, seed=i)
        assert opt.param_groups[0]["lambda_pop"] == 1.0

    def test_changes_only_on_interval(self):
        model = _make_model()
        opt = SNRAdamW(
            model.parameters(), lr=1e-2, gate="snr", lambda_pop=1.0,
            adaptive_threshold=AdaptiveThresholdConfig(
                mode="target_mean_gate", target_mean_gate=0.9,
                warmup_steps=0, update_interval=10, tolerance=0.0, adaptation_lr=0.3,
            ),
        )
        seen = []
        for i in range(35):
            _train_step(model, opt, seed=i)
            seen.append(opt.param_groups[0]["lambda_pop"])
        # Value should only change right after steps 10, 20, 30.
        changes = [i for i in range(1, len(seen)) if seen[i] != seen[i - 1]]
        assert changes == [9, 19, 29]


# ---------------------------------------------------------------------------
# 6. State dict roundtrip
# ---------------------------------------------------------------------------

class TestStateDict:
    def test_roundtrip_restores_and_continues(self):
        torch.manual_seed(0)
        model = _make_model()
        cfg = dict(
            mode="target_active_fraction", target_active_fraction=0.2,
            warmup_steps=5, update_interval=5, tolerance=0.0,
        )
        opt = SNRAdamW(model.parameters(), lr=1e-2, gate="snr", adaptive_threshold=cfg)
        for i in range(40):
            _train_step(model, opt, seed=i)

        sd = copy.deepcopy(opt.state_dict())
        w_snapshot = model.weight.data.clone()

        for i in range(40, 50):
            _train_step(model, opt, seed=i)
        w_continued = model.weight.data.clone()
        state_continued = opt.get_threshold_state()

        # Fresh optimizer + restore.
        model.weight.data.copy_(w_snapshot)
        opt2 = SNRAdamW(model.parameters(), lr=1e-2, gate="snr", adaptive_threshold=cfg)
        opt2.load_state_dict(sd)
        restored = opt2.get_threshold_state()["group_0"]
        snapshot = opt.get_threshold_state()  # opt has moved on; use sd-derived values instead

        # lambda_pop / alpha / EMA / step restored from checkpoint.
        assert restored["lambda_pop"] == pytest.approx(sd["param_groups"][0]["lambda_pop"])
        assert restored["step"] == sd["param_groups"][0]["_adaptive_state"]["step"]
        assert restored["ema_active_fraction"] == pytest.approx(
            sd["param_groups"][0]["_adaptive_state"]["ema_active_fraction"]
        )

        for i in range(40, 50):
            _train_step(model, opt2, seed=i)
        w_restored = model.weight.data.clone()

        assert torch.allclose(w_continued, w_restored, atol=1e-5)
        assert opt2.get_threshold_state()["group_0"]["lambda_pop"] == pytest.approx(
            state_continued["group_0"]["lambda_pop"]
        )

    def test_state_dict_contains_adaptive_keys(self):
        model = _make_model()
        opt = SNRAdamW(
            model.parameters(), lr=1e-2,
            adaptive_threshold={"mode": "target_mean_gate", "warmup_steps": 0, "update_interval": 1},
        )
        _train_step(model, opt, seed=0)
        sd = opt.state_dict()
        assert "_adaptive_state" in sd["param_groups"][0]
        assert "base_lambda_pop" in sd["param_groups"][0]


# ---------------------------------------------------------------------------
# 7. Per-param-group independence
# ---------------------------------------------------------------------------

class TestParamGroupIndependence:
    def test_groups_diverge(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(64, 64, bias=False), nn.Linear(64, 16, bias=False))
        opt = SNRAdamW(
            [
                {"params": model[0].parameters(),
                 "adaptive_threshold": {"mode": "target_active_fraction",
                                        "target_active_fraction": 0.1,
                                        "warmup_steps": 5, "update_interval": 5, "tolerance": 0.0}},
                {"params": model[1].parameters(),
                 "adaptive_threshold": {"mode": "target_active_fraction",
                                        "target_active_fraction": 0.6,
                                        "warmup_steps": 5, "update_interval": 5, "tolerance": 0.0}},
            ],
            lr=1e-2, gate="snr",
        )
        for i in range(150):
            x = torch.randn(16, 64)
            y = torch.randn(16, 16)
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        st = opt.get_threshold_state()
        # Higher target active fraction => lower lambda_pop (more passes through).
        assert st["group_1"]["lambda_pop"] < st["group_0"]["lambda_pop"]

    def test_off_group_untouched(self):
        model = nn.Sequential(nn.Linear(64, 64, bias=False), nn.Linear(64, 16, bias=False))
        opt = SNRAdamW(
            [
                {"params": model[0].parameters(), "adaptive_threshold": {"mode": "off"}},
                {"params": model[1].parameters(),
                 "adaptive_threshold": {"mode": "target_mean_gate", "warmup_steps": 0,
                                        "update_interval": 5, "target_mean_gate": 0.9,
                                        "tolerance": 0.0, "adaptation_lr": 0.3}},
            ],
            lr=1e-2, gate="snr", lambda_pop=1.0,
        )
        for i in range(60):
            x = torch.randn(16, 64)
            y = torch.randn(16, 16)
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        assert opt.param_groups[0]["lambda_pop"] == 1.0
        st = opt.get_threshold_state()
        assert "group_0" not in st
        assert "group_1" in st


# ---------------------------------------------------------------------------
# MARS optimizer parity
# ---------------------------------------------------------------------------

class TestMARSAdaptive:
    def test_mars_mean_gate_adapts(self):
        model = _make_model(in_dim=128, out_dim=32)
        opt = MARSSNRAdamW(
            model.parameters(), lr=1e-2, gate="snr", lambda_pop=1.0,
            adaptive_threshold={"mode": "target_mean_gate", "target_mean_gate": 0.95,
                                "warmup_steps": 5, "update_interval": 5,
                                "tolerance": 0.0, "adaptation_lr": 0.2},
        )
        for i in range(150):
            _train_step(model, opt, seed=i)
        ema = opt.get_threshold_state()["group_0"]["ema_mean_gate"]
        assert ema is not None
        # Unreachable-high target => lambda driven down.
        assert opt.param_groups[0]["lambda_pop"] < 1.0

    def test_mars_active_fraction_adapts(self):
        model = _make_model(in_dim=128, out_dim=32)
        opt = MARSSNRAdamW(
            model.parameters(), lr=1e-2, gate="snr",
            adaptive_threshold={"mode": "target_active_fraction", "target_active_fraction": 0.3,
                                "warmup_steps": 5, "update_interval": 5, "tolerance": 0.0,
                                "beta": 0.5},
        )
        for i in range(200):
            _train_step(model, opt, seed=i)
        ema = opt.get_threshold_state()["group_0"]["ema_active_fraction"]
        assert ema is not None
        assert abs(ema - 0.3) < 0.2


# ---------------------------------------------------------------------------
# Default behaviour unchanged when adaptive is off
# ---------------------------------------------------------------------------

class TestDefaultsUnchanged:
    def test_no_adaptive_state_when_disabled(self):
        model = _make_model()
        opt = SNRAdamW(model.parameters(), lr=1e-2)
        _train_step(model, opt, seed=0)
        assert "_adaptive_state" not in opt.param_groups[0]
        assert opt.get_threshold_state() == {}
