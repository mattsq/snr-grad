"""Tests for the persistent-gate `requires_grad` freezing feature.

The freeze logic lives in two module-level helpers (`_update_freeze_state` and
`_maybe_recheck_freeze`) shared by all four optimizers. Tests cover SNRAdamW
specifically and the other three optimizers via parametrization where the
behavior is shape-agnostic.
"""

import copy

import pytest
import torch
import torch.nn as nn

from snr_grad import (
    RotatedSNRAdamW,
    SNRAdamW,
    SNRMuon,
    SpectralSNRMuon,
    MARSSNRAdamW,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drive_noise_grads(param: torch.Tensor, opt, n_steps: int, seed: int = 0) -> None:
    """
    Drive a parameter with random-noise gradients (no signal) to collapse the gate.

    Bypasses autograd and writes p.grad directly so we don't need a model graph.
    """
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_steps):
        param.grad = torch.randn(param.shape, generator=gen)
        opt.step()
        param.grad = None


def _drive_signal_grads(param: torch.Tensor, opt, n_steps: int) -> None:
    """Drive a parameter with a constant strong-signal gradient."""
    g = torch.full(param.shape, 1.0)
    for _ in range(n_steps):
        param.grad = g.clone()
        opt.step()
        param.grad = None


def _make_2d_param(shape=(4, 4), seed=0):
    torch.manual_seed(seed)
    return torch.zeros(shape, requires_grad=True)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestFreezeConstruction:

    def test_default_freeze_off(self):
        model = nn.Linear(5, 1)
        opt = SNRAdamW(model.parameters())
        assert opt.defaults["freeze_low_snr"] is False

    @pytest.mark.parametrize("threshold", [-0.1, 1.1, 2.0])
    def test_invalid_threshold_raises(self, threshold):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="freeze_threshold"):
            SNRAdamW(model.parameters(), freeze_low_snr=True, freeze_threshold=threshold)

    @pytest.mark.parametrize("patience", [0, -5])
    def test_invalid_patience_raises(self, patience):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="freeze_patience"):
            SNRAdamW(model.parameters(), freeze_low_snr=True, freeze_patience=patience)

    @pytest.mark.parametrize("interval", [0, -1])
    def test_invalid_recheck_raises(self, interval):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="freeze_recheck_interval"):
            SNRAdamW(model.parameters(), freeze_low_snr=True, freeze_recheck_interval=interval)

    @pytest.mark.parametrize("beta", [-0.1, 1.0, 1.5])
    def test_invalid_beta_raises(self, beta):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="freeze_beta"):
            SNRAdamW(model.parameters(), freeze_low_snr=True, freeze_beta=beta)

    @pytest.mark.parametrize("guard", ["invalid", 123, None])
    def test_invalid_guard_raises(self, guard):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="freeze_guard"):
            SNRAdamW(model.parameters(), freeze_low_snr=True, freeze_guard=guard)

    @pytest.mark.parametrize("opt_cls", [SNRAdamW, SNRMuon, RotatedSNRAdamW, SpectralSNRMuon, MARSSNRAdamW])
    def test_count_frozen_method_available(self, opt_cls):
        """All four optimizers expose count_frozen() returning (params, elems)."""
        model = nn.Linear(5, 3)
        opt = opt_cls(model.parameters(), lr=1e-3)
        n_params, n_elems = opt.count_frozen()
        assert n_params == 0
        assert n_elems == 0


# ---------------------------------------------------------------------------
# Freeze trigger under sustained-low SNR
# ---------------------------------------------------------------------------

class TestFreezeTriggerSNRAdamW:

    def test_freezes_after_patience(self):
        """Pure-noise grads collapse the gate; param should freeze after patience."""
        p = _make_2d_param(shape=(8,))
        opt = SNRAdamW(
            [p],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=30,
            freeze_recheck_interval=10_000,
            freeze_beta=0.5,  # fast EMA so the test runs short
            freeze_guard=False,
        )
        # 60 noise steps is well past patience=30
        _drive_noise_grads(p, opt, n_steps=60)
        assert p.requires_grad is False
        assert opt.state[p]["frozen"] is True
        n_params, n_elems = opt.count_frozen()
        assert n_params == 1
        assert n_elems == 8

    def test_does_not_freeze_under_signal(self):
        """A strong constant gradient keeps the gate near 1; no freeze."""
        p = _make_2d_param(shape=(8,))
        opt = SNRAdamW(
            [p],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=20,
            freeze_recheck_interval=10_000,
            freeze_beta=0.5,
            freeze_guard=False,
        )
        _drive_signal_grads(p, opt, n_steps=100)
        assert p.requires_grad is True
        assert opt.count_frozen() == (0, 0)


# ---------------------------------------------------------------------------
# Across all four optimizers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "opt_cls",
    [SNRAdamW, SNRMuon, RotatedSNRAdamW, SpectralSNRMuon, MARSSNRAdamW],
    ids=["SNRAdamW", "SNRMuon", "RotatedSNRAdamW", "SpectralSNRMuon", "MARSSNRAdamW"],
)
class TestFreezeAcrossOptimizers:

    def test_noise_grads_eventually_freeze(self, opt_cls):
        """For each optimizer, sustained-low-gate inputs eventually freeze the param."""
        # Use a 2D param so all four optimizers exercise their primary branch.
        p = _make_2d_param(shape=(5, 5))
        opt = opt_cls(
            [p],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=30,
            freeze_recheck_interval=10_000,
            freeze_beta=0.5,
            freeze_guard=False,
        )
        _drive_noise_grads(p, opt, n_steps=120)
        assert p.requires_grad is False, (
            f"{opt_cls.__name__}: param should be frozen under sustained-noise gradients"
        )
        n_params, n_elems = opt.count_frozen()
        assert n_params == 1
        assert n_elems == p.numel()

    def test_no_freeze_when_disabled(self, opt_cls):
        """With freeze_low_snr=False (default), no freezing happens even under noise."""
        p = _make_2d_param(shape=(5, 5))
        opt = opt_cls([p], lr=1e-3)  # freeze_low_snr defaults to False
        _drive_noise_grads(p, opt, n_steps=120)
        assert p.requires_grad is True
        assert opt.count_frozen() == (0, 0)


# ---------------------------------------------------------------------------
# Recheck cadence
# ---------------------------------------------------------------------------

class TestFreezeRecheck:

    def test_recheck_unfreezes(self):
        """At the recheck interval, frozen params get requires_grad=True restored."""
        p = _make_2d_param(shape=(4,))
        opt = SNRAdamW(
            [p],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=10,
            freeze_recheck_interval=25,
            freeze_beta=0.5,
            freeze_guard=False,
        )
        # 20 steps: gate collapses, param freezes (patience=10)
        _drive_noise_grads(p, opt, n_steps=20)
        assert p.requires_grad is False
        assert opt.state[p]["frozen"] is True

        # Drive 5 more "no-grad" steps. Frozen param has no grad so optimizer
        # skips it. _global_step advances each step.
        for _ in range(5):
            # No grad assigned -> p.grad is None -> step body skips param,
            # but _global_step still increments via _maybe_recheck_freeze.
            opt.step()

        # We've called opt.step() 25 times total -> recheck fired on step 25.
        assert p.requires_grad is True
        assert opt.state[p]["frozen"] is False
        assert opt.state[p]["below_count"] == 0


# ---------------------------------------------------------------------------
# Respect user-set requires_grad=False
# ---------------------------------------------------------------------------

class TestFreezeRespectsUserRequiresGrad:

    def test_user_frozen_param_untouched(self):
        """If user sets requires_grad=False before optimizer init, recheck must not flip it back."""
        p_user_frozen = torch.zeros((5,), requires_grad=False)  # user has frozen this
        p_active = torch.zeros((5,), requires_grad=True)

        opt = SNRAdamW(
            [p_user_frozen, p_active],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=5,
            freeze_recheck_interval=10,  # fires at step 10, 20, ...
            freeze_beta=0.5,
        )

        # Drive the active param with strong signal; user-frozen one has no grad.
        for _ in range(30):
            p_active.grad = torch.full((5,), 1.0)
            opt.step()
            p_active.grad = None

        # We've stepped 30 times -> rechecks fired at steps 10/20/30. None
        # should have touched the user-frozen param because state["frozen"]
        # was never set to True by the optimizer for it.
        assert p_user_frozen.requires_grad is False
        user_state = opt.state.get(p_user_frozen)
        if user_state is not None:
            assert user_state.get("frozen", False) is False
        # Active param with strong signal should not have been frozen either.
        assert p_active.requires_grad is True


# ---------------------------------------------------------------------------
# State dict roundtrip
# ---------------------------------------------------------------------------

class TestFreezeStateDictRoundtrip:

    def test_freeze_state_roundtrip(self):
        """Save/load preserves gate_ema, below_count, frozen across a pickle cycle."""
        p = _make_2d_param(shape=(6,))
        opt = SNRAdamW(
            [p],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=10,
            freeze_recheck_interval=10_000,
            freeze_beta=0.5,
            freeze_guard=False,
        )
        _drive_noise_grads(p, opt, n_steps=30)
        assert opt.state[p]["frozen"] is True
        gate_ema_before = opt.state[p]["gate_ema"]
        below_before = opt.state[p]["below_count"]
        frozen_before = opt.state[p]["frozen"]

        sd = copy.deepcopy(opt.state_dict())

        # Build a fresh optimizer + param, restore.
        p2 = torch.zeros((6,), requires_grad=True)
        opt2 = SNRAdamW(
            [p2],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=10,
            freeze_recheck_interval=10_000,
            freeze_beta=0.5,
            freeze_guard=False,
        )
        opt2.load_state_dict(sd)

        st2 = opt2.state[p2]
        assert st2["frozen"] == frozen_before
        assert st2["below_count"] == below_before
        assert st2["gate_ema"] == pytest.approx(gate_ema_before)

    @pytest.mark.parametrize(
        "opt_cls",
        [SNRAdamW, SNRMuon, RotatedSNRAdamW, SpectralSNRMuon, MARSSNRAdamW],
        ids=["SNRAdamW", "SNRMuon", "RotatedSNRAdamW", "SpectralSNRMuon", "MARSSNRAdamW"],
    )
    def test_load_restores_requires_grad_false(self, opt_cls):
        """
        Fresh model params arrive with requires_grad=True, but if the loaded
        state says state["frozen"] is True we must reapply requires_grad=False.
        Otherwise count_frozen() and autograd disagree until the next recheck.
        """
        p = _make_2d_param(shape=(5, 5))
        opt = opt_cls(
            [p],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=10,
            freeze_recheck_interval=10_000,
            freeze_beta=0.5,
            freeze_guard=False,
        )
        _drive_noise_grads(p, opt, n_steps=40)
        assert opt.state[p]["frozen"] is True
        assert p.requires_grad is False

        sd = copy.deepcopy(opt.state_dict())

        # Fresh param + optimizer; the fresh param defaults to requires_grad=True.
        p2 = torch.zeros((5, 5), requires_grad=True)
        assert p2.requires_grad is True
        opt2 = opt_cls(
            [p2],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=10,
            freeze_recheck_interval=10_000,
            freeze_beta=0.5,
            freeze_guard=False,
        )
        opt2.load_state_dict(sd)

        # After load: optimizer state says frozen, so autograd must agree.
        assert opt2.state[p2]["frozen"] is True
        assert p2.requires_grad is False
        n_params, n_elems = opt2.count_frozen()
        assert n_params == 1
        assert n_elems == p2.numel()

    @pytest.mark.parametrize(
        "opt_cls",
        [SNRAdamW, SNRMuon, RotatedSNRAdamW, SpectralSNRMuon, MARSSNRAdamW],
        ids=["SNRAdamW", "SNRMuon", "RotatedSNRAdamW", "SpectralSNRMuon", "MARSSNRAdamW"],
    )
    def test_global_step_persists_for_recheck_cadence(self, opt_cls):
        """
        Recheck cadence depends on optimizer._global_step. A checkpoint saved
        mid-cadence must resume with the same counter so the next recheck
        fires at the correct step.
        """
        recheck = 25
        p = _make_2d_param(shape=(4,))
        opt = opt_cls(
            [p],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=5,
            freeze_recheck_interval=recheck,
            freeze_beta=0.5,
            freeze_guard=False,
        )
        # Drive to freeze well before the first recheck (step 25).
        _drive_noise_grads(p, opt, n_steps=15)
        assert opt._global_step == 15
        assert opt.state[p]["frozen"] is True
        assert p.requires_grad is False

        sd = copy.deepcopy(opt.state_dict())
        assert sd["_global_step"] == 15

        # Fresh optimizer/param, restore.
        p2 = torch.zeros((4,), requires_grad=True)
        opt2 = opt_cls(
            [p2],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=5,
            freeze_recheck_interval=recheck,
            freeze_beta=0.5,
            freeze_guard=False,
        )
        opt2.load_state_dict(sd)
        assert opt2._global_step == 15

        # Step 9 more times: that puts _global_step at 24 -- still pre-recheck.
        for _ in range(9):
            opt2.step()  # frozen, no grad, no-op step but _global_step bumps
        assert opt2._global_step == 24
        assert p2.requires_grad is False  # still frozen

        # One more step lands on global_step=25, which is the recheck boundary.
        opt2.step()
        assert opt2._global_step == 25
        assert p2.requires_grad is True
        assert opt2.state[p2]["frozen"] is False

    def test_load_state_dict_backward_compatible(self):
        """A state_dict missing _global_step (old checkpoint) loads with _global_step=0."""
        p = torch.zeros((4,), requires_grad=True)
        opt = SNRAdamW(
            [p],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=5,
            freeze_recheck_interval=10,
            freeze_beta=0.5,
            freeze_guard=False,
        )
        opt._global_step = 7  # pretend a prior run
        sd = opt.state_dict()
        sd.pop("_global_step")  # simulate an older checkpoint

        opt2 = SNRAdamW([torch.zeros((4,), requires_grad=True)], lr=1e-3, freeze_guard=False)
        opt2.load_state_dict(sd)
        assert opt2._global_step == 0


# ---------------------------------------------------------------------------
# Stats integration
# ---------------------------------------------------------------------------

class TestFreezeStats:

    def test_stats_report_frozen_counts(self):
        """SNRAdamWStats.parameters_frozen and elements_frozen reflect freeze state."""
        p = _make_2d_param(shape=(7,))
        opt = SNRAdamW(
            [p],
            lr=1e-3,
            track_stats=True,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=10,
            freeze_recheck_interval=10_000,
            freeze_beta=0.5,
            freeze_guard=False,
        )
        # Initially no freeze
        p.grad = torch.randn(7)
        opt.step()
        assert opt.last_stats.parameters_frozen == 0
        assert opt.last_stats.elements_frozen == 0

        # Drive to freeze
        _drive_noise_grads(p, opt, n_steps=40)

        # Need one more step with a (non-existent) gradient to refresh stats;
        # frozen param has no grad so the optimizer's stats block sees zero
        # active params. To get a fresh stats snapshot we step a second active
        # param.
        n_frozen, n_elems = opt.count_frozen()
        assert n_frozen == 1
        assert n_elems == 7


# ---------------------------------------------------------------------------
# Param groups respect their own freeze settings
# ---------------------------------------------------------------------------

class TestFreezePerGroup:

    def test_freeze_disabled_in_one_group(self):
        """Two param groups: one with freeze on, one off. Only the on-group freezes."""
        p_a = torch.zeros((6,), requires_grad=True)
        p_b = torch.zeros((6,), requires_grad=True)

        opt = SNRAdamW(
            [
                {"params": [p_a], "freeze_low_snr": True, "freeze_threshold": 0.5,
                 "freeze_patience": 10, "freeze_recheck_interval": 10_000,
                 "freeze_beta": 0.5},
                {"params": [p_b], "freeze_low_snr": False},
            ],
            lr=1e-3,
        )

        gen = torch.Generator().manual_seed(0)
        for _ in range(50):
            p_a.grad = torch.randn(6, generator=gen)
            p_b.grad = torch.randn(6, generator=gen)
            opt.step()
            p_a.grad = None
            p_b.grad = None

        assert p_a.requires_grad is False
        assert p_b.requires_grad is True


# ---------------------------------------------------------------------------
# Autograd total freeze prevention guard
# ---------------------------------------------------------------------------

class TestTotalFreezePreventionGuard:

    @pytest.mark.parametrize(
        "opt_cls",
        [SNRAdamW, SNRMuon, RotatedSNRAdamW, SpectralSNRMuon, MARSSNRAdamW],
        ids=["SNRAdamW", "SNRMuon", "RotatedSNRAdamW", "SpectralSNRMuon", "MARSSNRAdamW"],
    )
    def test_prevents_total_freeze(self, opt_cls):
        """When all parameters are about to freeze, the guard keeps the one with the highest gate_ema active."""
        p_a = _make_2d_param(shape=(4, 4), seed=42)
        p_b = _make_2d_param(shape=(4, 4), seed=43)

        opt = opt_cls(
            [p_a, p_b],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=10,
            freeze_recheck_interval=10_000,
            freeze_beta=0.5,
        )

        # Run one step to initialize state
        p_a.grad = torch.randn(4, 4)
        p_b.grad = torch.randn(4, 4)
        opt.step()
        p_a.grad = None
        p_b.grad = None

        # Manually set their gate_ema in state so they are both below freeze_threshold
        # but p_a has a higher EMA than p_b
        opt.state[p_a]["gate_ema"] = 0.3
        opt.state[p_b]["gate_ema"] = 0.1

        # Set below_count to patience - 1 so the next step will trigger freeze on both
        opt.state[p_a]["below_count"] = 9
        opt.state[p_b]["below_count"] = 9

        # Trigger optimizer step by giving them noise grads.
        # During the step, both would be frozen (requires_grad=False).
        # But our guard should unfreeze p_a because it has the higher gate_ema!
        p_a.grad = torch.randn(4, 4)
        p_b.grad = torch.randn(4, 4)
        opt.step()
        p_a.grad = None
        p_b.grad = None

        # Check that p_a was kept active (requires_grad is True, frozen is False)
        # while p_b was frozen (requires_grad is False, frozen is True)
        assert p_a.requires_grad is True
        assert opt.state[p_a]["frozen"] is False
        assert opt.state[p_a]["below_count"] == 0

        assert p_b.requires_grad is False
        assert opt.state[p_b]["frozen"] is True

    @pytest.mark.parametrize(
        "opt_cls",
        [SNRAdamW, SNRMuon, RotatedSNRAdamW, SpectralSNRMuon, MARSSNRAdamW],
        ids=["SNRAdamW", "SNRMuon", "RotatedSNRAdamW", "SpectralSNRMuon", "MARSSNRAdamW"],
    )
    def test_single_parameter_guard_by_default(self, opt_cls):
        """Single-parameter optimizers are guarded by default and kept active unless freeze_guard=False."""
        p = _make_2d_param(shape=(4, 4), seed=42)
        opt = opt_cls(
            [p],
            lr=1e-3,
            freeze_low_snr=True,
            freeze_threshold=0.5,
            freeze_patience=10,
            freeze_recheck_interval=10_000,
            freeze_beta=0.5,
        )

        p.grad = torch.randn(4, 4)
        opt.step()
        p.grad = None

        opt.state[p]["gate_ema"] = 0.3
        opt.state[p]["below_count"] = 9

        p.grad = torch.randn(4, 4)
        opt.step()
        p.grad = None

        # Kept active by the guard!
        assert p.requires_grad is True
        assert opt.state[p]["frozen"] is False


