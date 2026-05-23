"""Tests for the ScheduleFree variants of the SNR optimizers."""

import copy

import pytest
import torch
import torch.nn as nn

from snr_grad import (
    SNRScheduleFreeAdamW,
    SNRScheduleFreeMuon,
    RotatedSNRScheduleFreeAdamW,
    SpectralSNRScheduleFreeMuon,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear(dim=8, seed=0):
    torch.manual_seed(seed)
    model = nn.Linear(dim, 1, bias=False)
    target = torch.randn(1, dim)
    return model, target


def _train_step(model, target, optimizer):
    x = torch.ones(1, model.in_features)
    loss = ((model(x) - (target @ x.T).squeeze()) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


def _make_2d(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16, bias=False), nn.Linear(16, 4, bias=False))
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    return model, x, y


def _train_step_2d(model, x, y, opt):
    opt.zero_grad(set_to_none=True)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
    return loss.item()


ALL_SF_CLASSES = [
    SNRScheduleFreeAdamW,
    SNRScheduleFreeMuon,
    RotatedSNRScheduleFreeAdamW,
    SpectralSNRScheduleFreeMuon,
]


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestConstruction:

    @pytest.mark.parametrize("cls", ALL_SF_CLASSES)
    def test_default_construction(self, cls):
        model = nn.Linear(5, 1)
        opt = cls(model.parameters())
        assert opt.defaults["sf_beta"] == 0.9
        assert opt.defaults["sf_warmup_steps"] == 0
        assert opt.defaults["sf_lr_power"] == 2.0
        assert opt.defaults["sf_r"] == 0.0
        assert opt.defaults["train_mode"] is True

    @pytest.mark.parametrize("cls", ALL_SF_CLASSES)
    def test_invalid_sf_beta_zero(self, cls):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid sf_beta"):
            cls(model.parameters(), sf_beta=0.0)

    @pytest.mark.parametrize("cls", ALL_SF_CLASSES)
    def test_invalid_sf_beta_one(self, cls):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid sf_beta"):
            cls(model.parameters(), sf_beta=1.0)

    @pytest.mark.parametrize("cls", ALL_SF_CLASSES)
    def test_invalid_warmup_negative(self, cls):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid sf_warmup_steps"):
            cls(model.parameters(), sf_warmup_steps=-1)

    @pytest.mark.parametrize("cls", ALL_SF_CLASSES)
    def test_invalid_grokfast_alpha_negative(self, cls):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid grokfast_alpha"):
            cls(model.parameters(), grokfast_alpha=-0.1)


# ---------------------------------------------------------------------------
# State machine after first step
# ---------------------------------------------------------------------------

class TestFirstStepState:

    def test_z_initialized_to_param(self):
        model, target = _make_linear()
        initial_param = list(model.parameters())[0].data.clone()
        opt = SNRScheduleFreeAdamW(model.parameters(), lr=1e-3)
        _train_step(model, target, opt)
        p = list(model.parameters())[0]
        st = opt.state[p]
        assert "z" in st
        # z was initialized from p.data and then updated; should differ slightly from initial.
        # But the diff should be on the order of lr.
        assert (st["z"] - initial_param).abs().max().item() < 1e-1

    @pytest.mark.parametrize("cls", ALL_SF_CLASSES)
    def test_state_keys_after_step(self, cls):
        model, target = _make_linear()
        opt = cls(model.parameters(), lr=1e-3)
        _train_step(model, target, opt)
        p = list(model.parameters())[0]
        st = opt.state[p]
        assert "z" in st
        assert "step" in st
        assert st["step"] == 1

    @pytest.mark.parametrize("cls", ALL_SF_CLASSES)
    def test_weight_sum_grows_monotonically(self, cls):
        model, target = _make_linear()
        opt = cls(model.parameters(), lr=1e-3)
        weight_sums = []
        for _ in range(5):
            _train_step(model, target, opt)
            weight_sums.append(opt.param_groups[0]["weight_sum"])
        # Strictly increasing.
        assert all(b > a for a, b in zip(weight_sums, weight_sums[1:]))

    @pytest.mark.parametrize("cls", ALL_SF_CLASSES)
    def test_lr_max_tracks_max(self, cls):
        model, target = _make_linear()
        opt = cls(model.parameters(), lr=5e-3)
        _train_step(model, target, opt)
        assert opt.param_groups[0]["lr_max"] == pytest.approx(5e-3)


# ---------------------------------------------------------------------------
# train() / eval() swap
# ---------------------------------------------------------------------------

class TestTrainEvalSwap:

    def test_eval_then_train_roundtrips(self):
        model, target = _make_linear()
        opt = SNRScheduleFreeAdamW(model.parameters(), lr=1e-2)
        for _ in range(5):
            _train_step(model, target, opt)
        p = list(model.parameters())[0]
        y_train = p.data.clone()
        opt.eval()
        opt.train()
        y_after = p.data.clone()
        assert torch.allclose(y_train, y_after, atol=1e-6)

    def test_eval_swaps_to_x_identity(self):
        model, target = _make_linear()
        opt = SNRScheduleFreeAdamW(model.parameters(), lr=1e-2, sf_beta=0.9)
        for _ in range(5):
            _train_step(model, target, opt)
        p = list(model.parameters())[0]
        y = p.data.clone()
        z = opt.state[p]["z"]
        opt.eval()
        x_actual = p.data.clone()
        x_expected = (y - (1 - 0.9) * z) / 0.9
        assert torch.allclose(x_actual, x_expected, atol=1e-6)

    def test_eval_idempotent(self):
        model, target = _make_linear()
        opt = SNRScheduleFreeAdamW(model.parameters(), lr=1e-2)
        for _ in range(3):
            _train_step(model, target, opt)
        opt.eval()
        x_first = list(model.parameters())[0].data.clone()
        opt.eval()  # second call should be no-op
        x_second = list(model.parameters())[0].data.clone()
        assert torch.allclose(x_first, x_second)

    def test_train_idempotent(self):
        model, target = _make_linear()
        opt = SNRScheduleFreeAdamW(model.parameters(), lr=1e-2)
        for _ in range(3):
            _train_step(model, target, opt)
        opt.train()  # already in train mode; should be no-op
        y_first = list(model.parameters())[0].data.clone()
        opt.train()
        y_second = list(model.parameters())[0].data.clone()
        assert torch.allclose(y_first, y_second)

    def test_step_in_eval_mode_raises(self):
        model, target = _make_linear()
        opt = SNRScheduleFreeAdamW(model.parameters(), lr=1e-2)
        _train_step(model, target, opt)
        opt.eval()
        x = torch.ones(1, model.in_features)
        loss = ((model(x) - (target @ x.T).squeeze()) ** 2).mean()
        loss.backward()
        with pytest.raises(RuntimeError, match="eval mode"):
            opt.step()

    @pytest.mark.parametrize("cls", ALL_SF_CLASSES)
    def test_2d_train_eval_roundtrips(self, cls):
        model, x, y = _make_2d()
        opt = cls(model.parameters(), lr=1e-3)
        for _ in range(3):
            _train_step_2d(model, x, y, opt)
        snapshots_y = [p.data.clone() for p in model.parameters()]
        opt.eval()
        opt.train()
        snapshots_after = [p.data.clone() for p in model.parameters()]
        for a, b in zip(snapshots_y, snapshots_after):
            assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------------
# Warmup scaling
# ---------------------------------------------------------------------------

class TestWarmup:

    def test_warmup_scales_lr_max_linearly(self):
        model, target = _make_linear()
        opt = SNRScheduleFreeAdamW(
            model.parameters(), lr=1e-2, sf_warmup_steps=10
        )
        # At step 1, lr_t should be lr * 1/10.
        _train_step(model, target, opt)
        assert opt.param_groups[0]["lr_max"] == pytest.approx(1e-3)
        # By step 10, lr_max should reach full lr.
        for _ in range(9):
            _train_step(model, target, opt)
        assert opt.param_groups[0]["lr_max"] == pytest.approx(1e-2)


# ---------------------------------------------------------------------------
# Loss decreases on a simple fitting task
# ---------------------------------------------------------------------------

class TestConvergence:

    @pytest.mark.parametrize("cls", ALL_SF_CLASSES)
    def test_loss_decreases_2d(self, cls):
        model, x, y = _make_2d()
        opt = cls(model.parameters(), lr=1e-2)
        first_loss = _train_step_2d(model, x, y, opt)
        for _ in range(30):
            last_loss = _train_step_2d(model, x, y, opt)
        assert last_loss < first_loss


# ---------------------------------------------------------------------------
# Grokfast + ScheduleFree combination
# ---------------------------------------------------------------------------

class TestGrokfastCombo:

    def test_grokfast_state_populates_alongside_z(self):
        model, target = _make_linear()
        opt = SNRScheduleFreeAdamW(
            model.parameters(), lr=1e-3,
            grokfast_alpha=0.9, grokfast_lamb=2.0,
        )
        _train_step(model, target, opt)
        p = list(model.parameters())[0]
        st = opt.state[p]
        assert "g_slow" in st
        assert "z" in st
        assert st["g_slow"].abs().sum().item() > 0


# ---------------------------------------------------------------------------
# State dict save / load
# ---------------------------------------------------------------------------

class TestStateDict:

    def test_save_load_preserves_z_and_weight_sum(self):
        torch.manual_seed(0)
        model = nn.Linear(5, 1)
        opt = SNRScheduleFreeAdamW(model.parameters(), lr=1e-3)
        x = torch.randn(4, 5)
        y = torch.randn(4, 1)

        for _ in range(5):
            opt.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            opt.step()

        sd = copy.deepcopy(opt.state_dict())
        weight_sum_before = opt.param_groups[0]["weight_sum"]

        # Build fresh optimizer and load state.
        model2 = nn.Linear(5, 1)
        model2.load_state_dict(model.state_dict())
        opt2 = SNRScheduleFreeAdamW(model2.parameters(), lr=1e-3)
        opt2.load_state_dict(sd)

        assert opt2.param_groups[0]["weight_sum"] == pytest.approx(weight_sum_before)
        p_orig = list(model.parameters())[0]
        p_new = list(model2.parameters())[0]
        assert torch.allclose(opt2.state[p_new]["z"], opt.state[p_orig]["z"])

    def test_save_load_train_mode_preserved(self):
        model, target = _make_linear()
        opt = SNRScheduleFreeAdamW(model.parameters(), lr=1e-3)
        _train_step(model, target, opt)
        opt.eval()
        sd = copy.deepcopy(opt.state_dict())

        model2, _ = _make_linear()
        opt2 = SNRScheduleFreeAdamW(model2.parameters(), lr=1e-3)
        opt2.load_state_dict(sd)
        assert opt2.param_groups[0]["train_mode"] is False
