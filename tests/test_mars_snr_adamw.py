"""Tests for the MARSSNRAdamW optimizer: construction, stepping, and integration."""

import pytest
import torch
import torch.nn as nn

from snr_grad import MARSSNRAdamW, SNRAdamWStats


def _make_model_and_loss(dim=10, seed=0):
    """Create a simple linear model with a quadratic loss for testing."""
    torch.manual_seed(seed)
    model = nn.Linear(dim, 1, bias=True)  # bias is True so we have a 1D param
    target = torch.randn(1, dim)
    return model, target


def _do_step(model, target, optimizer):
    """Forward + backward + step."""
    x = torch.ones(1, model.in_features)
    loss = ((model(x) - (target @ x.T).squeeze()) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestMARSSNRAdamWConstruction:

    def test_default_construction(self):
        model = nn.Linear(5, 1)
        opt = MARSSNRAdamW(model.parameters())
        assert opt.defaults["gate"] == "snr"
        assert opt.defaults["alpha"] == "online"
        assert opt.defaults["gamma"] == 0.025
        assert opt.defaults["mars_clip"] == 1.0
        assert opt.defaults["optimize_1d"] is False
        assert opt.defaults["caution"] is False

    @pytest.mark.parametrize("gate", ["soft", "snr", "hard"])
    def test_valid_gate_types(self, gate):
        model = nn.Linear(5, 1)
        opt = MARSSNRAdamW(model.parameters(), gate=gate)
        assert opt.defaults["gate"] == gate

    def test_invalid_gamma_raises(self):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid gamma"):
            MARSSNRAdamW(model.parameters(), gamma=-0.01)

    def test_invalid_mars_clip_raises(self):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid mars_clip"):
            MARSSNRAdamW(model.parameters(), mars_clip=-1.0)


# ---------------------------------------------------------------------------
# Basic stepping and optimization dynamics
# ---------------------------------------------------------------------------

class TestMARSSNRAdamWStep:

    @pytest.mark.parametrize("gate", ["soft", "snr", "hard"])
    def test_step_runs_all_gates(self, gate):
        model, target = _make_model_and_loss()
        opt = MARSSNRAdamW(model.parameters(), lr=1e-3, gate=gate)
        loss = _do_step(model, target, opt)
        assert loss >= 0

    def test_parameters_change_after_step(self):
        model, target = _make_model_and_loss()
        opt = MARSSNRAdamW(model.parameters(), lr=1e-2, gate="snr")
        w_before = model.weight.data.clone()
        b_before = model.bias.data.clone()
        _do_step(model, target, opt)
        assert not torch.equal(model.weight.data, w_before)
        assert not torch.equal(model.bias.data, b_before)

    def test_caution_flag_runs(self):
        model, target = _make_model_and_loss()
        opt = MARSSNRAdamW(model.parameters(), lr=1e-2, gate="snr", caution=True)
        w_before = model.weight.data.clone()
        _do_step(model, target, opt)
        assert not torch.equal(model.weight.data, w_before)

    def test_optimize_1d_flag_behaves_correctly(self):
        # We verify optimize_1d=True and False both run and update bias
        for optimize_1d in (False, True):
            model, target = _make_model_and_loss()
            opt = MARSSNRAdamW(model.parameters(), lr=1e-2, gate="snr", optimize_1d=optimize_1d)
            b_before = model.bias.data.clone()
            _do_step(model, target, opt)
            assert not torch.equal(model.bias.data, b_before)

    def test_sparse_grad_raises(self):
        emb = nn.Embedding(10, 3, sparse=True)
        opt = MARSSNRAdamW(emb.parameters(), lr=1e-3)
        idx = torch.tensor([0, 1, 2])
        loss = emb(idx).sum()
        loss.backward()
        with pytest.raises(RuntimeError, match="sparse"):
            opt.step()

    def test_exact_variance_override(self):
        model, target = _make_model_and_loss(dim=5)
        opt = MARSSNRAdamW(model.parameters(), lr=1e-3, gate="soft")
        x = torch.ones(1, 5)
        loss = model(x).sum()
        loss.backward()
        param = list(model.parameters())[0]
        fake_var = torch.ones_like(param) * 0.01
        opt.step(grad_variances={param: fake_var})  # should not crash


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestMARSSNRAdamWStats:

    def test_stats_populated_after_step(self):
        model, target = _make_model_and_loss()
        opt = MARSSNRAdamW(model.parameters(), lr=1e-3, track_stats=True)
        _do_step(model, target, opt)
        stats = opt.last_stats
        assert stats is not None
        assert stats.parameters_seen == 2  # weight and bias
        assert stats.min_gate >= 0
        assert stats.max_gate <= 1.0
        assert stats.min_gate <= stats.max_gate + 1e-6
        assert stats.mean_s_hat >= 0
        assert stats.mean_m2 >= 0

    def test_stats_none_when_disabled(self):
        model, target = _make_model_and_loss()
        opt = MARSSNRAdamW(model.parameters(), lr=1e-3, track_stats=False)
        _do_step(model, target, opt)
        assert opt.last_stats is None
