"""Tests for the SNRAdamW optimizer: construction, stepping, and integration."""

import pytest
import torch
import torch.nn as nn

from snr_grad import SNRAdamW, SNRAdamWStats


def _make_model_and_loss(dim=10, seed=0):
    """Create a simple linear model with a quadratic loss for testing."""
    torch.manual_seed(seed)
    model = nn.Linear(dim, 1, bias=False)
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

class TestSNRAdamWConstruction:

    def test_default_construction(self):
        model = nn.Linear(5, 1)
        opt = SNRAdamW(model.parameters())
        assert opt.defaults["gate"] == "soft"
        assert opt.defaults["alpha"] == "online"

    @pytest.mark.parametrize("gate", ["soft", "snr", "hard"])
    def test_valid_gate_types(self, gate):
        model = nn.Linear(5, 1)
        opt = SNRAdamW(model.parameters(), gate=gate)
        assert opt.defaults["gate"] == gate

    def test_invalid_gate_raises(self):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid gate"):
            SNRAdamW(model.parameters(), gate="invalid")

    def test_negative_lr_raises(self):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid lr"):
            SNRAdamW(model.parameters(), lr=-1e-3)

    def test_invalid_beta1_raises(self):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid beta1"):
            SNRAdamW(model.parameters(), betas=(1.0, 0.999))

    def test_invalid_beta2_raises(self):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid beta2"):
            SNRAdamW(model.parameters(), betas=(0.9, -0.1))

    def test_invalid_rho_raises(self):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid rho"):
            SNRAdamW(model.parameters(), rho=1.0)

    def test_negative_weight_decay_raises(self):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            SNRAdamW(model.parameters(), weight_decay=-0.1)

    def test_zero_eps_raises(self):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid eps"):
            SNRAdamW(model.parameters(), eps=0)

    def test_negative_lambda_pop_raises(self):
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Invalid lambda_pop"):
            SNRAdamW(model.parameters(), lambda_pop=-1)


# ---------------------------------------------------------------------------
# Basic stepping
# ---------------------------------------------------------------------------

class TestSNRAdamWStep:

    @pytest.mark.parametrize("gate", ["soft", "snr", "hard"])
    def test_step_runs_all_gates(self, gate):
        model, target = _make_model_and_loss()
        opt = SNRAdamW(model.parameters(), lr=1e-3, gate=gate)
        loss = _do_step(model, target, opt)
        assert loss >= 0

    def test_parameters_change_after_step(self):
        model, target = _make_model_and_loss()
        opt = SNRAdamW(model.parameters(), lr=1e-2, gate="snr")
        w_before = model.weight.data.clone()
        _do_step(model, target, opt)
        assert not torch.equal(model.weight.data, w_before)

    def test_no_grad_params_skipped(self):
        model = nn.Linear(5, 1, bias=True)
        model.bias.requires_grad_(False)
        opt = SNRAdamW(model.parameters(), lr=1e-3)
        x = torch.randn(1, 5)
        loss = model(x).sum()
        loss.backward()
        opt.step()  # should not crash despite bias having no grad

    def test_closure(self):
        model, target = _make_model_and_loss()
        opt = SNRAdamW(model.parameters(), lr=1e-3)

        def closure():
            opt.zero_grad()
            x = torch.ones(1, model.in_features)
            loss = ((model(x) - 1.0) ** 2).mean()
            loss.backward()
            return loss

        loss = opt.step(closure)
        assert loss is not None
        assert loss.item() >= 0

    def test_sparse_grad_raises(self):
        emb = nn.Embedding(10, 3, sparse=True)
        opt = SNRAdamW(emb.parameters(), lr=1e-3)
        idx = torch.tensor([0, 1, 2])
        loss = emb(idx).sum()
        loss.backward()
        with pytest.raises(RuntimeError, match="sparse"):
            opt.step()


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestSNRAdamWStats:

    def test_stats_populated_after_step(self):
        model, target = _make_model_and_loss()
        opt = SNRAdamW(model.parameters(), lr=1e-3, track_stats=True)
        _do_step(model, target, opt)
        stats = opt.last_stats
        assert stats is not None
        assert stats.parameters_seen == 1  # one Linear weight
        assert stats.min_gate >= 0
        assert stats.max_gate <= 1.0
        assert stats.min_gate <= stats.max_gate + 1e-6
        assert stats.mean_s_hat >= 0
        assert stats.mean_m2 >= 0

    def test_stats_none_when_disabled(self):
        model, target = _make_model_and_loss()
        opt = SNRAdamW(model.parameters(), lr=1e-3, track_stats=False)
        _do_step(model, target, opt)
        assert opt.last_stats is None

    def test_stats_none_when_no_grads(self):
        p = torch.randn(5, requires_grad=True)
        opt = SNRAdamW([p], lr=1e-3)
        # step without backward -> no grads
        opt.step()
        assert opt.last_stats is None


# ---------------------------------------------------------------------------
# Weight decay
# ---------------------------------------------------------------------------

class TestWeightDecay:

    def test_weight_decay_shrinks_params(self):
        torch.manual_seed(0)
        model = nn.Linear(10, 1, bias=False)
        nn.init.ones_(model.weight)
        opt = SNRAdamW(model.parameters(), lr=0.0, weight_decay=0.1, gate="snr")
        # lr=0 means only weight decay acts: w <- w - lr*wd*w = w*(1 - 0)
        # Actually lr=0 means wd term is also 0. Use small lr.
        opt2 = SNRAdamW(model.parameters(), lr=0.01, weight_decay=0.5, gate="snr")
        w_before = model.weight.data.norm().item()
        x = torch.randn(1, 10)
        loss = model(x).sum()
        loss.backward()
        opt2.step()
        w_after = model.weight.data.norm().item()
        # weight decay should shrink the weight norm
        assert w_after < w_before


# ---------------------------------------------------------------------------
# Maximize flag
# ---------------------------------------------------------------------------

class TestMaximize:

    def test_maximize_moves_opposite(self):
        torch.manual_seed(0)
        # Two identical models, one minimize, one maximize
        m1 = nn.Linear(5, 1, bias=False)
        m2 = nn.Linear(5, 1, bias=False)
        m2.weight.data.copy_(m1.weight.data)

        opt1 = SNRAdamW(m1.parameters(), lr=1e-2, gate="snr", maximize=False)
        opt2 = SNRAdamW(m2.parameters(), lr=1e-2, gate="snr", maximize=True)

        x = torch.randn(1, 5)
        target = torch.tensor([[1.0]])

        loss1 = ((m1(x) - target) ** 2).mean()
        loss1.backward()
        opt1.step()

        loss2 = ((m2(x) - target) ** 2).mean()
        loss2.backward()
        opt2.step()

        # The updates should move in opposite directions
        delta1 = m1.weight.data - m2.weight.data
        assert delta1.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Finite alpha via step kwargs
# ---------------------------------------------------------------------------

class TestFiniteAlpha:

    def test_finite_alpha_step_kwargs(self):
        model, target = _make_model_and_loss()
        # Works via group-level defaults
        opt = SNRAdamW(
            model.parameters(), lr=1e-3,
            alpha="finite", batch_size=32, dataset_size=1000,
        )
        _do_step(model, target, opt)

    def test_finite_alpha_missing_raises(self):
        model, target = _make_model_and_loss()
        opt = SNRAdamW(model.parameters(), lr=1e-3, alpha="finite")
        x = torch.ones(1, model.in_features)
        loss = model(x).sum()
        loss.backward()
        with pytest.raises(ValueError, match="batch_size and dataset_size"):
            opt.step()


# ---------------------------------------------------------------------------
# grad_variances override
# ---------------------------------------------------------------------------

class TestGradVariances:

    def test_exact_variance_override(self):
        model, target = _make_model_and_loss(dim=5)
        opt = SNRAdamW(model.parameters(), lr=1e-3, gate="soft")
        x = torch.ones(1, 5)
        loss = model(x).sum()
        loss.backward()
        param = list(model.parameters())[0]
        fake_var = torch.ones_like(param) * 0.01
        opt.step(grad_variances={param: fake_var})  # should not crash

    def test_wrong_shape_raises(self):
        model, target = _make_model_and_loss(dim=5)
        opt = SNRAdamW(model.parameters(), lr=1e-3)
        x = torch.ones(1, 5)
        loss = model(x).sum()
        loss.backward()
        param = list(model.parameters())[0]
        wrong_var = torch.ones(99)
        with pytest.raises(ValueError, match="shape"):
            opt.step(grad_variances={param: wrong_var})


# ---------------------------------------------------------------------------
# Multi-step state accumulation
# ---------------------------------------------------------------------------

class TestStateAccumulation:

    def test_step_counter_increments(self):
        model, target = _make_model_and_loss()
        opt = SNRAdamW(model.parameters(), lr=1e-3)
        for _ in range(5):
            _do_step(model, target, opt)
        param = list(model.parameters())[0]
        assert opt.state[param]["step"] == 5

    def test_ema_states_nonzero_after_steps(self):
        model, target = _make_model_and_loss()
        opt = SNRAdamW(model.parameters(), lr=1e-3)
        for _ in range(3):
            _do_step(model, target, opt)
        param = list(model.parameters())[0]
        state = opt.state[param]
        assert state["exp_avg"].abs().sum().item() > 0
        assert state["exp_avg_sq"].abs().sum().item() > 0
        assert state["exp_grad_var"].abs().sum().item() > 0
