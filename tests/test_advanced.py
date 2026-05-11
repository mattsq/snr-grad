"""Advanced tests: state dict, param groups, reproducibility, integration, edge cases."""

import copy

import pytest
import torch
import torch.nn as nn

from snr_grad import SNRAdamW, compute_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


def _mlp_step(model, opt, x, y):
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
    return loss.item()


# ---------------------------------------------------------------------------
# State dict save / load (checkpoint-resume)
# ---------------------------------------------------------------------------

class TestStateDict:

    def test_save_load_roundtrip(self):
        """Optimizer state survives save/load and produces identical training."""
        torch.manual_seed(0)
        model = nn.Linear(5, 1)
        opt = SNRAdamW(model.parameters(), lr=1e-3, gate="soft")
        x = torch.randn(4, 5)
        y = torch.randn(4, 1)

        # Run a few steps
        for _ in range(5):
            _mlp_step(model, opt, x, y)

        # Snapshot
        sd = copy.deepcopy(opt.state_dict())
        w_snapshot = model.weight.data.clone()

        # Run 3 more steps
        for _ in range(3):
            _mlp_step(model, opt, x, y)
        w_continued = model.weight.data.clone()

        # Restore and replay
        model.weight.data.copy_(w_snapshot)
        opt.load_state_dict(sd)
        for _ in range(3):
            _mlp_step(model, opt, x, y)
        w_restored = model.weight.data.clone()

        assert torch.allclose(w_continued, w_restored, atol=1e-4)

    def test_state_dict_contains_custom_keys(self):
        """Verify our extra EMA state is in the state dict."""
        model = nn.Linear(3, 1, bias=False)
        opt = SNRAdamW(model.parameters(), lr=1e-3)
        x = torch.randn(1, 3)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        sd = opt.state_dict()
        state_0 = sd["state"][0]
        assert "exp_grad_var" in state_0
        assert "step" in state_0


# ---------------------------------------------------------------------------
# Multiple parameter groups
# ---------------------------------------------------------------------------

class TestParamGroups:

    def test_different_gates_per_group(self):
        """Each param group can use a different gate type."""
        model = _make_mlp()
        opt = SNRAdamW(
            [
                {"params": model[0].parameters(), "gate": "soft"},
                {"params": model[2].parameters(), "gate": "snr"},
            ],
            lr=1e-3,
        )
        x = torch.randn(4, 8)
        y = torch.randn(4, 1)
        # Should not crash; both groups update
        for _ in range(5):
            _mlp_step(model, opt, x, y)

    def test_different_lr_per_group(self):
        """Groups with different learning rates should produce different updates."""
        torch.manual_seed(0)
        m1 = nn.Linear(5, 1, bias=False)
        m2 = nn.Linear(5, 1, bias=False)
        m2.weight.data.copy_(m1.weight.data)

        # Single group, uniform lr
        opt1 = SNRAdamW(m1.parameters(), lr=1e-2, gate="snr")
        # Two-group, but effectively same since only one param
        opt2 = SNRAdamW(
            [{"params": m2.parameters(), "lr": 1e-4}],
            lr=1e-2,
            gate="snr",
        )

        x = torch.randn(1, 5)
        for _ in range(10):
            _mlp_step(m1, opt1, x, torch.ones(1, 1))
            _mlp_step(m2, opt2, x, torch.ones(1, 1))

        # Different lr -> different weights
        assert not torch.allclose(m1.weight.data, m2.weight.data, atol=1e-4)

    def test_different_alpha_per_group(self):
        """One group online, one group finite."""
        model = _make_mlp()
        opt = SNRAdamW(
            [
                {"params": model[0].parameters(), "alpha": "online"},
                {
                    "params": model[2].parameters(),
                    "alpha": "finite",
                    "batch_size": 32,
                    "dataset_size": 1000,
                },
            ],
            lr=1e-3,
        )
        x = torch.randn(4, 8)
        y = torch.randn(4, 1)
        for _ in range(3):
            _mlp_step(model, opt, x, y)


# ---------------------------------------------------------------------------
# Reproducibility / determinism
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_deterministic_given_same_seed(self):
        """Same seed + same data -> identical weights after N steps."""
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            model = nn.Linear(5, 1, bias=False)
            opt = SNRAdamW(model.parameters(), lr=1e-3, gate="soft")
            x = torch.randn(4, 5)
            y = torch.randn(4, 1)
            for _ in range(20):
                _mlp_step(model, opt, x, y)
            results.append(model.weight.data.clone())

        assert torch.equal(results[0], results[1])


# ---------------------------------------------------------------------------
# Integration: train a small MLP to convergence
# ---------------------------------------------------------------------------

class TestIntegration:

    @pytest.mark.parametrize("gate", ["soft", "snr", "hard"])
    def test_mlp_learns_xor(self, gate):
        """SNRAdamW can train a small MLP to solve XOR."""
        torch.manual_seed(1)
        model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        opt = SNRAdamW(model.parameters(), lr=1e-2, gate=gate, weight_decay=0.0)

        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

        for _ in range(2000):
            _mlp_step(model, opt, x, y)

        with torch.no_grad():
            preds = model(x)
            rounded = (preds > 0.5).float()
        assert torch.equal(rounded, y), f"XOR not solved: preds={preds.squeeze().tolist()}"

    def test_loss_decreases_over_training(self):
        """Loss should generally decrease over many steps."""
        torch.manual_seed(0)
        model = _make_mlp()
        opt = SNRAdamW(model.parameters(), lr=1e-3, gate="soft")
        x = torch.randn(16, 8)
        y = torch.randn(16, 1)

        early_loss = sum(_mlp_step(model, opt, x, y) for _ in range(10)) / 10
        for _ in range(200):
            _mlp_step(model, opt, x, y)
        late_loss = sum(_mlp_step(model, opt, x, y) for _ in range(10)) / 10
        assert late_loss < early_loss


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_zero_gradient_no_crash(self):
        """Zero gradients should not cause NaN or crash."""
        p = nn.Parameter(torch.ones(5))
        opt = SNRAdamW([p], lr=1e-3, gate="soft")
        for _ in range(5):
            opt.zero_grad()
            p.grad = torch.zeros(5)
            opt.step()
        assert torch.isfinite(p).all()

    def test_very_large_gradient(self):
        """Large gradients should not produce NaN."""
        p = nn.Parameter(torch.zeros(5))
        opt = SNRAdamW([p], lr=1e-6, gate="snr")
        opt.zero_grad()
        p.grad = torch.ones(5) * 1e6
        opt.step()
        assert torch.isfinite(p).all()

    def test_very_small_gradient(self):
        """Tiny gradients should not produce NaN."""
        p = nn.Parameter(torch.zeros(5))
        opt = SNRAdamW([p], lr=1e-3, gate="snr")
        opt.zero_grad()
        p.grad = torch.ones(5) * 1e-30
        opt.step()
        assert torch.isfinite(p).all()

    def test_gradient_accumulation(self):
        """Multiple backward() calls before step() should work."""
        model = nn.Linear(5, 1, bias=False)
        opt = SNRAdamW(model.parameters(), lr=1e-3)
        x1 = torch.randn(1, 5)
        x2 = torch.randn(1, 5)

        opt.zero_grad()
        loss1 = model(x1).sum()
        loss1.backward()
        loss2 = model(x2).sum()
        loss2.backward()  # accumulates into existing .grad
        opt.step()
        # Should not crash; grad is sum of two backward passes

    def test_single_element_parameter(self):
        """Scalar-like parameter (single element) should work."""
        p = nn.Parameter(torch.tensor([1.0]))
        opt = SNRAdamW([p], lr=1e-2, gate="soft")
        for _ in range(200):
            opt.zero_grad()
            loss = (p - 3.0) ** 2
            loss.backward()
            opt.step()
        assert abs(p.item() - 3.0) < 1.0

    def test_high_dimensional_parameter(self):
        """Large tensor should work without issues."""
        p = nn.Parameter(torch.randn(256, 256))
        opt = SNRAdamW([p], lr=1e-3, gate="soft")
        opt.zero_grad()
        loss = p.sum()
        loss.backward()
        opt.step()
        assert torch.isfinite(p).all()


# ---------------------------------------------------------------------------
# gate_eps prevents division by zero
# ---------------------------------------------------------------------------

class TestGateEps:

    def test_zero_s_hat_zero_m_hat_no_nan(self):
        """When both m_hat and s_hat are zero, gate_eps prevents 0/0."""
        m_hat = torch.zeros(5)
        s_hat = torch.zeros(5)
        for gate in ("soft", "snr"):
            q = compute_gate(m_hat, s_hat, gate=gate, gate_eps=1e-12)
            assert torch.isfinite(q).all()
            assert (q == 0).all()  # numerator is 0

    def test_gate_eps_affects_output(self):
        """Larger gate_eps should produce slightly smaller gate values."""
        m_hat = torch.tensor([1.0])
        s_hat = torch.tensor([0.001])
        q_small_eps = compute_gate(m_hat, s_hat, gate="snr", gate_eps=1e-12)
        q_large_eps = compute_gate(m_hat, s_hat, gate="snr", gate_eps=1.0)
        assert q_small_eps.item() > q_large_eps.item()


# ---------------------------------------------------------------------------
# Bias correction convergence
# ---------------------------------------------------------------------------

class TestBiasCorrection:

    def test_step_count_matches_bias_correction_window(self):
        """After many steps, bias correction factors approach 1."""
        p = nn.Parameter(torch.zeros(3))
        opt = SNRAdamW([p], lr=0.0, gate="snr", betas=(0.9, 0.999), rho=0.99)

        for _ in range(1000):
            opt.zero_grad()
            p.grad = torch.ones(3)
            opt.step()

        state = opt.state[p]
        step = state["step"]
        assert step == 1000

        # At step 1000, bias correction for beta1=0.9 is 1/(1-0.9^1000) ~ 1.0
        beta1_corr = 1.0 - 0.9 ** 1000
        assert beta1_corr == pytest.approx(1.0, abs=1e-10)

    def test_early_steps_have_significant_bias_correction(self):
        """At step 1, bias correction factors are far from 1."""
        p = nn.Parameter(torch.zeros(3))
        opt = SNRAdamW([p], lr=1e-3, gate="snr", betas=(0.9, 0.999), rho=0.99)
        opt.zero_grad()
        p.grad = torch.ones(3)
        opt.step()

        state = opt.state[p]
        m = state["exp_avg"]
        # Uncorrected m = 0.1 * grad = 0.1
        # Corrected m_hat = 0.1 / (1 - 0.9) = 1.0
        m_hat = m / (1 - 0.9 ** 1)
        assert m_hat[0].item() == pytest.approx(1.0, abs=1e-6)
        assert m[0].item() == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# Weight decay is decoupled (applies regardless of gate)
# ---------------------------------------------------------------------------

class TestDecoupledWeightDecay:

    def test_weight_decay_with_zero_gate(self):
        """Even when gate = 0 (no Adam update), weight decay should still apply."""
        torch.manual_seed(0)
        p = nn.Parameter(torch.ones(5) * 10.0)
        # Use hard gate with very high alpha so gate is always 0
        opt = SNRAdamW([p], lr=0.1, weight_decay=0.5, gate="hard", alpha=1e10)

        opt.zero_grad()
        p.grad = torch.ones(5) * 0.001  # tiny gradient, won't pass hard gate
        w_before = p.data.clone()
        opt.step()

        # Weight decay: w <- w - lr * wd * w = w * (1 - 0.1*0.5) = w * 0.95
        expected = w_before * (1 - 0.1 * 0.5)
        # The Adam update term should be ~0 due to gate, so weight decay dominates
        assert torch.allclose(p.data, expected, atol=0.01)


# ---------------------------------------------------------------------------
# Stats reflect actual gate values
# ---------------------------------------------------------------------------

class TestStatsAccuracy:

    def test_stats_reflect_gate_type(self):
        """Hard gate stats should show mean_gate as 0 or 1 (binary)."""
        torch.manual_seed(0)
        p = nn.Parameter(torch.zeros(100))
        opt = SNRAdamW([p], lr=1e-3, gate="hard", track_stats=True)

        # Give consistent gradient so gates become 1 after EMA builds up
        for _ in range(50):
            opt.zero_grad()
            p.grad = torch.ones(100) * 5.0
            opt.step()

        stats = opt.last_stats
        assert stats is not None
        # With constant large gradient, all hard gates should be 1
        assert stats.mean_gate == pytest.approx(1.0, abs=0.01)

    def test_stats_update_each_step(self):
        """Stats should change between steps as EMAs evolve."""
        torch.manual_seed(0)
        p = nn.Parameter(torch.zeros(10))
        opt = SNRAdamW([p], lr=1e-3, gate="snr", track_stats=True)

        opt.zero_grad()
        p.grad = torch.randn(10)
        opt.step()
        stats1_gate = opt.last_stats.mean_gate

        opt.zero_grad()
        p.grad = torch.randn(10) * 10  # very different gradient
        opt.step()
        stats2_gate = opt.last_stats.mean_gate

        # Stats should differ between steps
        assert stats1_gate != stats2_gate
