"""
Theoretical / mathematical property tests for the SNR gate and optimizer.

These verify invariants from the paper (arXiv:2605.01172) and expected
mathematical behaviour rather than input/output correctness.
"""

import pytest
import torch
import torch.nn as nn

from snr_grad import SNRAdamW, compute_gate, per_sample_variance_term, resolve_alpha


# ---------------------------------------------------------------------------
# Gate monotonicity: higher SNR -> higher gate value
# ---------------------------------------------------------------------------

class TestGateMonotonicity:
    """The gate should be monotonically non-decreasing in signal strength."""

    @pytest.mark.parametrize("gate", ["soft", "snr"])
    def test_gate_increases_with_signal(self, gate):
        """Fixing noise s, increasing |m| should increase the gate."""
        s_hat = torch.ones(1)
        m_values = torch.linspace(0.0, 5.0, 50)
        prev_q = -1.0
        for m in m_values:
            q = compute_gate(m.unsqueeze(0), s_hat, gate=gate).item()
            assert q >= prev_q - 1e-7, (
                f"gate={gate}: q decreased from {prev_q} to {q} at m={m.item()}"
            )
            prev_q = q

    @pytest.mark.parametrize("gate", ["soft", "snr"])
    def test_gate_decreases_with_noise(self, gate):
        """Fixing signal m, increasing s should decrease the gate."""
        m_hat = torch.tensor([1.0])
        s_values = torch.linspace(0.01, 10.0, 50)
        prev_q = float("inf")
        for s in s_values:
            q = compute_gate(m_hat, s.unsqueeze(0), gate=gate).item()
            assert q <= prev_q + 1e-7, (
                f"gate={gate}: q increased from {prev_q} to {q} at s={s.item()}"
            )
            prev_q = q


# ---------------------------------------------------------------------------
# Gate boundary behaviour
# ---------------------------------------------------------------------------

class TestGateBoundaries:
    """Test limiting behaviour of gates."""

    @pytest.mark.parametrize("gate", ["soft", "snr"])
    def test_pure_signal_gate_near_one(self, gate):
        """When noise is zero, gate should approach 1."""
        m_hat = torch.tensor([10.0])
        s_hat = torch.zeros(1)
        q = compute_gate(m_hat, s_hat, gate=gate, gate_eps=1e-30)
        assert q.item() > 0.999

    @pytest.mark.parametrize("gate", ["soft", "snr", "hard"])
    def test_pure_noise_gate_near_zero(self, gate):
        """When signal is zero, gate should be zero."""
        m_hat = torch.zeros(1)
        s_hat = torch.tensor([10.0])
        q = compute_gate(m_hat, s_hat, gate=gate)
        assert q.item() == pytest.approx(0.0, abs=1e-10)

    def test_hard_gate_exact_threshold(self):
        """Hard gate transitions exactly at m^2 = alpha * s."""
        s_hat = torch.tensor([1.0])
        alpha = 2.0
        # m^2 = 1.99 < 2.0 -> gate = 0
        q_below = compute_gate(
            torch.tensor([1.99 ** 0.5]), s_hat, gate="hard", alpha=alpha
        )
        # m^2 = 2.01 > 2.0 -> gate = 1
        q_above = compute_gate(
            torch.tensor([2.01 ** 0.5]), s_hat, gate="hard", alpha=alpha
        )
        assert q_below.item() == 0.0
        assert q_above.item() == 1.0


# ---------------------------------------------------------------------------
# SNR gate closed-form identity: q = m^2 / (m^2 + lambda*s + eps)
# ---------------------------------------------------------------------------

class TestSNRGateIdentity:

    def test_snr_gate_half_at_boundary_when_lambda_one(self):
        """At m^2 = alpha*s and lambda_pop=1, snr gate is exactly 1/2."""
        alpha = 2.0
        s_hat = torch.tensor([3.0])
        m_hat = torch.tensor([(alpha * s_hat.item()) ** 0.5])
        q = compute_gate(m_hat, s_hat, gate="snr", alpha=alpha, lambda_pop=1.0)
        assert q.item() == pytest.approx(0.5, abs=1e-8)

    def test_snr_gate_changes_with_alpha(self):
        """Finite alpha must affect the snr gate."""
        m_hat = torch.tensor([2.0])
        s_hat = torch.tensor([1.0])
        q_alpha1 = compute_gate(m_hat, s_hat, gate="snr", alpha=1.0, lambda_pop=1.0)
        q_alpha3 = compute_gate(m_hat, s_hat, gate="snr", alpha=3.0, lambda_pop=1.0)
        assert q_alpha3.item() < q_alpha1.item()

    def test_snr_formula_batch(self):
        """Verify the SNR gate matches its closed-form for random inputs."""
        torch.manual_seed(123)
        m_hat = torch.randn(200)
        s_hat = torch.rand(200) + 0.01
        lam = 2.5
        eps = 1e-12
        q = compute_gate(m_hat, s_hat, gate="snr", lambda_pop=lam, gate_eps=eps)
        m2 = m_hat.square()
        expected = m2 / (m2 + lam * s_hat + eps)
        assert torch.allclose(q, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# Soft gate closed-form identity
# ---------------------------------------------------------------------------

class TestSoftGateIdentity:

    def test_soft_formula_batch(self):
        """Verify the soft gate matches its closed-form."""
        torch.manual_seed(456)
        m_hat = torch.randn(200)
        s_hat = torch.rand(200) + 0.01
        alpha = 1.5
        lam = 0.8
        eps = 1e-12
        q = compute_gate(
            m_hat, s_hat, gate="soft", alpha=alpha, lambda_pop=lam, gate_eps=eps
        )
        m2 = m_hat.square()
        delta = torch.relu(m2 - alpha * s_hat)
        expected = delta / (delta + lam * s_hat + eps)
        assert torch.allclose(q, expected, atol=1e-7)

    def test_soft_gate_zero_below_threshold(self):
        """Soft gate should be exactly 0 when m^2 < alpha * s."""
        m_hat = torch.tensor([0.5])  # m^2 = 0.25
        s_hat = torch.tensor([1.0])
        alpha = 1.0  # threshold = 1.0 > 0.25
        q = compute_gate(m_hat, s_hat, gate="soft", alpha=alpha)
        assert q.item() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Finite-dataset alpha: alpha = b / (n - b)
# ---------------------------------------------------------------------------

class TestFiniteAlphaTheoretical:

    def test_alpha_equals_ratio(self):
        """alpha should be exactly b/(n-b) as per the paper."""
        b, n = 64, 50000
        alpha = resolve_alpha("finite", batch_size=b, dataset_size=n)
        assert alpha == pytest.approx(b / (n - b))

    def test_alpha_approaches_zero_large_dataset(self):
        """For n >> b, alpha -> 0."""
        alpha = resolve_alpha("finite", batch_size=32, dataset_size=10_000_000)
        assert alpha < 1e-4

    def test_alpha_approaches_one_half_dataset(self):
        """For b = n/2, alpha = 1."""
        alpha = resolve_alpha("finite", batch_size=500, dataset_size=1000)
        assert alpha == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Variance estimator: unbiased and correct scaling
# ---------------------------------------------------------------------------

class TestVarianceEstimatorTheory:

    def test_unbiased_estimator(self):
        """
        Over many draws from a known distribution, the average of
        per_sample_variance_term should converge to true_var / batch_size.
        """
        torch.manual_seed(7)
        true_var = 4.0  # variance of N(0, 2)
        b = 32
        n_trials = 5000
        estimates = []
        for _ in range(n_trials):
            grads = torch.randn(b, 1) * (true_var ** 0.5)
            estimates.append(per_sample_variance_term(grads).item())
        mean_est = sum(estimates) / len(estimates)
        expected = true_var / b
        # Should be within ~5% with 5000 trials
        assert mean_est == pytest.approx(expected, rel=0.1)

    def test_scales_inversely_with_batch_size(self):
        """Variance term should halve when batch size doubles (approx)."""
        torch.manual_seed(99)
        n_trials = 2000
        est_small = []
        est_large = []
        for _ in range(n_trials):
            g16 = torch.randn(16, 10)
            g32 = torch.randn(32, 10)
            est_small.append(per_sample_variance_term(g16).mean().item())
            est_large.append(per_sample_variance_term(g32).mean().item())
        ratio = (sum(est_small) / len(est_small)) / (sum(est_large) / len(est_large))
        # Should be ~2.0
        assert ratio == pytest.approx(2.0, rel=0.15)


# ---------------------------------------------------------------------------
# Optimizer convergence on a simple quadratic
# ---------------------------------------------------------------------------

class TestConvergence:

    @pytest.mark.parametrize("gate", ["soft", "snr", "hard"])
    def test_quadratic_convergence(self, gate):
        """
        SNRAdamW should minimize f(x) = ||x - x*||^2 to near zero
        on a noise-free quadratic.
        """
        torch.manual_seed(0)
        dim = 10
        x_star = torch.randn(dim)
        param = nn.Parameter(torch.zeros(dim))
        opt = SNRAdamW([param], lr=1e-2, gate=gate, rho=0.99)

        for _ in range(500):
            opt.zero_grad()
            loss = ((param - x_star) ** 2).sum()
            loss.backward()
            opt.step()

        final_loss = ((param - x_star) ** 2).sum().item()
        assert final_loss < 0.1, f"gate={gate}: final_loss={final_loss}"


# ---------------------------------------------------------------------------
# Gate suppresses noise-dominated parameters
# ---------------------------------------------------------------------------

class TestGateSuppressionTheory:
    """
    The core premise: parameters with high gradient noise relative to
    signal should get smaller updates than those with clear signal.
    """

    def test_noisy_param_gets_lower_gate(self):
        """
        Compare gate values for a high-SNR vs low-SNR parameter after
        several steps with controlled gradients.
        """
        torch.manual_seed(42)

        # High-signal param: consistent gradient of 1.0
        p_signal = nn.Parameter(torch.zeros(10))
        # Low-signal param: gradient is pure noise
        p_noise = nn.Parameter(torch.zeros(10))

        opt_signal = SNRAdamW([p_signal], lr=1e-3, gate="soft", track_stats=True)
        opt_noise = SNRAdamW([p_noise], lr=1e-3, gate="soft", track_stats=True)

        for _ in range(50):
            opt_signal.zero_grad()
            p_signal.grad = torch.ones(10)  # consistent signal
            opt_signal.step()

            opt_noise.zero_grad()
            p_noise.grad = torch.randn(10)  # random noise each step
            opt_noise.step()

        assert opt_signal.last_stats is not None
        assert opt_noise.last_stats is not None
        assert opt_signal.last_stats.mean_gate > opt_noise.last_stats.mean_gate


# ---------------------------------------------------------------------------
# Symmetry: gate is symmetric in sign of m_hat
# ---------------------------------------------------------------------------

class TestGateSymmetry:

    @pytest.mark.parametrize("gate", ["soft", "snr", "hard"])
    def test_sign_symmetry(self, gate):
        """Gate depends on m_hat^2, so sign of m_hat should not matter."""
        torch.manual_seed(0)
        m_hat = torch.randn(50)
        s_hat = torch.rand(50) + 0.01
        q_pos = compute_gate(m_hat, s_hat, gate=gate)
        q_neg = compute_gate(-m_hat, s_hat, gate=gate)
        assert torch.allclose(q_pos, q_neg, atol=1e-7)


# ---------------------------------------------------------------------------
# lambda_pop scaling
# ---------------------------------------------------------------------------

class TestLambdaPopScaling:

    @pytest.mark.parametrize("gate", ["soft", "snr"])
    def test_higher_lambda_shrinks_gate(self, gate):
        """Increasing lambda_pop should shrink the gate (more conservative)."""
        # Use m^2 > alpha*s so the soft gate's delta is nonzero
        m_hat = torch.tensor([2.0])
        s_hat = torch.tensor([1.0])
        q_low = compute_gate(m_hat, s_hat, gate=gate, lambda_pop=0.1, alpha=1.0)
        q_high = compute_gate(m_hat, s_hat, gate=gate, lambda_pop=10.0, alpha=1.0)
        assert q_low.item() > q_high.item()
