"""Functional tests for resolve_alpha, compute_gate, and per_sample_variance_term."""

import pytest
import torch

from snr_grad import compute_gate, per_sample_variance_term, resolve_alpha


# ---------------------------------------------------------------------------
# resolve_alpha
# ---------------------------------------------------------------------------

class TestResolveAlpha:
    """Tests for the alpha-resolution helper."""

    @pytest.mark.parametrize("value", [0.5, 1, 1.0, 0, 2.5])
    def test_numeric_passthrough(self, value):
        result = resolve_alpha(value)
        assert result == float(value)
        assert isinstance(result, float)

    @pytest.mark.parametrize("key", ["online", "fresh", "fresh_batch"])
    def test_online_aliases(self, key):
        assert resolve_alpha(key) == 1.0

    @pytest.mark.parametrize("key", ["Online", "FRESH", "Fresh_Batch"])
    def test_case_insensitive(self, key):
        assert resolve_alpha(key) == 1.0

    def test_finite_basic(self):
        # alpha = b / (n - b) = 32 / (1000 - 32)
        result = resolve_alpha("finite", batch_size=32, dataset_size=1000)
        assert result == pytest.approx(32.0 / 968.0)

    def test_finite_dataset_alias(self):
        result = resolve_alpha("finite_dataset", batch_size=64, dataset_size=5000)
        assert result == pytest.approx(64.0 / 4936.0)

    def test_finite_missing_batch_size(self):
        with pytest.raises(ValueError, match="batch_size and dataset_size"):
            resolve_alpha("finite", dataset_size=1000)

    def test_finite_missing_dataset_size(self):
        with pytest.raises(ValueError, match="batch_size and dataset_size"):
            resolve_alpha("finite", batch_size=32)

    def test_finite_zero_batch_size(self):
        with pytest.raises(ValueError, match="positive"):
            resolve_alpha("finite", batch_size=0, dataset_size=1000)

    def test_finite_negative_batch_size(self):
        with pytest.raises(ValueError, match="positive"):
            resolve_alpha("finite", batch_size=-1, dataset_size=1000)

    def test_finite_dataset_too_small(self):
        with pytest.raises(ValueError, match="larger than batch_size"):
            resolve_alpha("finite", batch_size=100, dataset_size=100)

    def test_finite_dataset_smaller_than_batch(self):
        with pytest.raises(ValueError, match="larger than batch_size"):
            resolve_alpha("finite", batch_size=200, dataset_size=100)

    def test_unknown_string(self):
        with pytest.raises(ValueError, match="Unknown alpha spec"):
            resolve_alpha("bogus")


# ---------------------------------------------------------------------------
# compute_gate
# ---------------------------------------------------------------------------

class TestComputeGate:
    """Functional tests for the gate computation."""

    @pytest.fixture
    def tensors(self):
        torch.manual_seed(42)
        m_hat = torch.randn(100)
        s_hat = torch.rand(100) + 0.01  # positive
        return m_hat, s_hat

    # -- hard gate --

    def test_hard_gate_binary(self, tensors):
        m_hat, s_hat = tensors
        q = compute_gate(m_hat, s_hat, gate="hard", alpha=1.0)
        unique = q.unique()
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_hard_gate_correctness(self):
        m_hat = torch.tensor([2.0, 0.1, 0.0])
        s_hat = torch.tensor([1.0, 1.0, 1.0])
        q = compute_gate(m_hat, s_hat, gate="hard", alpha=1.0)
        assert q[0].item() == 1.0  # 4.0 > 1.0
        assert q[1].item() == 0.0  # 0.01 < 1.0
        assert q[2].item() == 0.0  # 0.0 < 1.0

    # -- soft gate --

    def test_soft_gate_bounded(self, tensors):
        m_hat, s_hat = tensors
        q = compute_gate(m_hat, s_hat, gate="soft")
        assert (q >= 0).all()
        assert (q <= 1).all()

    def test_soft_gate_zero_signal(self):
        m_hat = torch.zeros(10)
        s_hat = torch.ones(10)
        q = compute_gate(m_hat, s_hat, gate="soft")
        assert torch.allclose(q, torch.zeros(10), atol=1e-10)

    def test_soft_gate_zero_noise(self):
        m_hat = torch.tensor([3.0])
        s_hat = torch.zeros(1)
        q = compute_gate(m_hat, s_hat, gate="soft", gate_eps=1e-12)
        assert q.item() == pytest.approx(1.0, abs=1e-6)

    # -- snr gate --

    def test_snr_gate_bounded(self, tensors):
        m_hat, s_hat = tensors
        q = compute_gate(m_hat, s_hat, gate="snr")
        assert (q >= 0).all()
        assert (q <= 1).all()

    def test_snr_gate_zero_signal(self):
        m_hat = torch.zeros(10)
        s_hat = torch.ones(10)
        q = compute_gate(m_hat, s_hat, gate="snr")
        assert torch.allclose(q, torch.zeros(10), atol=1e-10)

    def test_snr_gate_zero_noise(self):
        m_hat = torch.tensor([5.0])
        s_hat = torch.zeros(1)
        q = compute_gate(m_hat, s_hat, gate="snr", gate_eps=1e-12)
        assert q.item() == pytest.approx(1.0, abs=1e-6)

    def test_snr_gate_formula(self):
        m_hat = torch.tensor([2.0, 0.5])
        s_hat = torch.tensor([1.0, 1.0])
        eps = 1e-12
        q = compute_gate(m_hat, s_hat, gate="snr", lambda_pop=1.0, gate_eps=eps)
        expected = m_hat.square() / (m_hat.square() + s_hat + eps)
        assert torch.allclose(q, expected)

    # -- alpha scaling --

    def test_soft_gate_alpha_scales_threshold(self):
        m_hat = torch.tensor([1.0])  # m^2 = 1.0
        s_hat = torch.tensor([1.0])
        # alpha=0.5: delta = relu(1 - 0.5) = 0.5 > 0 -> gate open
        q_low = compute_gate(m_hat, s_hat, gate="soft", alpha=0.5)
        # alpha=2.0: delta = relu(1 - 2.0) = 0 -> gate shut
        q_high = compute_gate(m_hat, s_hat, gate="soft", alpha=2.0)
        assert q_low.item() > 0
        assert q_high.item() == pytest.approx(0.0, abs=1e-10)

    # -- error handling --

    def test_unknown_gate_raises(self, tensors):
        m_hat, s_hat = tensors
        with pytest.raises(ValueError, match="Unknown gate"):
            compute_gate(m_hat, s_hat, gate="invalid")

    # -- dtype preservation --

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_output_dtype(self, dtype):
        m_hat = torch.randn(5, dtype=dtype)
        s_hat = torch.rand(5, dtype=dtype) + 0.01
        for gate in ("soft", "snr", "hard"):
            q = compute_gate(m_hat, s_hat, gate=gate)
            assert q.dtype == dtype


# ---------------------------------------------------------------------------
# per_sample_variance_term
# ---------------------------------------------------------------------------

class TestPerSampleVarianceTerm:
    """Tests for the exact variance helper."""

    def test_output_shape(self):
        grads = torch.randn(16, 10, 5)
        result = per_sample_variance_term(grads)
        assert result.shape == (10, 5)

    def test_1d_params(self):
        grads = torch.randn(8, 20)
        result = per_sample_variance_term(grads)
        assert result.shape == (20,)

    def test_known_variance(self):
        # All samples identical -> variance = 0
        grads = torch.ones(10, 5) * 3.0
        result = per_sample_variance_term(grads)
        assert torch.allclose(result, torch.zeros(5), atol=1e-7)

    def test_known_nonzero_variance(self):
        # Two samples: [0, 2] -> var(unbiased)=2, divide by b=2 -> 1.0
        grads = torch.tensor([[0.0], [2.0]])
        result = per_sample_variance_term(grads)
        assert result.item() == pytest.approx(1.0)

    def test_scalar_input_raises(self):
        with pytest.raises(ValueError, match="batch dimension"):
            per_sample_variance_term(torch.tensor(1.0))

    def test_single_sample_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            per_sample_variance_term(torch.randn(1, 5))

    def test_nonnegative(self):
        torch.manual_seed(0)
        grads = torch.randn(32, 64)
        result = per_sample_variance_term(grads)
        assert (result >= 0).all()
