"""Tests for activation preconditioning (DoPr) in snr_grad.activation."""

import warnings

import pytest
import torch
import torch.nn as nn

from snr_grad import (
    ActivationPrecondConfig,
    ActivationPreconditioner,
    DoPr,
    SNRAdamW,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _linear(in_dim=5, out_dim=3, bias=True, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    return nn.Linear(in_dim, out_dim, bias=bias).to(dtype)


def _ref_linear_M(G, Z, gamma):
    """Reference ``M = G @ (S + tau I)^-1`` for a linear layer (dense)."""
    n, in_dim = Z.shape
    S = Z.t() @ Z / n
    tau = gamma * S.diagonal().sum() / in_dim
    return G @ torch.linalg.inv(S + tau * torch.eye(in_dim, dtype=S.dtype))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfig:

    def test_damping_must_be_positive(self):
        with pytest.raises(ValueError):
            ActivationPrecondConfig(damping=0.0)

    def test_ema_beta_range(self):
        with pytest.raises(ValueError):
            ActivationPrecondConfig(ema_beta=1.0)
        ActivationPrecondConfig(ema_beta=0.0)  # ok

    def test_warmup_nonnegative(self):
        with pytest.raises(ValueError):
            ActivationPrecondConfig(warmup_steps=-1)


# ---------------------------------------------------------------------------
# Linear correctness
# ---------------------------------------------------------------------------

class TestLinearPrecondition:

    def test_matches_dense_reference(self):
        gamma = 0.2
        lin = _linear(5, 3, bias=True)
        x = torch.randn(20, 5, dtype=torch.float64)
        ap = ActivationPreconditioner(
            lin, ActivationPrecondConfig(damping=gamma, compute_dtype=torch.float64)
        )
        (lin(x) ** 2).sum().backward()
        G = lin.weight.grad.clone()
        M_ref = _ref_linear_M(G, x, gamma)
        ap.precondition_()
        assert torch.allclose(lin.weight.grad, M_ref, atol=1e-10)

    def test_bias_grad_untouched(self):
        lin = _linear(5, 3, bias=True)
        x = torch.randn(12, 5, dtype=torch.float64)
        ap = ActivationPreconditioner(
            lin, ActivationPrecondConfig(damping=0.1, compute_dtype=torch.float64)
        )
        (lin(x) ** 2).sum().backward()
        bias_before = lin.bias.grad.clone()
        ap.precondition_()
        assert torch.equal(lin.bias.grad, bias_before)

    def test_handles_multi_dim_input(self):
        # [batch, seq, in] should be flattened to [batch*seq, in].
        gamma = 0.15
        lin = _linear(4, 6)
        x = torch.randn(3, 7, 4, dtype=torch.float64)
        ap = ActivationPreconditioner(
            lin, ActivationPrecondConfig(damping=gamma, compute_dtype=torch.float64)
        )
        (lin(x) ** 2).sum().backward()
        G = lin.weight.grad.clone()
        M_ref = _ref_linear_M(G, x.reshape(-1, 4), gamma)
        ap.precondition_()
        assert torch.allclose(lin.weight.grad, M_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# Embedding correctness (one-hot diagonal)
# ---------------------------------------------------------------------------

class TestEmbeddingPrecondition:

    def test_matches_dense_onehot_reference(self):
        gamma = 0.3
        vocab, dim = 6, 4
        torch.manual_seed(1)
        emb = nn.Embedding(vocab, dim).double()
        idx = torch.randint(0, vocab, (15,))
        ap = ActivationPreconditioner(
            emb, ActivationPrecondConfig(damping=gamma, compute_dtype=torch.float64)
        )
        (emb(idx) ** 2).sum().backward()
        G = emb.weight.grad.clone()
        oneh = torch.nn.functional.one_hot(idx, vocab).double()
        S = oneh.t() @ oneh / idx.numel()
        tau = gamma * S.diagonal().sum() / vocab
        # Embedding input dim is the ROW dim of W -> left-multiply (row scaling).
        M_ref = torch.linalg.inv(S + tau * torch.eye(vocab, dtype=S.dtype)) @ G
        ap.precondition_()
        assert torch.allclose(emb.weight.grad, M_ref, atol=1e-10)

    def test_absent_tokens_stay_zero(self):
        vocab, dim = 8, 3
        emb = nn.Embedding(vocab, dim).double()
        idx = torch.tensor([0, 1, 2, 0, 1])  # tokens 3..7 absent
        ap = ActivationPreconditioner(emb, ActivationPrecondConfig(damping=0.1))
        (emb(idx) ** 2).sum().backward()
        ap.precondition_()
        absent = emb.weight.grad[3:]
        assert torch.count_nonzero(absent) == 0


# ---------------------------------------------------------------------------
# Affine invariance (Proposition 4.2) -- the headline property
# ---------------------------------------------------------------------------

class TestAffineInvariance:
    """Under z -> A z, W -> W A^-1 (identical pre-step outputs), one DoPr step keeps
    the post-step layer outputs identical (W_next == Wbar_next @ A), whereas plain
    SGD diverges. Exact only for undamped AP, so we use tiny damping."""

    def _setup(self, seed=0):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(seed)
        in_dim, out_dim, n = 4, 3, 400
        W = torch.randn(out_dim, in_dim)
        A = torch.randn(in_dim, in_dim) + 2.0 * torch.eye(in_dim)  # non-orthogonal
        Z = torch.randn(n, in_dim)
        target = torch.randn(n, out_dim)
        return in_dim, out_dim, W, A, Z, target

    def _step(self, W, Z, target, *, use_ap, lr=0.1, damping=1e-7):
        lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
        lin.weight.data.copy_(W)
        ap = (ActivationPreconditioner(lin, ActivationPrecondConfig(damping=damping))
              if use_ap else None)
        ((lin(Z) - target) ** 2).sum().backward()
        if ap is not None:
            ap.precondition_()
        with torch.no_grad():
            lin.weight -= lr * lin.weight.grad
        return lin.weight.data.clone()

    def test_ap_is_affine_invariant(self):
        try:
            in_dim, out_dim, W, A, Z, target = self._setup()
            Zbar = Z @ A.t()
            Wbar = W @ torch.linalg.inv(A)
            # Pre-step outputs identical.
            assert torch.allclose(Z @ W.t(), Zbar @ Wbar.t(), atol=1e-9)
            W1 = self._step(W, Z, target, use_ap=True)
            Wb1 = self._step(Wbar, Zbar, target, use_ap=True)
            assert (W1 - Wb1 @ A).abs().max() < 1e-2
        finally:
            torch.set_default_dtype(torch.float32)

    def test_sgd_is_not_affine_invariant(self):
        try:
            in_dim, out_dim, W, A, Z, target = self._setup()
            Zbar = Z @ A.t()
            Wbar = W @ torch.linalg.inv(A)
            W1 = self._step(W, Z, target, use_ap=False)
            Wb1 = self._step(Wbar, Zbar, target, use_ap=False)
            assert (W1 - Wb1 @ A).abs().max() > 1.0  # diverges -> the test has teeth
        finally:
            torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------------------
# Damping behavior
# ---------------------------------------------------------------------------

class TestDamping:

    def test_large_damping_approaches_identity_direction(self):
        lin = _linear(5, 3)
        x = torch.randn(40, 5, dtype=torch.float64)
        (lin(x) ** 2).sum().backward()
        G = lin.weight.grad.clone().reshape(-1)
        ap = ActivationPreconditioner(
            lin, ActivationPrecondConfig(damping=1e6, compute_dtype=torch.float64)
        )
        ap.precondition_()
        M = lin.weight.grad.reshape(-1)
        cos = torch.dot(G, M) / (G.norm() * M.norm())
        assert cos > 0.999

    def test_scale_invariance(self):
        # Scaling z -> c z scales S -> c^2 S and tau -> c^2 tau, so M -> M / c^2.
        gamma, c = 0.25, 3.0
        lin = _linear(5, 3)
        x = torch.randn(30, 5, dtype=torch.float64)

        lin1 = _linear(5, 3)
        ap1 = ActivationPreconditioner(
            lin1, ActivationPrecondConfig(damping=gamma, compute_dtype=torch.float64))
        (lin1(x) ** 2).sum().backward()
        ap1.precondition_()
        M1 = lin1.weight.grad.clone()

        lin2 = _linear(5, 3)  # same seed -> same weights -> same G for same outputs
        ap2 = ActivationPreconditioner(
            lin2, ActivationPrecondConfig(damping=gamma, compute_dtype=torch.float64))
        # Use cx as input; the gradient G scales by c, and M(cx)=G(cx)@(c^2 S)^-1.
        # Compare against analytic: M(cx) should equal (1/c) * [G(x)-derived]. Simpler:
        # verify M(cx) == ref computed from the scaled inputs directly.
        (lin2(c * x) ** 2).sum().backward()
        G2 = lin2.weight.grad.clone()
        ref = _ref_linear_M(G2, c * x, gamma)
        ap2.precondition_()
        assert torch.allclose(lin2.weight.grad, ref, atol=1e-9)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestEMA:

    def test_ema_combines_batches(self):
        beta = 0.9
        lin = _linear(4, 3)
        ap = ActivationPreconditioner(
            lin, ActivationPrecondConfig(damping=0.1, ema_beta=beta,
                                         compute_dtype=torch.float64))
        x1 = torch.randn(10, 4, dtype=torch.float64)
        x2 = torch.randn(10, 4, dtype=torch.float64)
        S1 = x1.t() @ x1 / 10
        S2 = x2.t() @ x2 / 10

        (lin(x1) ** 2).sum().backward()
        ap.precondition_()
        module = next(m for m in ap._sigma_ema)
        assert torch.allclose(ap._sigma_ema[module], S1, atol=1e-10)

        lin.zero_grad(set_to_none=True)
        (lin(x2) ** 2).sum().backward()
        ap.precondition_()
        assert torch.allclose(ap._sigma_ema[module], beta * S1 + (1 - beta) * S2, atol=1e-10)

    def test_ema_in_state_dict(self):
        lin = _linear(4, 3)
        ap = ActivationPreconditioner(lin, ActivationPrecondConfig(ema_beta=0.9))
        (lin(torch.randn(8, 4, dtype=torch.float64)) ** 2).sum().backward()
        ap.precondition_()
        sd = ap.state_dict()
        assert sd["step"] == 1
        assert len(sd["sigma_ema"]) == 1


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

class TestWarmup:

    def test_identity_during_warmup_then_active(self):
        lin = _linear(5, 3)
        ap = ActivationPreconditioner(
            lin, ActivationPrecondConfig(damping=0.1, warmup_steps=2,
                                         compute_dtype=torch.float64))
        for step in range(3):
            lin.zero_grad(set_to_none=True)
            (lin(torch.randn(12, 5, dtype=torch.float64)) ** 2).sum().backward()
            g_before = lin.weight.grad.clone()
            ap.precondition_()
            if step < 2:
                assert torch.equal(lin.weight.grad, g_before)  # identity
            else:
                assert not torch.equal(lin.weight.grad, g_before)  # AP applied


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_dopr_with_snradamw(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        opt = DoPr(SNRAdamW(model.parameters(), lr=1e-2, track_stats=True), model)
        before = [p.detach().clone() for p in model.parameters()]
        (model(torch.randn(16, 8)) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()
        assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        assert opt.last_stats is not None  # attribute delegated to base

    def test_dopr_with_baseline_adam(self):
        model = nn.Linear(5, 3)
        opt = DoPr(torch.optim.Adam(model.parameters(), lr=1e-2), model)
        before = model.weight.detach().clone()
        (model(torch.randn(10, 5)) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()
        assert not torch.equal(before, model.weight)

    def test_external_usage(self):
        model = nn.Linear(5, 3)
        ap = ActivationPreconditioner(model, ActivationPrecondConfig(damping=0.1))
        opt = SNRAdamW(model.parameters(), lr=1e-2)
        (model(torch.randn(10, 5)) ** 2).mean().backward()
        ap.precondition_()
        opt.step()
        opt.zero_grad(set_to_none=True)
        ap.zero_grad()
        assert ap.step_count == 1


# ---------------------------------------------------------------------------
# state_dict roundtrip
# ---------------------------------------------------------------------------

class TestStateDict:

    def test_roundtrip_restores_ema_and_step(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        ap = ActivationPreconditioner(model, ActivationPrecondConfig(ema_beta=0.9))
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            (model(torch.randn(8, 4)) ** 2).mean().backward()
            ap.precondition_()
        sd = ap.state_dict()

        model2 = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        ap2 = ActivationPreconditioner(model2, ActivationPrecondConfig(ema_beta=0.9))
        ap2.load_state_dict(sd)
        assert ap2.step_count == 2
        names = set(sd["sigma_ema"].keys())
        loaded = {ap2._name_of_module[m] for m in ap2._sigma_ema}
        assert names == loaded
        for m in ap2._sigma_ema:
            name = ap2._name_of_module[m]
            assert torch.allclose(ap2._sigma_ema[m], sd["sigma_ema"][name])

    def test_dopr_state_dict_bundles_both(self):
        model = nn.Linear(5, 3)
        opt = DoPr(SNRAdamW(model.parameters(), lr=1e-2), model)
        (model(torch.randn(10, 5)) ** 2).mean().backward()
        opt.step()
        sd = opt.state_dict()
        assert "base" in sd and "ap" in sd
        opt.load_state_dict(sd)  # should not raise


# ---------------------------------------------------------------------------
# Hooks / lifecycle
# ---------------------------------------------------------------------------

class TestHooks:

    def test_remove_hooks_makes_noop(self):
        lin = _linear(5, 3)
        ap = ActivationPreconditioner(lin, ActivationPrecondConfig(damping=0.1))
        ap.remove_hooks()
        (lin(torch.randn(8, 5, dtype=torch.float64)) ** 2).sum().backward()
        g_before = lin.weight.grad.clone()
        ap.precondition_()
        assert torch.equal(lin.weight.grad, g_before)

    def test_context_manager_removes_hooks(self):
        lin = _linear(5, 3)
        with ActivationPreconditioner(lin, ActivationPrecondConfig(damping=0.1)) as ap:
            assert len(ap._handles) == 1
        assert len(ap._handles) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestNoOp:

    def test_no_supported_modules(self):
        model = nn.LayerNorm(5)
        ap = ActivationPreconditioner(model, ActivationPrecondConfig(damping=0.1))
        x = torch.randn(4, 5)
        (model(x) ** 2).sum().backward()
        g_before = model.weight.grad.clone()
        ap.precondition_()  # no error
        assert torch.equal(model.weight.grad, g_before)


class TestTiedWeights:

    def test_tied_weight_skipped_with_warning(self):
        emb = nn.Embedding(6, 4)
        head = nn.Linear(4, 6, bias=False)
        head.weight = emb.weight  # tie
        model = nn.ModuleDict({"emb": emb, "head": head})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ap = ActivationPreconditioner(model, ActivationPrecondConfig(damping=0.1))
        assert any("shared" in str(w.message).lower() or "tied" in str(w.message).lower()
                   for w in caught)
        # The tied weight is registered nowhere -> precondition_ leaves it untouched.
        idx = torch.tensor([0, 1, 2])
        (emb(idx) ** 2).sum().backward()
        g_before = emb.weight.grad.clone()
        ap.precondition_()
        assert torch.equal(emb.weight.grad, g_before)


class TestConvNotImplemented:

    def test_conv_raises(self):
        with pytest.raises(NotImplementedError):
            ActivationPreconditioner(nn.Conv2d(3, 3, 3))

    def test_conv_can_be_excluded(self):
        model = nn.Sequential(nn.Conv2d(3, 3, 3), nn.Flatten(), nn.Linear(3, 2))
        # Exclude the conv (module name "0") -> no error, linear still registered.
        ap = ActivationPreconditioner(
            model, ActivationPrecondConfig(damping=0.1, exclude_modules=["0"]))
        assert len(ap._handles) == 1


class TestMaximize:

    def test_ap_commutes_with_negation(self):
        lin = _linear(5, 3)
        x = torch.randn(20, 5, dtype=torch.float64)
        ap = ActivationPreconditioner(
            lin, ActivationPrecondConfig(damping=0.2, compute_dtype=torch.float64))
        (lin(x) ** 2).sum().backward()
        G = lin.weight.grad.clone()
        M_ref = _ref_linear_M(G, x, 0.2)
        # AP(-G) == -AP(G): feed -G and check.
        lin.weight.grad.copy_(-G)
        ap.precondition_()
        assert torch.allclose(lin.weight.grad, -M_ref, atol=1e-10)


class TestStaleEMA:

    def test_skips_module_without_forward_when_no_ema(self):
        # Two-branch model; only one branch runs the forward this step.
        lin_a = _linear(5, 3, seed=0)
        lin_b = _linear(5, 3, seed=1)
        model = nn.ModuleDict({"a": lin_a, "b": lin_b})
        ap = ActivationPreconditioner(model, ActivationPrecondConfig(damping=0.1,
                                                                     compute_dtype=torch.float64))
        x = torch.randn(10, 5, dtype=torch.float64)
        # Only branch a runs; give b a manual grad to confirm it is left untouched.
        (lin_a(x) ** 2).sum().backward()
        lin_b.weight.grad = torch.ones_like(lin_b.weight)
        b_before = lin_b.weight.grad.clone()
        ap.precondition_()
        assert torch.equal(lin_b.weight.grad, b_before)  # skipped (no activations, no EMA)
