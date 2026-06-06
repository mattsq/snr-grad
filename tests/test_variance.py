"""Tests for the variance-estimation backends in snr_grad.variance."""

import warnings

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call

from snr_grad import (
    ExactVarianceEstimator,
    MicrobatchVarianceEstimator,
    SNRAdamW,
    backward_with_microbatch_variance,
    compare_gate_with_external_variance,
    per_sample_grad_variances,
    per_sample_variance_term,
)
from snr_grad.variance import tree_batch_size, tree_split


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _linear_model(in_dim=4, out_dim=3, seed=0, bias=True):
    torch.manual_seed(seed)
    return nn.Linear(in_dim, out_dim, bias=bias)


def _make_batch(b=8, in_dim=4, out_dim=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, in_dim, generator=g)
    y = torch.randn(b, out_dim, generator=g)
    return x, y


def _loss_one_sample(model):
    """Build a per-sample loss closure for a regression model + MSE (summed)."""
    def loss_one_sample(params, buffers, sample):
        x, y = sample
        pred = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        # Per-example loss: sum so its gradient is the per-example gradient.
        return ((pred.squeeze(0) - y) ** 2).sum()
    return loss_one_sample


def _manual_per_sample_grads(model, x, y):
    """Per-sample gradients of the summed squared-error loss for a linear model."""
    grads_w = []
    grads_b = []
    for i in range(x.shape[0]):
        model.zero_grad(set_to_none=True)
        pred = model(x[i:i + 1])
        loss = ((pred.squeeze(0) - y[i]) ** 2).sum()
        loss.backward()
        grads_w.append(model.weight.grad.detach().clone())
        if model.bias is not None:
            grads_b.append(model.bias.grad.detach().clone())
    model.zero_grad(set_to_none=True)
    out = {model.weight: torch.stack(grads_w)}
    if model.bias is not None:
        out[model.bias] = torch.stack(grads_b)
    return out


# ---------------------------------------------------------------------------
# Pytree helpers
# ---------------------------------------------------------------------------

class TestTreeHelpers:

    def test_tree_batch_size_tuple_dict_tensor(self):
        x = torch.randn(7, 3)
        assert tree_batch_size(x) == 7
        assert tree_batch_size((x, torch.randn(7))) == 7
        assert tree_batch_size({"a": x, "b": torch.randn(7, 2)}) == 7

    def test_tree_split_tuple_reconstructs(self):
        x = torch.arange(8).reshape(8, 1).float()
        y = torch.arange(8).float()
        chunks = tree_split((x, y), 4)
        assert len(chunks) == 4
        assert all(c[0].shape[0] == 2 for c in chunks)
        # Concatenating chunks recovers the original.
        x_back = torch.cat([c[0] for c in chunks], dim=0)
        assert torch.equal(x_back, x)

    def test_tree_split_dict(self):
        batch = {"inputs": torch.randn(6, 2), "targets": torch.randn(6)}
        chunks = tree_split(batch, 3)
        assert len(chunks) == 3
        assert all(set(c) == {"inputs", "targets"} for c in chunks)
        assert all(c["inputs"].shape[0] == 2 for c in chunks)


# ---------------------------------------------------------------------------
# Exact backend: formula correctness
# ---------------------------------------------------------------------------

class TestExactVariance:

    def test_matches_per_sample_variance_term(self):
        model = _linear_model()
        x, y = _make_batch()
        manual = _manual_per_sample_grads(model, x, y)
        expected = {p: per_sample_variance_term(g) for p, g in manual.items()}

        got = per_sample_grad_variances(model, _loss_one_sample(model), (x, y))

        assert set(got) == set(expected)
        for p in expected:
            assert torch.allclose(got[p], expected[p], atol=1e-6, rtol=1e-5)

    def test_estimator_matches_low_level(self):
        model = _linear_model()
        x, y = _make_batch()
        manual = _manual_per_sample_grads(model, x, y)
        expected = {p: per_sample_variance_term(g) for p, g in manual.items()}

        est = ExactVarianceEstimator()
        got = est.estimate(model, _loss_one_sample(model), (x, y))

        for p in expected:
            assert torch.allclose(got[p], expected[p], atol=1e-6, rtol=1e-5)

    def test_shapes_match_parameters(self):
        model = _linear_model(in_dim=5, out_dim=2)
        x, y = _make_batch(b=6, in_dim=5, out_dim=2)
        got = ExactVarianceEstimator().estimate(model, _loss_one_sample(model), (x, y))
        for p, var in got.items():
            assert var.shape == p.shape

    def test_variance_is_nonnegative(self):
        model = _linear_model()
        x, y = _make_batch()
        got = ExactVarianceEstimator().estimate(model, _loss_one_sample(model), (x, y))
        for var in got.values():
            assert torch.all(var >= 0)
            assert torch.all(torch.isfinite(var))

    def test_dtype_returned_matches_parameter(self):
        model = _linear_model().double()
        x, y = _make_batch()
        x, y = x.double(), y.double()
        # Compute in fp32 even though params are fp64; output must match param dtype.
        est = ExactVarianceEstimator(dtype=torch.float32)
        got = est.estimate(model, _loss_one_sample(model), (x, y))
        for p, var in got.items():
            assert var.dtype == p.dtype == torch.float64

    def test_chunk_size_gives_same_result(self):
        model = _linear_model()
        x, y = _make_batch(b=8)
        full = per_sample_grad_variances(model, _loss_one_sample(model), (x, y))
        chunked = per_sample_grad_variances(
            model, _loss_one_sample(model), (x, y), chunk_size=2
        )
        for p in full:
            assert torch.allclose(full[p], chunked[p], atol=1e-6)


# ---------------------------------------------------------------------------
# Exact backend: parameter filtering
# ---------------------------------------------------------------------------

class TestParameterFiltering:

    def test_exclude_params_skips_named(self):
        model = _linear_model()
        x, y = _make_batch()
        est = ExactVarianceEstimator(exclude_params=["bias"])
        got = est.estimate(model, _loss_one_sample(model), (x, y))
        assert model.weight in got
        assert model.bias not in got

    def test_include_params_only_keeps_named(self):
        model = _linear_model()
        x, y = _make_batch()
        est = ExactVarianceEstimator(include_params=["weight"])
        got = est.estimate(model, _loss_one_sample(model), (x, y))
        assert set(got) == {model.weight}

    def test_filtered_value_matches_unfiltered(self):
        # Restricting which params are differentiated must not change the values
        # for the params that remain.
        model = _linear_model()
        x, y = _make_batch()
        full = ExactVarianceEstimator().estimate(model, _loss_one_sample(model), (x, y))
        only_w = ExactVarianceEstimator(include_params=["weight"]).estimate(
            model, _loss_one_sample(model), (x, y)
        )
        assert torch.allclose(full[model.weight], only_w[model.weight], atol=1e-6)

    def test_exclude_norm_skips_batchnorm_params(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
        model.eval()  # avoid BatchNorm train-mode coupling
        x = torch.randn(6, 4)
        y = torch.randn(6, 2)

        def loss_one_sample(params, buffers, sample):
            xi, yi = sample
            pred = functional_call(model, (params, buffers), (xi.unsqueeze(0),))
            return ((pred.squeeze(0) - yi) ** 2).sum()

        est = ExactVarianceEstimator(exclude_norm=True)
        got = est.estimate(model, loss_one_sample, (x, y))
        bn = model[1]
        assert bn.weight not in got
        assert bn.bias not in got
        # Linear weights still present.
        assert model[0].weight in got


# ---------------------------------------------------------------------------
# Microbatch (split-batch) backend
# ---------------------------------------------------------------------------

def _microbatch_loss_fn():
    def loss_fn(model, sub_batch):
        x, y = sub_batch
        return ((model(x) - y) ** 2).mean()
    return loss_fn


class TestMicrobatchVariance:

    def test_two_split_identity(self):
        model = _linear_model()
        x, y = _make_batch(b=8)
        loss_fn = _microbatch_loss_fn()

        # Reference: manual halves.
        xa, xb = x[:4], x[4:]
        ya, yb = y[:4], y[4:]
        model.zero_grad(set_to_none=True)
        ((model(xa) - ya) ** 2).mean().backward()
        ha_w = model.weight.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        ((model(xb) - yb) ** 2).mean().backward()
        hb_w = model.weight.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        expected_w = (ha_w - hb_w) ** 2 / 4

        _, grad_variances = backward_with_microbatch_variance(
            model, loss_fn, (x, y), num_splits=2
        )
        assert torch.allclose(grad_variances[model.weight], expected_w, atol=1e-6)

    def test_accumulate_full_grad_sets_mean_gradient(self):
        model = _linear_model()
        x, y = _make_batch(b=8)
        loss_fn = _microbatch_loss_fn()

        # Full-batch mean gradient reference.
        model.zero_grad(set_to_none=True)
        ((model(x) - y) ** 2).mean().backward()
        full_grad_w = model.weight.grad.detach().clone()
        model.zero_grad(set_to_none=True)

        _, _ = backward_with_microbatch_variance(
            model, loss_fn, (x, y), num_splits=2, accumulate_full_grad=True
        )
        # For equal-size chunks the average of chunk means equals the full mean.
        assert torch.allclose(model.weight.grad, full_grad_w, atol=1e-6)

    def test_estimator_wrapper_returns_variance_and_loss(self):
        model = _linear_model()
        x, y = _make_batch(b=8)
        est = MicrobatchVarianceEstimator(num_splits=4)
        got = est.estimate(model, _microbatch_loss_fn(), (x, y))
        assert model.weight in got
        assert est.last_loss is not None
        for var in got.values():
            assert torch.all(var >= 0)

    def test_variance_scales_inversely_with_batch_size(self):
        # For IID gradients, the variance of the minibatch mean ~ 1/b. Doubling the
        # batch size should roughly halve the estimate (loose statistical tolerance).
        torch.manual_seed(0)
        model = nn.Linear(6, 1, bias=False)
        loss_fn = _microbatch_loss_fn()

        def mean_estimate(b, trials=40):
            total = 0.0
            for t in range(trials):
                g = torch.Generator().manual_seed(1000 + t)
                x = torch.randn(b, 6, generator=g)
                y = torch.randn(b, 1, generator=g)
                _, gv = backward_with_microbatch_variance(model, loss_fn, (x, y), num_splits=2)
                total += float(gv[model.weight].mean())
            return total / trials

        s_small = mean_estimate(16)
        s_large = mean_estimate(32)
        ratio = s_small / s_large
        # Expect ~2x; allow a wide band given the noisy K=2 estimator.
        assert 1.3 < ratio < 3.0

    def test_num_splits_too_large_raises(self):
        model = _linear_model()
        x, y = _make_batch(b=3)
        with pytest.raises(ValueError, match="batch size"):
            backward_with_microbatch_variance(model, _microbatch_loss_fn(), (x, y), num_splits=4)

    def test_unsupported_reduction_raises(self):
        model = _linear_model()
        x, y = _make_batch(b=4)
        with pytest.raises(ValueError, match="loss_reduction"):
            backward_with_microbatch_variance(
                model, _microbatch_loss_fn(), (x, y), num_splits=2, loss_reduction="sum"
            )


# ---------------------------------------------------------------------------
# Optimizer integration
# ---------------------------------------------------------------------------

class TestOptimizerIntegration:

    def test_exact_variance_step_runs(self):
        model = _linear_model()
        opt = SNRAdamW(model.parameters(), lr=1e-2, track_stats=True)
        x, y = _make_batch()

        ((model(x) - y) ** 2).mean().backward()
        gv = ExactVarianceEstimator().estimate(model, _loss_one_sample(model), (x, y))
        before = model.weight.detach().clone()
        opt.step(grad_variances=gv)
        assert not torch.equal(before, model.weight)
        assert opt.last_stats is not None

    def test_microbatch_integrated_path_steps(self):
        model = _linear_model()
        opt = SNRAdamW(model.parameters(), lr=1e-2)
        x, y = _make_batch()
        before = model.weight.detach().clone()

        _, gv = backward_with_microbatch_variance(
            model, _microbatch_loss_fn(), (x, y), num_splits=4
        )
        opt.step(grad_variances=gv)
        assert not torch.equal(before, model.weight)

    def test_wrong_variance_shape_raises(self):
        model = _linear_model()
        opt = SNRAdamW(model.parameters(), lr=1e-2)
        x, y = _make_batch()
        ((model(x) - y) ** 2).mean().backward()
        bad = {model.weight: torch.zeros(99)}
        with pytest.raises(ValueError, match="shape"):
            opt.step(grad_variances=bad)

    def test_state_dict_roundtrip_with_external_variance(self):
        model = _linear_model()
        opt = SNRAdamW(model.parameters(), lr=1e-2)
        x, y = _make_batch()
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            ((model(x) - y) ** 2).mean().backward()
            gv = ExactVarianceEstimator().estimate(model, _loss_one_sample(model), (x, y))
            opt.step(grad_variances=gv)

        sd = opt.state_dict()
        model2 = _linear_model()
        opt2 = SNRAdamW(model2.parameters(), lr=1e-2)
        opt2.load_state_dict(sd)
        # State restored for the stepped parameters.
        assert len(opt2.state) == len(opt.state)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:

    def test_compare_gate_reports_fields(self):
        model = _linear_model()
        opt = SNRAdamW(model.parameters(), lr=1e-2)
        x, y = _make_batch()
        # Take a step so EMA state exists.
        ((model(x) - y) ** 2).mean().backward()
        opt.step()

        gv = ExactVarianceEstimator().estimate(model, _loss_one_sample(model), (x, y))
        report = compare_gate_with_external_variance(opt, gv)
        for key in [
            "mean_internal_s",
            "mean_external_s",
            "mean_variance_ratio",
            "mean_gate_internal",
            "mean_gate_external",
            "frac_gate_changed",
            "elements_compared",
        ]:
            assert key in report
        assert report["elements_compared"] > 0
        assert 0.0 <= report["frac_gate_changed"] <= 1.0

    def test_compare_gate_identical_variance_changes_nothing(self):
        model = _linear_model()
        opt = SNRAdamW(model.parameters(), lr=1e-2)
        x, y = _make_batch()
        ((model(x) - y) ** 2).mean().backward()
        opt.step()

        # Feed back the internal EMA s_hat as the "external" estimate.
        identical = {}
        for group in opt.param_groups:
            rho = group["rho"]
            for p in group["params"]:
                st = opt.state[p]
                bc_s = 1.0 - rho ** st["step"]
                identical[p] = st["exp_grad_var"] / bc_s
        report = compare_gate_with_external_variance(opt, identical)
        assert report["frac_gate_changed"] == pytest.approx(0.0, abs=1e-6)
        assert report["mean_variance_ratio"] == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# BatchNorm warning
# ---------------------------------------------------------------------------

class TestBatchNormHandling:

    def test_training_batchnorm_warns(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        model.train()
        x = torch.randn(6, 4)
        y = torch.randn(6, 4)

        def loss_one_sample(params, buffers, sample):
            xi, yi = sample
            pred = functional_call(model, (params, buffers), (xi.unsqueeze(0),))
            return ((pred.squeeze(0) - yi) ** 2).sum()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # The warning is emitted up front. The per-sample (batch-of-1) forward
            # through train-mode BatchNorm then raises, which is expected.
            with pytest.raises(Exception):
                ExactVarianceEstimator(exclude_norm=True).estimate(model, loss_one_sample, (x, y))
        assert any("BatchNorm" in str(w.message) for w in caught)
