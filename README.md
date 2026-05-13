# snr-grad

A PyTorch optimizer that adds an SNR / population-risk gate to AdamW, based on [arXiv:2605.01172](https://arxiv.org/abs/2605.01172).

The gate suppresses parameter updates that are dominated by gradient noise, allowing only updates with a strong signal-to-noise ratio to pass through.

## Installation

Install directly from GitHub:

```bash
uv pip install git+https://github.com/mattsq/snr-grad.git
```

Or clone and install in editable mode for development:

```bash
git clone https://github.com/mattsq/snr-grad.git
cd snr-grad
uv pip install -e .
```

## Quick start

`SNRAdamW` can be used in place of `torch.optim.AdamW` in standard training loops:

```python
from snr_grad import SNRAdamW

optimizer = SNRAdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01,
)

# Standard training loop
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

## Gate types

Three gating strategies are available via the `gate` parameter:

| Gate     | Formula | Notes |
|----------|---------|-------|
| `"snr"`  | `m^2 / (m^2 + l*s + eps)` | **Default.** Smooth SNR shrinker, robust out of the box |
| `"soft"` | `relu(m^2 - a*s) / (relu(m^2 - a*s) + l*s + eps)` | Paper Algorithm 1. Has hard threshold floor at `m^2 = a*s` |
| `"hard"` | `1[m^2 > a*s]` | Binary gate for ablations |

Where `m` = bias-corrected first moment, `s` = bias-corrected gradient variance EMA, `a` = alpha, `l` = lambda_pop.

> **Why `"snr"` is the default instead of the paper's `"soft"`:** The soft gate has a hard
> threshold floor — when `m^2 < alpha * s`, the gate is exactly zero. In practice this can
> shut down too many parameters, especially in overparameterized models where most gradients
> are noisy. The SNR gate degrades gracefully: every parameter gets an update proportional to
> its signal-to-noise ratio, with no cliff. It is also less sensitive to `alpha` and
> `lambda_pop`, making it easier to use without tuning. Use `gate="soft"` if you want the
> paper's exact Algorithm 1 formulation.

```python
optimizer = SNRAdamW(model.parameters(), lr=3e-4, gate="soft")  # paper default
```

## Finite-dataset correction

For finite datasets, set `alpha="finite"` with dataset metadata to use the leave-one-out coefficient `alpha = b / (n - b)`:

```python
optimizer = SNRAdamW(
    model.parameters(),
    lr=3e-4,
    alpha="finite",
    batch_size=128,
    dataset_size=len(train_dataset),
)
```

`alpha` also accepts `"online"` (equivalent to `1.0`, the default) or any numeric value.

## Exact gradient variance

If you have access to per-sample gradients (e.g. via `torch.func.vmap`), you can supply exact variance estimates instead of relying on the streaming EMA:

```python
from snr_grad import per_sample_variance_term
import torch.func as F

# Compute per-example gradients with vmap
def compute_loss(params, buffers, sample, target):
    prediction = torch.func.functional_call(model, (params, buffers), (sample.unsqueeze(0),))
    return loss_fn(prediction, target.unsqueeze(0))

ft_compute = F.grad(compute_loss)
ft_compute_vmap = F.vmap(ft_compute, in_dims=(None, None, 0, 0))
per_sample_grads = ft_compute_vmap(params, buffers, batch_inputs, batch_targets)

# Build grad_variances dict for each parameter
grad_variances = {p: per_sample_variance_term(g) for p, g in zip(model.parameters(), per_sample_grads.values())}
optimizer.step(grad_variances=grad_variances)
```


## Experimental extensions

This repo includes several experimental optimizers that extend SNR gating to matrix-aware update strategies for 2D weight parameters. Non-2D parameters (biases, norms, vectors) fall back to standard SNR-gated AdamW-style updates in all variants.

### `SNRMuon` -- SNR-gated Muon orthogonalization

Combines SNR gating with Muon-style Newton-Schulz orthogonalization for 2D parameters:

- `muon_mode="post"` (default): `q * Ortho(update)` -- gate the orthogonalized update
- `muon_mode="pre"`: `Ortho(q * update)` -- orthogonalize the gated update

```python
from snr_grad import SNRMuon

optimizer = SNRMuon(
    model.parameters(),
    lr=3e-4,
    gate="snr",
    muon_mode="post",
    muon_ns_steps=5,
)
```

### `RotatedSNRAdamW` -- Eigenbasis-rotated SNR gating

SOAP-style optimizer that maintains running estimates of the left and right gradient covariance matrices, periodically computes their eigenbases, and applies SNR gating in the rotated coordinate frame. This allows the gate to operate along the natural axes of gradient variation rather than the parameter axes.

```python
from snr_grad import RotatedSNRAdamW

optimizer = RotatedSNRAdamW(
    model.parameters(),
    lr=1e-3,
    gate="snr",
    basis_beta=0.95,            # EMA for covariance tracking
    basis_update_interval=50,   # re-compute eigenbasis every N steps
)
```

### `SpectralSNRMuon` -- SVD-basis SNR gating

Applies SNR gating in the singular value decomposition (SVD) basis of the momentum matrix. Two modes control the granularity of gating, and two variants control how the gated coefficients are used:

| Parameter | Options | Description |
|-----------|---------|-------------|
| `mode` | `"diag"` / `"full"` | Gate per singular value, or gate the full spectral coefficient matrix |
| `variant` | `"adam_spectral_gate"` / `"muon_spectral_gate"` | Include Adam-style v_hat normalisation, or use raw gated coefficients |

```python
from snr_grad import SpectralSNRMuon

optimizer = SpectralSNRMuon(
    model.parameters(),
    lr=1e-3,
    gate="snr",
    mode="diag",                    # "diag" or "full"
    variant="adam_spectral_gate",   # "adam_spectral_gate" or "muon_spectral_gate"
)
```

### When to use which optimizer

Benchmarks on synthetic low-rank matrix recovery with anisotropic inputs reveal clear regimes where each method excels:

| Regime | Best method | Why |
|--------|------------|-----|
| **Axis-aligned sparsity + anisotropic inputs** | `RotatedSNRAdamW` | Eigenbasis rotation compensates for input covariance mismatch; per-coordinate SNR is confused by correlated gradient noise |
| **Dense signal (randomly rotated)** | `SNRAdamW` | Signal is distributed across all parameters; per-coordinate gating correctly treats all entries as having signal |
| **General 2D weights, mild overparameterization** | `SpectralSNRMuon (full)` | Full spectral gating captures cross-singular-value interactions |
| **Non-2D parameters** | `SNRAdamW` | All matrix-basis methods fall back to SNRAdamW for 1D params |

The matrix-basis optimizers are **preconditioners**: they add value when there is structured sparsity in the gradient covariance eigenbasis. When signal is uniformly distributed across parameters, standard per-coordinate `SNRAdamW` is preferred.

## Benchmarks

The repo includes three benchmark scripts that can be run to reproduce all figures:

```bash
python benchmark.py           # SNRAdamW vs AdamW on sparse regression
python benchmark_muon.py      # SNRMuon vs SNRAdamW vs AdamW (two-layer network)
python benchmark_spectral.py  # RotatedSNRAdamW & SpectralSNRMuon vs baselines
python benchmark_hard.py      # Low-rank matrix recovery stress test
```

### `benchmark.py` -- Core SNR gating evaluation

Sparse linear regression (d=200, k=5 signal features, n=100 training samples, high noise). Demonstrates that SNR gating suppresses updates to the 195 irrelevant features while allowing signal features through. Compares both `"snr"` and `"soft"` gate types.

**Output:** `benchmarks/benchmark_main_*.png`, `benchmark_weights_*.png`, `benchmark_summary_*.png`

### `benchmark_muon.py` -- SNRMuon hybrid evaluation

Two-layer linear network so 2D weight matrices trigger Muon's Newton-Schulz orthogonalization path. Compares SNRMuon (post/pre modes), SNRAdamW, and AdamW.

**Output:** `benchmarks/benchmark_muon_*.png`

### `benchmark_spectral.py` -- Spectral & rotated optimizer evaluation

Same two-layer sparse regression task, comparing RotatedSNRAdamW, SpectralSNRMuon (diag/full, adam/muon variants), SNRAdamW, and AdamW. Includes a per-seed heatmap showing improvement ratios vs AdamW.

**Output:** `benchmarks/benchmark_spectral_*.png`

### `benchmark_hard.py` -- Low-rank matrix recovery stress test

The most demanding benchmark. Recovers a rank-5 matrix in R^{100x100} from noisy observations with anisotropic input covariance (condition number 100). Tests both axis-aligned and randomly-rotated signal conditions to delineate when matrix-basis gating helps vs hurts.

Metrics tracked: relative Frobenius error, left/right singular subspace alignment, effective (stable) rank, and singular value spectrum of the learned matrix.

**Key finding:** RotatedSNRAdamW reduces Frobenius error by ~34% vs AdamW in the aligned+anisotropic regime, but per-coordinate methods are ~2x better when the signal is dense (rotated case). This clearly shows the matrix-basis methods are preconditioners for structured problems, not universal improvements.

**Output:** `benchmarks/benchmark_hardrot_*.png`

## Diagnostics

Enable `track_stats=True` (the default) to inspect gate behaviour after each step:

```python
stats = optimizer.last_stats  # SNRAdamWStats or None
if stats:
    print(f"mean gate: {stats.mean_gate:.4f}")
    print(f"gate range: [{stats.min_gate:.4f}, {stats.max_gate:.4f}]")
    print(f"mean s_hat: {stats.mean_s_hat:.6f}")
    print(f"mean m^2: {stats.mean_m2:.6f}")
```

## API reference

### `SNRAdamW(params, **kwargs)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | `1e-3` | Learning rate |
| `betas` | `tuple[float, float]` | `(0.9, 0.999)` | Adam moment coefficients |
| `rho` | `float` | `0.99` | EMA coefficient for gradient variance |
| `eps` | `float` | `1e-8` | Adam denominator epsilon |
| `gate_eps` | `float` | `1e-12` | Gate denominator epsilon |
| `weight_decay` | `float` | `0.0` | Decoupled weight decay |
| `gate` | `"soft" \| "snr" \| "hard"` | `"snr"` | Gate type (see Gate types) |
| `lambda_pop` | `float` | `1.0` | Population-risk scaling factor |
| `alpha` | `float \| "online" \| "finite"` | `"online"` | Leave-one-out coefficient |
| `batch_size` | `int \| None` | `None` | Required when `alpha="finite"` |
| `dataset_size` | `int \| None` | `None` | Required when `alpha="finite"` |
| `maximize` | `bool` | `False` | Maximize the objective instead of minimizing |
| `track_stats` | `bool` | `True` | Collect per-step gate diagnostics |

### Helper functions

- **`resolve_alpha(alpha, *, batch_size, dataset_size)`** -- Resolve an alpha spec to a float.
- **`compute_gate(m_hat, s_hat, *, gate, alpha, lambda_pop, gate_eps)`** -- Compute the gate tensor from bias-corrected moments.
- **`per_sample_variance_term(per_sample_grads)`** -- Compute exact diagonal variance from per-example gradients.

## Scope

This package implements the **diagonal SNR / population-risk gated AdamW update** (Algorithm 1 from the paper) along with experimental extensions to matrix-aware gating strategies (rotated eigenbasis, SVD-basis, and Muon-style orthogonalization). It does not implement:

- Automatic per-example gradient computation (use `torch.func.vmap` and pass results via `grad_variances`)
- Multi-epoch total-variation corrections for replayed batches
- The full leave-one-out estimator pipeline (only the derived diagonal gate is implemented)

## Citation

```bibtex
@article{litman2026theory,
    title   = {A Theory of Generalization in Deep Learning},
    author  = {Litman, Elon and Guo, Gabe},
    journal = {arXiv preprint arXiv:2605.01172},
    year    = {2026},
}
```

## License

MIT
