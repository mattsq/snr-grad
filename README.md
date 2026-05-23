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

### Grokfast integration -- Slow-gradient pre-amplification

We have integrated **Grokfast** (a technique that tracks and amplifies slow-varying components of the gradients to accelerate generalization and grokking) into all four optimizer classes: `SNRAdamW`, `SNRMuon`, `RotatedSNRAdamW`, and `SpectralSNRMuon`.

Grokfast operates via a **Pre-Gate Amplification** strategy: the slow-gradient moving average is computed and added to the gradient *before* the moments and SNR gates are calculated. This helps push parameters out of sharp local minima towards flat, broad valleys that generalize better.

To enable Grokfast slow-gradient amplification, pass `grokfast_alpha` and `grokfast_lamb` when initializing any of the optimizers:

```python
from snr_grad import SNRAdamW

optimizer = SNRAdamW(
    model.parameters(),
    lr=1e-3,
    grokfast_alpha=0.98,  # EMA decay factor for slow-gradient tracking
    grokfast_lamb=2.0,    # slow-gradient amplification strength
)
```

#### Important Guidance: The Underdetermined vs. Overdetermined Trade-off

Based on rigorous regime sweeps on high-dimensional sparse regression, Grokfast slow-gradient amplification exhibits a critical trade-off:

1. **Underdetermined Regime ($n < d$, e.g., small datasets, high noise):**
   - **Do not use standard Grokfast.** In underdetermined settings, spurious correlations create persistent, static noise-fitting gradients. Grokfast will track and *amplify* this persistent noise-fitting component, leading to severe overfitting.
   - If Grokfast is required, always pair it with the SNR gate (`grokfast_lamb > 0` and `lambda_pop > 0.0`), which acts as an adaptive filter to block the noise-amplifying updates.
2. **Overdetermined Regime ($n > d$, e.g., large clean datasets):**
   - **Highly Recommended.** When training data is abundant, the true signals have consistent, slow-moving gradients while noise cancels out. Grokfast excels in this well-determined regime, dramatically accelerating signal learning and slashing test MSE.

### ScheduleFree integration -- Iterate averaging without LR schedules

We provide ScheduleFree (Defazio et al., [arXiv:2405.15682](https://arxiv.org/abs/2405.15682)) variants of all four optimizer classes:

- `SNRScheduleFreeAdamW`
- `SNRScheduleFreeMuon`
- `RotatedSNRScheduleFreeAdamW`
- `SpectralSNRScheduleFreeMuon`

ScheduleFree replaces the need for LR schedules (warmup/cosine/linear) with built-in Polyak-Ruppert iterate averaging. The model parameters hold the gradient-evaluation point `y = (1 - sf_beta) * z + sf_beta * x` during training; the averaged iterate `x` (used at inference) is reconstructed on demand. The SNR gate is computed exactly as before and filters the per-step Adam-normalized gradient that drives the base sequence `z`. Adam's first moment `m_hat` is used *only* to compute the gate -- it does not appear in the update direction, because ScheduleFree's `y`-interpolation already provides the momentum role.

```python
from snr_grad import SNRScheduleFreeAdamW

optimizer = SNRScheduleFreeAdamW(
    model.parameters(),
    lr=3e-4,
    sf_beta=0.9,             # y-interpolation factor (Polyak averaging strength)
    sf_warmup_steps=500,     # optional linear warmup of effective lr (0 disables)
    sf_lr_power=2.0,         # weight_t = lr_max ** sf_lr_power (Defazio default)
    weight_decay=0.01,
)

# Training loop
optimizer.train()
for batch in train_loader:
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

# Switch parameters to the averaged iterate x for validation:
optimizer.eval()
model.eval()
with torch.no_grad():
    validate(model)
# Restore y for resumed training:
optimizer.train()
model.train()
```

Calling `.step()` while in eval mode raises an explicit error. `optimizer.eval()` and `optimizer.train()` are idempotent and survive `state_dict()` / `load_state_dict()` roundtrips.

Defaults follow the ScheduleFree paper: `sf_beta=0.9`, `sf_warmup_steps=0`, `sf_lr_power=2.0`, `sf_r=0.0`. Decoupled weight decay is applied through `y` (Defazio's choice). The Grokfast slow-gradient amplification described above composes with the ScheduleFree variants -- pass `grokfast_alpha` and `grokfast_lamb` as usual.

### When to use which optimizer

Benchmarks on synthetic low-rank matrix recovery with anisotropic inputs reveal clear regimes where each method excels:

| Regime | Best method | Why |
|--------|------------|-----|
| **Axis-aligned sparsity + anisotropic inputs** | `RotatedSNRAdamW` | Eigenbasis rotation compensates for input covariance mismatch; per-coordinate SNR is confused by correlated gradient noise |
| **Dense signal (randomly rotated)** | `SNRAdamW` | Signal is distributed across all parameters; per-coordinate gating correctly treats all entries as having signal |
| **General 2D weights, mild overparameterization** | `SpectralSNRMuon (full)` | Full spectral gating captures cross-singular-value interactions |
| **Non-2D parameters** | `SNRAdamW` | All matrix-basis methods fall back to SNRAdamW for 1D params |

The matrix-basis optimizers are **preconditioners**: they add value when there is structured sparsity in the gradient covariance eigenbasis. When signal is uniformly distributed across parameters, standard per-coordinate `SNRAdamW` is preferred.


## Experimental design playbook: finding ideal SNR settings

This section outlines a rigorous experiment program to identify strong default SNR parameters, understand when they transfer, and map interactions with optimizer, schedule, and data properties.

### 1) Core hypotheses to test

1. There is no single globally-optimal `lambda_pop` / `alpha`; best settings depend on gradient noise scale, effective batch size, and data anisotropy.
2. `gate="snr"` should be more robust across tasks than `"soft"`/`"hard"`, with slightly lower peak performance in highly structured regimes.
3. Matrix-aware variants (`RotatedSNRAdamW`, `SpectralSNRMuon`) should dominate only when gradient covariance is strongly structured and low-rank.
4. Finite-dataset correction (`alpha="finite"`) should help most in small-`n`, high-reuse regimes and can over-regularize in large-`n` settings.

### 2) Factor space (what to sweep)

Use a staged DOE (design of experiments) strategy:

- **Stage A (screening):** broad Latin-hypercube / Sobol exploration.
- **Stage B (interaction):** factorial sweeps around top 10--20% configs.
- **Stage C (local refinement):** Bayesian optimization per task family.

Recommended factors:

- **SNR-specific:**
  - `gate` ∈ {`snr`, `soft`, `hard`}
  - `lambda_pop` ∈ logspace [1e-3, 1e2]
  - `alpha` ∈ {`online`, `finite`, numeric logspace [1e-3, 10]}
  - `rho` ∈ {0.9, 0.95, 0.99, 0.995, 0.999}
  - `gate_eps` ∈ {1e-14, 1e-12, 1e-10}
- **Base optimizer/training:**
  - `lr` (log sweep), `weight_decay`, `betas`, `eps`
  - scheduler type (cosine, linear, constant), warmup ratio
  - gradient clipping threshold, EMA/SWA on/off
- **Batching/noise controls:**
  - batch size, gradient accumulation, label noise, augmentation strength
- **Model/data structure:**
  - parameter count vs sample size ratio (overparameterization index)
  - input covariance condition number and feature correlation
  - target sparsity / low-rankness / rotation (aligned vs rotated signal)

### 3) Benchmark matrix (tasks x regimes)

For each domain, define low/medium/high-noise and low/medium/high-data regimes.

- **Synthetic controlled tasks** (must-have for mechanism clarity):
  - sparse linear regression (already in `benchmark.py`)
  - low-rank matrix recovery with anisotropy + random rotations (`benchmark_hard.py`)
  - heteroscedastic noise variants (feature-dependent variance)
- **Vision:** CIFAR-10/100 with ResNet-18/50 at multiple data fractions (10%, 50%, 100%).
- **NLP:** small transformer on WikiText-103 / OpenWebText subset; vary sequence length and token budget.
- **Tabular:** medium-scale UCI/OpenML tasks with correlated features.

Use at least 5 seeds for screening, then 10--20 seeds for final confirmation on shortlisted settings.

### 4) Measurement protocol (what to record every run)

Record both quality and mechanism metrics:

- **Primary outcomes:** best validation metric, final test metric, time-to-target, area under learning curve.
- **Stability:** divergence rate, NaN incidence, worst-seed percentile, variance across seeds.
- **Efficiency:** tokens/s or samples/s, wall-clock to target, extra optimizer overhead.
- **Gate diagnostics** (from `track_stats`):
  - `mean_gate`, gate quantiles, fraction of near-zero gates
  - layer-wise gate distributions
  - correlation of gate values with gradient norm and update norm
- **Noise diagnostics:** estimated gradient noise scale, signal/noise decomposition by layer.

Persist all metrics as structured tables (CSV/Parquet) keyed by: task, seed, step, and full hyperparameter config hash.

### 5) Statistical analysis plan

1. **Hierarchical mixed-effects model** across all runs:
   - response ~ SNR params + optimizer params + data descriptors + interactions + (1|task) + (1|seed)
2. **Global sensitivity analysis** (Sobol/functional ANOVA): rank which knobs matter most.
3. **Partial dependence / ICE plots:** identify monotonic vs non-monotonic ranges for `lambda_pop`, `rho`, and `alpha`.
4. **Regime clustering:** cluster tasks by gradient covariance statistics and fit per-cluster defaults.
5. **Pareto frontiers:** accuracy vs wall-clock vs stability; pick defaults on Pareto knee.

### 6) Practical output targets

Produce three deliverables:

- **Universal safe default** (max robustness): e.g., `gate="snr"`, conservative `lambda_pop`, high `rho`.
- **Regime-conditional defaults** keyed by measurable quantities:
  - small-data/high-noise
  - anisotropic low-rank structure
  - dense isotropic signal
- **Tuning recipe** (2--3 knobs only): ordered search over `lr` → `lambda_pop` → `rho`, with decision thresholds based on gate diagnostics.

### 7) Minimal reproducible execution plan for this repo

1. Extend existing benchmark scripts to emit per-step CSV logs (loss, metrics, `last_stats`).
2. Add sweep driver (Hydra/W&B/Optuna) with a shared config schema.
3. Run Stage A screening on:
   - `benchmark.py`
   - `benchmark_spectral.py`
   - `benchmark_hard.py`
4. Run Stage B focused factorial sweeps around top configs.
5. Fit analysis notebooks to produce:
   - interaction heatmaps (`lambda_pop` x `lr`, `rho` x batch size)
   - regime recommendation table
   - confidence intervals for suggested defaults.

### 8) Decision criteria for “ideal” SNR parameters

Treat a setting as ideal only if it is:

- **Consistently strong:** top quartile mean performance across benchmark families.
- **Stable:** low variance and low failure rate across seeds.
- **Efficient:** no large wall-clock penalty for the achieved gain.
- **Interpretable:** gate statistics align with expected noise suppression behavior.

This prevents overfitting to one benchmark and yields deployable parameter guidance.

## Hyperparameter tuning notes

Empirical results from controlled sweeps on sparse linear regression (d=200, k=5, n=100, high noise). See `studies/hyperparameter_study/` for full experiment code and data.

### `lambda_pop` (regularization strength)

`lambda_pop` controls how aggressively the gate suppresses noisy parameters. For the SNR gate, `q = 0.5` when `m^2/s = lambda_pop`, so it directly sets the decision boundary.

- **Soft gate:** robust across `lambda_pop` 0.01--2.0 (best around 0.5). Degrades when `lambda_pop >= 10` as it starts suppressing signal.
- **SNR gate:** U-shaped response; too low (0.01) leaves noise ungated, too high (100) suppresses signal. Best around 5.0 in this regime.
- The soft gate achieves much better signal/noise separation (noise gates near zero) because its `relu` threshold creates a hard floor. The SNR gate never fully zeros out noise parameters.

### `alpha` (leave-one-out threshold)

`alpha` plays different roles depending on the gate type:

- **Soft gate:** `alpha` sets the threshold `m^2 > alpha * s` below which parameters are fully gated off. Best at `alpha` = 1.0--2.0; too low (0.1) under-thresholds, too high (5.0) over-thresholds and increases variance across seeds.
- **SNR gate:** `alpha` multiplies `lambda_pop` in the denominator (`alpha * lambda_pop * s`), so it's effectively a second scaling knob. Varying `alpha` with fixed `lambda_pop` produces the same effect as varying `lambda_pop` with fixed `alpha`.

### `rho` (variance EMA decay)

`rho` controls the effective memory window for gradient variance estimation: `1/(1-rho)` steps.

- Higher `rho` gives smoother, lower-variance estimates at the cost of slower adaptation.
- **Soft gate:** best at `rho` = 0.995. Slight degradation at 0.999 suggests over-smoothing.
- **SNR gate:** monotonically improves up to `rho` = 0.999 in stationary settings.
- Under distribution shift, `rho` = 0.999 takes ~250 steps to re-adapt vs ~50 for `rho` = 0.95. Choose lower `rho` if non-stationarity is expected.

### `alpha="finite"` (finite-dataset correction)

The correction `alpha = b/(n-b)` accounts for data reuse in finite datasets:

- **Small datasets (n < 500):** `"online"` (alpha=1.0) slightly outperforms `"finite"` -- the correction over-adjusts when data is scarce.
- **Large datasets (n >= 2000):** `"finite"` wins (e.g., 10.9 vs 12.5 MSE at n=10000) as the `b/(n-b)` term properly compensates for batch overlap.
- Rule of thumb: use `alpha="finite"` when `dataset_size / batch_size > 50`.

### Interaction with learning rate

SNR gating benefits increase with higher learning rates. Across 262 sweep trials on three benchmarks, SNR won 84% of trials at `lr > 3e-3` vs 67% at `lr < 1e-3`. Higher learning rates amplify gradient noise, giving the gate more room to suppress noisy updates. If using SNR gating, you can push `lr` slightly higher than you would with plain AdamW.

### Recommended starting points

| Setting | Conservative default | Notes |
|---------|---------------------|-------|
| `gate` | `"snr"` | More robust; switch to `"soft"` for peak performance with tuning |
| `lambda_pop` | 1.0 | Increase for noisier problems, decrease for cleaner signal |
| `alpha` | `"online"` | Use `"finite"` for large finite datasets with high reuse |
| `rho` | 0.99 | Increase to 0.995 for stationary problems; decrease to 0.95 for non-stationary |

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

Enable `track_stats=True` to inspect gate behaviour after each step (disabled by default to avoid potential device-sync overhead):

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
| `track_stats` | `bool` | `False` | Collect per-step gate diagnostics |

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
