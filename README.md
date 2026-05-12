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


## Experimental extension: `SNRMuon`

This repo now includes an experimental hybrid optimizer that combines SNR gating with Muon-style orthogonalization for 2D parameters:

- `muon_mode="post"` (default): `q ⊙ Ortho(update)`
- `muon_mode="pre"`: `Ortho(q ⊙ update)`

Non-2D parameters (biases, norms, vectors) fall back to SNR-gated AdamW-style updates.

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

This package implements the **diagonal SNR / population-risk gated AdamW update** (Algorithm 1 from the paper). It does not implement:

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
