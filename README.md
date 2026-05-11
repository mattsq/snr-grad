# snr-grad

A PyTorch optimizer that adds an SNR / population-risk gate to AdamW, based on [arXiv:2605.01172](https://arxiv.org/abs/2605.01172).

The gate suppresses parameter updates that are dominated by gradient noise, allowing only updates with a strong signal-to-noise ratio to pass through.

## Installation

```bash
uv pip install snr-grad
```

Or install from source:

```bash
git clone https://github.com/<your-org>/snr-grad.git
cd snr-grad
uv pip install -e .
```

## Quick start

`SNRAdamW` is a drop-in replacement for `torch.optim.AdamW`:

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
| `"soft"` | `relu(m^2 - a*s) / (relu(m^2 - a*s) + l*s + eps)` | Paper default (Algorithm 1) |
| `"snr"`  | `m^2 / (m^2 + l*s + eps)` | Smoother SNR shrinker |
| `"hard"` | `1[m^2 > a*s]` | Binary gate for ablations |

Where `m` = bias-corrected first moment, `s` = bias-corrected gradient variance EMA, `a` = alpha, `l` = lambda_pop.

```python
optimizer = SNRAdamW(model.parameters(), lr=3e-4, gate="snr")
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

If you have access to per-sample gradients, you can supply exact variance estimates instead of relying on the streaming EMA:

```python
from snr_grad import per_sample_variance_term

# per_sample_grads: [batch, *param_shape]
var_term = per_sample_variance_term(per_sample_grads)
optimizer.step(grad_variances={param: var_term})
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
| `gate` | `"soft" \| "snr" \| "hard"` | `"soft"` | Gate type |
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
