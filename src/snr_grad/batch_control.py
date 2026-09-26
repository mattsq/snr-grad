"""Training-loop batch-size diagnostics and a conservative discrete controller.

The probe owns separate backward passes at fixed parameters. It does not change
optimizer gradients or optimizer state. Call it before the training step, then
apply the controller's recommendation to the *next* training batch.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .variance import tree_batch_size, tree_split


@dataclass(frozen=True)
class NoiseScale:
    signal: float
    noise: float
    scale: float


@dataclass(frozen=True)
class BatchProbe:
    batch_size: int
    microbatches: int
    elapsed_seconds: float
    euclidean: NoiseScale
    adamw: NoiseScale
    muon: Optional[NoiseScale]
    muon_fallback: Optional[NoiseScale]


def _ratio(signal: Tensor, noise: Tensor) -> NoiseScale:
    s, n = float(signal), float(noise)
    scale = n / s if s > 0 else math.inf
    return NoiseScale(s, n, scale)


def _squared_norms(grads: Sequence[Tensor], batch_per_split: int) -> tuple[Tensor, Tensor]:
    # The unbiased variance of means of b independent examples is C / b.
    stack = torch.stack(grads).float()
    mean = stack.mean(dim=0)
    return mean.square().sum(), stack.var(dim=0, unbiased=True).sum() * batch_per_split


def _adam_denominator(optimizer: Optional[Optimizer], parameter: Tensor) -> Tensor:
    if optimizer is None:
        return torch.ones_like(parameter, dtype=torch.float32)
    for group in optimizer.param_groups:
        if any(p is parameter for p in group["params"]):
            state = optimizer.state.get(parameter, {})
            v = state.get("exp_avg_sq")
            if v is None:
                return torch.ones_like(parameter, dtype=torch.float32)
            step = state.get("step", 0)
            if isinstance(step, Tensor):
                step = int(step.item())
            beta2 = group.get("betas", (0.9, 0.999))[1]
            correction = 1 - beta2 ** int(step)
            if correction <= 0:
                return torch.ones_like(parameter, dtype=torch.float32)
            return (v.detach().float() / correction).sqrt() + group.get("eps", 1e-8)
    raise ValueError("A probed model parameter is missing from the optimizer.")


def _nuclear_statistics(grads: Sequence[Tensor], b: int) -> tuple[Tensor, Tensor]:
    """Nuclear signal and squared trace of the square root of per-example covariance."""
    stack = torch.stack(grads).float()
    mean = stack.mean(0)
    residual = stack - mean
    # Nonzero singular values of C^(1/2) equal those of the centered K x (m*n)
    # matrix, scaled by sqrt(b/(K-1)); this avoids an m x m Gram allocation.
    noise_root = torch.linalg.svdvals(residual.reshape(len(grads), -1)).sum()
    noise = noise_root.square() * (b / (len(grads) - 1))
    signal = torch.linalg.svdvals(mean).sum().square()
    return signal, noise


def _l1_statistics(grads: Sequence[Tensor], b: int) -> tuple[Tensor, Tensor]:
    stack = torch.stack(grads).float()
    return stack.mean(0).abs().sum().square(), (stack.var(0, unbiased=True) * b).sqrt().sum().square()


def probe_batch(
    model: nn.Module,
    loss_fn: Callable[[nn.Module, Any], Tensor],
    batch: Any,
    *,
    splits: int = 2,
    optimizer: Optional[Optimizer] = None,
) -> BatchProbe:
    """Measure Euclidean, frozen AdamW, and spectral dual-norm noise scales.

    Split sizes must match. Each chunk is a disjoint, mean-reduced sample, with
    identical model parameters. The normal approximation Var(mean_B) = C/B
    presumes independent examples; correlated data or BatchNorm breaks it.
    AdamW uses detached, bias-corrected second moments (or unit weights before
    initialization). Muon uses 2D matrices only; other tensors are reported as
    a separate L1 fallback diagnostic. No gate values enter these sensors.
    """
    B = tree_batch_size(batch)
    if splits < 2 or B < splits or B % splits:
        raise ValueError("Probe requires at least two equal, nonempty microbatches.")
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training for m in model.modules()):
        raise ValueError("Training BatchNorm makes split gradients dependent on the split.")
    if any(isinstance(m, nn.modules.dropout._DropoutNd) and m.training for m in model.modules()):
        raise ValueError("Disable training Dropout during a probe to avoid augmentation noise.")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Probe requires trainable model parameters.")
    start = time.perf_counter()
    chunks = tree_split(batch, splits)
    gradients: dict[Tensor, list[Tensor]] = {p: [] for p in params}
    # autograd.grad leaves existing .grad untouched, including accumulated steps.
    for chunk in chunks:
        with torch.enable_grad():
            loss = loss_fn(model, chunk)
            if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
                raise ValueError("Probe loss must be a finite, mean-reduced scalar.")
            gs = torch.autograd.grad(loss, params, allow_unused=True)
        for p, g in zip(params, gs):
            gradients[p].append(torch.zeros_like(p) if g is None else g.detach())

    eu_s = eu_n = ad_s = ad_n = mu_s = mu_n = fb_s = fb_n = 0.0
    matrices = fallback = 0
    with torch.no_grad():
        for p, gs in gradients.items():
            b = B // splits
            s, n = _squared_norms(gs, b)
            eu_s += float(s)
            eu_n += float(n)
            denom = _adam_denominator(optimizer, p)
            s, n = _squared_norms([g.float() / denom for g in gs], b)
            ad_s += float(s)
            ad_n += float(n)
            if p.ndim == 2:
                s, n = _nuclear_statistics(gs, b)
                mu_s += float(s)
                mu_n += float(n)
                matrices += 1
            else:
                s, n = _l1_statistics(gs, b)
                fb_s += float(s)
                fb_n += float(n)
                fallback += 1
    return BatchProbe(B, splits, time.perf_counter() - start,
                      _ratio(torch.tensor(eu_s), torch.tensor(eu_n)),
                      _ratio(torch.tensor(ad_s), torch.tensor(ad_n)),
                      _ratio(torch.tensor(mu_s), torch.tensor(mu_n)) if matrices else None,
                      _ratio(torch.tensor(fb_s), torch.tensor(fb_n)) if fallback else None)


@dataclass(frozen=True)
class BatchDecision:
    batch_size: int
    reason: str
    estimated_scale: Optional[float]
    probe_seconds: Optional[float]


class BatchController:
    """Choose one rung per decision; the training loop supplies the actual batch.

    ``target_multiplier`` calibrates the chosen scale against a measured local
    batch-efficiency curve. It is not implied by gate density or noise alone.
    Call ``observe(None)`` on scheduled probe failures to hold the batch.
    """

    def __init__(self, sizes: Sequence[int], *, initial: int, sensor: str = "euclidean",
                 ema: float = 0.8, warmup: int = 5, dwell: int = 5,
                 deadband: float = 0.2, target_multiplier: float = 1.0,
                 max_probe_fraction: float = 0.5) -> None:
        if not sizes or tuple(sorted(set(sizes))) != tuple(sizes) or any(b <= 0 for b in sizes):
            raise ValueError("sizes must be strictly increasing positive integers.")
        if initial not in sizes or sensor not in ("euclidean", "adamw", "muon"):
            raise ValueError("Invalid initial size or sensor.")
        if not 0 <= ema < 1 or warmup < 0 or dwell < 0 or deadband < 0 or target_multiplier <= 0:
            raise ValueError("Invalid controller smoothing or threshold.")
        if max_probe_fraction <= 0:
            raise ValueError("max_probe_fraction must be positive.")
        self.sizes = tuple(sizes)
        self.current = initial
        self.sensor = sensor
        self.ema = ema
        self.warmup = warmup
        self.dwell = dwell
        self.deadband = deadband
        self.target_multiplier = target_multiplier
        self.max_probe_fraction = max_probe_fraction
        self.observations = 0
        self.last_change = 0
        self.signal: Optional[float] = None
        self.noise: Optional[float] = None
        self._reason = "no_probe"
        self._probe_seconds: Optional[float] = None
        self._decision: Optional[BatchDecision] = None

    def observe(self, probe: Optional[BatchProbe], *, step_seconds: Optional[float] = None) -> None:
        self.observations += 1
        self._decision = None
        self._probe_seconds = None if probe is None else probe.elapsed_seconds
        self._reason = "no_probe"
        if probe is None:
            return
        if step_seconds is not None and probe.elapsed_seconds > step_seconds * self.max_probe_fraction:
            self._reason = "expensive_probe"
            return
        value = getattr(probe, self.sensor)
        if value is None or not all(math.isfinite(v) for v in (value.signal, value.noise, value.scale)) or value.signal <= 0 or value.noise < 0:
            self._reason = "unstable_probe"
            return
        self.signal = value.signal if self.signal is None else self.ema * self.signal + (1 - self.ema) * value.signal
        self.noise = value.noise if self.noise is None else self.ema * self.noise + (1 - self.ema) * value.noise
        self._reason = "ready"

    def recommend(self) -> BatchDecision:
        if self._decision is not None:
            return self._decision
        scale = None if self.signal is None else self.noise / self.signal
        reason = self._reason
        if reason == "ready":
            if self.observations <= self.warmup:
                reason = "warmup"
            elif self.observations - self.last_change < self.dwell:
                reason = "dwell"
            else:
                target = scale * self.target_multiplier
                index = self.sizes.index(self.current)
                if target > self.current * (1 + self.deadband) and index < len(self.sizes) - 1:
                    self.current = self.sizes[index + 1]
                    self.last_change = self.observations
                    reason = "increase"
                elif target < self.current * (1 - self.deadband) and index > 0:
                    self.current = self.sizes[index - 1]
                    self.last_change = self.observations
                    reason = "decrease"
                else:
                    reason = "deadband_or_bound"
        self._decision = BatchDecision(self.current, reason, scale, self._probe_seconds)
        return self._decision
