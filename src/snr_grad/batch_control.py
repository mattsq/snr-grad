"""Training-loop batch-size diagnostics and a conservative discrete controller.

The probe owns separate backward passes at fixed parameters. It does not change
optimizer gradients or optimizer state. Call it before the training step, then
apply the controller's recommendation to the *next* training batch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .variance import tree_batch_size, tree_split

__all__ = ["BatchController", "BatchDecision", "BatchProbe", "CostAwareBatchController",
           "GradientNoiseAccumulator", "NoiseScale", "StepTimeModel", "coupled_lr", "probe_batch"]


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


def _ratio(signal: float, noise: float) -> NoiseScale:
    s, n = float(signal), float(noise)
    scale = n / s if s > 0 else math.inf
    return NoiseScale(s, n, scale)


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


class _Moments:
    """Streaming (Welford) moments of K equal-size microbatch gradients.

    Memory is one running mean and one per-coordinate second moment per
    parameter, plus an ``m x m`` row co-moment for each 2D matrix when the
    spectral sensor is enabled. Welford updates avoid the cancellation of
    ``sum g**2 - K mean**2`` when the signal dominates the noise.
    """

    def __init__(self, params: Sequence[Tensor], matrix_sensor: bool = True) -> None:
        self.params = list(params)
        self.count = 0
        self.mean = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.m2 = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.rows = [torch.zeros(p.shape[0], p.shape[0], dtype=torch.float32, device=p.device)
                     if matrix_sensor and p.ndim == 2 else None for p in self.params]

    @torch.no_grad()
    def update(self, gradients: Sequence[Tensor]) -> None:
        self.count += 1
        for i, g in enumerate(gradients):
            g = g.float()
            old = g - self.mean[i]
            self.mean[i].add_(old, alpha=1 / self.count)
            new = g - self.mean[i]
            self.m2[i].addcmul_(old, new)
            if self.rows[i] is not None:
                # Column-wise co-moment update: sum_k R_k R_k^T for m x n R_k.
                # Flattening each matrix would instead measure covariance
                # across microbatches, a different geometry.
                self.rows[i].addmm_(old, new.t())

    @torch.no_grad()
    def finish(self, batch_size: int, denominators: Sequence[Tensor],
               elapsed: float) -> BatchProbe:
        K = self.count
        if K < 2 or batch_size % K:
            raise ValueError("Noise statistics need at least two equal microbatches.")
        b = batch_size // K
        eu_s = eu_n = ad_s = ad_n = mu_s = mu_n = fb_s = fb_n = 0.0
        matrices = fallback = 0
        for mean, m2, rows, d in zip(self.mean, self.m2, self.rows, denominators):
            # Per-example covariance diagonal: Var(mean of b examples) = C / b.
            var = m2.clamp_min(0) / (K - 1) * b
            eu_s += float(mean.square().sum())
            eu_n += float(var.sum())
            ad_s += float((mean / d).square().sum())
            ad_n += float((var / d.square()).sum())
            if mean.ndim == 2:
                if rows is not None:
                    # Eigenvalues of sum_k R_k R_k^T are the squared singular
                    # values of the column-concatenated residuals.
                    eig = torch.linalg.eigvalsh(0.5 * (rows + rows.t())).clamp_min(0)
                    mu_n += float(eig.sqrt().sum().square()) * b / (K - 1)
                    mu_s += float(torch.linalg.svdvals(mean).sum().square())
                    matrices += 1
            else:
                fb_s += float(mean.abs().sum().square())
                fb_n += float(var.sqrt().sum().square())
                fallback += 1
        # For quadratic norms, E||g_B||² = ||E g||² + tr(C)/B. Correct the
        # finite-probe upward bias in the signal; a nonpositive result means the
        # probe cannot identify signal and the controller must hold. The
        # nuclear and L1 dual norms have no such simple correction.
        eu_s = max(0., eu_s - eu_n / batch_size)
        ad_s = max(0., ad_s - ad_n / batch_size)
        return BatchProbe(batch_size, K, elapsed,
                          _ratio(eu_s, eu_n), _ratio(ad_s, ad_n),
                          _ratio(mu_s, mu_n) if matrices else None,
                          _ratio(fb_s, fb_n) if fallback else None)


def _nuclear_statistics(grads: Sequence[Tensor], b: int) -> tuple[float, float]:
    """Nuclear signal and squared trace of the square root of row covariance."""
    moments = _Moments([grads[0]])
    for g in grads:
        moments.update([g])
    probe = moments.finish(b * len(grads), [torch.ones_like(grads[0])], 0.)
    return probe.muon.signal, probe.muon.noise


def _check_probe_model(model: nn.Module) -> list[Tensor]:
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training for m in model.modules()):
        raise ValueError("Training BatchNorm makes split gradients dependent on the split.")
    if any(isinstance(m, nn.modules.dropout._DropoutNd) and m.training for m in model.modules()):
        raise ValueError("Disable training Dropout during a probe to avoid augmentation noise.")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Probe requires trainable model parameters.")
    return params


def probe_batch(
    model: nn.Module,
    loss_fn: Callable[[nn.Module, Any], Tensor],
    batch: Any,
    *,
    splits: int = 2,
    optimizer: Optional[Optimizer] = None,
    matrix_sensor: bool = True,
) -> BatchProbe:
    """Measure Euclidean, frozen AdamW, and spectral dual-norm noise scales.

    Split sizes must match. Each chunk is a disjoint, mean-reduced sample, with
    identical model parameters. The normal approximation Var(mean_B) = C/B
    presumes independent examples; correlated data or BatchNorm breaks it.
    AdamW uses detached, bias-corrected second moments (or unit weights before
    initialization). Muon uses 2D matrices only; other tensors are reported as
    a separate L1 fallback diagnostic. No gate values enter these sensors.

    This costs extra forward and backward passes on examples that do not train
    the model. Prefer :class:`GradientNoiseAccumulator`, which measures the same
    statistics from the training step's own accumulation microbatches.
    """
    B = tree_batch_size(batch)
    if splits < 2 or B < splits or B % splits:
        raise ValueError("Probe requires at least two equal, nonempty microbatches.")
    params = _check_probe_model(model)
    start = time.perf_counter()
    moments = _Moments(params, matrix_sensor)
    # autograd.grad leaves existing .grad untouched, including accumulated steps.
    for chunk in tree_split(batch, splits):
        with torch.enable_grad():
            loss = loss_fn(model, chunk)
            if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
                raise ValueError("Probe loss must be a finite, mean-reduced scalar.")
            gs = torch.autograd.grad(loss, params, allow_unused=True)
        moments.update([torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, gs)])
    denominators = [_adam_denominator(optimizer, p) for p in params]
    return moments.finish(B, denominators, time.perf_counter() - start)


class GradientNoiseAccumulator:
    """Measure noise scales from the training step's own accumulation passes.

    Use it with ordinary gradient accumulation, dividing each microbatch loss
    by ``microbatches``::

        noise = GradientNoiseAccumulator(model.parameters(), microbatches=K,
                                         optimizer=optimizer)
        for chunk in tree_split(batch, K):
            (loss_fn(model, chunk) / K).backward()
            noise.record()
        probe = noise.finish(batch_size)
        optimizer.step()

    Each microbatch gradient is recovered from the change in ``.grad``, so no
    extra forward or backward pass is needed and every example trains the
    model. The overhead is two parameter-sized buffers, elementwise updates,
    and, for the spectral sensor, one ``m x m`` product per matrix per
    microbatch plus one eigendecomposition per matrix in :meth:`finish`.
    Construct it after ``zero_grad`` or with the gradients you intend to keep;
    AdamW denominators are frozen at construction, i.e. before the step.

    With a ``GradScaler``, pass the current scale to :meth:`record`. Under DDP,
    the backward that synchronizes gradients averages across ranks, so record
    only inside ``no_sync`` or aggregate per-rank statistics separately.
    """

    def __init__(self, params: Any, microbatches: int, *, optimizer: Optional[Optimizer] = None,
                 matrix_sensor: bool = True) -> None:
        if microbatches < 2:
            raise ValueError("At least two microbatches are needed to estimate noise.")
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("No trainable parameters.")
        self.microbatches = microbatches
        start = time.perf_counter()
        self.denominators = [_adam_denominator(optimizer, p) for p in self.params]
        self.previous = [torch.zeros_like(p, dtype=torch.float32) if p.grad is None
                         else p.grad.detach().float().clone() for p in self.params]
        self.moments = _Moments(self.params, matrix_sensor)
        self.elapsed = time.perf_counter() - start

    @torch.no_grad()
    def record(self, grad_scale: float = 1.0) -> None:
        """Record the microbatch whose backward pass has just finished."""
        if self.moments.count >= self.microbatches:
            raise RuntimeError("More microbatches recorded than declared.")
        start = time.perf_counter()
        gradients = []
        for p, prev in zip(self.params, self.previous):
            current = torch.zeros_like(prev) if p.grad is None else p.grad.detach().float()
            # The loss was divided by K, so K times the change is this
            # microbatch's own mean gradient.
            gradients.append((current - prev) * (self.microbatches / grad_scale))
            prev.copy_(current)
        self.moments.update(gradients)
        self.elapsed += time.perf_counter() - start

    def finish(self, batch_size: int) -> BatchProbe:
        """Return the probe for a step of ``batch_size`` equally split examples."""
        if self.moments.count != self.microbatches:
            raise RuntimeError("Record every declared microbatch before finishing.")
        start = time.perf_counter()
        probe = self.moments.finish(batch_size, self.denominators, 0.)
        elapsed = self.elapsed + time.perf_counter() - start
        return BatchProbe(probe.batch_size, probe.microbatches, elapsed, probe.euclidean,
                          probe.adamw, probe.muon, probe.muon_fallback)


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

    This multiplier rule has no notion of what a step or an example costs, so
    it can only be tuned for one budget at a time. Prefer
    :class:`CostAwareBatchController`, which makes the cost explicit.
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


def coupled_lr(base_lr: float, batch_size: int, reference_batch: int, rule: str = "sqrt") -> float:
    """Scale a learning rate tuned at ``reference_batch`` to ``batch_size``.

    ``"linear"`` is the usual SGD rule, ``"sqrt"`` the usual Adam-family rule,
    and ``"none"`` keeps the rate fixed. None of them is guaranteed to be right
    for a given optimizer; treat the rule as an experimental factor.
    """
    if base_lr <= 0 or batch_size <= 0 or reference_batch <= 0:
        raise ValueError("Learning rate and batch sizes must be positive.")
    ratio = batch_size / reference_batch
    if rule == "linear":
        return base_lr * ratio
    if rule == "sqrt":
        return base_lr * math.sqrt(ratio)
    if rule == "none":
        return base_lr
    raise ValueError("rule must be 'linear', 'sqrt', or 'none'.")


class StepTimeModel:
    """Seconds per optimizer step for each candidate batch size.

    Seed it with measured times (for example a short timing pass before
    training) and keep it current with :meth:`record`. Unmeasured sizes are
    linearly interpolated between measured ones, and extrapolated from the
    nearest two; accelerator step time is usually flat and then linear in the
    batch, so measure every candidate when possible.
    """

    def __init__(self, initial: Optional[Mapping[int, float]] = None, *, ema: float = 0.9) -> None:
        if not 0 <= ema < 1:
            raise ValueError("ema must lie in [0, 1).")
        self.ema = ema
        self.seconds: dict[int, float] = {}
        for batch, seconds in (initial or {}).items():
            self.record(int(batch), float(seconds))

    def record(self, batch_size: int, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            return
        old = self.seconds.get(batch_size)
        self.seconds[batch_size] = seconds if old is None else self.ema * old + (1 - self.ema) * seconds

    def __call__(self, batch_size: int) -> Optional[float]:
        if batch_size in self.seconds:
            return self.seconds[batch_size]
        known = sorted(self.seconds)
        if len(known) < 2:
            return None
        lower = [b for b in known if b < batch_size]
        upper = [b for b in known if b > batch_size]
        # Interpolate between neighbours, or extrapolate from the nearest two.
        a, c = ((lower[-1], upper[0]) if lower and upper else
                tuple(lower[-2:]) if lower else tuple(upper[:2]))
        ta, tc = self.seconds[a], self.seconds[c]
        value = ta + (tc - ta) * (batch_size - a) / (c - a)
        return value if value > 0 else None


class CostAwareBatchController(BatchController):
    """Choose the batch that maximizes expected progress per unit of cost.

    With gradient noise scale ``B_noise``, a step at batch ``B`` makes about
    ``B / (B + B_noise)`` of the progress of a noise-free step (McCandlish et
    al., 2018). The controller prices one step at

        ``cost(B) = time_price * seconds(B) + example_price * B + step_price``

    and moves one rung toward ``argmax_B progress(B) / cost(B)``. This replaces
    the fitted ``target_multiplier``: under a pure example price the smallest
    batch always wins, under a pure step price the largest does, and only a
    cost with both a per-step and a per-example part, as on an accelerator
    whose step time is flat until it saturates, gives an interior optimum
    near ``sqrt(B_noise * fixed_cost / per_example_cost)``.

    The progress law is derived for SGD with a locally optimal step size. For
    Adam or Muon the sensor argument selects which geometry supplies
    ``B_noise``; whether that improves decisions is the experimental question.
    ``deadband`` is the minimum relative efficiency gain before moving.
    """

    def __init__(self, sizes: Sequence[int], *, initial: int, sensor: str = "euclidean",
                 step_times: Optional[StepTimeModel] = None, time_price: float = 1.0,
                 example_price: float = 0.0, step_price: float = 0.0,
                 ema: float = 0.9, warmup: int = 5, dwell: int = 5, deadband: float = 0.05,
                 max_probe_fraction: float = 0.5) -> None:
        super().__init__(sizes, initial=initial, sensor=sensor, ema=ema, warmup=warmup,
                         dwell=dwell, deadband=deadband, max_probe_fraction=max_probe_fraction)
        if min(time_price, example_price, step_price) < 0 or not (time_price or example_price or step_price):
            raise ValueError("Prices must be nonnegative and not all zero.")
        if time_price and step_times is None:
            raise ValueError("A time price needs a StepTimeModel.")
        self.step_times = step_times
        self.time_price = time_price
        self.example_price = example_price
        self.step_price = step_price

    def cost(self, batch_size: int) -> Optional[float]:
        cost = self.example_price * batch_size + self.step_price
        if self.time_price:
            seconds = self.step_times(batch_size)
            if seconds is None:
                return None
            cost += self.time_price * seconds
        return cost

    def efficiency(self, batch_size: int, noise_scale: float) -> Optional[float]:
        cost = self.cost(batch_size)
        if cost is None or cost <= 0:
            return None
        return batch_size / (batch_size + noise_scale) / cost

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
                scores = {b: self.efficiency(b, scale) for b in self.sizes}
                scores = {b: v for b, v in scores.items() if v is not None}
                here = scores.get(self.current)
                if here is None or not scores:
                    reason = "unknown_cost"
                else:
                    best = max(scores, key=lambda b: (scores[b], -b))
                    index = self.sizes.index(self.current)
                    if best != self.current and scores[best] > here * (1 + self.deadband):
                        step = 1 if best > self.current else -1
                        self.current = self.sizes[index + step]
                        self.last_change = self.observations
                        reason = "increase" if step > 0 else "decrease"
                    else:
                        reason = "deadband_or_optimal"
        self._decision = BatchDecision(self.current, reason, scale, self._probe_seconds)
        return self._decision
