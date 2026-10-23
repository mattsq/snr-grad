"""
Checkpoint-local stochastic continuation simulator (research tool).

Section 5 of ``docs/proposals/curvature-aware-stability-control.md``. At a
checkpoint ``theta0`` the loss is replaced by a quadratic surrogate and the
gradient by a stochastic synthetic one::

    L_hat(theta) = L0 + g0^T (theta - theta0) + 0.5 (theta - theta0)^T H (theta - theta0)
    g_sim(theta) = g0 + H (theta - theta0) + noise(theta, t, seed)

The simulator clones the parameters and **all** optimizer state, then executes
the optimizer's *real* ``step`` (Adam moments, SNR variance EMA and gate,
adaptive thresholds, bias correction, weight decay, Muon orthogonalization...)
on the synthetic gradients. No model forward/backward runs per simulated step,
and the live model and optimizer are never touched.

It answers *distributional* questions ("how often / how large an excursion under
this configuration?"). It does not predict when a real spike will occur, and a
collection of independent layer surrogates is not the full network loss.
Validate rankings against held-out real continuations before trusting them.
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import torch
from torch import Tensor, nn

from snr_grad.activation import ActivationCovarianceSnapshot, DoPr
from snr_grad.curvature import CurvatureProxy, LowRankCurvature, flatten_params, unflatten_like
from snr_grad.stability import ExcursionReport, StabilityProbe, detect_excursions

__all__ = [
    "SimulationNoise",
    "GaussianProbeNoise",
    "ResampledProbeNoise",
    "CovarianceFactorNoise",
    "SimulationDistribution",
    "QuadraticStabilitySimulator",
    "clone_optimizer",
]


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------

class SimulationNoise(Protocol):
    """``noise(delta, loss_ratio, step, generator) -> flat noise vector``."""

    def __call__(
        self, delta: Tensor, loss_ratio: float, step: int, generator: torch.Generator
    ) -> Tensor:
        ...


def _heavy_tail_multiplier(dof: Optional[float], generator: torch.Generator) -> float:
    """Variance-preserving multivariate Student-t scale (1 for Gaussian)."""
    if dof is None:
        return 1.0
    # chi2(dof) = 2 * Gamma(dof/2); sampled from our private generator.
    u = _gamma_sample(dof / 2.0, generator) * 2.0
    return math.sqrt((dof - 2.0) / u)


def _gamma_sample(shape: float, generator: torch.Generator) -> float:
    """Marsaglia-Tsang Gamma(shape, 1) sample using only ``generator``."""
    if shape < 1.0:
        u = float(torch.rand((), generator=generator))
        return _gamma_sample(shape + 1.0, generator) * u ** (1.0 / shape)
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = float(torch.randn((), generator=generator))
        v = (1.0 + c * x) ** 3
        if v <= 0:
            continue
        u = float(torch.rand((), generator=generator))
        if math.log(u) < 0.5 * x * x + d - d * v + d * math.log(v):
            return d * v


class _ScaledNoise:
    def __init__(
        self,
        *,
        batch_size: int,
        heavy_tail_dof: Optional[float],
        scaling: str,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if heavy_tail_dof is not None and heavy_tail_dof <= 2:
            raise ValueError("heavy_tail_dof must be > 2 (finite variance).")
        if scaling not in {"fixed", "loss_proportional"}:
            raise ValueError(f"scaling must be 'fixed' or 'loss_proportional', got {scaling!r}.")
        self.batch_size = batch_size
        self.heavy_tail_dof = heavy_tail_dof
        self.scaling = scaling

    def _scale(self, loss_ratio: float, generator: torch.Generator) -> float:
        s = _heavy_tail_multiplier(self.heavy_tail_dof, generator)
        if self.scaling == "loss_proportional":
            s *= math.sqrt(max(loss_ratio, 0.0))
        return s


class GaussianProbeNoise(_ScaledNoise):
    """Noise with covariance ``Sigma_hat / B`` fitted from probe deviations.

    ``Sigma_hat = b/(K-1) * sum_k (g_k - g_bar)(g_k - g_bar)^T`` (rank ``<= K-1``);
    a sample is ``sqrt(b / ((K-1) B)) * sum_k z_k (g_k - g_bar)`` with ``z ~ N(0, I)``,
    which has exactly that covariance without materializing it.

    Args:
        deviations: ``[K, n]`` flattened raw deviations ``g_k - g_bar``.
        microbatch_size: ``b``.
        batch_size: Simulated training batch ``B``.
        heavy_tail_dof: If set, a variance-preserving multivariate Student-t
            with these degrees of freedom (failure-mode check for Gaussianity).
        scaling: ``"fixed"`` (baseline) or ``"loss_proportional"`` (std scales with
            ``sqrt(L_hat / L0)``, as in random-feature models where per-example
            gradient variance tracks the residual).
    """

    def __init__(
        self,
        deviations: Tensor,
        microbatch_size: int,
        batch_size: int,
        *,
        heavy_tail_dof: Optional[float] = None,
        scaling: str = "fixed",
    ) -> None:
        super().__init__(batch_size=batch_size, heavy_tail_dof=heavy_tail_dof, scaling=scaling)
        if deviations.ndim != 2 or deviations.shape[0] < 2:
            raise ValueError("deviations must be [K >= 2, n].")
        self.deviations = deviations.detach()
        k = deviations.shape[0]
        self._coef = math.sqrt(microbatch_size / ((k - 1) * batch_size))

    def __call__(
        self, delta: Tensor, loss_ratio: float, step: int, generator: torch.Generator
    ) -> Tensor:
        # Sampled on the (CPU) private generator, then moved to the stored tensor's device.
        z = torch.randn(
            self.deviations.shape[0], generator=generator, dtype=self.deviations.dtype
        ).to(self.deviations.device)
        return self._scale(loss_ratio, generator) * self._coef * (z @ self.deviations)


class ResampledProbeNoise(_ScaledNoise):
    """Non-Gaussian noise that resamples the observed microbatch deviations.

    Gaussian noise with the right covariance can badly understate tail risk when
    the real gradient noise is heavy-tailed (e.g. products of Gaussians in random
    feature models). This model keeps the empirical shape: a sample is

        sqrt(K/(K-1)) * sqrt(b / (B m)) * sum_{j=1..m} s_j d_{k_j},   m = max(1, round(B/b))

    with ``k_j`` drawn uniformly with replacement from the ``K`` deviations
    ``d_k = g_k - g_bar`` and ``s_j = 1`` (or random signs if ``sign_flip``). Its
    covariance is ``Sigma_hat / B`` like :class:`GaussianProbeNoise`; when ``B == b``
    it is exactly the (centering-corrected) empirical distribution of microbatch
    noise. Like any probe-fitted model it only spans the ``K - 1`` sampled
    directions and is frozen at the checkpoint apart from ``scaling``.

    Args:
        deviations: ``[K, n]`` flattened raw deviations ``g_k - g_bar``.
        microbatch_size: ``b``.
        batch_size: Simulated training batch ``B``.
        sign_flip: Symmetrize by random signs (removes skew; off by default).
        scaling: ``"fixed"`` or ``"loss_proportional"``.
    """

    def __init__(
        self,
        deviations: Tensor,
        microbatch_size: int,
        batch_size: int,
        *,
        sign_flip: bool = False,
        scaling: str = "fixed",
    ) -> None:
        super().__init__(batch_size=batch_size, heavy_tail_dof=None, scaling=scaling)
        if deviations.ndim != 2 or deviations.shape[0] < 2:
            raise ValueError("deviations must be [K >= 2, n].")
        self.deviations = deviations.detach()
        k = deviations.shape[0]
        self.m = max(1, round(batch_size / microbatch_size))
        self.sign_flip = sign_flip
        self._coef = math.sqrt(k / (k - 1)) * math.sqrt(microbatch_size / (batch_size * self.m))

    def __call__(
        self, delta: Tensor, loss_ratio: float, step: int, generator: torch.Generator
    ) -> Tensor:
        k = self.deviations.shape[0]
        idx = torch.randint(0, k, (self.m,), generator=generator).to(self.deviations.device)
        picked = self.deviations[idx]
        if self.sign_flip:
            signs = torch.randint(0, 2, (self.m, 1), generator=generator).to(picked.device, picked.dtype) * 2 - 1
            picked = picked * signs
        return self._scale(loss_ratio, generator) * self._coef * picked.sum(0)


class CovarianceFactorNoise(_ScaledNoise):
    """Noise ``F z / sqrt(B)`` for a known per-example covariance ``Sigma = F F^T``.

    Use on quadratic toys where the true per-example gradient covariance is
    available (e.g. PLRF), to separate simulator error from probe error.
    """

    def __init__(
        self,
        factor: Tensor,
        batch_size: int,
        *,
        heavy_tail_dof: Optional[float] = None,
        scaling: str = "fixed",
    ) -> None:
        super().__init__(batch_size=batch_size, heavy_tail_dof=heavy_tail_dof, scaling=scaling)
        self.factor = factor.detach()

    def __call__(
        self, delta: Tensor, loss_ratio: float, step: int, generator: torch.Generator
    ) -> Tensor:
        z = torch.randn(self.factor.shape[1], generator=generator, dtype=self.factor.dtype).to(self.factor.device)
        return self._scale(loss_ratio, generator) * (self.factor @ z) / math.sqrt(self.batch_size)


class _SubspaceNoise:
    """Run a delta-independent noise model in ``r`` subspace coordinates.

    ``inner`` produces coefficients on the basis ``V`` (its deviations / factor were
    pre-multiplied by ``V^T``); a sample is ``V @ inner(...)``. Sampling and
    projection then cost ``O(K r + n r)`` per step instead of ``O(K n + 2 n r)``.
    """

    def __init__(self, inner: Any, basis: Tensor) -> None:
        self.inner = inner
        self.basis = basis

    def __call__(
        self, delta: Tensor, loss_ratio: float, step: int, generator: torch.Generator
    ) -> Tensor:
        coef = self.inner(None, loss_ratio, step, generator)
        return self.basis @ coef.to(self.basis.dtype)


def _subspace_noise(noise: Any, basis: Tensor) -> Optional[_SubspaceNoise]:
    """Projected copy of a known, delta-independent noise model (else ``None``)."""
    if isinstance(noise, (GaussianProbeNoise, ResampledProbeNoise)):
        inner = copy.copy(noise)
        inner.deviations = noise.deviations.to(basis.dtype) @ basis
        return _SubspaceNoise(inner, basis)
    if isinstance(noise, CovarianceFactorNoise):
        inner = copy.copy(noise)
        inner.factor = basis.t() @ noise.factor.to(basis.dtype)
        return _SubspaceNoise(inner, basis)
    return None


# ---------------------------------------------------------------------------
# Optimizer cloning
# ---------------------------------------------------------------------------

def clone_optimizer(optimizer: Any, params: Sequence[Tensor]) -> tuple[Any, list[Tensor]]:
    """Deep-copy ``optimizer`` onto fresh leaf clones of ``params``.

    Every tensor in the optimizer state and param groups is copied; the copy's
    param groups reference the clones, so stepping it never touches the live
    parameters or state. Parameters of the optimizer that are not in ``params``
    are cloned too (they simply never receive simulated gradients).

    Returns:
        ``(optimizer_clone, clones)`` with ``clones[i]`` the clone of ``params[i]``.
    """
    if isinstance(optimizer, DoPr):
        raise TypeError("clone the DoPr base optimizer instead (see QuadraticStabilitySimulator).")
    memo: dict[int, Any] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) not in memo:
                c = p.detach().clone().requires_grad_(p.requires_grad)
                memo[id(p)] = c
    for p in params:
        if id(p) not in memo:
            raise ValueError("every simulated parameter must belong to the optimizer.")
    clone = copy.deepcopy(optimizer, memo)
    return clone, [memo[id(p)] for p in params]


class _FlatClone:
    """Optimizer clone whose simulated parameters / grads are views of flat buffers.

    Avoids a concatenate and a gradient clone per simulated step. ``aliased()``
    checks (by data pointer, O(#params)) that the optimizer still updates the
    views in place; callers fall back to concatenation otherwise.
    """

    def __init__(self, optimizer: Any, params: Sequence[Tensor]) -> None:
        params = list(params)
        dtypes = {p.dtype for p in params}
        devices = {p.device for p in params}
        if len(dtypes) != 1 or len(devices) != 1:
            raise ValueError("flat cloning needs a single dtype and device.")
        self.flat = torch.cat([p.detach().reshape(-1) for p in params]).clone()
        self.grad = torch.zeros_like(self.flat)
        memo: dict[int, Any] = {}
        self.clones: list[Tensor] = []
        self.grad_views: list[Optional[Tensor]] = []
        o = 0
        for p in params:
            n = p.numel()
            c = nn.Parameter(self.flat[o:o + n].view(p.shape), requires_grad=p.requires_grad)
            gv = self.grad[o:o + n].view(p.shape) if p.requires_grad else None
            memo[id(p)] = c
            self.clones.append(c)
            self.grad_views.append(gv)
            o += n
        for group in optimizer.param_groups:
            for p in group["params"]:
                if id(p) not in memo:
                    memo[id(p)] = p.detach().clone().requires_grad_(p.requires_grad)
        self.optimizer = copy.deepcopy(optimizer, memo)
        self._ptrs = [c.data_ptr() for c in self.clones]

    def attach_grads(self) -> None:
        for c, gv in zip(self.clones, self.grad_views):
            if c.grad is not gv:
                c.grad = gv

    def aliased(self) -> bool:
        return all(c.data_ptr() == ptr for c, ptr in zip(self.clones, self._ptrs))

    def params_flat(self) -> Tensor:
        if self.aliased():
            return self.flat
        return torch.cat([c.detach().reshape(-1) for c in self.clones])


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class SimulationDistribution:
    """Distribution of simulated continuations from one checkpoint.

    Attributes:
        losses: ``[n_seeds, horizon + 1]`` surrogate loss (index 0 = checkpoint).
        deterministic: ``[horizon + 1]`` noise-free continuation with the same
            optimizer (the expected-progress reference).
        max_rel_excess: ``[n_seeds]`` ``max_t (L_t / L_det_t - 1)`` -- largest
            noise-driven excess over the noiseless path.
        excursions: Per-seed :class:`ExcursionReport` (shared spike definition).
        mean_gate: ``[n_seeds, horizon]`` mean SNR gate per step when the optimizer
            tracks stats, else ``None``.
        mode_projections: ``[n_seeds, horizon + 1, m]`` displacement projected on
            the supplied modes, else ``None``.
        thresholds: The preregistered excess thresholds for ``exceedance``.
        config: Simulation settings for reproducibility.
    """

    losses: Tensor
    deterministic: Tensor
    max_rel_excess: Tensor
    excursions: list[ExcursionReport]
    mean_gate: Optional[Tensor]
    mode_projections: Optional[Tensor]
    thresholds: tuple[float, ...]
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        n = self.losses.shape[0]
        finite = torch.isfinite(self.max_rel_excess)
        exc = self.max_rel_excess.clone()
        exc[~finite] = float("inf")
        q = torch.quantile(exc.clamp_max(1e30), torch.tensor([0.5, 0.9, 0.95], dtype=exc.dtype))
        final = self.losses[:, -1]
        return {
            "n_seeds": n,
            "spike_fraction": sum(r.spiked for r in self.excursions) / n,
            "excursion_severity_mean": sum(
                min(r.max_excursion, 100.0) for r in self.excursions
            ) / n,
            "mean_events": sum(r.num_events for r in self.excursions) / n,
            "divergence_fraction": sum(r.diverged for r in self.excursions) / n,
            "exceedance": {
                t: float((exc > t).to(torch.float64).mean()) for t in self.thresholds
            },
            "max_rel_excess_median": float(q[0]),
            "max_rel_excess_q90": float(q[1]),
            "max_rel_excess_q95": float(q[2]),
            "final_loss_median": float(final[torch.isfinite(final)].median())
            if bool(torch.isfinite(final).any()) else float("nan"),
            "deterministic_final_loss": float(self.deterministic[-1]),
            "expected_progress": float(self.losses[0, 0] - self.deterministic[-1]),
            "deterministic_min_loss": float(self.deterministic.min()),
            # A non-negative loss whose noiseless surrogate path goes below zero means
            # the quadratic model is unreliable here (e.g. truncated curvature).
            "surrogate_below_zero": bool(self.losses[0, 0] >= 0 and float(self.deterministic.min()) < 0),
            "note": "rank configurations; not a calibrated spike probability",
        }


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class QuadraticStabilitySimulator:
    """Seeded optimizer-state replay on a checkpoint-local quadratic surrogate.

    Args:
        params: Live parameters spanning the surrogate (their checkpoint values
            are ``theta0``). Only read.
        optimizer: The live optimizer (or :class:`DoPr`). Cloned per seed, never
            stepped. For DoPr the activation covariance is frozen at
            ``activation_snapshot`` (default: its current snapshot) and applied
            before the cloned base step.
        curvature: Curvature proxy supplying ``matvec`` over ``params``.
        grad0: Gradient ``g0`` at the checkpoint (dict or flat vector).
        noise: A :class:`SimulationNoise` (``None`` = deterministic).
        loss0: Surrogate loss at the checkpoint (only shifts the curve; must be
            positive for relative excursions to be meaningful).
        subspace: ``[n, r]`` orthonormal basis onto which ``g0`` and the noise are
            projected, or ``"auto"`` (default: the eigenvectors of a
            :class:`~snr_grad.curvature.LowRankCurvature`, otherwise no projection).
            Without it a truncated curvature leaves ``g0`` components with almost no
            restoring curvature and the surrogate loss falls without bound (on an
            MLP it went from 0.06 to -1.08), which makes relative excursions
            meaningless. ``None`` disables projection.
    """

    def __init__(
        self,
        params: Sequence[Tensor],
        optimizer: Any,
        curvature: CurvatureProxy,
        grad0: Mapping[Tensor, Tensor] | Tensor,
        noise: Optional[SimulationNoise],
        *,
        loss0: float = 1.0,
        activation_snapshot: Optional[Mapping[str, ActivationCovarianceSnapshot]] = None,
        subspace: Optional[Tensor] | str = "auto",
    ) -> None:
        self.params = list(params)
        self.optimizer = optimizer
        self.curvature = curvature
        self.theta0 = torch.cat([p.detach().reshape(-1).clone() for p in self.params])
        if isinstance(grad0, Tensor):
            self.grad0 = grad0.detach().reshape(-1).clone()
        else:
            self.grad0 = flatten_params(grad0, self.params).detach().clone()
        if self.grad0.numel() != self.theta0.numel():
            raise ValueError("grad0 does not match params.")
        if isinstance(subspace, str):
            if subspace != "auto":
                raise ValueError(f"subspace must be a tensor, None or 'auto', got {subspace!r}.")
            subspace = getattr(curvature, "eigenvectors", None)
        self.subspace = None if subspace is None else subspace.detach().to(self.grad0.dtype)
        if self.subspace is not None:
            if self.subspace.shape[0] != self.grad0.numel():
                raise ValueError("subspace must be [n, r] with n matching params.")
            self.grad0 = self._project(self.grad0)
        self.noise = noise
        # Delta-independent probe noise is pre-projected onto the subspace once.
        self._step_noise = noise
        self._noise_projected = False
        if noise is not None and self.subspace is not None:
            sub = _subspace_noise(noise, self.subspace)
            if sub is not None:
                self._step_noise, self._noise_projected = sub, True
        self.loss0 = float(loss0)
        self._ap = None
        self._ap_sigma: dict[Tensor, ActivationCovarianceSnapshot] = {}
        if isinstance(optimizer, DoPr):
            self._ap = optimizer.ap
            snap = activation_snapshot or optimizer.ap.covariance_snapshot(source="auto")
            self._ap_sigma = {s.weight: s for s in snap.values()}
            self._ap_active = optimizer.ap.step_count + 1 > optimizer.ap.config.warmup_steps

    @classmethod
    def from_checkpoint(
        cls,
        optimizer: Any,
        probe: StabilityProbe,
        curvature: CurvatureProxy,
        *,
        model: Optional[nn.Module] = None,
        params: Optional[Sequence[Tensor]] = None,
        batch_size: Optional[int] = None,
        loss0: float = 1.0,
        grad0: Optional[Mapping[Tensor, Tensor]] = None,
        heavy_tail_dof: Optional[float] = None,
        noise_scaling: str = "fixed",
        noise_model: str = "gaussian",
        **kwargs: Any,
    ) -> "QuadraticStabilitySimulator":
        """Fit the noise model from ``probe`` (``g0`` defaults to the probe mean).

        ``batch_size`` is the simulated training batch ``B`` (default ``K*b``).
        ``noise_model`` is ``"gaussian"`` (:class:`GaussianProbeNoise`) or
        ``"resample"`` (:class:`ResampledProbeNoise`, keeps non-Gaussian tails).
        """
        if params is None:
            params = probe.params
        params = list(params)
        if model is not None:
            live = set(id(p) for p in model.parameters())
            if any(id(p) not in live for p in params):
                raise ValueError("probe parameters do not belong to `model`.")
        devs = torch.stack([
            flatten_params(d, params).to(torch.float64) for d in probe.deviations()
        ]).to(params[0].dtype)
        big_b = batch_size if batch_size is not None else probe.num_microbatches * probe.microbatch_size
        if noise_model == "gaussian":
            noise: SimulationNoise = GaussianProbeNoise(
                devs, probe.microbatch_size, big_b,
                heavy_tail_dof=heavy_tail_dof, scaling=noise_scaling,
            )
        elif noise_model == "resample":
            if heavy_tail_dof is not None:
                raise ValueError("heavy_tail_dof applies to the Gaussian model only.")
            noise = ResampledProbeNoise(devs, probe.microbatch_size, big_b, scaling=noise_scaling)
        else:
            raise ValueError(f"noise_model must be 'gaussian' or 'resample', got {noise_model!r}.")
        return cls(
            params, optimizer, curvature,
            grad0 if grad0 is not None else probe.mean_gradient(),
            noise, loss0=loss0, **kwargs,
        )

    # -- surrogate ------------------------------------------------------------

    def _project(self, v: Tensor) -> Tensor:
        if self.subspace is None:
            return v
        basis = self.subspace.to(v.dtype)
        return basis @ (basis.t() @ v)

    def _hvp(self, delta: Tensor) -> Tensor:
        flat_mv = getattr(self.curvature, "matvec_flat", None)
        if flat_mv is not None:
            return flat_mv(delta).to(delta.dtype)
        hv = self.curvature.matvec(unflatten_like(delta, self.params))
        return flatten_params(hv, self.params).to(delta.dtype)

    def surrogate_loss(self, delta: Tensor, hdelta: Optional[Tensor] = None) -> float:
        """``L_hat(theta0 + delta)`` (accumulated in the parameters' dtype)."""
        if hdelta is None:
            hdelta = self._hvp(delta)
        return self.loss0 + float(torch.dot(self.grad0, delta)) + 0.5 * float(
            torch.dot(delta, hdelta)
        )

    # -- execution ------------------------------------------------------------

    def _clone(self) -> _FlatClone:
        base = self.optimizer.base if isinstance(self.optimizer, DoPr) else self.optimizer
        return _FlatClone(base, self.params)

    def _apply_ap(self, clones: list[Tensor]) -> None:
        if self._ap is None or not self._ap_active:
            return
        for p, c in zip(self.params, clones):
            snap = self._ap_sigma.get(p)
            if snap is None or c.grad is None:
                continue
            if snap.kind == "linear":
                m = self._ap._solve_linear(snap.sigma.to(c.device), c.grad)
            else:
                m = self._ap._apply_embedding(snap.sigma.to(c.device), c.grad)
            c.grad.copy_(m.to(c.grad.dtype))

    def _set_lr(self, opt: Any, lr: Optional[float], mult: float) -> None:
        for group in opt.param_groups:
            if "_sim_base_lr" not in group:
                group["_sim_base_lr"] = group["lr"] if lr is None else lr
            group["lr"] = group["_sim_base_lr"] * mult

    def _trajectory(
        self,
        horizon: int,
        generator: Optional[torch.Generator],
        lr: Optional[float],
        lr_schedule: Optional[Callable[[int], float]],
        modes: Optional[Tensor],
        divergence_factor: float,
    ) -> tuple[list[float], list[float], Optional[Tensor]]:
        if self._fused_low_rank(generator):
            return self._trajectory_low_rank(horizon, generator, lr, lr_schedule, modes,
                                             divergence_factor)
        fc = self._clone()
        opt, clones, gbuf = fc.optimizer, fc.clones, fc.grad
        delta = torch.zeros_like(self.theta0)
        hdelta = torch.zeros_like(self.theta0)
        losses = [self.surrogate_loss(delta, hdelta)]
        gates: list[float] = []
        projs = [delta @ modes] if modes is not None else None
        for t in range(horizon):
            torch.add(self.grad0, hdelta, out=gbuf)
            if self.noise is not None and generator is not None:
                ratio = losses[-1] / self.loss0 if self.loss0 > 0 else 1.0
                eps = self._step_noise(delta, ratio, t, generator).to(gbuf.dtype)
                gbuf.add_(eps if self._noise_projected else self._project(eps))
            fc.attach_grads()
            self._apply_ap(clones)
            self._set_lr(opt, lr, lr_schedule(t) if lr_schedule is not None else 1.0)
            opt.step()
            torch.sub(fc.params_flat(), self.theta0, out=delta)
            hdelta = self._hvp(delta)
            loss = self.surrogate_loss(delta, hdelta)
            if not math.isfinite(loss):
                losses.extend([float("inf")] * (horizon - t))
                break
            losses.append(loss)
            stats = getattr(opt, "last_stats", None)
            if stats is not None and hasattr(stats, "mean_gate"):
                gates.append(float(stats.mean_gate))
            if projs is not None:
                projs.append(delta @ modes)
            if not math.isfinite(loss) or abs(loss) > divergence_factor * max(abs(self.loss0), 1e-30):
                losses.extend([float("inf")] * (horizon - t - 1))
                break
        proj_t = None
        if projs is not None:
            while len(projs) < horizon + 1:
                projs.append(torch.full_like(projs[0], float("nan")))
            proj_t = torch.stack(projs)
        return losses, gates, proj_t

    def _fused_low_rank(self, generator: Optional[torch.Generator]) -> bool:
        """Whether the r-dimensional fused path applies (same results, fewer passes)."""
        return (
            isinstance(self.curvature, LowRankCurvature)
            and self.subspace is not None
            and self.subspace.shape == self.curvature.eigenvectors.shape
            and torch.equal(self.subspace, self.curvature.eigenvectors.to(self.subspace.dtype))
            and self._ap is None
            and (self.noise is None or generator is None or self._noise_projected)
        )

    def _trajectory_low_rank(
        self,
        horizon: int,
        generator: Optional[torch.Generator],
        lr: Optional[float],
        lr_schedule: Optional[Callable[[int], float]],
        modes: Optional[Tensor],
        divergence_factor: float,
    ) -> tuple[list[float], list[float], Optional[Tensor]]:
        """Low-rank fast path: two passes over ``V`` per step instead of three or four.

        With ``H = V diag(lam) V^T + rho (I - V V^T)`` and ``g0 = V a0`` (projected),
        ``c = V^T delta`` gives ``grad = g0 + rho delta + V (lam c - rho c + z)`` (``z``
        the noise coefficients) and ``L = L0 + a0.c + (c.lam c + rho (|d|^2 - |c|^2))/2``.
        """
        basis = self.subspace
        lam = self.curvature.eigenvalues.to(basis.dtype)
        rho = self.curvature.residual
        a0 = basis.t() @ self.grad0
        inner = self._step_noise.inner if self._noise_projected else None
        fc = self._clone()
        opt, gbuf = fc.optimizer, fc.grad
        delta = torch.zeros_like(self.theta0)
        c = torch.zeros_like(a0)
        losses = [self.loss0]
        gates: list[float] = []
        projs = [delta @ modes] if modes is not None else None
        for t in range(horizon):
            coef = a0 + (lam - rho) * c
            if inner is not None and generator is not None:
                ratio = losses[-1] / self.loss0 if self.loss0 > 0 else 1.0
                coef = coef + inner(None, ratio, t, generator).to(coef.dtype)
            torch.mv(basis, coef, out=gbuf)
            if rho:
                gbuf.add_(delta, alpha=rho)
            fc.attach_grads()
            self._set_lr(opt, lr, lr_schedule(t) if lr_schedule is not None else 1.0)
            opt.step()
            torch.sub(fc.params_flat(), self.theta0, out=delta)
            c = basis.t() @ delta
            loss = self.loss0 + float(a0 @ c) + 0.5 * float(c @ (lam * c))
            if rho:
                loss += 0.5 * rho * (float(torch.dot(delta, delta)) - float(c @ c))
            if not math.isfinite(loss):
                losses.extend([float("inf")] * (horizon - t))
                break
            losses.append(loss)
            stats = getattr(opt, "last_stats", None)
            if stats is not None and hasattr(stats, "mean_gate"):
                gates.append(float(stats.mean_gate))
            if projs is not None:
                projs.append(delta @ modes)
            if abs(loss) > divergence_factor * max(abs(self.loss0), 1e-30):
                losses.extend([float("inf")] * (horizon - t - 1))
                break
        proj_t = None
        if projs is not None:
            while len(projs) < horizon + 1:
                projs.append(torch.full_like(projs[0], float("nan")))
            proj_t = torch.stack(projs)
        return losses, gates, proj_t

    def replay(self, gradients: Sequence[Tensor], *, lr: Optional[float] = None) -> list[Tensor]:
        """Apply a fixed sequence of flat gradients to a cloned optimizer.

        Returns the flat parameter vector after each step. Used to check
        state-replay parity against the real optimizer.
        """
        fc = self._clone()
        out = []
        for g in gradients:
            fc.grad.copy_(g.reshape(-1).to(fc.grad.dtype))
            fc.attach_grads()
            self._apply_ap(fc.clones)
            if lr is not None:
                self._set_lr(fc.optimizer, lr, 1.0)
            fc.optimizer.step()
            out.append(fc.params_flat().clone())
        return out

    def run(
        self,
        n_seeds: int,
        horizon: int,
        *,
        seed: int = 0,
        lr: Optional[float] = None,
        lr_schedule: Optional[Callable[[int], float]] = None,
        thresholds: Sequence[float] = (0.1, 0.5, 1.0),
        modes: Optional[Tensor] = None,
        divergence_factor: float = 100.0,
        excursion_kwargs: Optional[dict[str, Any]] = None,
    ) -> SimulationDistribution:
        """Simulate ``n_seeds`` independent continuations of ``horizon`` steps.

        Args:
            n_seeds, horizon: Number of trajectories / steps each.
            seed: Base seed; trajectory ``i`` uses ``seed + i`` on a private
                generator (the global RNG is not used).
            lr: Override every group's learning rate (default: the optimizer's).
            lr_schedule: Optional multiplier ``f(step)`` on the learning rate.
            thresholds: Preregistered ``max_rel_excess`` thresholds.
            modes: Optional ``[n, m]`` directions to project displacement onto.
            divergence_factor: Stop a trajectory once ``|L| > factor * |L0|``.
            excursion_kwargs: Overrides for :func:`detect_excursions`.
        """
        if n_seeds < 1 or horizon < 1:
            raise ValueError("n_seeds and horizon must be >= 1.")
        if modes is not None:
            modes = modes.to(self.theta0.dtype)
        det, _, _ = self._trajectory(horizon, None, lr, lr_schedule, None, divergence_factor)
        det_t = torch.tensor(det, dtype=torch.float64)
        if self.loss0 >= 0 and float(det_t.min()) < 0:
            warnings.warn(
                f"QuadraticStabilitySimulator: the noiseless surrogate loss fell to "
                f"{float(det_t.min()):.3g} (< 0, from {self.loss0:.3g}). The curvature has "
                f"directions with almost no restoring force (e.g. a GGN on fewer samples x "
                f"outputs than parameters), so excursion statistics are unreliable. Pass "
                f"subspace= (e.g. LowRankCurvature.from_proxy(...).eigenvectors).",
                RuntimeWarning,
                stacklevel=2,
            )
        ex_kw = {"window": max(5, horizon // 20), "divergence_factor": divergence_factor}
        ex_kw.update(excursion_kwargs or {})
        all_losses, all_gates, all_proj, reports, excess = [], [], [], [], []
        for i in range(n_seeds):
            gen = torch.Generator().manual_seed(seed + i)
            losses, gates, proj = self._trajectory(
                horizon, gen, lr, lr_schedule, modes, divergence_factor
            )
            lt = torch.tensor(losses, dtype=torch.float64)
            all_losses.append(lt)
            if gates:
                gates = gates + [float("nan")] * (horizon - len(gates))
                all_gates.append(torch.tensor(gates))
            if proj is not None:
                all_proj.append(proj)
            reports.append(detect_excursions(lt, **ex_kw))
            ratio = lt / det_t.clamp_min(1e-30) - 1.0
            ratio[~torch.isfinite(lt)] = float("inf")
            excess.append(float(ratio[1:].max()))
        return SimulationDistribution(
            losses=torch.stack(all_losses),
            deterministic=det_t,
            max_rel_excess=torch.tensor(excess, dtype=torch.float64),
            excursions=reports,
            mean_gate=torch.stack(all_gates) if len(all_gates) == n_seeds else None,
            mode_projections=torch.stack(all_proj) if all_proj else None,
            thresholds=tuple(thresholds),
            config={
                "n_seeds": n_seeds,
                "horizon": horizon,
                "seed": seed,
                "lr": lr,
                "loss0": self.loss0,
                "noise": type(self.noise).__name__ if self.noise is not None else None,
                "curvature": self.curvature.metadata.name,
                "subspace_rank": None if self.subspace is None else int(self.subspace.shape[1]),
                "optimizer": type(self.optimizer).__name__,
            },
        )

    # -- validity -------------------------------------------------------------

    def check_validity(
        self,
        *,
        params_now: Optional[Sequence[Tensor]] = None,
        steps_since_checkpoint: Optional[int] = None,
        probe_now: Optional[StabilityProbe] = None,
        probe_then: Optional[StabilityProbe] = None,
        max_rel_displacement: float = 0.05,
        max_age: Optional[int] = None,
        max_rel_stat_drift: float = 0.5,
        real_loss_fn: Optional[Callable[[Tensor], float]] = None,
        fd_direction: Optional[Tensor] = None,
        max_fd_rel_error: float = 0.2,
    ) -> list[str]:
        """Return reasons to refit or abort (empty list = still valid).

        Checks parameter drift from ``theta0``, checkpoint age, drift of the
        Euclidean noise trace between two probes, and optionally the
        finite-difference error of the surrogate along ``fd_direction`` using
        ``real_loss_fn(flat_theta) -> loss``.
        """
        reasons: list[str] = []
        if params_now is not None:
            now = torch.cat([p.detach().reshape(-1) for p in params_now])
            rel = float((now - self.theta0).norm() / self.theta0.norm().clamp_min(1e-30))
            if rel > max_rel_displacement:
                reasons.append(f"parameter_drift:{rel:.3g}")
        if max_age is not None and steps_since_checkpoint is not None \
                and steps_since_checkpoint > max_age:
            reasons.append(f"checkpoint_age:{steps_since_checkpoint}")
        if probe_now is not None and probe_then is not None:
            a = probe_then.measure().trace_sigma
            b = probe_now.measure().trace_sigma
            drift = abs(b - a) / max(abs(a), 1e-30)
            if drift > max_rel_stat_drift:
                reasons.append(f"gradient_statistic_drift:{drift:.3g}")
        if real_loss_fn is not None and fd_direction is not None:
            d = fd_direction.reshape(-1).to(self.theta0.dtype)
            real = real_loss_fn(self.theta0 + d) - real_loss_fn(self.theta0)
            surr = self.surrogate_loss(d) - self.loss0
            err = abs(real - surr) / max(abs(real), 1e-30)
            if err > max_fd_rel_error:
                reasons.append(f"surrogate_fd_error:{err:.3g}")
        return reasons
