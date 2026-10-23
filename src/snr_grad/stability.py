"""
Read-only, curvature-aware stochastic stability sensor.

Implements the diagnostics part of
``docs/proposals/curvature-aware-stability-control.md``. Nothing here takes an
optimizer step, writes ``.grad``, changes ``requires_grad``, parameters or
optimizer state, or consumes the global RNG without restoring it.

Local model at a checkpoint (per-example gradient mean ``mu``, covariance
``Sigma``; batch-mean gradient ``g_B = mu + eps_B`` with ``Cov(eps_B) = Sigma/B``
for *independent* examples; frozen linearized optimizer response ``P``; PSD
curvature proxy ``H_plus``; update ``Delta = -eta P g_B``)::

    E[Delta L] ~= -eta g^T P mu + (eta^2/2) mu^T P^T H P mu
                  + (eta^2/(2B)) tr(H P Sigma P^T)

    A     = H^(1/2) P Sigma P^T H^(1/2)
    R     = tr(A)                     # curvature-weighted noise trace
    D_eff = R^2 / tr(A^2)             # concentration (small = few modes dominate)

For Gaussian noise the noise-only penalty has mean ``eta^2 R/(2B)`` and relative
SD ``sqrt(2/D_eff)``. These are *local surrogate* quantities, not calibrated spike
probabilities; :class:`StabilityMetrics` therefore exposes an uncalibrated
``risk_score`` and never a ``spike_probability``.

Main entry points:

* :func:`collect_microbatch_gradients` -- ``K`` disjoint microbatch mean
  gradients at fixed parameters, leaving ``.grad`` and the RNG as they were.
* :func:`frozen_update_map` -- the one-step linearized update map ``P`` of an
  optimizer at its current state (AdamW, SNRAdamW gated/ungated, SGD, DoPr), or
  an explicit *unsupported* map for nonlinear optimizers (Muon family, ...).
* :class:`StabilityProbe` -- turns microbatch gradients + ``P`` + a
  :class:`~snr_grad.curvature.CurvatureProxy` into :class:`StabilityMetrics`
  (Euclidean and optimizer-aware GNS, ``R`` with jackknife uncertainty,
  bias-corrected concentration with abstention, mean progress terms, cost).
* :func:`exact_noise_geometry` -- ground-truth ``R`` / ``D_eff`` from dense
  matrices, for quadratic toys.
* :func:`spectrum_metrics` -- cheap activation-spectrum proxies (labelled as
  such: they are *not* curvature-weighted update-noise concentration).
* :func:`detect_excursions` -- the shared, preregistered loss-spike definition.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from snr_grad._core import (
    MARSSNRAdamW,
    RotatedSNRAdamW,
    RotatedSNRScheduleFreeAdamW,
    SNRAdamW,
    SNRMuon,
    SNRScheduleFreeAdamW,
    SNRScheduleFreeMuon,
    SpectralSNRMuon,
    SpectralSNRScheduleFreeMuon,
    compute_gate,
    resolve_alpha,
)
from snr_grad.activation import ActivationCovarianceSnapshot, DoPr, isolated_activation_capture
from snr_grad.curvature import CurvatureProxy, ParamDict, preserved_rng
from snr_grad.variance import tree_batch_size, tree_split

__all__ = [
    "per_example_variance_from_batch_mean",
    "collect_microbatch_gradients",
    "FrozenUpdateMap",
    "frozen_update_map",
    "StabilityMetrics",
    "StabilityProbe",
    "aggregate_probe_groups",
    "exact_noise_geometry",
    "spectrum_metrics",
    "ExcursionReport",
    "detect_excursions",
]


# ---------------------------------------------------------------------------
# Scale conventions
# ---------------------------------------------------------------------------

def per_example_variance_from_batch_mean(s: Tensor, batch_size: int) -> Tensor:
    """Convert a batch-mean gradient variance (``grad_variances`` scale) to per-example.

    ``snr_grad.variance`` and ``SNRAdamW(grad_variances=...)`` use the variance
    of the *minibatch mean* gradient, ``Var_per_example / b``. Multiply back by
    ``b`` to recover the per-example scale used by batch-size reasoning. Valid
    only for independent examples.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    return s * batch_size


# ---------------------------------------------------------------------------
# Microbatch gradient collection
# ---------------------------------------------------------------------------

def collect_microbatch_gradients(
    model: nn.Module,
    loss_fn: Callable[[nn.Module, Any], Tensor],
    batch: Any,
    *,
    num_splits: int,
    params: Optional[Sequence[Tensor]] = None,
    preserve_rng: bool = True,
    group_ids: Optional[Tensor] = None,
) -> list[dict[Tensor, Tensor]]:
    """Return ``K`` microbatch mean gradients at the current parameters.

    Splits ``batch`` into ``num_splits`` equal contiguous chunks (the batch size
    must be divisible by ``num_splits`` so every chunk mean has the same
    variance). The existing ``.grad`` of every parameter and (when
    ``preserve_rng``) the CPU / CUDA RNG state are restored afterwards, and the probe
    forwards do not reach a live :class:`DoPr`'s activation cache, so a probe
    between training steps does not change training.

    **Correlated rows must stay in the same microbatch.** If copies of an example,
    or tokens / windows of one sequence, are spread across microbatches, the
    microbatch means become positively correlated and the probe *underestimates*
    the noise (to zero in the limit; see ``studies/stability_study``). Keep
    correlated rows together (pass ``group_ids``) and count *groups* as the
    independent units (``microbatch_size`` in :class:`StabilityProbe`).

    Args:
        model: Module to differentiate.
        loss_fn: ``loss_fn(model, chunk) -> mean-reduced scalar loss``.
        batch: Pytree with a leading batch dimension.
        num_splits: ``K >= 2``.
        params: Parameters to record (default: all with ``requires_grad``).
        preserve_rng: Restore RNG state after the probe (dropout etc.).
        group_ids: Optional ``[N]`` integer ids of correlated rows (sequence /
            document / duplicate id). Rows are reordered so each group is
            contiguous, and a ``ValueError`` is raised if any group would straddle
            two microbatches (choose ``num_splits`` so groups pack evenly).

    Returns:
        A list of ``K`` dicts ``{param: chunk_mean_grad}`` (detached clones).
    """
    if num_splits < 2:
        raise ValueError(f"num_splits must be >= 2, got {num_splits}.")
    b = tree_batch_size(batch)
    if b % num_splits != 0:
        raise ValueError(
            f"batch size {b} is not divisible by num_splits={num_splits}; unequal "
            f"microbatches have unequal variance."
        )
    if group_ids is not None:
        batch = _group_contiguous(batch, group_ids, num_splits)
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]
    params = list(params)
    all_params = list(model.parameters())
    saved_grads = {p: (None if p.grad is None else p.grad.detach().clone()) for p in all_params}
    out: list[dict[Tensor, Tensor]] = []
    try:
        with preserved_rng(preserve_rng), isolated_activation_capture():
            for chunk in tree_split(batch, num_splits):
                for p in all_params:
                    p.grad = None
                with torch.enable_grad():
                    loss = loss_fn(model, chunk)
                    loss.backward()
                out.append({
                    p: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                    for p in params
                })
    finally:
        for p in all_params:
            p.grad = saved_grads[p]
    return out


def _tree_index(batch: Any, idx: Tensor) -> Any:
    """Index every tensor leaf of a pytree along dim 0; share other leaves."""
    if isinstance(batch, Tensor):
        return batch[idx.to(batch.device)]
    if isinstance(batch, Mapping):
        return type(batch)({k: _tree_index(v, idx) for k, v in batch.items()})
    if isinstance(batch, (tuple, list)):
        return type(batch)(_tree_index(v, idx) for v in batch)
    return batch


def _group_contiguous(batch: Any, group_ids: Tensor, num_splits: int) -> Any:
    """Reorder rows so groups are contiguous; refuse splits that cut a group."""
    ids = torch.as_tensor(group_ids).reshape(-1).cpu()
    n = tree_batch_size(batch)
    if ids.numel() != n:
        raise ValueError(f"group_ids has {ids.numel()} entries, batch has {n} rows.")
    order = torch.sort(ids, stable=True).indices
    sorted_ids = ids[order]
    chunk = n // num_splits
    for k in range(1, num_splits):
        if bool(sorted_ids[k * chunk - 1] == sorted_ids[k * chunk]):
            raise ValueError(
                f"group {int(sorted_ids[k * chunk])} would straddle microbatches {k - 1} "
                f"and {k}; choose num_splits so that whole groups fill each microbatch."
            )
    return _tree_index(batch, order)


# ---------------------------------------------------------------------------
# Frozen linearized update maps
# ---------------------------------------------------------------------------

@dataclass
class FrozenUpdateMap:
    """A linear map ``g -> P g`` approximating one optimizer step at frozen state.

    The parameter update is ``Delta = -lr * P g`` (weight decay and other
    deterministic terms are excluded: they do not change noise deviations).

    Attributes:
        name: Which adapter produced the map.
        supported: ``False`` means *no valid frozen-linear sensor exists* for this
            optimizer (e.g. Muon's polar update is nonlinear); calling it raises.
        reason: Why the map is unsupported or approximate.
        exact_linear: ``True`` when the map is the exact Jacobian of the next step
            with respect to the gradient, holding all *other* state fixed.
        details: Hyperparameters and summary statistics (``beta2``, ``rho``,
            ``lambda_pop``, mean gate, preconditioner scale, ...).
    """

    name: str
    supported: bool
    reason: str = ""
    exact_linear: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    fn: Optional[Callable[[ParamDict], dict[Tensor, Tensor]]] = None

    def __call__(self, grads: ParamDict) -> dict[Tensor, Tensor]:
        if not self.supported or self.fn is None:
            raise RuntimeError(f"frozen update map {self.name!r} is unsupported: {self.reason}")
        return self.fn(grads)


def _identity_map() -> FrozenUpdateMap:
    return FrozenUpdateMap(
        name="identity",
        supported=True,
        exact_linear=True,
        reason="plain gradient (P = I)",
        fn=lambda g: {p: v.detach().clone() for p, v in g.items()},
    )


def _scaled_map(scales: Mapping[Tensor, Tensor | float]) -> Callable[[ParamDict], dict]:
    def fn(grads: ParamDict) -> dict[Tensor, Tensor]:
        out: dict[Tensor, Tensor] = {}
        for p, g in grads.items():
            s = scales.get(p)
            out[p] = torch.zeros_like(g) if s is None else g.detach() * s
        return out

    return fn


def _momentum_factor(beta1: float, bias_correction1: float, linearization: str) -> float:
    if linearization == "one_step":
        return (1.0 - beta1) / bias_correction1
    if linearization == "stationary":
        # Std of an EMA of iid noise relative to one sample: sqrt((1-b)/(1+b)).
        return math.sqrt((1.0 - beta1) / (1.0 + beta1))
    raise ValueError(f"linearization must be 'one_step' or 'stationary', got {linearization!r}.")


def _sgd_map(opt: torch.optim.SGD, linearization: str) -> FrozenUpdateMap:
    scales: dict[Tensor, float] = {}
    for group in opt.param_groups:
        if group.get("maximize", False):
            return FrozenUpdateMap("sgd", False, "maximize=True is not supported")
        mu = group.get("momentum", 0.0)
        damp = group.get("dampening", 0.0)
        nesterov = group.get("nesterov", False)
        for p in group["params"]:
            if not p.requires_grad:
                continue
            has_buf = opt.state.get(p, {}).get("momentum_buffer") is not None
            if mu == 0:
                f = 1.0
            elif linearization == "stationary":
                f = (1.0 - damp) / math.sqrt(1.0 - mu * mu)
            else:
                f = (1.0 - damp) if has_buf else 1.0
                if nesterov:
                    f = 1.0 + mu * f
            scales[p] = f
    return FrozenUpdateMap(
        name="sgd",
        supported=True,
        exact_linear=linearization == "one_step",
        reason="" if linearization == "one_step" else "stationary momentum noise factor",
        details={"linearization": linearization},
        fn=_scaled_map(scales),
    )


def _adam_map(
    opt: torch.optim.Optimizer,
    linearization: str,
    at_gradient: Optional[ParamDict],
) -> FrozenUpdateMap:
    scales: dict[Tensor, Tensor] = {}
    precond_means = []
    for group in opt.param_groups:
        if group.get("maximize", False):
            return FrozenUpdateMap("adam", False, "maximize=True is not supported")
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        amsgrad = group.get("amsgrad", False)
        for p in group["params"]:
            st = opt.state.get(p)
            if not st or "exp_avg_sq" not in st:
                continue
            t = int(st["step"]) + 1  # the probe gradient feeds the *next* step
            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t
            v = st["max_exp_avg_sq"] if amsgrad and "max_exp_avg_sq" in st else st["exp_avg_sq"]
            g = at_gradient.get(p) if at_gradient is not None else None
            if g is None or linearization != "one_step":
                denom = (v / bc2).sqrt() + eps
                pre = _momentum_factor(beta1, bc1, linearization) / denom
            else:
                pre = _adam_exact_jacobian(st, v, g.detach(), beta1, beta2, eps, bc1, bc2, amsgrad)
            scales[p] = pre
            precond_means.append(float(pre.mean()))
    if not scales:
        return FrozenUpdateMap("adam", False, "optimizer has no Adam state yet (take a step first)")
    g0 = opt.param_groups[0]
    exact = at_gradient is not None and linearization == "one_step"
    return FrozenUpdateMap(
        name="adam",
        supported=True,
        exact_linear=exact,
        reason=(
            "exact diagonal Jacobian of the next step at the given gradient" if exact
            else "second moment v held fixed (its g^2 dependence is ignored)"
        ),
        details={
            "linearization": linearization,
            "beta1": g0["betas"][0],
            "beta2": g0["betas"][1],
            "mean_preconditioner": sum(precond_means) / len(precond_means),
        },
        fn=_scaled_map(scales),
    )


def _adam_exact_jacobian(
    st: Mapping[str, Any],
    v_state: Tensor,
    g: Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    bc1: float,
    bc2: float,
    amsgrad: bool,
) -> Tensor:
    """Diagonal ``d(m_hat / (sqrt(v_hat) + eps)) / dg`` of the next Adam step at ``g``.

    Adam is coordinate-wise, so the Jacobian is diagonal and exact (including the
    ``g^2`` dependence of the second moment, which matters early in training when
    the bias-corrected ``v_hat`` gives the new gradient a large weight).
    """
    m_new = beta1 * st["exp_avg"] + (1.0 - beta1) * g
    v_new = beta2 * st["exp_avg_sq"] + (1.0 - beta2) * g * g
    dv = 2.0 * (1.0 - beta2) * g
    if amsgrad and "max_exp_avg_sq" in st:
        use_new = v_new > v_state
        v_new = torch.maximum(v_state, v_new)
        dv = torch.where(use_new, dv, torch.zeros_like(dv))
    s = (v_new / bc2).sqrt()
    denom = s + eps
    ds = torch.where(s > 0, dv / (2.0 * bc2 * s.clamp_min(1e-30)), torch.zeros_like(s))
    return (1.0 - beta1) / bc1 / denom - (m_new / bc1) * ds / (denom * denom)


def _snr_adamw_map(opt: SNRAdamW, linearization: str, gated: bool) -> FrozenUpdateMap:
    scales: dict[Tensor, Tensor] = {}
    gate_sum = 0.0
    gate_n = 0
    for group in opt.param_groups:
        if group.get("maximize", False):
            return FrozenUpdateMap("snr_adamw", False, "maximize=True is not supported")
        beta1, beta2 = group["betas"]
        rho = group["rho"]
        eps = group["eps"]
        alpha_value = resolve_alpha(
            group["alpha"],
            batch_size=group.get("batch_size"),
            dataset_size=group.get("dataset_size"),
        )
        g_alpha = group.get("grokfast_alpha", 0.0)
        g_lamb = group.get("grokfast_lamb", 0.0)
        grok = 1.0 + g_lamb * (1.0 - g_alpha) if g_alpha > 0 and g_lamb > 0 else 1.0
        for p in group["params"]:
            st = opt.state.get(p)
            if not st or "step" not in st or st.get("frozen", False):
                continue
            t = int(st["step"])
            bc1_next = 1.0 - beta1 ** (t + 1)
            bc2_next = 1.0 - beta2 ** (t + 1)
            denom = (st["exp_avg_sq"] / bc2_next).sqrt() + eps
            pre = grok * _momentum_factor(beta1, bc1_next, linearization) / denom
            if gated:
                # The gate of the most recent step, from the stored state.
                m_hat = st["exp_avg"] / (1.0 - beta1 ** t)
                s_hat = st["exp_grad_var"] / (1.0 - rho ** t)
                q = compute_gate(
                    m_hat, s_hat, gate=group["gate"], alpha=alpha_value,
                    lambda_pop=group["lambda_pop"], gate_eps=group["gate_eps"],
                )
                pre = pre * q
                gate_sum += float(q.sum())
                gate_n += q.numel()
            scales[p] = pre
    if not scales:
        return FrozenUpdateMap(
            "snr_adamw", False, "optimizer has no SNRAdamW state yet (take a step first)"
        )
    g0 = opt.param_groups[0]
    reason = "v held fixed"
    if gated:
        reason += "; gate q frozen at its last value (it actually responds to noise)"
    return FrozenUpdateMap(
        name="snr_adamw_gated" if gated else "snr_adamw_ungated",
        supported=True,
        exact_linear=False,
        reason=reason,
        details={
            "linearization": linearization,
            "beta1": g0["betas"][0],
            "beta2": g0["betas"][1],
            "rho": g0["rho"],
            "gate": g0["gate"],
            "lambda_pop": g0["lambda_pop"],
            "alpha": g0["alpha"],
            "adaptive_threshold": g0.get("adaptive_threshold") is not None,
            "mean_gate": gate_sum / gate_n if gate_n else None,
        },
        fn=_scaled_map(scales),
    )


def _dopr_map(
    opt: DoPr,
    base_map: FrozenUpdateMap,
    snapshot: Optional[Mapping[str, ActivationCovarianceSnapshot]],
) -> FrozenUpdateMap:
    ap = opt.ap
    if not base_map.supported:
        return FrozenUpdateMap("dopr", False, f"base optimizer: {base_map.reason}")
    warm = ap.step_count + 1 <= ap.config.warmup_steps
    if snapshot is None:
        snapshot = ap.covariance_snapshot(source="auto")
    by_weight = {snap.weight: snap for snap in snapshot.values()}
    registered = list(ap._weight_of_module.values())
    missing = [w for w in registered if w not in by_weight]
    if not warm and missing:
        return FrozenUpdateMap(
            "dopr", False,
            f"{len(missing)} registered layer(s) have no activation covariance; probe "
            f"after a grad-enabled forward (before precondition_) or enable ema_beta",
        )

    def fn(grads: ParamDict) -> dict[Tensor, Tensor]:
        pre: dict[Tensor, Tensor] = {}
        for p, g in grads.items():
            snap = by_weight.get(p)
            if warm or snap is None:
                pre[p] = g.detach()
            elif snap.kind == "linear":
                pre[p] = ap._solve_linear(snap.sigma.to(g.device), g.detach()).to(g.dtype)
            else:
                pre[p] = ap._apply_embedding(snap.sigma.to(g.device), g.detach()).to(g.dtype)
        return base_map(pre)

    return FrozenUpdateMap(
        name=f"dopr[{base_map.name}]",
        supported=True,
        exact_linear=False,
        reason=(
            f"activation covariance frozen; {base_map.reason}. Rank-deficient covariances "
            f"can amplify noisy directions"
        ),
        details={**base_map.details, "ap_warmup": warm, "ap_damping": ap.config.damping},
        fn=fn,
    )


_NONLINEAR = (
    SNRMuon, SpectralSNRMuon, SNRScheduleFreeMuon, SpectralSNRScheduleFreeMuon,
)
_UNSUPPORTED = (
    RotatedSNRAdamW, RotatedSNRScheduleFreeAdamW, SNRScheduleFreeAdamW, MARSSNRAdamW,
)


def frozen_update_map(
    optimizer: Any = None,
    *,
    gated: bool = True,
    linearization: Literal["one_step", "stationary"] = "one_step",
    activation_snapshot: Optional[Mapping[str, ActivationCovarianceSnapshot]] = None,
    at_gradient: Optional[ParamDict] = None,
) -> FrozenUpdateMap:
    """Return the frozen linearized update map ``P`` for ``optimizer``.

    Args:
        optimizer: ``None`` (plain gradient), ``torch.optim.SGD``, ``Adam``,
            ``AdamW``, :class:`SNRAdamW`, or :class:`DoPr` wrapping one of these.
            Anything else returns an *unsupported* map with a reason; the Muon
            family is unsupported because its orthogonalized update is nonlinear
            (use :class:`snr_grad.stability_sim.QuadraticStabilitySimulator`,
            which replays the real step, instead).
        gated: For SNRAdamW, include the frozen gate ``q`` (``True``) or report
            the ungated AdamW-equivalent map (``False``).
        linearization: ``"one_step"`` is the Jacobian of the next update with
            respect to the next gradient (momentum factor ``(1-beta1)/bc1``).
            ``"stationary"`` uses the noise std of a converged momentum EMA,
            ``sqrt((1-beta1)/(1+beta1))``, closer to the per-step noise actually
            carried by the update over many steps.
        activation_snapshot: For DoPr, override the covariances (default: the
            wrapper's current batch cache, falling back to its EMA).
        at_gradient: For Adam/AdamW with ``"one_step"``, linearize at this
            gradient (typically the probe mean) to get the *exact* diagonal
            Jacobian of the next step, including the second moment's ``g^2``
            dependence. Without it ``v`` is held fixed, which is a poor
            approximation early in training (small ``1 - beta2**t``).
    """
    if optimizer is None:
        return _identity_map()
    if isinstance(optimizer, DoPr):
        base = frozen_update_map(optimizer.base, gated=gated, linearization=linearization)
        # at_gradient is a raw gradient; after AP the base sees a different one, so
        # the exact Adam Jacobian is not used under DoPr.
        return _dopr_map(optimizer, base, activation_snapshot)
    if isinstance(optimizer, _NONLINEAR):
        return FrozenUpdateMap(
            type(optimizer).__name__, False,
            "orthogonalized/polar update is nonlinear; a diagonal Adam linearization "
            "would be wrong. Use a dual-norm GNS baseline or simulator replay.",
        )
    if isinstance(optimizer, _UNSUPPORTED):
        return FrozenUpdateMap(
            type(optimizer).__name__, False,
            "no frozen-linear adapter implemented for this optimizer",
        )
    if isinstance(optimizer, SNRAdamW):
        return _snr_adamw_map(optimizer, linearization, gated)
    if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        return _adam_map(optimizer, linearization, at_gradient)
    if isinstance(optimizer, torch.optim.SGD):
        return _sgd_map(optimizer, linearization)
    return FrozenUpdateMap(
        type(optimizer).__name__, False, "unknown optimizer type"
    )


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

@dataclass
class StabilityMetrics:
    """Diagnostics from one :meth:`StabilityProbe.measure` call.

    All variance-like quantities are on the *per-example* scale unless the name
    says otherwise. ``None`` means "not estimable here" (see ``flags``).

    Attributes:
        num_microbatches, microbatch_size, units: Probe shape (``K``, ``b``).
        independent: Whether examples were declared independent; when ``False``
            every ``1/B`` extrapolation is disabled.
        trace_sigma: ``tr(Sigma)`` (Euclidean).
        grad_norm_sq: Unbiased ``|mu|^2``.
        gns_euclidean: ``tr(Sigma) / |mu|^2`` (simple noise scale, in units).
        trace_sigma_precond: ``tr(P Sigma P^T)``.
        grad_norm_sq_precond: Unbiased ``|P mu|^2``.
        gns_optimizer: ``tr(P Sigma P^T) / |P mu|^2`` (optimizer-aware GNS).
        risk_trace_per_example: ``R_hat`` estimate of ``tr(H P Sigma P^T)``.
        risk_trace_se: Delete-one jackknife standard error of ``R_hat``. Tends to be
            optimistic for small ``K`` and heavy-tailed noise (``R_hat`` is right-skewed).
        risk_trace_ci: ``(lower, upper)`` interval at ``ci_level`` from a jackknife on
            ``log R`` with a Student-t quantile on ``K - 1`` degrees of freedom; better
            calibrated than ``R_hat +/- 2 SE`` for skewed ``R_hat``.
        directional_progress: Unbiased ``g^T P mu`` (``g`` = validation gradient
            if supplied, else ``mu``).
        mean_curvature_term: Unbiased ``mu^T P^T H P mu``.
        concentration: Bias-corrected ``D_eff`` estimate, or ``None`` on abstention.
        concentration_se: Jackknife SE of the bias-corrected ``D_eff``.
        concentration_plugin: Plug-in ``D_eff`` of the sample matrix (always
            ``<= K-1``; *never* evidence of concentration on its own).
        concentration_status: ``"ok"``, ``"insufficient_samples"``,
            ``"rank_not_resolved"``, ``"indefinite_curvature"`` or ``"unresolved"``.
        batch_size, lr: The ``B`` and ``eta`` used for the derived terms below.
        noise_penalty: ``eta^2 R / (2B)``.
        noise_penalty_rel_sd: ``sqrt(2/D_eff)`` (noise-only, Gaussian).
        expected_loss_change: Sum of the three local terms.
        risk_score: Uncalibrated ranking score ``noise_penalty / (eta |g^T P mu|)``
            (noise penalty relative to first-order progress). Not a probability.
        update_map, update_map_details, curvature_name, curvature_psd: Provenance.
        flags: Violated / unchecked assumptions.
        cost: ``seconds``, ``curvature_calls``, ``gradient_bytes``.
    """

    num_microbatches: int
    microbatch_size: int
    units: str
    independent: bool
    trace_sigma: float
    grad_norm_sq: float
    gns_euclidean: Optional[float]
    trace_sigma_precond: float
    grad_norm_sq_precond: float
    gns_optimizer: Optional[float]
    risk_trace_per_example: Optional[float]
    risk_trace_se: Optional[float]
    risk_trace_ci: Optional[tuple[float, float]]
    directional_progress: float
    mean_curvature_term: Optional[float]
    concentration: Optional[float]
    concentration_se: Optional[float]
    concentration_plugin: Optional[float]
    concentration_status: str
    batch_size: Optional[int]
    lr: Optional[float]
    noise_penalty: Optional[float]
    noise_penalty_rel_sd: Optional[float]
    expected_loss_change: Optional[float]
    risk_score: Optional[float]
    update_map: str
    update_map_details: dict[str, Any]
    curvature_name: Optional[str]
    curvature_psd: Optional[bool]
    flags: list[str]
    cost: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _offdiag_mean(m: Tensor) -> float:
    k = m.shape[0]
    return float((m.sum() - m.diagonal().sum()) / (k * (k - 1)))


def _centered_gram(raw: Tensor) -> Tensor:
    k = raw.shape[0]
    c = torch.eye(k, dtype=raw.dtype) - 1.0 / k
    return c @ raw @ c


def _concentration_from_gram(gram_c: Tensor, scale: float) -> tuple[float, float, float]:
    """Return (plug-in D, bias-corrected tr(A)^2, bias-corrected tr(A^2))."""
    n = gram_c.shape[0] - 1
    s_eig = scale * gram_c
    t1 = float(s_eig.diagonal().sum()) ** 2
    t2 = float((s_eig * s_eig).sum())
    plugin = t1 / t2 if t2 > 0 else float("nan")
    if n < 2:
        return plugin, float("nan"), float("nan")
    # Unbiased under Gaussian sampling with n degrees of freedom:
    #   E[tr S^2] = (1 + 1/n) tr A^2 + tr(A)^2 / n,  E[tr(S)^2] = tr(A)^2 + 2 tr A^2 / n.
    c = (t2 - t1 / n) * n * n / ((n - 1) * (n + 2))
    a = t1 - 2.0 * c / n
    return plugin, a, c


class StabilityProbe:
    """Curvature-aware stability probe from ``K`` microbatch mean gradients.

    Args:
        gradients: ``K >= 2`` dicts ``{param: microbatch_mean_grad}`` taken at
            identical parameters and optimizer state, from disjoint, equally
            sized, randomly assigned microbatches.
        microbatch_size: Independent units per microbatch (``b``).
        update_map: Frozen linearized optimizer map (default identity).
        curvature: PSD curvature proxy (optional; without it only GNS terms are
            reported).
        independent: Declare examples independent. ``False`` disables ``1/B``
            extrapolation and per-example rescaling claims (flags it).
        units: Name of the independent unit (``"examples"``, ``"sequences"``).
        validation_gradient: Optional held-out gradient ``g`` for the linear term.
        min_concentration_dof: Minimum ``K-1`` before a concentration estimate is
            reported (default 15, i.e. ``K >= 16``; on PLRF the bias-corrected
            estimate is unbiased to within 1% at ``K = 16`` but biased by about
            -24% at ``K = 8``).
        max_rank_fraction: Abstain if the estimated ``D_eff`` exceeds this
            fraction of ``K-1`` (the sample cannot resolve such ranks).
        ci_level: Coverage level of ``risk_trace_ci``.
    """

    def __init__(
        self,
        gradients: Sequence[ParamDict],
        microbatch_size: int,
        *,
        update_map: Optional[FrozenUpdateMap] = None,
        curvature: Optional[CurvatureProxy] = None,
        independent: bool = True,
        units: str = "examples",
        validation_gradient: Optional[ParamDict] = None,
        min_concentration_dof: int = 15,
        max_rank_fraction: float = 0.25,
        ci_level: float = 0.95,
    ) -> None:
        if len(gradients) < 2:
            raise ValueError("need at least K=2 microbatch gradients.")
        if microbatch_size < 1:
            raise ValueError(f"microbatch_size must be >= 1, got {microbatch_size}.")
        keys = list(gradients[0].keys())
        for g in gradients[1:]:
            if set(g.keys()) != set(keys):
                raise ValueError("all microbatch gradients must cover the same parameters.")
        self.params = keys
        self.gradients = [{p: g[p].detach() for p in keys} for g in gradients]
        self.microbatch_size = int(microbatch_size)
        self.update_map = update_map if update_map is not None else _identity_map()
        self.curvature = curvature
        self.independent = independent
        self.units = units
        self.validation_gradient = validation_gradient
        if not 0.0 < ci_level < 1.0:
            raise ValueError(f"ci_level must be in (0, 1), got {ci_level}.")
        self.min_concentration_dof = min_concentration_dof
        self.ci_level = ci_level
        self.max_rank_fraction = max_rank_fraction
        self._cache: Optional[dict[str, Any]] = None

    @classmethod
    def from_microbatches(
        cls,
        gradients: Sequence[ParamDict],
        microbatch_size: int,
        optimizer: Any = None,
        curvature: Optional[CurvatureProxy] = None,
        *,
        gated: bool = True,
        linearization: Literal["one_step", "stationary"] = "one_step",
        **kwargs: Any,
    ) -> "StabilityProbe":
        """Build a probe, deriving the frozen update map from ``optimizer``."""
        k = len(gradients)
        mean = {p: sum(g[p] for g in gradients) / k for p in gradients[0]}
        umap = frozen_update_map(
            optimizer, gated=gated, linearization=linearization, at_gradient=mean
        )
        return cls(gradients, microbatch_size, update_map=umap, curvature=curvature, **kwargs)

    @property
    def num_microbatches(self) -> int:
        return len(self.gradients)

    def mean_gradient(self) -> dict[Tensor, Tensor]:
        """Probe mean gradient ``g_bar`` (batch of ``K*b`` units)."""
        k = len(self.gradients)
        return {p: sum(g[p] for g in self.gradients) / k for p in self.params}

    def deviations(self) -> list[dict[Tensor, Tensor]]:
        """Raw microbatch deviations ``g_k - g_bar``."""
        mean = self.mean_gradient()
        return [{p: g[p] - mean[p] for p in self.params} for g in self.gradients]

    @staticmethod
    def _flat(d: ParamDict, params: Sequence[Tensor]) -> Tensor:
        return torch.cat([d[p].reshape(-1).to(torch.float64) for p in params])

    def _grams(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache
        t0 = time.perf_counter()
        cost0 = dict(self.curvature.metadata.cost) if self.curvature is not None else {}
        k = len(self.gradients)
        flat_g = torch.stack([self._flat(g, self.params) for g in self.gradients])
        euclid = flat_g @ flat_g.t()
        cache: dict[str, Any] = {"euclid": euclid}
        if self.update_map.supported:
            pg = [self.update_map(g) for g in self.gradients]
            flat_pg = torch.stack([self._flat(x, self.params) for x in pg])
            cache["precond"] = flat_pg @ flat_pg.t()
            # Symmetrized <g_j, P g_k>, used for the unbiased first-order term.
            gp = flat_g @ flat_pg.t()
            cache["cross"] = 0.5 * (gp + gp.t())
            if self.validation_gradient is not None:
                v = self._flat(
                    {p: self.validation_gradient.get(p, torch.zeros_like(p)) for p in self.params},
                    self.params,
                )
                cache["val_cross"] = flat_pg @ v
            if self.curvature is not None:
                hp = [self.curvature.matvec(x) for x in pg]
                flat_hp = torch.stack([
                    self._flat({p: h.get(p, torch.zeros_like(p)) for p in self.params}, self.params)
                    for h in hp
                ])
                hraw = flat_pg @ flat_hp.t()
                cache["curv"] = 0.5 * (hraw + hraw.t())
        cache["seconds"] = time.perf_counter() - t0
        cache["curvature_calls"] = (
            sum(self.curvature.metadata.cost.get(key, 0.0) - cost0.get(key, 0.0)
                for key in ("quad_calls", "matvec_calls"))
            if self.curvature is not None else 0.0
        )
        cache["gradient_bytes"] = float(flat_g.numel() * flat_g.element_size() * (2 if "precond" in cache else 1))
        del flat_g
        self._cache = cache
        return cache

    def _trace_cov(self, raw: Tensor, idx: Optional[Sequence[int]] = None) -> float:
        """Per-example covariance trace from a raw Gram (optionally on a subset)."""
        if idx is not None:
            raw = raw[idx][:, idx]
        k = raw.shape[0]
        return self.microbatch_size * float(_centered_gram(raw).diagonal().sum()) / (k - 1)

    def measure(
        self,
        *,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> StabilityMetrics:
        """Compute :class:`StabilityMetrics`. Never steps or mutates anything.

        Args:
            lr: Learning rate ``eta`` for the derived penalty terms.
            batch_size: Training batch ``B`` (in ``units``) for the ``1/B`` terms;
                defaults to the probe batch ``K*b``.
        """
        cache = self._grams()
        k = len(self.gradients)
        b = self.microbatch_size
        flags: list[str] = []
        if self.independent:
            flags.append("independence_assumed")
        else:
            flags.append("independence_violated:1/B_extrapolation_disabled")
        if not self.update_map.supported:
            flags.append(f"update_map_unsupported:{self.update_map.reason}")
        elif not self.update_map.exact_linear:
            flags.append(f"update_map_approximate:{self.update_map.reason}")

        euclid = cache["euclid"]
        trace_sigma = self._trace_cov(euclid)
        grad_norm_sq = _offdiag_mean(euclid)
        gns_e = trace_sigma / grad_norm_sq if grad_norm_sq > 0 else None
        if gns_e is None:
            flags.append("gradient_norm_not_resolved")

        trace_p = float("nan")
        norm_p = float("nan")
        gns_o = None
        progress = float("nan")
        mean_curv = None
        r_hat = None
        r_se = None
        r_ci = None
        conc = conc_se = conc_plugin = None
        conc_status = "unresolved"
        if "precond" in cache:
            trace_p = self._trace_cov(cache["precond"])
            norm_p = _offdiag_mean(cache["precond"])
            gns_o = trace_p / norm_p if norm_p > 0 else None
            if "val_cross" in cache:
                progress = float(cache["val_cross"].mean())
            else:
                progress = _offdiag_mean(cache["cross"])

        curv_name = curv_psd = None
        if "curv" in cache:
            meta = self.curvature.metadata
            curv_name, curv_psd = meta.name, meta.psd
            if not meta.psd:
                flags.append("curvature_not_psd")
            if meta.uncovered:
                flags.append(f"curvature_uncovered_params:{len(meta.uncovered)}")
            curv = cache["curv"]
            r_hat = self._trace_cov(curv)
            mean_curv = _offdiag_mean(curv)
            scale = b / (k - 1)
            if k >= 3:
                loo = [self._trace_cov(curv, [j for j in range(k) if j != i]) for i in range(k)]
                r_se = _jackknife_se(loo)
                if r_hat > 0 and all(v > 0 for v in loo):
                    se_log = _jackknife_se([math.log(v) for v in loo])
                    tq = _student_t_quantile(0.5 + 0.5 * self.ci_level, k - 1)
                    r_ci = (r_hat * math.exp(-tq * se_log), r_hat * math.exp(tq * se_log))
            if not meta.psd:
                conc_status = "indefinite_curvature"
            else:
                plugin, a, c = _concentration_from_gram(_centered_gram(curv), scale)
                conc_plugin = plugin
                n = k - 1
                d_bc = a / c if (c > 0 and a > 0) else None
                if n < self.min_concentration_dof:
                    conc_status = "insufficient_samples"
                elif d_bc is None:
                    conc_status = "unresolved"
                elif d_bc > self.max_rank_fraction * n:
                    conc_status = "rank_not_resolved"
                else:
                    conc_status = "ok"
                    conc = d_bc
                    loo_d = []
                    for i in range(k):
                        idx = [j for j in range(k) if j != i]
                        sub = curv[idx][:, idx]
                        _, a_i, c_i = _concentration_from_gram(_centered_gram(sub), b / (k - 2))
                        if c_i > 0 and a_i > 0:
                            loo_d.append(a_i / c_i)
                    conc_se = _jackknife_se(loo_d) if len(loo_d) == k else None

        big_b = batch_size if batch_size is not None else k * b
        noise_pen = rel_sd = exp_change = risk = None
        if lr is not None and r_hat is not None:
            if self.independent:
                noise_pen = lr * lr * r_hat / (2.0 * big_b)
                if mean_curv is not None and math.isfinite(progress):
                    exp_change = -lr * progress + 0.5 * lr * lr * mean_curv + noise_pen
                if math.isfinite(progress) and progress != 0:
                    risk = noise_pen / (lr * abs(progress))
            if conc is not None:
                rel_sd = math.sqrt(2.0 / conc)

        return StabilityMetrics(
            num_microbatches=k,
            microbatch_size=b,
            units=self.units,
            independent=self.independent,
            trace_sigma=trace_sigma,
            grad_norm_sq=grad_norm_sq,
            gns_euclidean=gns_e,
            trace_sigma_precond=trace_p,
            grad_norm_sq_precond=norm_p,
            gns_optimizer=gns_o,
            risk_trace_per_example=r_hat,
            risk_trace_se=r_se,
            risk_trace_ci=r_ci,
            directional_progress=progress,
            mean_curvature_term=mean_curv,
            concentration=conc,
            concentration_se=conc_se,
            concentration_plugin=conc_plugin,
            concentration_status=conc_status,
            batch_size=big_b,
            lr=lr,
            noise_penalty=noise_pen,
            noise_penalty_rel_sd=rel_sd,
            expected_loss_change=exp_change,
            risk_score=risk,
            update_map=self.update_map.name,
            update_map_details=dict(self.update_map.details),
            curvature_name=curv_name,
            curvature_psd=curv_psd,
            flags=flags,
            cost={
                "seconds": cache["seconds"],
                "curvature_calls": cache["curvature_calls"],
                "gradient_bytes": cache["gradient_bytes"],
            },
        )


def _student_t_quantile(p: float, dof: int) -> float:
    """Student-t quantile via the Cornish-Fisher expansion (accurate for dof >= 3)."""
    z = float(torch.special.ndtri(torch.tensor(p, dtype=torch.float64)))
    v = float(dof)
    return (z + (z ** 3 + z) / (4 * v) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * v * v)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * v ** 3))


def _jackknife_se(values: Sequence[float]) -> Optional[float]:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    return math.sqrt((n - 1) / n * sum((v - mean) ** 2 for v in values))


def aggregate_probe_groups(metrics: Sequence[StabilityMetrics]) -> dict[str, Optional[float]]:
    """Mean and standard error of key metrics across *independent* probe groups.

    Prefer this over the within-probe jackknife when you can afford several
    independent probes at the same checkpoint (fresh data for each).
    """
    out: dict[str, Optional[float]] = {"groups": float(len(metrics))}
    for key in ("risk_trace_per_example", "gns_euclidean", "gns_optimizer", "trace_sigma"):
        vals = [getattr(m, key) for m in metrics if getattr(m, key) is not None]
        if not vals:
            out[key] = out[f"{key}_se"] = None
            continue
        mean = sum(vals) / len(vals)
        out[key] = mean
        if len(vals) >= 2:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            out[f"{key}_se"] = math.sqrt(var / len(vals))
        else:
            out[f"{key}_se"] = None
    return out


# ---------------------------------------------------------------------------
# Ground truth and cheap spectral proxies
# ---------------------------------------------------------------------------

def exact_noise_geometry(
    hessian: Tensor,
    sigma: Tensor,
    precond: Optional[Tensor] = None,
) -> dict[str, float]:
    """Exact ``R = tr(H P Sigma P^T)`` and ``D_eff = R^2 / tr(A^2)`` from dense matrices.

    Args:
        hessian: PSD ``[n, n]`` curvature (raises if materially indefinite).
        sigma: Per-example gradient covariance ``[n, n]``.
        precond: ``P`` as ``[n, n]`` or a diagonal ``[n]`` (default identity).
    """
    h = 0.5 * (hessian + hessian.t()).to(torch.float64)
    evals = torch.linalg.eigvalsh(h)
    if float(evals.min()) < -1e-8 * float(evals.abs().max().clamp_min(1e-30)):
        raise ValueError("hessian is indefinite; exact_noise_geometry needs a PSD proxy.")
    s = sigma.to(torch.float64)
    if precond is None:
        m = s
    elif precond.ndim == 1:
        p = precond.to(torch.float64)
        m = p[:, None] * s * p[None, :]
    else:
        p = precond.to(torch.float64)
        m = p @ s @ p.t()
    hm = h @ m
    r = float(hm.diagonal().sum())
    tr_a2 = float((hm * hm.t()).sum())
    return {
        "risk_trace": r,
        "trace_a2": tr_a2,
        "concentration": r * r / tr_a2 if tr_a2 > 0 else float("nan"),
        "lambda_max": float(evals.max()),
    }


def spectrum_metrics(
    matrix_or_eigs: Tensor,
    *,
    sample_count: Optional[int] = None,
    fit_range: tuple[float, float] = (0.01, 0.5),
) -> dict[str, Any]:
    """Cheap spectral-concentration proxies of an activation correlation ``C_h``.

    Labelled ``kind="activation_spectrum"``: a steep activation spectrum is *not*
    the same as concentrated curvature-weighted update noise (a sharp mode with
    little update variance is harmless).

    Args:
        matrix_or_eigs: Symmetric PSD matrix, or a 1-D tensor of eigenvalues.
        sample_count: Rows behind an estimated matrix; ``status`` warns when the
            matrix is rank-limited by sampling (``sample_count < dim``).
        fit_range: Fraction-of-rank window for the log-log power-law slope fit.

    Returns:
        ``participation_ratio`` ((sum l)^2 / sum l^2), ``entropy_rank``,
        ``top1_share``, ``top10_share``, ``powerlaw_slope`` (of ``log l_i`` vs
        ``log i``), ``lambda_max``, ``dim``, ``status``.
    """
    t = matrix_or_eigs.detach().to(torch.float64)
    eigs = torch.linalg.eigvalsh(0.5 * (t + t.t())) if t.ndim == 2 else t.reshape(-1)
    eigs = torch.sort(eigs.clamp_min(0.0), descending=True).values
    dim = eigs.numel()
    total = float(eigs.sum())
    status = "ok"
    if sample_count is not None and sample_count < dim:
        status = "rank_limited_by_samples"
    if total <= 0:
        return {"kind": "activation_spectrum", "dim": dim, "status": "zero_spectrum"}
    p = eigs / total
    nz = p[p > 0]
    lo = max(0, int(fit_range[0] * dim))
    hi = max(lo + 2, int(fit_range[1] * dim))
    if sample_count is not None:
        hi = min(hi, max(lo + 2, sample_count))
    idx = torch.arange(1, dim + 1, dtype=torch.float64)[lo:hi]
    vals = eigs[lo:hi]
    keep = vals > total * 1e-12
    slope = float("nan")
    if int(keep.sum()) >= 2:
        x = idx[keep].log()
        y = vals[keep].log()
        x_c = x - x.mean()
        slope = float((x_c * (y - y.mean())).sum() / (x_c * x_c).sum())
    return {
        "kind": "activation_spectrum",
        "dim": dim,
        "participation_ratio": total * total / float((eigs * eigs).sum()),
        "entropy_rank": float(torch.exp(-(nz * nz.log()).sum())),
        "top1_share": float(p[0]),
        "top10_share": float(p[: min(10, dim)].sum()),
        "powerlaw_slope": slope,
        "lambda_max": float(eigs[0]),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Excursion (spike) detection
# ---------------------------------------------------------------------------

@dataclass
class ExcursionReport:
    """Loss excursions of one trajectory under the preregistered definition.

    Attributes:
        num_events: Transient excursion events (merged contiguous runs).
        max_excursion: Largest relative rise above the trailing baseline
            ``(L_t - base_t) / base_t`` (0 when none), over finite pre-divergence steps.
        event_steps: Start step of each event.
        durations: Length (steps) of each event.
        diverged: Sustained divergence (non-finite or above the divergence
            threshold at the end of the trajectory).
        divergence_step: First step at which divergence began, or ``None``.
        noise_scale: Robust scale of the relative residuals used in the test.
    """

    num_events: int
    max_excursion: float
    event_steps: list[int]
    durations: list[int]
    diverged: bool
    divergence_step: Optional[int]
    noise_scale: float

    @property
    def spiked(self) -> bool:
        return self.num_events > 0


def detect_excursions(
    losses: Sequence[float] | Tensor,
    *,
    window: int = 50,
    k_mad: float = 6.0,
    rel_floor: float = 0.25,
    abs_floor: float = 0.0,
    divergence_factor: float = 100.0,
    merge_gap: int = 5,
    warmup: Optional[int] = None,
) -> ExcursionReport:
    """Detect transient loss excursions and sustained divergence.

    An excursion at step ``t`` is a relative rise ``r_t = (L_t - base_t)/base_t``
    above the trailing-window median ``base_t`` (of the previous ``window``
    finite losses) that exceeds ``k_mad * robust_scale + rel_floor`` and whose
    absolute rise exceeds ``abs_floor``. ``robust_scale`` is ``1.4826 * MAD`` of
    all relative residuals. Runs closer than ``merge_gap`` steps merge into one
    event. Divergence: a non-finite loss or one above ``divergence_factor *
    L_0``; if it persists to the end of the trajectory it is *sustained* and the
    steps from its onset are excluded from excursion statistics.

    Fix these thresholds before looking at predictions.
    """
    x = torch.as_tensor(losses, dtype=torch.float64).reshape(-1)
    n = x.numel()
    if warmup is None:
        warmup = window
    finite = torch.isfinite(x)
    l0 = float(x[0]) if n and bool(finite[0]) else float("nan")
    bad = ~finite
    if math.isfinite(l0):
        bad = bad | (x > divergence_factor * abs(l0))
    diverged = False
    div_step: Optional[int] = None
    if bool(bad.any()):
        if bool(bad[-1]):
            # Sustained: find the onset of the final bad run.
            j = n - 1
            while j > 0 and bool(bad[j - 1]):
                j -= 1
            diverged = True
            div_step = j
        else:
            div_step = int(torch.nonzero(bad)[0])
    end = div_step if diverged else n

    rel = torch.full((n,), float("nan"), dtype=torch.float64)
    for t in range(warmup, end):
        lo = max(0, t - window)
        prev = x[lo:t]
        prev = prev[torch.isfinite(prev)]
        if prev.numel() == 0 or not bool(finite[t]):
            continue
        base = float(prev.median())
        if base <= 0:
            continue
        rel[t] = (x[t] - base) / base
        if x[t] - base <= abs_floor:
            rel[t] = min(float(rel[t]), 0.0)
    valid = rel[torch.isfinite(rel)]
    if valid.numel() == 0:
        return ExcursionReport(0, 0.0, [], [], diverged, div_step, float("nan"))
    med = valid.median()
    scale = float(1.4826 * (valid - med).abs().median())
    thresh = k_mad * scale + rel_floor
    hits = [t for t in range(n) if math.isfinite(float(rel[t])) and float(rel[t]) > thresh]
    events: list[list[int]] = []
    for t in hits:
        if events and t - events[-1][-1] <= merge_gap:
            events[-1].append(t)
        else:
            events.append([t])
    return ExcursionReport(
        num_events=len(events),
        max_excursion=max(0.0, float(valid.max())),
        event_steps=[e[0] for e in events],
        durations=[e[-1] - e[0] + 1 for e in events],
        diverged=diverged,
        divergence_step=div_step,
        noise_scale=scale,
    )
