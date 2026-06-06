"""
Variance-estimation backends for the SNR / population-risk gate.

The SNR gate compares the squared bias-corrected first moment ``m_hat**2`` to an
estimate of the variance of the *minibatch mean* gradient, ``s``. ``SNRAdamW``
maintains a cheap streaming EMA of that quantity internally, but the optimizer
also exposes a hook::

    optimizer.step(grad_variances={param: s_tensor})

that lets you replace the EMA with a better per-step estimate while still
updating the internal EMA for continuity. This module provides interchangeable
backends that produce such ``grad_variances`` dictionaries:

* :class:`ExactVarianceEstimator` -- per-sample gradients via ``torch.func``
  (``grad`` + ``vmap``), giving the exact diagonal variance of the minibatch
  mean gradient.
* :class:`MicrobatchVarianceEstimator` /
  :func:`backward_with_microbatch_variance` -- a cheap split-batch estimator that
  uses ``K`` ordinary backward passes instead of per-example gradients.

All estimators return ``s`` on the *same scale as the internal ``s_hat``*: the
unbiased per-example gradient variance divided by the batch size. This matches
:func:`snr_grad.per_sample_variance_term`.

Limitations (see README for details):

* Exact per-sample gradients may not work cleanly with all modules or custom ops,
  and require a *deterministic* model (control dropout / RNG) for clean estimates.
* BatchNorm in train mode couples examples through batch statistics, so it is
  excluded from exact probes by default (``exclude_norm=True``) and a warning is
  emitted if a BatchNorm layer is found in training mode.
* Microbatch variance is biased if the chunks are not comparable (e.g. strong
  augmentation differences or stateful layers).
* Variance estimates are local to the current process; distributed aggregation is
  out of scope for this module.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

import torch
from torch import Tensor, nn

from snr_grad._core import (
    GateType,
    compute_gate,
    per_sample_variance_term,
    resolve_alpha,
)

__all__ = [
    "VarianceEstimator",
    "per_sample_grad_variances",
    "ExactVarianceEstimator",
    "MicrobatchVarianceEstimator",
    "backward_with_microbatch_variance",
    "tree_batch_size",
    "tree_split",
    "compare_gate_with_external_variance",
]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class VarianceEstimator(Protocol):
    """Structural type for variance backends.

    An estimator maps a model + loss + batch to a dict ``{param_tensor: s}`` where
    ``s`` is the variance of the minibatch mean gradient for that parameter, on the
    same scale as ``SNRAdamW``'s internal ``s_hat``.
    """

    def estimate(
        self,
        model: nn.Module,
        loss_fn: Callable,
        batch: Any,
        *,
        params: Optional[Mapping[str, Tensor]] = None,
    ) -> Mapping[Tensor, Tensor]:
        ...


# ---------------------------------------------------------------------------
# Pytree helpers (minimal: Tensor / tuple / list / dict)
# ---------------------------------------------------------------------------

def tree_batch_size(batch: Any) -> int:
    """Return the leading (batch) dimension size of the first tensor leaf found."""
    if isinstance(batch, Tensor):
        if batch.ndim < 1:
            raise ValueError("Cannot infer batch size from a 0-d tensor leaf.")
        return batch.shape[0]
    if isinstance(batch, Mapping):
        for v in batch.values():
            return tree_batch_size(v)
    if isinstance(batch, (tuple, list)):
        for v in batch:
            return tree_batch_size(v)
    raise ValueError(
        f"Could not find a tensor leaf to infer batch size from in {type(batch).__name__}."
    )


def tree_split(batch: Any, num_splits: int) -> list:
    """Split a batch pytree into ``num_splits`` contiguous chunks along dim 0.

    Chunks are as even as possible (``torch.tensor_split`` semantics). Non-tensor
    leaves are shared by reference across all chunks. Returns a list of sub-batches
    with the same structure as ``batch``.
    """
    if num_splits < 1:
        raise ValueError(f"num_splits must be >= 1, got {num_splits}.")

    if isinstance(batch, Tensor):
        return list(torch.tensor_split(batch, num_splits, dim=0))
    if isinstance(batch, Mapping):
        split_values = {k: tree_split(v, num_splits) for k, v in batch.items()}
        return [type(batch)({k: split_values[k][i] for k in batch}) for i in range(num_splits)]
    if isinstance(batch, (tuple, list)):
        split_items = [tree_split(v, num_splits) for v in batch]
        rebuilt = [type(batch)(item[i] for item in split_items) for i in range(num_splits)]
        return rebuilt
    # Non-tensor leaf: share across chunks.
    return [batch for _ in range(num_splits)]


def _tree_cast_floating(batch: Any, dtype: torch.dtype) -> Any:
    """Cast floating-point tensor leaves of a pytree to ``dtype``; share other leaves."""
    if isinstance(batch, Tensor):
        return batch.to(dtype) if torch.is_floating_point(batch) else batch
    if isinstance(batch, Mapping):
        return type(batch)({k: _tree_cast_floating(v, dtype) for k, v in batch.items()})
    if isinstance(batch, (tuple, list)):
        return type(batch)(_tree_cast_floating(v, dtype) for v in batch)
    return batch


# ---------------------------------------------------------------------------
# Exact per-sample variance backend
# ---------------------------------------------------------------------------

def per_sample_grad_variances(
    model: nn.Module,
    loss_one_sample_fn: Callable[[Mapping[str, Tensor], Mapping[str, Tensor], Any], Tensor],
    batch: Any,
    *,
    params: Optional[Mapping[str, Tensor]] = None,
    buffers: Optional[Mapping[str, Tensor]] = None,
    chunk_size: Optional[int] = None,
) -> dict[Tensor, Tensor]:
    """Exact diagonal variance of the minibatch mean gradient, per parameter.

    Uses ``torch.func.grad`` + ``torch.vmap`` to compute per-example gradients, then
    returns the unbiased per-example variance divided by the batch size
    (:func:`snr_grad.per_sample_variance_term`) for each differentiated parameter.

    Args:
        model:
            The module whose parameters are differentiated. Used to source default
            params/buffers and to key the output by the *live* parameter tensors.
        loss_one_sample_fn:
            ``loss_one_sample_fn(params, buffers, sample) -> scalar``. Receives the
            functional ``params``/``buffers`` dicts and a single example ``sample``
            (one ``vmap`` slice of ``batch``), and returns that example's scalar loss.
            Typically calls ``torch.func.functional_call(model, (params, buffers), ...)``.
        batch:
            A pytree (tensor / tuple / list / dict) whose tensor leaves have a leading
            batch dimension; ``vmap`` maps over dim 0 of each leaf.
        params:
            Optional subset of named parameters to differentiate. Defaults to all of
            ``model.named_parameters()``. Restricting this saves memory for large heads
            or embeddings. Parameters not in ``params`` should be supplied through
            ``buffers`` so ``functional_call`` can still see them.
        buffers:
            Optional buffers dict for ``functional_call``. Defaults to
            ``model.named_buffers()``.
        chunk_size:
            Optional ``vmap`` chunk size, trading memory for speed.

    Returns:
        ``{param_tensor: variance_tensor}`` keyed by the model's live parameter
        tensors, with each variance tensor matching that parameter's shape.
    """
    from torch.func import grad, vmap

    param_by_name = dict(model.named_parameters())
    if params is None:
        params = param_by_name
    else:
        params = dict(params)
    if buffers is None:
        buffers = dict(model.named_buffers())
    else:
        buffers = dict(buffers)

    grad_fn = grad(loss_one_sample_fn)  # differentiate w.r.t. params (argnums=0)
    per_sample_grads = vmap(
        grad_fn,
        in_dims=(None, None, 0),
        chunk_size=chunk_size,
    )(params, buffers, batch)

    out: dict[Tensor, Tensor] = {}
    for name, g in per_sample_grads.items():
        out[param_by_name[name]] = per_sample_variance_term(g)
    return out


def _is_norm_module(module: nn.Module) -> bool:
    """True for normalization layers whose params we exclude from exact probes."""
    norm_types: tuple = (
        nn.modules.batchnorm._BatchNorm,
        nn.GroupNorm,
        nn.LayerNorm,
        nn.LocalResponseNorm,
    )
    # InstanceNorm subclasses _BatchNorm, so it is already covered.
    return isinstance(module, norm_types)


def _norm_param_names(model: nn.Module) -> set:
    """Names of parameters that belong to normalization layers."""
    names: set = set()
    for mod_name, module in model.named_modules():
        if _is_norm_module(module):
            for p_name, _ in module.named_parameters(recurse=False):
                full = f"{mod_name}.{p_name}" if mod_name else p_name
                names.add(full)
    return names


def _has_training_batchnorm(model: nn.Module) -> bool:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
            return True
    return False


def _cast_floating(d: Mapping[str, Tensor], dtype: torch.dtype) -> dict[str, Tensor]:
    """Cast floating-point tensors in a dict to ``dtype``; leave others untouched."""
    return {
        k: (v.to(dtype) if torch.is_floating_point(v) else v)
        for k, v in d.items()
    }


def _name_matches(name: str, patterns: Sequence[str]) -> bool:
    return any(pat in name for pat in patterns)


class ExactVarianceEstimator:
    """Exact per-sample-gradient variance backend.

    Computes the exact diagonal variance of the minibatch mean gradient via
    ``torch.func`` per-sample gradients. By default the computation runs in fp32
    (robust under mixed precision) and the result is cast back to each parameter's
    dtype before being handed to the optimizer.

    Args:
        chunk_size:
            Optional ``vmap`` chunk size, trading memory for speed.
        include_params:
            If given, only parameters whose name contains one of these substrings are
            differentiated. All other parameters fall back to the optimizer's EMA.
        exclude_params:
            Substrings; parameters whose name contains any of them are skipped.
        exclude_norm:
            If True (default), normalization-layer parameters (BatchNorm, GroupNorm,
            LayerNorm, ...) are skipped. Exact per-sample gradients through BatchNorm in
            train mode are ill-defined because examples interact via batch statistics.
        dtype:
            Compute dtype for the probe (default fp32). Set to ``None`` to compute in the
            model's native dtype.
        warn_batchnorm:
            If True (default), emit a warning when a BatchNorm layer is in training mode.
    """

    def __init__(
        self,
        *,
        chunk_size: Optional[int] = None,
        include_params: Optional[Iterable[str]] = None,
        exclude_params: Optional[Iterable[str]] = None,
        exclude_norm: bool = True,
        dtype: Optional[torch.dtype] = torch.float32,
        warn_batchnorm: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.include_params = list(include_params) if include_params is not None else None
        self.exclude_params = list(exclude_params) if exclude_params is not None else None
        self.exclude_norm = exclude_norm
        self.dtype = dtype
        self.warn_batchnorm = warn_batchnorm

    def _selected_names(self, model: nn.Module, all_names: Iterable[str]) -> set:
        norm_names = _norm_param_names(model) if self.exclude_norm else set()
        selected = set()
        for name in all_names:
            if self.include_params is not None and not _name_matches(name, self.include_params):
                continue
            if self.exclude_params is not None and _name_matches(name, self.exclude_params):
                continue
            if name in norm_names:
                continue
            selected.add(name)
        return selected

    def estimate(
        self,
        model: nn.Module,
        loss_one_sample_fn: Callable,
        batch: Any,
        *,
        params: Optional[Mapping[str, Tensor]] = None,
    ) -> dict[Tensor, Tensor]:
        """Return ``{param_tensor: s}`` for the selected parameters.

        Parameters that are filtered out are simply absent from the returned dict, so
        the optimizer falls back to its internal EMA for them.
        """
        if self.warn_batchnorm and _has_training_batchnorm(model):
            warnings.warn(
                "ExactVarianceEstimator: model has a BatchNorm layer in training mode. "
                "Per-sample gradients couple examples through batch statistics; BatchNorm "
                "parameters are excluded by default. Consider model.eval() for clean probes.",
                RuntimeWarning,
                stacklevel=2,
            )

        all_params = dict(model.named_parameters()) if params is None else dict(params)
        selected = self._selected_names(model, all_params.keys())
        if not selected:
            return {}

        diff_params = {n: p for n, p in all_params.items() if n in selected}
        fixed_params = {n: p for n, p in all_params.items() if n not in selected}
        # Fixed (non-differentiated) params are passed through functional_call via the
        # buffers dict so the forward pass still sees the full parameter set.
        merged_buffers = {**dict(model.named_buffers()), **fixed_params}

        compute_params: Mapping[str, Tensor] = diff_params
        compute_batch = batch
        if self.dtype is not None:
            compute_params = _cast_floating(diff_params, self.dtype)
            merged_buffers = _cast_floating(merged_buffers, self.dtype)
            # Cast floating inputs to the compute dtype so the forward pass matches the
            # (possibly upcast) parameters under mixed precision.
            compute_batch = _tree_cast_floating(batch, self.dtype)

        raw = per_sample_grad_variances(
            model,
            loss_one_sample_fn,
            compute_batch,
            params=compute_params,
            buffers=merged_buffers,
            chunk_size=self.chunk_size,
        )
        # raw is keyed by the model's live parameter tensors; cast back to their dtype.
        return {p: var.to(dtype=p.dtype) for p, var in raw.items()}


# ---------------------------------------------------------------------------
# Cheap split-batch (microbatch) backend
# ---------------------------------------------------------------------------

def backward_with_microbatch_variance(
    model: nn.Module,
    loss_fn: Callable[[nn.Module, Any], Tensor],
    batch: Any,
    *,
    num_splits: int = 2,
    accumulate_full_grad: bool = True,
    loss_reduction: str = "mean",
) -> tuple[float, dict[Tensor, Tensor]]:
    """Cheap split-batch variance estimator that owns the backward pass.

    Splits ``batch`` into ``num_splits`` (``K``) contiguous chunks, runs one ordinary
    backward per chunk to get chunk-mean gradients ``h_1, ..., h_K``, and estimates the
    variance of the full-batch mean gradient as ``Var_unbiased(h_j) / K``. For ``K=2``
    this reduces to ``(h_1 - h_2)**2 / 4``.

    When ``accumulate_full_grad`` is True, each parameter's ``.grad`` is set to the mean
    of the chunk gradients (the full-batch mean gradient, for equal chunk sizes and
    mean-reduced loss), so the caller can ``optimizer.step(grad_variances=...)`` directly
    without a separate backward.

    Args:
        model:
            The module to differentiate.
        loss_fn:
            ``loss_fn(model, sub_batch) -> scalar``. Performs the forward pass for a chunk
            and returns its (mean-reduced) scalar loss. Keeping the forward inside the
            callback avoids any universal batch parser.
        batch:
            A pytree split along dim 0 into ``num_splits`` chunks.
        num_splits:
            Number of chunks ``K`` (>= 2). Each chunk needs >= 1 example.
        accumulate_full_grad:
            If True, write the averaged chunk gradient into each ``param.grad``.
        loss_reduction:
            Only ``"mean"`` is supported in this version (the chunk loss must already be
            mean-reduced so chunk gradients are directly averageable).

    Returns:
        ``(mean_loss, grad_variances)`` where ``grad_variances`` maps each parameter to
        its estimated ``s`` (variance of the minibatch mean gradient).
    """
    if num_splits < 2:
        raise ValueError(f"num_splits must be >= 2 to estimate a variance, got {num_splits}.")
    if loss_reduction != "mean":
        raise ValueError(
            f"backward_with_microbatch_variance only supports loss_reduction='mean', "
            f"got {loss_reduction!r}. Ensure loss_fn returns a mean-reduced scalar."
        )

    b = tree_batch_size(batch)
    if b < num_splits:
        raise ValueError(
            f"batch size ({b}) must be >= num_splits ({num_splits}) so each chunk is non-empty."
        )

    chunks = tree_split(batch, num_splits)
    trainable = [p for p in model.parameters() if p.requires_grad]
    per_param_grads: dict[Tensor, list[Tensor]] = {p: [] for p in trainable}

    total_loss = 0.0
    for chunk in chunks:
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, chunk)
        loss.backward()
        for p in trainable:
            if p.grad is not None:
                per_param_grads[p].append(p.grad.detach().clone())
        total_loss += float(loss.detach())

    model.zero_grad(set_to_none=True)

    grad_variances: dict[Tensor, Tensor] = {}
    K = len(chunks)
    for p, grads in per_param_grads.items():
        if len(grads) < 2:
            continue
        stacked = torch.stack(grads, dim=0)  # [K, *p.shape]
        if accumulate_full_grad:
            p.grad = stacked.mean(dim=0)
        # Var of the full-batch mean gradient = Var_unbiased(chunk means) / K.
        grad_variances[p] = stacked.var(dim=0, unbiased=True) / K

    return total_loss / K, grad_variances


class MicrobatchVarianceEstimator:
    """Cheap split-batch variance backend.

    Thin wrapper around :func:`backward_with_microbatch_variance`. Because the split-batch
    estimator requires its own backward passes, ``estimate`` *owns the backward* and (by
    default) leaves the full-batch mean gradient in each ``param.grad`` ready for
    ``optimizer.step``. The most recent loss is stored on ``last_loss``.

    Args:
        num_splits:
            Number of chunks ``K`` (>= 2).
        accumulate_full_grad:
            If True, write the averaged chunk gradient into each ``param.grad``.
        loss_reduction:
            Only ``"mean"`` is supported.
    """

    def __init__(
        self,
        num_splits: int = 2,
        *,
        accumulate_full_grad: bool = True,
        loss_reduction: str = "mean",
    ) -> None:
        if num_splits < 2:
            raise ValueError(f"num_splits must be >= 2, got {num_splits}.")
        self.num_splits = num_splits
        self.accumulate_full_grad = accumulate_full_grad
        self.loss_reduction = loss_reduction
        self.last_loss: Optional[float] = None

    def estimate(
        self,
        model: nn.Module,
        loss_fn: Callable[[nn.Module, Any], Tensor],
        batch: Any,
        *,
        params: Optional[Mapping[str, Tensor]] = None,
    ) -> dict[Tensor, Tensor]:
        """Run ``num_splits`` backward passes and return ``{param: s}``.

        ``params`` is accepted for protocol compatibility but ignored; the microbatch
        estimator always differentiates the model's trainable parameters.
        """
        loss, grad_variances = backward_with_microbatch_variance(
            model,
            loss_fn,
            batch,
            num_splits=self.num_splits,
            accumulate_full_grad=self.accumulate_full_grad,
            loss_reduction=self.loss_reduction,
        )
        self.last_loss = loss
        return grad_variances


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compare_gate_with_external_variance(
    optimizer: Any,
    grad_variances: Mapping[Tensor, Tensor],
    *,
    gate: Optional[GateType] = None,
) -> dict[str, float]:
    """Compare the gate computed from internal EMA vs. an external variance estimate.

    Answers "would the supplied ``grad_variances`` actually change the gate?" without
    mutating any optimizer state. Uses the current bias-corrected first moment and the
    EMA ``s_hat`` already stored on the optimizer.

    Returns a dict with, where defined:
        ``mean_internal_s``, ``mean_external_s``, ``mean_variance_ratio``
            (external / internal, EMA-weighted means over covered elements),
        ``mean_gate_internal``, ``mean_gate_external``,
        ``frac_gate_changed`` (fraction of elements where the two gates differ by > 0.25),
        ``log_corr`` (Pearson correlation of log internal vs log external variance),
        ``elements_compared``.
    """
    eps = 1e-12
    internal_vals: list[Tensor] = []
    external_vals: list[Tensor] = []
    gate_internal_vals: list[Tensor] = []
    gate_external_vals: list[Tensor] = []

    for group in optimizer.param_groups:
        beta1 = group["betas"][0]
        rho = group["rho"]
        gate_type: GateType = gate if gate is not None else group["gate"]
        lambda_pop = group["lambda_pop"]
        gate_eps = group["gate_eps"]
        alpha_value = resolve_alpha(
            group["alpha"],
            batch_size=group.get("batch_size"),
            dataset_size=group.get("dataset_size"),
        )
        for p in group["params"]:
            if p not in grad_variances:
                continue
            state = optimizer.state.get(p)
            if not state or "step" not in state:
                continue
            step_num = state["step"]
            bc1 = 1.0 - beta1 ** step_num
            bc_s = 1.0 - rho ** step_num
            m_hat = state["exp_avg"] / bc1
            s_internal = state["exp_grad_var"] / bc_s
            s_external = grad_variances[p].to(device=p.device, dtype=s_internal.dtype)

            q_int = compute_gate(
                m_hat, s_internal, gate=gate_type, alpha=alpha_value,
                lambda_pop=lambda_pop, gate_eps=gate_eps,
            )
            q_ext = compute_gate(
                m_hat, s_external, gate=gate_type, alpha=alpha_value,
                lambda_pop=lambda_pop, gate_eps=gate_eps,
            )
            internal_vals.append(s_internal.reshape(-1))
            external_vals.append(s_external.reshape(-1))
            gate_internal_vals.append(q_int.reshape(-1))
            gate_external_vals.append(q_ext.reshape(-1))

    if not internal_vals:
        return {"elements_compared": 0.0}

    s_int = torch.cat(internal_vals)
    s_ext = torch.cat(external_vals)
    q_int = torch.cat(gate_internal_vals)
    q_ext = torch.cat(gate_external_vals)

    ratio = s_ext / (s_int + eps)
    log_int = torch.log(s_int.clamp_min(eps))
    log_ext = torch.log(s_ext.clamp_min(eps))
    if log_int.numel() > 1 and torch.std(log_int) > 0 and torch.std(log_ext) > 0:
        log_corr = float(torch.corrcoef(torch.stack([log_int, log_ext]))[0, 1])
    else:
        log_corr = float("nan")

    return {
        "mean_internal_s": float(s_int.mean()),
        "mean_external_s": float(s_ext.mean()),
        "mean_variance_ratio": float(ratio.mean()),
        "mean_gate_internal": float(q_int.mean()),
        "mean_gate_external": float(q_ext.mean()),
        "frac_gate_changed": float(((q_int - q_ext).abs() > 0.25).float().mean()),
        "log_corr": log_corr,
        "elements_compared": float(s_int.numel()),
    }
