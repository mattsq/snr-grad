"""
Activation preconditioning (AP) and Double Preconditioning (DoPr).

This module implements the **activation preconditioner** from "Double
Preconditioning (DoPr): Optimization for Test-Time Performance, not Validation
Loss" (arXiv:2606.06418) as a drop-in stage in front of *any* gradient-based
optimizer (a "gradient preconditioner" / GP, such as the SNR-gated optimizers in
this package, or plain ``torch.optim.Adam`` / Muon).

For a linear layer with weight ``W`` (shape ``[out, in]``) whose input
activations are ``z`` (shape ``[..., in]``), AP left-conditions the gradient by
the inverse uncentered covariance of the layer *inputs*::

    G    = dL/dW                                     # [out, in]
    S_z  = (1/n) sum_i z_i z_i^T                     # [in, in], uncentered cov
    M    = G @ (S_z + gamma * tr(S_z)/d_z * I)^-1    # AP step (damped solve)
    D    = GP(M)                                     # base optimizer consumes M
    W   <- (1 - eta*lambda) W - eta * D              # base optimizer's own update

AP debiases the gradient from anisotropic activation statistics, which encourages
more uniform / isotropic feature learning. The paper shows this mitigates
*test-time feedback* (TTF) -- the compounding distribution shift that arises when
a model trained on a one-step prediction loss is deployed by rolling out on its
own outputs (autoregressive language models, robot policies) -- and that the
benefit often does not show up in validation loss.

Design (mirrors :mod:`snr_grad.variance`): AP needs model/module awareness, which
the optimizers deliberately lack. So, like the variance estimators, it is an
*external* helper. :class:`ActivationPreconditioner` registers forward-pre-hooks
to capture layer inputs, then :meth:`ActivationPreconditioner.precondition_`
rewrites each registered weight's ``.grad`` in place **after ``backward()`` and
before ``optimizer.step()``**. Because the map ``G -> G @ S^-1`` is linear and
only touches ``.grad``, it commutes with ``maximize`` (negation) and is
orthogonal to decoupled weight decay and schedule-free parameter swapping (which
act on ``p.data`` inside ``step``). The base optimizer needs no changes.

Usage::

    from snr_grad import SNRAdamW, DoPr, ActivationPrecondConfig

    # Convenience wrapper (precondition_ then base.step()):
    opt = DoPr(SNRAdamW(model.parameters(), lr=3e-4), model,
               ActivationPrecondConfig(damping=0.1))
    loss.backward(); opt.step(); opt.zero_grad()

    # Or the external form, which works with ANY optimizer:
    ap = ActivationPreconditioner(model, ActivationPrecondConfig(damping=0.1))
    loss.backward(); ap.precondition_()
    optimizer.step(); optimizer.zero_grad(set_to_none=True); ap.zero_grad()

Because a GP normalizes the update magnitude, substituting the AP gradient for
the raw gradient does not change the update norm, so existing maximal-update
parameterization (muP) / learning-rate / weight-decay scaling rules of the base
GP transfer directly to DoPr.

Limitations:

* Supported layers: :class:`torch.nn.Linear` (covers attention Q/K/V/out
  projections) and :class:`torch.nn.Embedding` (one-hot inputs => diagonal
  covariance = per-token counts, handled efficiently). Convolutions raise
  ``NotImplementedError`` unless explicitly excluded; conv (unfold / SUA) support
  is a future extension.
* Tied / shared weights (e.g. a weight that is the ``.weight`` of more than one
  registered module, such as tied input/output embeddings) are skipped with a
  warning, because the correct AP is ambiguous (different, even differently
  shaped, input covariances).
* The covariance and its solve run in fp32 by default for numerical stability.
* The covariance is formed from *all* positions in the layer input
  (``z.reshape(-1, d_in)``). The module hook cannot see an attention/padding mask,
  so padded positions in a ``[batch, seq, d]`` input are folded into ``S_z``; with
  heavy right-padding this biases the preconditioner toward pad-token statistics.
  Mask/pad-aware capture is a future extension.
* Activation/gradient checkpointing (``torch.utils.checkpoint`` with the modern
  ``use_reentrant=False``) recomputes the wrapped forward during backward, so the
  capture hook fires twice for a checkpointed module. When checkpointing is applied
  *uniformly* (every registered module recomputed once -- the usual "checkpoint each
  block" pattern) both ``gram`` and ``count`` double and ``S_z`` is unchanged, so AP
  is correct. But if the *same* module runs both inside and outside a checkpoint in
  one step (e.g. checkpointing the blocks but not the final ``nn.Linear`` head, or a
  shared module reused across checkpointed and non-checkpointed regions), the
  checkpointed call's samples are weighted 2x in ``S_z`` -- a silent bias (no crash).
  Keep checkpointing uniform across registered modules, or exclude the offending
  module via ``exclude_modules``.
* Covariances are computed from *local* activations and, under DistributedDataParallel,
  are all-reduced across the process group before the solve (auto-detected via
  ``torch.distributed.is_initialized()``; set ``ActivationPrecondConfig(distributed=False)``
  to opt out). This is required for correctness: ``.grad`` is already all-reduced by the
  time ``precondition_`` runs, so every rank must solve with the same covariance to stay
  in sync. **Requirements for the distributed path** (else it will hang or mis-precondition):
  (a) every rank runs an identical model so the same modules are registered in the same
  order; (b) the step counter / ``warmup_steps`` are in lockstep, so every rank either
  warms up or preconditions on a given step (a desynced ``_step`` makes one rank skip the
  collective and deadlock the others); (c) ``.grad`` really is the all-reduced gradient when
  ``precondition_`` runs -- so call it only on a synchronizing step, NOT inside DDP
  ``no_sync()`` / on non-final gradient-accumulation microbatches; (d) auto-detection assumes
  any initialized process group is the DDP group whose grads were averaged -- if you
  initialize a process group for some *other* purpose (and grads are not DDP-averaged), pass
  ``distributed=False``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn

__all__ = [
    "ActivationPrecondConfig",
    "ActivationPreconditioner",
    "DoPr",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ActivationPrecondConfig:
    """Configuration for :class:`ActivationPreconditioner`.

    Args:
        damping:
            ``gamma`` in ``S_z + (gamma * tr(S_z)/d_z + damping_floor) * I``.
            Scale-invariant: this term is set relative to the magnitude of
            ``S_z``. Must be > 0. Note that the *relative* term alone does not
            guarantee invertibility (it vanishes when ``S_z`` is all-zero, e.g. a
            layer whose inputs are fully masked); the absolute ``damping_floor``
            provides that guarantee.
        damping_floor:
            Absolute floor added to the damping (default ``1e-8``), so the damped
            covariance is positive-definite even when ``S_z`` is zero or
            rank-deficient. Must be >= 0; internally clamped to a strictly
            positive minimum (``1e-12``) so even ``damping_floor == 0`` cannot
            produce a singular solve.
        ema_beta:
            If set, the activation covariance is an exponential moving average
            ``S <- ema_beta * S + (1 - ema_beta) * S_batch`` instead of the raw
            batch covariance. ``None`` (default) uses the batch covariance, as
            recommended by the paper (the base GP already smooths the gradient;
            EMA is applied to the covariance only, never to the gradient).
        warmup_steps:
            Number of initial :meth:`ActivationPreconditioner.precondition_`
            calls during which AP is the identity (plain GP). Useful when early
            activations are ill-conditioned (e.g. zero-initialized blocks).
            Warmup is evaluated against the (possibly restored) internal step
            counter, so after :meth:`load_state_dict` it counts from the resumed
            step, not from zero -- changing ``warmup_steps`` on resume takes
            effect relative to that restored counter.
        include_modules:
            If given, only modules whose qualified name contains one of these
            substrings are preconditioned.
        exclude_modules:
            Modules whose qualified name contains one of these substrings are
            left untouched (and, for unsupported layers like conv, bypass the
            ``NotImplementedError``).
        include_linear:
            Register :class:`torch.nn.Linear` layers (default True).
        include_embedding:
            Register :class:`torch.nn.Embedding` layers (default True).
        compute_dtype:
            Dtype for forming the covariance and its solve (default fp32). Set to
            ``None`` to use the activation's native dtype. The linear Cholesky solve
            is internally promoted to fp32 when this dtype is fp16/bf16 (which
            ``torch.linalg.cholesky`` does not support), then cast back to the grad
            dtype.
        use_stale_ema_on_missing:
            When ``ema_beta`` is set and a registered module did not run a forward
            pass this step (so there is no fresh covariance) but its weight still
            has a gradient, reuse the most recent EMA covariance (so calling
            ``precondition_`` twice without an intervening backward re-applies that
            stale covariance). If False, such a weight is left as plain GP for that
            step. Ignored when ``ema_beta`` is
            ``None``.
        distributed:
            Whether to all-reduce the activation covariance across the process
            group before solving. Required for correctness under DDP: ``.grad`` is
            already all-reduced (identical on every rank) by the time
            :meth:`ActivationPreconditioner.precondition_` runs, but the
            covariance is computed from *local* activations, so without this every
            rank would solve with a different ``S_z`` and the ranks would diverge.
            ``None`` (default) auto-detects (enabled iff
            ``torch.distributed.is_initialized()``); pass ``True``/``False`` to
            force. Requires standard DDP with the ranks in lockstep (identical
            model, identical step counter / ``warmup_steps``, and ``.grad`` already
            all-reduced) -- see the module docstring's distributed requirements; a
            desynced step counter or a non-DDP process group will hang or
            mis-precondition. Set ``False`` for non-DDP process groups.
        process_group:
            Process group for the covariance all-reduce (default group when
            ``None``). Ignored unless distributed reduction is active.
    """

    damping: float = 0.1
    damping_floor: float = 1e-8
    ema_beta: Optional[float] = None
    warmup_steps: int = 0
    include_modules: Optional[Sequence[str]] = None
    exclude_modules: Optional[Sequence[str]] = None
    include_linear: bool = True
    include_embedding: bool = True
    compute_dtype: Optional[torch.dtype] = torch.float32
    use_stale_ema_on_missing: bool = True
    distributed: Optional[bool] = None
    process_group: Any = None

    def __post_init__(self) -> None:
        if self.damping <= 0:
            raise ValueError(f"damping must be > 0, got {self.damping}.")
        if self.damping_floor < 0:
            raise ValueError(f"damping_floor must be >= 0, got {self.damping_floor}.")
        if self.ema_beta is not None and not 0.0 <= self.ema_beta < 1.0:
            raise ValueError(f"ema_beta must be in [0, 1), got {self.ema_beta}.")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LINEAR = "linear"
_EMBEDDING = "embedding"

# Strictly-positive lower bound for the damping floor, so the damped covariance is
# always positive-definite even when ``S_z`` is exactly zero (fully masked inputs).
_TINY = 1e-12


def _name_matches(name: str, patterns: Optional[Sequence[str]]) -> bool:
    return patterns is not None and any(pat in name for pat in patterns)


def _is_conv(module: nn.Module) -> bool:
    return isinstance(module, nn.modules.conv._ConvNd)


# ---------------------------------------------------------------------------
# Activation preconditioner
# ---------------------------------------------------------------------------

class ActivationPreconditioner:
    """Activation-covariance preconditioner (AP) for DoPr.

    Registers forward-pre-hooks on the supported layers of ``model`` to capture
    their input activations, then :meth:`precondition_` rewrites each registered
    weight's ``.grad`` in place as ``G @ S_z^-1`` (damped). Call it after
    ``loss.backward()`` and before ``optimizer.step()``.

    Args:
        model:
            The module to precondition. Hooks are registered on its supported
            submodules at construction time.
        config:
            An :class:`ActivationPrecondConfig` (defaults are used if omitted).

    Raises:
        NotImplementedError: if the model contains a convolution layer that is not
            excluded via ``config.exclude_modules`` (conv AP is not implemented).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ActivationPrecondConfig] = None,
    ) -> None:
        self.model = model
        self.config = config if config is not None else ActivationPrecondConfig()

        # Per-module bookkeeping.
        self._kind: dict[nn.Module, str] = {}
        self._weight_of_module: dict[nn.Module, Tensor] = {}
        self._name_of_module: dict[nn.Module, str] = {}
        self._handles: list[Any] = []

        # Per-step capture (cleared every precondition_): accumulated Gram /
        # token counts keyed by module.
        # linear:    {"gram": Tensor[in, in], "count": int}
        # embedding: {"counts": Tensor[vocab], "count": int}
        self._accum: dict[nn.Module, dict[str, Any]] = {}

        # Persistent EMA covariance keyed by module (S for linear, diag for
        # embedding). Only populated when config.ema_beta is set.
        self._sigma_ema: dict[nn.Module, Tensor] = {}

        self._step = 0

        # One-time guard so a fp16 cast-back overflow warns at most once.
        self._warned_fp16_overflow = False

        self._register(model)

    # -- registration -------------------------------------------------------

    def _supported_kind(self, module: nn.Module) -> Optional[str]:
        cfg = self.config
        if cfg.include_linear and isinstance(module, nn.Linear):
            return _LINEAR
        if cfg.include_embedding and isinstance(module, nn.Embedding):
            return _EMBEDDING
        return None

    def _register(self, model: nn.Module) -> None:
        cfg = self.config
        # Tied-weight detection scans the FULL module tree (not just the modules
        # that survive include/exclude filtering): a weight tied to an excluded or
        # filtered-out module is still ambiguous and must be skipped.
        weight_owners: dict[int, list[str]] = {}
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if isinstance(weight, Tensor):
                weight_owners.setdefault(id(weight), []).append(name)

        # First pass: figure out which modules to register and reject unsupported
        # conv layers / warn about layers whose hook will not fire.
        candidates: list[tuple[str, nn.Module, str]] = []
        for name, module in model.named_modules():
            if _name_matches(name, cfg.exclude_modules):
                continue
            if (cfg.include_modules is not None
                    and not _name_matches(name, cfg.include_modules)):
                continue
            if _is_conv(module):
                raise NotImplementedError(
                    f"ActivationPreconditioner does not support convolution layer "
                    f"{name!r} ({type(module).__name__}). Conv (unfold / SUA) AP is "
                    f"not implemented; exclude it via "
                    f"ActivationPrecondConfig(exclude_modules=[...])."
                )
            if isinstance(module, nn.MultiheadAttention):
                warnings.warn(
                    f"ActivationPreconditioner: fused nn.MultiheadAttention {name!r} "
                    f"computes its projections via F.linear and bypasses module hooks, "
                    f"so activation preconditioning will NOT be applied to it. Use "
                    f"explicit nn.Linear Q/K/V/out projections for AP support.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                continue
            kind = self._supported_kind(module)
            if kind is None:
                continue
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            candidates.append((name, module, kind))

        for name, module, kind in candidates:
            weight = module.weight
            if len(weight_owners[id(weight)]) > 1:
                warnings.warn(
                    f"ActivationPreconditioner: weight of module {name!r} is shared "
                    f"with {weight_owners[id(weight)]}; activation preconditioning is "
                    f"ambiguous for tied weights and will be skipped (plain GP).",
                    RuntimeWarning,
                    stacklevel=3,
                )
                continue
            self._kind[module] = kind
            self._weight_of_module[module] = weight
            self._name_of_module[module] = name
            self._handles.append(
                module.register_forward_pre_hook(self._make_hook(module, kind))
            )

    def _make_hook(self, module: nn.Module, kind: str):
        def hook(_module: nn.Module, args: tuple) -> None:
            # Ignore forwards that cannot contribute to the gradient we will
            # precondition (inference / validation passes under torch.no_grad or
            # inference_mode). Without this guard, an eval forward run between
            # backward() and precondition_() would leak its activations into the
            # next step's covariance and silently mis-precondition.
            if not torch.is_grad_enabled():
                return
            if not args:
                return
            z = args[0]
            if not isinstance(z, Tensor):
                return
            if kind == _LINEAR:
                self._accumulate_linear(module, z)
            else:
                self._accumulate_embedding(module, z)

        return hook

    # -- capture ------------------------------------------------------------

    def _accumulate_linear(self, module: nn.Module, z: Tensor) -> None:
        z = z.detach().reshape(-1, z.shape[-1])
        n = z.shape[0]
        if n == 0:
            # Empty batch contributes nothing; skip so an all-empty step is a
            # clean "no forward" (plain GP) rather than a 0/0 -> NaN covariance.
            return
        if self.config.compute_dtype is not None:
            z = z.to(self.config.compute_dtype)
        gram = z.t() @ z  # [in, in]
        acc = self._accum.get(module)
        if acc is None:
            self._accum[module] = {"gram": gram, "count": n}
        else:
            acc["gram"] = acc["gram"] + gram
            acc["count"] += n

    def _accumulate_embedding(self, module: nn.Module, idx: Tensor) -> None:
        idx_flat = idx.detach().reshape(-1)
        vocab = module.num_embeddings
        if torch.is_floating_point(idx_flat) or idx_flat.is_complex():
            raise TypeError(
                f"ActivationPreconditioner: nn.Embedding input must be an integer "
                f"index tensor, got dtype {idx_flat.dtype}. Exclude this module via "
                f"ActivationPrecondConfig(exclude_modules=[...]) if it is not a "
                f"standard embedding lookup."
            )
        n = int(idx_flat.numel())
        if n == 0:
            return  # empty batch contributes nothing (see _accumulate_linear)
        counts = torch.bincount(idx_flat, minlength=vocab).to(
            dtype=self.config.compute_dtype or torch.float32
        )
        acc = self._accum.get(module)
        if acc is None:
            self._accum[module] = {"counts": counts, "count": n}
        else:
            acc["counts"] = acc["counts"] + counts
            acc["count"] += n

    # -- solve --------------------------------------------------------------

    def _solve_linear(self, sigma: Tensor, g: Tensor) -> Tensor:
        """Return ``M = G @ (S + tau I)^-1`` via a damped Cholesky solve.

        ``M = G S_d^-1`` and ``S_d`` symmetric give ``M^T = S_d^-1 G^T``, i.e. we
        solve ``S_d X = G^T`` for ``X`` and return ``X^T``.
        """
        # torch.linalg.cholesky does not support fp16/bf16, which is the covariance
        # dtype when compute_dtype=None on a half/bf16 model. Promote the solve to
        # fp32 (the result is cast back to the grad dtype by _copy_back anyway).
        if sigma.dtype in (torch.float16, torch.bfloat16):
            sigma = sigma.float()
        d_z = sigma.shape[0]
        eye = torch.eye(d_z, dtype=sigma.dtype, device=sigma.device)
        gt = g.to(sigma.dtype).t().contiguous()
        # Damping has a scale-invariant relative term plus an absolute floor; the
        # floor keeps S_d positive-definite even when S_z is zero/rank-deficient.
        tau = self._damping_tau(sigma.diagonal().sum(), d_z)
        sd = sigma + tau * eye
        L, info = torch.linalg.cholesky_ex(sd)
        if int(info) != 0:
            # Numerical safety net: grow the damping ADDITIVELY from the matrix
            # scale. A purely multiplicative bump (tau *= 10) cannot escape the
            # tau == 0 fixed point, so we add a positive, scale-aware increment.
            bump = sigma.diagonal().max().clamp_min(_TINY)
            for _ in range(8):
                tau = tau + bump
                bump = bump * 10.0
                L, info = torch.linalg.cholesky_ex(sigma + tau * eye)
                if int(info) == 0:
                    break
        if int(info) != 0:
            # Still not positive-definite after the retries: fall back to
            # isotropic scaling rather than running cholesky_solve on a failed
            # factorization (which would silently produce NaN gradients).
            denom = self._damping_tau(sigma.diagonal().sum(), d_z).clamp_min(_TINY)
            return g.to(sigma.dtype) / denom  # match the main path's solve dtype
        x = torch.cholesky_solve(gt, L)
        return x.t()

    def _damping_tau(self, trace: Tensor, d_z: int) -> Tensor:
        """Scale-invariant relative damping plus a strictly-positive floor.

        The floor is clamped to a positive minimum (``_TINY``) so the damped
        covariance is invertible for *every* allowed config, including
        ``damping_floor == 0`` on a layer whose inputs are entirely masked.
        """
        floor = max(self.config.damping_floor, _TINY)
        return self.config.damping * (trace / d_z) + floor

    # -- main API -----------------------------------------------------------

    @torch.no_grad()
    def precondition_(self) -> None:
        """Rewrite ``p.grad <- G @ S_z^-1`` in place for every registered weight.

        Advances the internal step counter, applies warmup (identity for the
        first ``warmup_steps`` calls), optional covariance EMA, and damping, then
        clears the per-step activation cache. Modules that did not run a forward
        this step (no captured activations) are skipped, unless EMA is enabled and
        ``use_stale_ema_on_missing`` is set.

        Note: the activation cache is populated by grad-enabled forward passes and
        consumed (and cleared) here. Inference/validation forwards (under
        ``torch.no_grad`` / ``inference_mode``) are ignored by the capture hook, so
        a validation loop between ``backward()`` and ``precondition_()`` is safe.
        Multiple *grad-enabled* forwards before a single ``precondition_()`` (e.g.
        gradient accumulation) intentionally accumulate into one covariance.
        """
        self._step += 1
        cfg = self.config

        if self._step <= cfg.warmup_steps:
            self._accum.clear()
            return

        distributed = self._distributed_enabled()
        # Under DDP the all-reduce must be issued in lockstep on every rank. Run a
        # deterministic pre-pass over the (rank-identical) registered modules that
        # reduces each covariance up front, decoupling the collective from per-rank
        # grad presence so a rank that later skips the solve cannot deadlock the
        # others.
        try:
            reduced: dict[nn.Module, Optional[Tensor]] = {}
            if distributed:
                for module, kind in self._kind.items():
                    reduced[module] = self._reduce_batch_sigma(module, kind)

            for module, kind in self._kind.items():
                weight = self._weight_of_module[module]
                g = weight.grad
                if g is None:
                    continue

                if distributed:
                    batch_sigma = reduced[module]
                else:
                    acc = self._accum.get(module)
                    batch_sigma = self._batch_sigma(kind, acc) if acc is not None else None

                if batch_sigma is None:
                    # No forward anywhere this step.
                    if cfg.ema_beta is not None and cfg.use_stale_ema_on_missing \
                            and module in self._sigma_ema:
                        sigma = self._sigma_ema[module]
                    else:
                        continue
                else:
                    # Move to the grad's device BEFORE the EMA combine: after a
                    # `map_location="cpu"` resume the stored EMA may live on CPU
                    # while the fresh batch covariance is on CUDA, and there is no
                    # implicit cross-device promotion.
                    sigma = batch_sigma.to(device=g.device)
                    if cfg.ema_beta is not None:
                        prev = self._sigma_ema.get(module)
                        if prev is not None:
                            prev = prev.to(device=g.device)
                            sigma = cfg.ema_beta * prev + (1.0 - cfg.ema_beta) * sigma
                        self._sigma_ema[module] = sigma

                sigma = sigma.to(device=g.device)
                if kind == _LINEAR:
                    m = self._solve_linear(sigma, g)
                else:
                    m = self._apply_embedding(sigma, g)
                self._copy_back(weight, m)
        finally:
            # Always clear the per-step cache, even if a collective/solve raised,
            # so stale activations cannot leak into the next step's covariance.
            self._accum.clear()

    def _copy_back(self, weight: Tensor, m: Tensor) -> None:
        """Write ``m`` into ``weight.grad``, warning once on a fp16 downcast overflow.

        The solve runs in ``compute_dtype`` (fp32 by default); casting the result
        back to a low-precision ``grad`` dtype can overflow (fp16 max ~6.5e4) when
        ``S_z`` is ill-conditioned, which would otherwise silently produce inf
        gradients. Detect that and warn once (no behavior change otherwise).
        """
        m_cast = m.to(weight.grad.dtype)
        if (not self._warned_fp16_overflow and m_cast.dtype != m.dtype
                and not torch.isfinite(m_cast).all()
                and bool(torch.isfinite(m).all())):
            self._warned_fp16_overflow = True
            warnings.warn(
                f"ActivationPreconditioner: preconditioned gradient overflowed when "
                f"cast from {m.dtype} to {weight.grad.dtype} (ill-conditioned "
                f"activation covariance). Increase `damping`/`damping_floor` or use a "
                f"wider `compute_dtype`. This warning is shown once.",
                RuntimeWarning,
                stacklevel=3,
            )
        weight.grad.copy_(m_cast)

    def _distributed_enabled(self) -> bool:
        """Whether to all-reduce the covariance across the process group."""
        if self.config.distributed is not None:
            return bool(self.config.distributed)
        return dist.is_available() and dist.is_initialized()

    def _reduce_batch_sigma(self, module: nn.Module, kind: str) -> Optional[Tensor]:
        """All-reduce a module's covariance across the group; return the batch sigma.

        Sums the raw accumulator (linear ``gram`` + ``count``, embedding ``counts``
        + ``count``) over all ranks with ``ReduceOp.SUM``, contributing zeros when
        this rank did not run a forward for the module (so the collective is issued
        on every rank). Returns ``S_z = stat_total / count_total``, or ``None`` if
        no rank produced any samples this step.
        """
        pg = self.config.process_group
        weight = self._weight_of_module[module]
        device = weight.device
        acc = self._accum.get(module)
        if kind == _LINEAR:
            in_dim = weight.shape[1]
            dtype = self.config.compute_dtype or weight.dtype
            if acc is None:
                stat = torch.zeros(in_dim, in_dim, dtype=dtype, device=device)
                count = 0.0
            else:
                stat = acc["gram"].to(device=device).clone()
                count = float(acc["count"])
        else:  # embedding
            vocab = weight.shape[0]
            dtype = self.config.compute_dtype or torch.float32
            if acc is None:
                stat = torch.zeros(vocab, dtype=dtype, device=device)
                count = 0.0
            else:
                stat = acc["counts"].to(device=device).clone()
                count = float(acc["count"])
        # Count is carried in fp64 (never the compute dtype): a single rank's
        # token count can exceed the fp16 max (~6.5e4) and the SUM all-reduce would
        # saturate to inf, corrupting the global mean.
        cnt = torch.tensor([count], dtype=torch.float64, device=device)
        dist.all_reduce(stat, op=dist.ReduceOp.SUM, group=pg)
        dist.all_reduce(cnt, op=dist.ReduceOp.SUM, group=pg)
        total = float(cnt.item())
        if total <= 0.0:
            return None
        return stat / total

    def _batch_sigma(self, kind: str, acc: dict[str, Any]) -> Optional[Tensor]:
        # Defense in depth (the accumulators already skip empty batches): never
        # divide by a zero count, matching the distributed reduce path's guard.
        if acc["count"] <= 0:
            return None
        if kind == _LINEAR:
            return acc["gram"] / acc["count"]
        return acc["counts"] / acc["count"]  # diagonal of S_z

    def _apply_embedding(self, sigma_diag: Tensor, g: Tensor) -> Tensor:
        """Diagonal AP for an embedding: scale each token (row) of ``G``.

        ``S_z = diag(p)`` with ``p`` the per-token frequency; the input dim is the
        vocab dim, which is the *row* dim of ``W``. So ``M = G @ diag(p + tau)^-1``
        scales row ``v`` of ``G`` by ``1/(p_v + tau)`` (rare tokens upweighted).
        """
        d_z = sigma_diag.shape[0]
        tau = self._damping_tau(sigma_diag.sum(), d_z)
        denom = (sigma_diag + tau).unsqueeze(1)  # [vocab, 1], in compute dtype
        # Divide in compute dtype and let _copy_back do the single downcast to the
        # grad dtype, so the fp16-overflow guard there also covers embeddings (a
        # fp16 division here would overflow to inf *before* the guard runs).
        return g.to(denom.dtype) / denom

    # -- lifecycle ----------------------------------------------------------

    def zero_grad(self) -> None:
        """Clear the per-step activation cache (call after ``optimizer.step``)."""
        self._accum.clear()

    def reset(self) -> None:
        """Clear EMA covariances, the activation cache, and the step counter."""
        self._accum.clear()
        self._sigma_ema.clear()
        self._step = 0
        # Fresh start: let a genuine new fp16 overflow warn again.
        self._warned_fp16_overflow = False

    def remove_hooks(self) -> None:
        """Remove all forward-pre-hooks. After this, ``precondition_`` is a no-op."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._kind.clear()
        self._weight_of_module.clear()
        self._name_of_module.clear()
        self._accum.clear()
        self._sigma_ema.clear()

    @property
    def step_count(self) -> int:
        """Number of :meth:`precondition_` calls so far."""
        return self._step

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpointable dict (step counter + EMA covariances).

        EMA tensors are keyed by qualified module name so they survive being
        re-loaded onto a freshly constructed preconditioner. The persisted step
        counter drives the warmup gate on resume (see ``warmup_steps``).
        """
        ema = {
            self._name_of_module[m]: sigma.detach().clone()
            for m, sigma in self._sigma_ema.items()
            if m in self._name_of_module
        }
        return {"step": self._step, "sigma_ema": ema}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore step counter and EMA covariances from :meth:`state_dict`.

        Emits a warning if saved EMA entries do not match any current module
        (e.g. loading onto a structurally different model), since those
        covariances would otherwise be silently dropped.
        """
        self._step = int(state.get("step", 0))
        self._sigma_ema.clear()
        ema = state.get("sigma_ema", {})
        name_to_module = {n: m for m, n in self._name_of_module.items()}
        unmatched = []
        for name, sigma in ema.items():
            module = name_to_module.get(name)
            if module is not None:
                self._sigma_ema[module] = sigma.clone()
            else:
                unmatched.append(name)
        if unmatched:
            warnings.warn(
                f"ActivationPreconditioner.load_state_dict: {len(unmatched)} saved EMA "
                f"covariance(s) had no matching module and were dropped: {unmatched}. "
                f"The model structure may differ from the checkpoint.",
                RuntimeWarning,
                stacklevel=2,
            )

    def __enter__(self) -> "ActivationPreconditioner":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove_hooks()


# ---------------------------------------------------------------------------
# DoPr wrapper
# ---------------------------------------------------------------------------

class DoPr:
    """Double Preconditioning: an AP stage in front of any base optimizer.

    A thin delegating wrapper that owns an :class:`ActivationPreconditioner` and a
    base optimizer. :meth:`step` runs ``ap.precondition_()`` and then
    ``base.step()``, so the base optimizer transparently consumes the
    activation-preconditioned gradient. Unknown attributes (e.g. ``train``/
    ``eval`` on schedule-free optimizers, ``last_stats`` on SNR optimizers) are
    forwarded to the base optimizer.

    ``DoPr`` is a delegating wrapper, **not** a ``torch.optim.Optimizer`` subclass,
    so it fails the ``isinstance(opt, Optimizer)`` check that stock
    ``torch.optim.lr_scheduler`` schedulers enforce. Attach any LR scheduler to the
    **base** optimizer instead -- ``DoPr.param_groups`` *is* the base's
    ``param_groups``, so the scheduler still drives the learning rate DoPr uses::

        base = SNRAdamW(model.parameters(), lr=3e-4)
        opt = DoPr(base, model)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=1000)
        loss.backward(); opt.step(); opt.zero_grad(); sched.step()

    Args:
        base_optimizer:
            Any optimizer with a ``step`` / ``zero_grad`` interface (an snr_grad
            optimizer or a ``torch.optim`` one).
        model:
            The model whose activations drive AP.
        config:
            An :class:`ActivationPrecondConfig`.

    Example::

        opt = DoPr(SNRAdamW(model.parameters(), lr=3e-4), model)
        loss.backward(); opt.step(); opt.zero_grad()

    Note: checkpoint via :meth:`state_dict` / :meth:`load_state_dict`. ``DoPr`` is
    not picklable because the activation hooks hold module closures; use the state
    dict rather than ``pickle``/``torch.save`` of the object itself.

    ``DoPr`` is also a context manager and exposes :meth:`remove_hooks`,
    :meth:`reset`, and :attr:`step_count` (delegated to its preconditioner), so a
    transient instance can clean up its forward hooks::

        with DoPr(SNRAdamW(model.parameters(), lr=3e-4), model) as opt:
            ...  # hooks removed on exit
    """

    def __init__(
        self,
        base_optimizer: Any,
        model: nn.Module,
        config: Optional[ActivationPrecondConfig] = None,
    ) -> None:
        self.base = base_optimizer
        self.ap = ActivationPreconditioner(model, config)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Precondition the gradients, then take one base-optimizer step.

        When a ``closure`` is given it is evaluated **here, exactly once** (with
        grad enabled) to populate ``.grad`` *before* activation preconditioning, and
        the base optimizer is then stepped without a closure. This is required for
        correctness: if the closure were forwarded to ``base.step``, the base would
        re-run ``backward()`` and overwrite the AP-rewritten gradient, silently
        making AP a no-op. Consequently DoPr is incompatible with optimizers that
        require *multiple* closure evaluations per step (e.g. ``torch.optim.LBFGS``).
        """
        if closure is None:
            self.ap.precondition_()
            return self.base.step()
        with torch.enable_grad():
            loss = closure()
        self.ap.precondition_()
        self.base.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero the base optimizer's grads and clear the AP activation cache."""
        self.base.zero_grad(set_to_none=set_to_none)
        self.ap.zero_grad()

    def state_dict(self) -> dict[str, Any]:
        """Bundle the base optimizer and AP state dicts."""
        return {"base": self.base.state_dict(), "ap": self.ap.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from a :meth:`state_dict` bundle."""
        self.base.load_state_dict(state["base"])
        self.ap.load_state_dict(state["ap"])

    @property
    def param_groups(self) -> Any:
        return self.base.param_groups

    @property
    def state(self) -> Any:
        return self.base.state

    def remove_hooks(self) -> None:
        """Remove the activation preconditioner's forward hooks.

        After this, :meth:`step` still runs the base optimizer but applies no
        activation preconditioning. Provided explicitly because ``__getattr__``
        delegates to the *base optimizer*, which has no ``remove_hooks``.
        """
        self.ap.remove_hooks()

    def reset(self) -> None:
        """Reset the activation preconditioner (EMA covariances, cache, step)."""
        self.ap.reset()

    @property
    def step_count(self) -> int:
        """Number of activation-preconditioning steps taken so far."""
        return self.ap.step_count

    def __enter__(self) -> "DoPr":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ap.remove_hooks()

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails; delegate to the base optimizer.
        # During copy/deepcopy/unpickle ``base`` may not be set yet -- raise
        # AttributeError (not KeyError) so those protocols work correctly.
        try:
            base = self.__dict__["base"]
        except KeyError:
            raise AttributeError(name)
        return getattr(base, name)
