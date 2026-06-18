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
* Covariances are computed from *local* activations; distributed (DDP)
  aggregation is out of scope (same stance as :mod:`snr_grad.variance`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
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
            rank-deficient. Must be >= 0.
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
            ``None`` to use the activation's native dtype.
        use_stale_ema_on_missing:
            When ``ema_beta`` is set and a registered module did not run a forward
            pass this step (so there is no fresh covariance) but its weight still
            has a gradient, reuse the most recent EMA covariance. If False, such a
            weight is left as plain GP for that step. Ignored when ``ema_beta`` is
            ``None``.
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
        if self.config.compute_dtype is not None:
            z = z.to(self.config.compute_dtype)
        gram = z.t() @ z  # [in, in]
        n = z.shape[0]
        acc = self._accum.get(module)
        if acc is None:
            self._accum[module] = {"gram": gram, "count": n}
        else:
            acc["gram"] = acc["gram"] + gram
            acc["count"] += n

    def _accumulate_embedding(self, module: nn.Module, idx: Tensor) -> None:
        idx_flat = idx.detach().reshape(-1)
        vocab = module.num_embeddings
        counts = torch.bincount(idx_flat, minlength=vocab).to(
            dtype=self.config.compute_dtype or torch.float32
        )
        n = int(idx_flat.numel())
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
        d_z = sigma.shape[0]
        eye = torch.eye(d_z, dtype=sigma.dtype, device=sigma.device)
        gt = g.to(sigma.dtype).t().contiguous()
        # Damping has a scale-invariant relative term plus an absolute floor; the
        # floor keeps S_d positive-definite even when S_z is zero/rank-deficient.
        tau = self._damping_tau(sigma.diagonal().sum(), d_z)
        sd = sigma + tau * eye
        L, info = torch.linalg.cholesky_ex(sd)
        if int(info) != 0:
            # Numerical safety net: bump the damping until the factorization
            # succeeds (guaranteed to terminate as tau grows).
            for _ in range(5):
                tau = tau * 10.0
                L, info = torch.linalg.cholesky_ex(sigma + tau * eye)
                if int(info) == 0:
                    break
        x = torch.cholesky_solve(gt, L)
        return x.t()

    def _damping_tau(self, trace: Tensor, d_z: int) -> Tensor:
        """Scale-invariant relative damping plus an absolute floor."""
        return self.config.damping * (trace / d_z) + self.config.damping_floor

    # -- main API -----------------------------------------------------------

    @torch.no_grad()
    def precondition_(self) -> None:
        """Rewrite ``p.grad <- G @ S_z^-1`` in place for every registered weight.

        Advances the internal step counter, applies warmup (identity for the
        first ``warmup_steps`` calls), optional covariance EMA, and damping, then
        clears the per-step activation cache. Modules that did not run a forward
        this step (no captured activations) are skipped, unless EMA is enabled and
        ``use_stale_ema_on_missing`` is set.

        Note: the activation cache is populated by forward passes and consumed (and
        cleared) here. Call this once per training step, in the usual
        ``loss.backward(); precondition_(); optimizer.step()`` loop. If a forward
        pass runs without a following ``precondition_()`` (or :meth:`zero_grad` /
        :meth:`reset`), its activations carry over into the next step's covariance.
        """
        self._step += 1
        cfg = self.config

        if self._step <= cfg.warmup_steps:
            self._accum.clear()
            return

        for module, kind in self._kind.items():
            weight = self._weight_of_module[module]
            g = weight.grad
            if g is None:
                continue

            acc = self._accum.get(module)
            if acc is None:
                # No forward this step.
                if cfg.ema_beta is not None and cfg.use_stale_ema_on_missing \
                        and module in self._sigma_ema:
                    sigma = self._sigma_ema[module]
                else:
                    continue
            else:
                sigma = self._batch_sigma(kind, acc)
                if cfg.ema_beta is not None:
                    prev = self._sigma_ema.get(module)
                    if prev is not None:
                        sigma = cfg.ema_beta * prev + (1.0 - cfg.ema_beta) * sigma
                    self._sigma_ema[module] = sigma

            sigma = sigma.to(device=g.device)
            if kind == _LINEAR:
                m = self._solve_linear(sigma, g)
            else:
                m = self._apply_embedding(sigma, g)
            weight.grad.copy_(m.to(weight.grad.dtype))

        self._accum.clear()

    def _batch_sigma(self, kind: str, acc: dict[str, Any]) -> Tensor:
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
        denom = (sigma_diag + tau).to(g.dtype).unsqueeze(1)  # [vocab, 1]
        return g / denom

    # -- lifecycle ----------------------------------------------------------

    def zero_grad(self) -> None:
        """Clear the per-step activation cache (call after ``optimizer.step``)."""
        self._accum.clear()

    def reset(self) -> None:
        """Clear EMA covariances, the activation cache, and the step counter."""
        self._accum.clear()
        self._sigma_ema.clear()
        self._step = 0

    def remove_hooks(self) -> None:
        """Remove all forward-pre-hooks. After this, ``precondition_`` is a no-op."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._kind.clear()
        self._weight_of_module.clear()
        self._accum.clear()

    @property
    def step_count(self) -> int:
        """Number of :meth:`precondition_` calls so far."""
        return self._step

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpointable dict (step counter + EMA covariances).

        EMA tensors are keyed by qualified module name so they survive being
        re-loaded onto a freshly constructed preconditioner.
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
        """Precondition the gradients, then take one base-optimizer step."""
        self.ap.precondition_()
        return self.base.step(closure)

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

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails; delegate to the base optimizer.
        # During copy/deepcopy/unpickle ``base`` may not be set yet -- raise
        # AttributeError (not KeyError) so those protocols work correctly.
        try:
            base = self.__dict__["base"]
        except KeyError:
            raise AttributeError(name)
        return getattr(base, name)
