"""
Curvature proxies for the curvature-aware stability sensor.

See ``docs/proposals/curvature-aware-stability-control.md``. A stability probe
needs a positive-semidefinite (PSD) curvature proxy ``H_plus`` to weight
stochastic update deviations: the curvature-weighted stochastic update penalty is
``(eta**2 / (2 B)) * tr(H_plus P Sigma P^T)``. This module provides a minimal
common interface and three backends of increasing cost / generality:

* :class:`ExactQuadraticCurvature` -- a known dense (or diagonal) Hessian over the
  flattened parameters. The gold standard for quadratic toys such as power-law
  random features (PLRF).
* :class:`ActivationCurvature` -- the layer-wise proxy ``H_W ~= I_out (x) C_h`` for
  ``y = W h``, built from the *uncentered* input correlation ``C_h`` that DoPr
  already collects (via :meth:`ActivationPreconditioner.covariance_snapshot`).
  Exact only under its model assumptions (squared loss, isotropic output
  curvature, no cross-layer terms); validate it against GGN.
* :class:`GGNCurvature` -- the generalized Gauss-Newton ``J^T H_out J`` of a
  model on a small held-out batch via ``torch.func`` Jacobian-vector products,
  PSD whenever the output loss is convex. :class:`HessianCurvature` gives the
  true (possibly indefinite) Hessian through Hessian-vector products; it is
  flagged ``psd=False`` and the stability code refuses to treat it as ``H_plus``.

All backends act on *parameter dicts* ``{live_param_tensor: tensor}`` (the same
keying as ``grad_variances`` elsewhere in this package). ``quad(u)`` returns
``u^T H u`` (no factor 1/2). Parameters a backend does not cover contribute
zero and are listed in ``metadata.uncovered``; check coverage before trusting a
number. None of these classes modifies parameters, ``.grad`` or optimizer state.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from snr_grad.activation import (
    ActivationCovarianceSnapshot,
    ActivationPrecondConfig,
    ActivationPreconditioner,
)

__all__ = [
    "ParamDict",
    "CurvatureMetadata",
    "CurvatureProxy",
    "ExactQuadraticCurvature",
    "ActivationCurvature",
    "GGNCurvature",
    "HessianCurvature",
    "LowRankCurvature",
    "collect_activation_snapshot",
    "flatten_params",
    "preserved_rng",
    "unflatten_like",
]


ParamDict = Mapping[Tensor, Tensor]


@contextlib.contextmanager
def preserved_rng(enabled: bool = True):
    """Restore the CPU (and CUDA) RNG state on exit, so probes do not shift training."""
    if not enabled:
        yield
        return
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


# ---------------------------------------------------------------------------
# Flattening helpers
# ---------------------------------------------------------------------------

def flatten_params(d: ParamDict, params: Sequence[Tensor]) -> Tensor:
    """Concatenate ``d[p]`` for ``p in params`` into one vector (zeros if missing)."""
    pieces = []
    for p in params:
        v = d.get(p)
        if v is None:
            pieces.append(torch.zeros(p.numel(), dtype=p.dtype, device=p.device))
        else:
            pieces.append(v.reshape(-1))
    return torch.cat(pieces) if pieces else torch.zeros(0)


def unflatten_like(vec: Tensor, params: Sequence[Tensor]) -> dict[Tensor, Tensor]:
    """Inverse of :func:`flatten_params`: split ``vec`` back into ``{p: tensor}``."""
    out: dict[Tensor, Tensor] = {}
    offset = 0
    for p in params:
        n = p.numel()
        out[p] = vec[offset:offset + n].reshape(p.shape)
        offset += n
    if offset != vec.numel():
        raise ValueError(f"vector has {vec.numel()} elements, params need {offset}.")
    return out


def _dot(a: ParamDict, b: ParamDict) -> Tensor:
    total = None
    for p, va in a.items():
        vb = b.get(p)
        if vb is None:
            continue
        term = (va.reshape(-1).to(torch.float64) * vb.reshape(-1).to(torch.float64)).sum()
        total = term if total is None else total + term
    return total if total is not None else torch.zeros((), dtype=torch.float64)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

@dataclass
class CurvatureMetadata:
    """What a curvature proxy is, and how far to trust it.

    Attributes:
        name: Backend name.
        psd: Whether ``quad`` is guaranteed non-negative (a valid ``H_plus``).
        assumptions: Human-readable modelling assumptions.
        covered: Number of parameter tensors the proxy acts on.
        uncovered: Names/shapes of parameters treated as zero curvature.
        rank: Rank of the proxy when known (``None`` otherwise).
        damping: Damping added to the proxy (0 when none).
        sample_count: Samples behind an estimated proxy (``None`` if exact/unknown).
        checkpoint_step: Optional training step the proxy was measured at.
        psd_projection: How indefiniteness was handled (``"none"``, ``"clip"``, ...).
        cost: Running counters (``quad_calls``, ``matvec_calls``, ``seconds``).
    """

    name: str
    psd: bool
    assumptions: list[str] = field(default_factory=list)
    covered: int = 0
    uncovered: list[str] = field(default_factory=list)
    rank: Optional[int] = None
    damping: float = 0.0
    sample_count: Optional[int] = None
    checkpoint_step: Optional[int] = None
    psd_projection: str = "none"
    cost: dict[str, float] = field(
        default_factory=lambda: {"quad_calls": 0.0, "matvec_calls": 0.0, "seconds": 0.0}
    )


class CurvatureProxy:
    """Minimal curvature interface.

    Subclasses implement :meth:`matvec`; :meth:`quad` defaults to
    ``<u, matvec(u)>`` and may be overridden with something cheaper.
    """

    metadata: CurvatureMetadata

    def matvec(self, update: ParamDict) -> dict[Tensor, Tensor]:
        raise NotImplementedError

    def quad(self, update: ParamDict) -> float:
        """Return ``u^T H u`` as a Python float (no factor 1/2)."""
        t0 = time.perf_counter()
        hv = self._matvec_uncounted(update)
        val = float(_dot(update, hv))
        self._count("quad_calls", t0)
        return val

    def bilinear(self, a: ParamDict, b: ParamDict) -> float:
        """Return ``a^T H b``."""
        t0 = time.perf_counter()
        hv = self._matvec_uncounted(b)
        val = float(_dot(a, hv))
        self._count("matvec_calls", t0)
        return val

    # Subclasses route through these so cost accounting stays in one place.
    def _matvec_uncounted(self, update: ParamDict) -> dict[Tensor, Tensor]:
        raise NotImplementedError

    def _count(self, key: str, t0: float) -> None:
        self.metadata.cost[key] = self.metadata.cost.get(key, 0.0) + 1.0
        self.metadata.cost["seconds"] = (
            self.metadata.cost.get("seconds", 0.0) + time.perf_counter() - t0
        )


# ---------------------------------------------------------------------------
# Exact quadratic
# ---------------------------------------------------------------------------

class ExactQuadraticCurvature(CurvatureProxy):
    """A known Hessian over the flattened concatenation of ``params``.

    Args:
        params: Parameter tensors defining the coordinate order.
        hessian: Dense symmetric ``[n, n]`` matrix, or ``None`` if ``diag`` given.
        diag: Diagonal ``[n]`` Hessian (alternative to ``hessian``).
        psd_projection:
            How to handle a negative eigenvalue below ``-psd_tol * max|eig|``:
            ``"error"`` (default) raises, ``"clip"`` sets negative eigenvalues to
            zero, ``"abs"`` takes absolute values, ``"none"`` keeps the matrix and
            marks the proxy ``psd=False``. The choice is recorded in metadata.
        psd_tol: Relative tolerance for treating small negative eigenvalues as 0.
    """

    def __init__(
        self,
        params: Sequence[Tensor],
        hessian: Optional[Tensor] = None,
        *,
        diag: Optional[Tensor] = None,
        psd_projection: Literal["error", "clip", "abs", "none"] = "error",
        psd_tol: float = 1e-6,
    ) -> None:
        if (hessian is None) == (diag is None):
            raise ValueError("Pass exactly one of `hessian` or `diag`.")
        self.params = list(params)
        n = sum(p.numel() for p in self.params)
        psd = True
        applied = "none"
        if diag is not None:
            d = diag.detach().reshape(-1).clone()
            if d.numel() != n:
                raise ValueError(f"diag has {d.numel()} entries, params have {n}.")
            neg = d < -psd_tol * d.abs().max().clamp_min(1e-30)
            if bool(neg.any()):
                psd, d = self._project(d, psd_projection)
                applied = psd_projection
            else:
                d = d.clamp_min(0.0)
            self._diag: Optional[Tensor] = d
            self._dense: Optional[Tensor] = None
            rank = int((d > 0).sum())
        else:
            h = hessian.detach().clone()
            if h.shape != (n, n):
                raise ValueError(f"hessian must be [{n}, {n}], got {tuple(h.shape)}.")
            h = 0.5 * (h + h.t())
            evals, evecs = torch.linalg.eigh(h.to(torch.float64))
            scale = evals.abs().max().clamp_min(1e-30)
            if bool((evals < -psd_tol * scale).any()):
                psd, evals = self._project(evals, psd_projection)
                applied = psd_projection
                h = (evecs * evals) @ evecs.t()
            else:
                evals = evals.clamp_min(0.0)
            self._dense = h.to(hessian.dtype)
            self._diag = None
            self._evals = evals
            self._evecs = evecs
            rank = int((evals > psd_tol * scale).sum())
        self.metadata = CurvatureMetadata(
            name="exact_quadratic",
            psd=psd,
            assumptions=["exact Hessian of a quadratic loss supplied by the caller"],
            covered=len(self.params),
            rank=rank,
            psd_projection=applied,
        )

    @staticmethod
    def _project(evals: Tensor, mode: str) -> tuple[bool, Tensor]:
        if mode == "error":
            raise ValueError(
                "Hessian is indefinite; pass psd_projection='clip'/'abs' to build an "
                "explicit PSD proxy, or 'none' to keep it (flagged psd=False)."
            )
        if mode == "clip":
            return True, evals.clamp_min(0.0)
        if mode == "abs":
            return True, evals.abs()
        if mode == "none":
            return False, evals
        raise ValueError(f"Unknown psd_projection {mode!r}.")

    def dense(self) -> Tensor:
        """Return the (projected) Hessian as a dense ``[n, n]`` matrix."""
        if self._dense is not None:
            return self._dense
        return torch.diag(self._diag)

    def eigenvalues(self) -> Tensor:
        """Eigenvalues of the (projected) Hessian, ascending."""
        if self._diag is not None:
            return torch.sort(self._diag).values
        return self._evals.to(self._dense.dtype)

    def sqrt(self) -> Tensor:
        """Dense PSD square root ``H_plus^(1/2)`` (requires ``psd``)."""
        if not self.metadata.psd:
            raise ValueError("sqrt() is undefined for an indefinite proxy.")
        if self._diag is not None:
            return torch.diag(self._diag.sqrt())
        ev = self._evals.clamp_min(0.0).sqrt()
        return ((self._evecs * ev) @ self._evecs.t()).to(self._dense.dtype)

    def matvec_flat(self, vec: Tensor) -> Tensor:
        if self._diag is not None:
            return self._diag.to(vec.dtype) * vec
        return self._dense.to(vec.dtype) @ vec

    def _matvec_uncounted(self, update: ParamDict) -> dict[Tensor, Tensor]:
        vec = flatten_params(update, self.params)
        return unflatten_like(self.matvec_flat(vec), self.params)

    def matvec(self, update: ParamDict) -> dict[Tensor, Tensor]:
        t0 = time.perf_counter()
        out = self._matvec_uncounted(update)
        self._count("matvec_calls", t0)
        return out


# ---------------------------------------------------------------------------
# Activation-correlation proxy
# ---------------------------------------------------------------------------

def collect_activation_snapshot(
    model: nn.Module,
    run_forward: Callable[[nn.Module], Any],
    *,
    config: Optional[ActivationPrecondConfig] = None,
    preserve_rng: bool = True,
) -> dict[str, ActivationCovarianceSnapshot]:
    """Collect uncentered input correlations for ``model``'s Linear/Embedding layers.

    Registers *temporary* capture hooks (an :class:`ActivationPreconditioner`
    that never preconditions), runs ``run_forward(model)`` with grad enabled (the
    capture hook ignores no-grad forwards) and removes the hooks. No backward is
    run, ``.grad`` is untouched and (by default) the RNG state is restored
    afterwards, since dropout in the forward would otherwise consume it. Use this when there is no live DoPr whose
    :meth:`~ActivationPreconditioner.covariance_snapshot` can be reused.
    """
    cfg = config if config is not None else ActivationPrecondConfig(distributed=False)
    ap = ActivationPreconditioner(model, cfg)
    try:
        with preserved_rng(preserve_rng), torch.enable_grad():
            run_forward(model)
        return ap.covariance_snapshot(source="batch")
    finally:
        ap.remove_hooks()


class ActivationCurvature(CurvatureProxy):
    """Layer-wise proxy ``H_W ~= output_scale * (I_out (x) C_h)`` per linear weight.

    For ``y = W h`` with squared loss and isotropic output curvature,
    ``u^T H u = output_scale * tr(U C_h U^T)``; for an embedding (one-hot inputs)
    ``C_h`` is the diagonal of token frequencies. Cross-layer terms, biases,
    normalization parameters and anything not in the snapshot are *not covered*
    (zero curvature, listed in ``metadata.uncovered``).

    ``C_out = I`` is a modelling assumption, not a fact about real networks. Use
    ``output_scale`` to match the loss reduction (e.g. ``2 / out_features`` for a
    per-output-mean MSE), and validate against :class:`GGNCurvature`.

    Args:
        snapshot: ``{name: ActivationCovarianceSnapshot}`` (from DoPr's
            ``ap.covariance_snapshot()`` or :func:`collect_activation_snapshot`).
        params: Optional full parameter list, only used to report uncovered ones.
        output_scale: Scalar output-curvature factor, or ``{name: factor}``.
        damping: Optional relative ridge ``damping * tr(C_h)/d`` added to ``C_h``.
    """

    def __init__(
        self,
        snapshot: Mapping[str, ActivationCovarianceSnapshot],
        *,
        params: Optional[Iterable[Tensor]] = None,
        param_names: Optional[Mapping[Tensor, str]] = None,
        output_scale: float | Mapping[str, float] = 1.0,
        damping: float = 0.0,
    ) -> None:
        if damping < 0:
            raise ValueError(f"damping must be >= 0, got {damping}.")
        self._entries: dict[Tensor, tuple[str, Tensor, float]] = {}
        counts = []
        for name, snap in snapshot.items():
            if snap.centered:
                raise ValueError(
                    f"snapshot {name!r} is centered; the activation Hessian proxy needs "
                    f"the uncentered correlation E[h h^T]."
                )
            sigma = snap.sigma.detach()
            if damping > 0:
                if snap.kind == "linear":
                    d = sigma.shape[0]
                    sigma = sigma + damping * sigma.diagonal().sum() / d * torch.eye(
                        d, dtype=sigma.dtype, device=sigma.device
                    )
                else:
                    sigma = sigma + damping * sigma.sum() / sigma.numel()
            scale = (
                float(output_scale.get(name, 1.0)) if isinstance(output_scale, Mapping)
                else float(output_scale)
            )
            self._entries[snap.weight] = (snap.kind, sigma, scale)
            if snap.sample_count is not None:
                counts.append(snap.sample_count)

        uncovered = []
        if params is not None:
            for p in params:
                if p not in self._entries:
                    label = param_names.get(p) if param_names else None
                    uncovered.append(label or f"param{tuple(p.shape)}")
        self.metadata = CurvatureMetadata(
            name="activation_correlation",
            psd=True,
            assumptions=[
                "per-layer H_W = output_scale * (I_out kron C_h) (squared loss, "
                "isotropic output curvature)",
                "no cross-layer coupling; biases and norm params not covered",
                "C_h is the uncentered input correlation from forward activations",
            ],
            covered=len(self._entries),
            uncovered=uncovered,
            damping=damping,
            sample_count=min(counts) if counts else None,
        )

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        run_forward: Callable[[nn.Module], Any],
        **kwargs: Any,
    ) -> "ActivationCurvature":
        """Collect a snapshot with temporary hooks and build the proxy."""
        snap = collect_activation_snapshot(model, run_forward)
        names = {p: n for n, p in model.named_parameters()}
        return cls(snap, params=list(model.parameters()), param_names=names, **kwargs)

    def covered_params(self) -> list[Tensor]:
        return list(self._entries.keys())

    def _apply(self, kind: str, sigma: Tensor, u: Tensor) -> Tensor:
        if kind == "linear":
            return u.to(sigma.dtype) @ sigma  # U C_h (C_h symmetric)
        return u.to(sigma.dtype) * sigma.unsqueeze(1)  # rows scaled by frequency

    def _matvec_uncounted(self, update: ParamDict) -> dict[Tensor, Tensor]:
        out: dict[Tensor, Tensor] = {}
        for p, u in update.items():
            entry = self._entries.get(p)
            if entry is None:
                out[p] = torch.zeros_like(u)
                continue
            kind, sigma, scale = entry
            out[p] = (scale * self._apply(kind, sigma, u)).to(u.dtype)
        return out

    def matvec(self, update: ParamDict) -> dict[Tensor, Tensor]:
        t0 = time.perf_counter()
        out = self._matvec_uncounted(update)
        self._count("matvec_calls", t0)
        return out

    def quad(self, update: ParamDict) -> float:
        t0 = time.perf_counter()
        total = 0.0
        for p, u in update.items():
            entry = self._entries.get(p)
            if entry is None:
                continue
            kind, sigma, scale = entry
            u64 = u.detach().to(torch.float64)
            s64 = sigma.to(torch.float64)
            if kind == "linear":
                total += scale * float(((u64 @ s64) * u64).sum())
            else:
                total += scale * float((u64.square().sum(dim=1) * s64).sum())
        self._count("quad_calls", t0)
        return total


# ---------------------------------------------------------------------------
# GGN / Hessian via torch.func
# ---------------------------------------------------------------------------

class _FunctionalCurvature(CurvatureProxy):
    """Shared plumbing for ``torch.func``-based backends."""

    def __init__(
        self,
        model: nn.Module,
        params: Optional[Sequence[Tensor]],
    ) -> None:
        self.model = model
        named = dict(model.named_parameters())
        if params is None:
            params = [p for p in named.values() if p.requires_grad]
        self.params = list(params)
        name_of = {p: n for n, p in named.items()}
        missing = [p for p in self.params if p not in name_of]
        if missing:
            raise ValueError("all `params` must be parameters of `model`.")
        self._names = [name_of[p] for p in self.params]
        selected = set(self._names)
        self._fixed = {n: p.detach() for n, p in named.items() if n not in selected}
        self._buffers = dict(model.named_buffers())
        self._primals = tuple(p.detach() for p in self.params)
        self._uncovered = [n for n in named if n not in selected]

    def _call(self, primals: tuple, inputs: Any) -> Any:
        from torch.func import functional_call

        state = {**self._buffers, **self._fixed, **dict(zip(self._names, primals))}
        return functional_call(self.model, state, (inputs,))

    def _tangents(self, update: ParamDict) -> tuple:
        return tuple(
            update[p].detach().to(p.dtype) if p in update else torch.zeros_like(p)
            for p in self.params
        )


class GGNCurvature(_FunctionalCurvature):
    """Generalized Gauss-Newton ``J^T H_out J`` on a fixed held-out batch.

    ``quad(u) = (J u)^T H_out (J u)`` costs one JVP plus an output-space HVP;
    ``matvec`` adds one VJP. PSD whenever ``output_loss`` is convex in the model
    outputs (MSE, cross-entropy). Run the model in eval mode (or otherwise
    control dropout) so repeated calls see the same function.

    Args:
        model: The network.
        inputs: Model input for the held-out batch (passed as ``model(inputs)``).
        output_loss: ``output_loss(outputs) -> scalar`` with targets closed over;
            its reduction sets the curvature scale (use the training reduction).
        params: Parameters to differentiate (default: all trainable).
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Any,
        output_loss: Callable[[Tensor], Tensor],
        *,
        params: Optional[Sequence[Tensor]] = None,
    ) -> None:
        super().__init__(model, params)
        self.inputs = inputs
        self.output_loss = output_loss
        n_samples = inputs.shape[0] if isinstance(inputs, Tensor) and inputs.ndim else None
        self.metadata = CurvatureMetadata(
            name="ggn",
            psd=True,
            assumptions=[
                "generalized Gauss-Newton on a fixed batch; PSD iff output loss is convex",
                "drops the residual-times-network-Hessian term of the true Hessian",
            ],
            covered=len(self.params),
            uncovered=self._uncovered,
            sample_count=n_samples,
        )

    def _jvp_and_hout(self, update: ParamDict) -> tuple[Tensor, Tensor, Any]:
        from torch.func import grad, jvp, vjp

        f = lambda *ps: self._call(ps, self.inputs)  # noqa: E731
        out, ju = jvp(f, self._primals, self._tangents(update))
        _, hju = jvp(grad(self.output_loss), (out.detach(),), (ju,))
        return ju, hju, f

    def _matvec_uncounted(self, update: ParamDict) -> dict[Tensor, Tensor]:
        from torch.func import vjp

        with torch.enable_grad():
            _, hju, f = self._jvp_and_hout(update)
            _, pullback = vjp(f, *self._primals)
            cot = pullback(hju)
        return {p: c.detach() for p, c in zip(self.params, cot)}

    def matvec(self, update: ParamDict) -> dict[Tensor, Tensor]:
        t0 = time.perf_counter()
        out = self._matvec_uncounted(update)
        self._count("matvec_calls", t0)
        return out

    def quad(self, update: ParamDict) -> float:
        t0 = time.perf_counter()
        with torch.enable_grad():
            ju, hju, _ = self._jvp_and_hout(update)
        val = float((ju.detach().to(torch.float64) * hju.detach().to(torch.float64)).sum())
        self._count("quad_calls", t0)
        return val


class HessianCurvature(_FunctionalCurvature):
    """True loss Hessian via forward-over-reverse HVP. **Possibly indefinite.**

    Marked ``psd=False``: the stability probe reports its ``quad`` values but will
    not derive PSD-only quantities (effective rank / concentration) from it.

    Args:
        model: The network.
        inputs: Model input for the held-out batch.
        output_loss: ``output_loss(outputs) -> scalar``.
        params: Parameters to differentiate (default: all trainable).
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Any,
        output_loss: Callable[[Tensor], Tensor],
        *,
        params: Optional[Sequence[Tensor]] = None,
    ) -> None:
        super().__init__(model, params)
        self.inputs = inputs
        self.output_loss = output_loss
        self.metadata = CurvatureMetadata(
            name="hessian",
            psd=False,
            assumptions=["true Hessian on a fixed batch; may be indefinite"],
            covered=len(self.params),
            uncovered=self._uncovered,
        )

    def _matvec_uncounted(self, update: ParamDict) -> dict[Tensor, Tensor]:
        from torch.func import grad, jvp

        loss = lambda *ps: self.output_loss(self._call(ps, self.inputs))  # noqa: E731
        g = grad(loss, argnums=tuple(range(len(self.params))))
        with torch.enable_grad():
            _, hv = jvp(g, self._primals, self._tangents(update))
        return {p: h.detach() for p, h in zip(self.params, hv)}

    def matvec(self, update: ParamDict) -> dict[Tensor, Tensor]:
        t0 = time.perf_counter()
        out = self._matvec_uncounted(update)
        self._count("matvec_calls", t0)
        return out


# ---------------------------------------------------------------------------
# Low-rank compression of any proxy
# ---------------------------------------------------------------------------

class LowRankCurvature(CurvatureProxy):
    """Top-``rank`` eigendecomposition of another proxy: ``H ~= V diag(l) V^T (+ c I)``.

    Built once by randomized subspace iteration on the source proxy's ``matvec``
    (``(power_iters + 1) * (rank + oversample)`` matvecs, plus ``trace_probes`` for
    the optional remainder), after which ``matvec`` costs two thin matrix products.
    Intended for the stability simulator, which needs one ``matvec`` per simulated
    step: with a GGN source that is a forward+backward pass per step, which can
    exceed the cost of a real training step.

    Args:
        params: Parameters spanned (flattened in this order).
        eigenvalues: ``[r]`` eigenvalues.
        eigenvectors: ``[n, r]`` orthonormal eigenvectors.
        residual: Isotropic remainder ``c`` added on the orthogonal complement.
        psd: Whether the result is PSD (eigenvalues clamped at 0 if so).
        source: Name of the source proxy (for metadata).
    """

    def __init__(
        self,
        params: Sequence[Tensor],
        eigenvalues: Tensor,
        eigenvectors: Tensor,
        *,
        residual: float = 0.0,
        psd: bool = True,
        source: str = "explicit",
        build_matvecs: int = 0,
        build_seconds: float = 0.0,
    ) -> None:
        self.params = list(params)
        n = sum(p.numel() for p in self.params)
        if eigenvectors.shape[0] != n or eigenvectors.shape[1] != eigenvalues.numel():
            raise ValueError("eigenvectors must be [n, r] matching params and eigenvalues.")
        if residual < 0 and psd:
            raise ValueError("a PSD low-rank proxy needs residual >= 0.")
        self.eigenvalues = eigenvalues.detach().clamp_min(0.0) if psd else eigenvalues.detach()
        self.eigenvectors = eigenvectors.detach()
        self.residual = float(residual)
        self.metadata = CurvatureMetadata(
            name=f"low_rank[{source}]",
            psd=psd,
            assumptions=[
                f"top-{eigenvalues.numel()} eigenpairs of {source}"
                + (f" plus isotropic remainder {residual:.3g}" if residual else "; rest set to 0"),
            ],
            covered=len(self.params),
            rank=int(eigenvalues.numel()),
        )
        self.metadata.cost["build_matvecs"] = float(build_matvecs)
        self.metadata.cost["build_seconds"] = float(build_seconds)

    @classmethod
    def from_proxy(
        cls,
        proxy: CurvatureProxy,
        params: Sequence[Tensor],
        rank: int,
        *,
        oversample: int = 8,
        power_iters: int = 2,
        trace_probes: int = 0,
        seed: int = 0,
    ) -> "LowRankCurvature":
        """Randomized subspace iteration on ``proxy.matvec`` (does not touch the RNG).

        ``trace_probes > 0`` estimates the trace left unexplained by the top ``rank``
        eigenpairs (Hutchinson on the deflated operator) and spreads it isotropically
        over the orthogonal complement.
        """
        params = list(params)
        n = sum(p.numel() for p in params)
        k = min(n, rank + oversample)
        gen = torch.Generator().manual_seed(seed)
        dtype = params[0].dtype
        t0 = time.perf_counter()
        calls = 0

        def apply(mat: Tensor) -> Tensor:
            nonlocal calls
            cols = []
            for j in range(mat.shape[1]):
                hv = proxy.matvec(unflatten_like(mat[:, j].to(dtype), params))
                cols.append(flatten_params(hv, params).to(torch.float64))
                calls += 1
            return torch.stack(cols, 1)

        q, _ = torch.linalg.qr(torch.randn(n, k, generator=gen, dtype=torch.float64))
        for _ in range(power_iters):
            q, _ = torch.linalg.qr(apply(q))
        hq = apply(q)
        t = q.t() @ hq
        evals, evecs = torch.linalg.eigh(0.5 * (t + t.t()))
        order = torch.argsort(evals, descending=True)[:rank]
        lam = evals[order]
        vec = q @ evecs[:, order]
        residual = 0.0
        if trace_probes > 0 and n > rank:
            # Hutchinson on the deflated operator (I - V V^T) H (I - V V^T), whose
            # trace is the unexplained mass; projecting first removes the variance
            # contributed by the dominant eigenvalues (the Hutch++ idea).
            z = (torch.randint(0, 2, (n, trace_probes), generator=gen) * 2 - 1).to(torch.float64)
            z = z - vec @ (vec.t() @ z)
            tr_rest = float((z * apply(z)).sum() / trace_probes)
            residual = max(0.0, tr_rest / (n - rank))
        return cls(
            params, lam.to(dtype), vec.to(dtype),
            residual=residual, psd=proxy.metadata.psd, source=proxy.metadata.name,
            build_matvecs=calls, build_seconds=time.perf_counter() - t0,
        )

    def matvec_flat(self, v: Tensor) -> Tensor:
        """``H v`` for a flat vector (no dict round trip; used by the simulator)."""
        vecs = self.eigenvectors if self.eigenvectors.dtype == v.dtype else self.eigenvectors.to(v.dtype)
        coef = vecs.t() @ v
        out = vecs @ (self.eigenvalues.to(v.dtype) * coef)
        if self.residual:
            out = out + self.residual * (v - vecs @ coef)
        return out

    def _matvec_uncounted(self, update: ParamDict) -> dict[Tensor, Tensor]:
        return unflatten_like(self.matvec_flat(flatten_params(update, self.params)), self.params)

    def matvec(self, update: ParamDict) -> dict[Tensor, Tensor]:
        t0 = time.perf_counter()
        out = self._matvec_uncounted(update)
        self._count("matvec_calls", t0)
        return out
