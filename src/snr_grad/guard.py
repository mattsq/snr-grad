"""
Adam memory guard: bound the update ratio of Adam-family (per coordinate) and Muon-style
(in spectral norm) optimizers.

Adam's step for a coordinate is ``lr * m_hat / (sqrt(v_hat) + eps)``. In steady,
noise-dominated training that ratio is well below 1, but when a coordinate's gradient
falls silent its second moment ``v_hat`` decays (at rate ``beta2``), and when the
gradient returns the step can reach

    A(beta1, beta2) = max_k (1 - beta1^k) / sqrt(1 - beta2^k)

times ``lr`` (6.6 for ``beta2 = 0.999``; see
:func:`snr_grad.adam_reactivation_gain`). In study E9 (``studies/stability_study``)
such overshoots in a few converged coordinates caused large, isolated loss spikes in
small Transformer LMs trained on confident targets. In study E10, bounding the ratio
at 1 removed the large spikes at no measurable loss cost; refilling the stale memory
did slightly better than clipping alone.

Two modes, applied **after** the base optimizer's step:

* ``"cap"``: clip each coordinate's displacement to ``limit * lr``. Decoupled weight
  decay is subtracted first and left untouched. This works for any per-coordinate
  optimizer, whatever it does to the gradient internally.
* ``"refill"`` (default): ``"cap"``, plus raise the stored second moment wherever
  ``|m_hat| / (sqrt(v_hat) + eps)`` exceeds ``limit``, to the value at which the ratio
  equals ``limit``. The memory is then no longer stale, so the following steps are
  damped too. This needs Adam-style state (``exp_avg``, ``exp_avg_sq``, ``step``);
  parameters without it fall back to ``"cap"``. AMSGrad's ``max_exp_avg_sq`` is
  raised as well.

The ratio uses the ungated moments, so for the gated SNR optimizers (whose step is
``q * m_hat / (sqrt(v_hat) + eps)`` with ``q <= 1``) the refill is a conservative
memory correction and the cap acts on the actual gated step.

Like :class:`snr_grad.ActivationPreconditioner`, the guard is an *external* helper
with a functional form and a delegating wrapper::

    from snr_grad import GuardedOptimizer, MemoryGuard

    # Wrapper: snapshot, base.step(), guard -- in one call.
    opt = GuardedOptimizer(torch.optim.AdamW(model.parameters(), lr=3e-4))
    loss.backward(); opt.step(); opt.zero_grad()

    # External form, around any optimizer:
    guard = MemoryGuard(optimizer)
    loss.backward()
    guard.snapshot_(); optimizer.step(); guard.guard_()

    # Monitoring only (free; reads the optimizer state):
    from snr_grad import adam_update_overshoot
    adam_update_overshoot(optimizer)   # {param: max |m_hat| / (sqrt(v_hat) + eps)}

**Muon-style optimizers.** An orthogonalized step is not on the ``lr`` scale per
coordinate; it is on the ``lr`` scale in spectral norm (Muon is steepest descent under the
spectral norm). So for matrices that a Muon-style optimizer orthogonalizes, the guard
bounds the step's *spectral norm* instead: ``||delta W||_2 <= spectral_limit * lr_eff``,
where ``lr_eff`` is the lr after Muon's shape-dependent adjustment. A step above the bound
is scaled down as a whole. Plain Muon keeps no second moment on its matrices, so nothing
is refilled there; an optimizer that keeps Adam state on them (:class:`SNRMuon`, whose
Newton-Schulz input is ``m_hat / (sqrt(v_hat) + eps)``) is refilled as usual, which stops a
stale coordinate from dominating the orthogonalized direction.
For plain Muon this is a safety net rather than a correction: Newton-Schulz output never
exceeds :func:`snr_grad.muon_update_gain` (1.20 for the standard coefficients), and the
default ``spectral_limit=1.25`` sits just above it. It bounds variants whose step can grow
(row- or element-wise second moments, gates, custom scalings). The parameters such
optimizers hand to Adam (embeddings, biases, norms, often the output head) keep the
per-coordinate guard and its refill. In study E11 (``studies/stability_study``) that Adam
half still overshot to 3-5 lr under Muon; guarding it left no loss events. Muon's own
failure at high lr, cumulative growth of weight norms and attention logits, happens
over many bounded steps; a per-step guard does not address it (weight decay does).

A parameter counts as orthogonalized (``geometry="auto"``) when its group has a
``use_muon`` flag set to True (Keller Jordan's ``MuonWithAuxAdam`` and its descendants),
or, without that flag, is 2-D and its group has a Newton-Schulz setting: ``ns_steps``
(``torch.optim.Muon``) or ``muon_ns_steps`` (:class:`SNRMuon`, which orthogonalizes only
2-D parameters). Keller's single-device ``Muon`` has neither key; use
``geometry="spectral"`` for it. ``geometry="spectral"`` or ``"coordinate"`` forces one
treatment for every parameter with two or more dimensions. As in Keller's Muon, a 3-D
parameter is a stack of matrices, each capped on its own, and a parameter with four or
more dimensions (a conv kernel) is flattened to ``(out, rest)``.

Scope: optimizers whose parameters hold the iterate. Per coordinate:
``torch.optim.Adam`` / ``AdamW`` (including ``amsgrad``, ``foreach`` and ``fused``),
:class:`SNRAdamW`, :class:`MARSSNRAdamW`, and :class:`DoPr` around one of these. Spectral:
``torch.optim.Muon``, ``MuonWithAuxAdam``-style optimizers (both halves) and
:class:`SNRMuon`. It is not meaningful for schedule-free optimizers, whose parameters hold
an interpolation rather than the iterate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import torch
from torch import Tensor

GuardMode = Literal["cap", "refill"]
Geometry = Literal["auto", "coordinate", "spectral"]


@dataclass
class MemoryGuardConfig:
    """Settings for :class:`MemoryGuard` / :class:`GuardedOptimizer`.

    Args:
        limit: Largest allowed update ratio ``|delta theta| / lr`` per coordinate
            (excluding decoupled weight decay). ``math.inf`` disables the guard.
            Steady Adam steps are below about 1, so 1 only touches stale coordinates.
        mode: ``"refill"`` (clip and correct the stale second moment) or ``"cap"``
            (clip only).
        decoupled_weight_decay: Whether the base applies weight decay as
            ``theta *= 1 - lr * wd`` (AdamW and the SNR optimizers) rather than through
            the gradient (``torch.optim.Adam``). ``None`` reads the group's
            ``decoupled_weight_decay`` flag when present, treats a plain
            ``torch.optim.Adam`` as coupled, and anything else as decoupled.
        track_stats: Record :attr:`MemoryGuard.last_stats` after each guarded step
            (one device sync per step). Off by default, as for the SNR optimizers.
        spectral_limit: Largest allowed ``||delta W||_2 / lr_eff`` for orthogonalized
            matrices (excluding decoupled weight decay). The default sits just above the
            1.20 that standard Muon can reach (:func:`snr_grad.muon_update_gain`), so plain
            Muon steps are untouched; 1 enforces Muon's nominal step exactly.
            ``math.inf`` disables the spectral cap.
        geometry: ``"auto"`` (orthogonalized matrices detected from the parameter group;
            see the module docstring), ``"spectral"`` (every parameter with two or more
            dimensions) or ``"coordinate"`` (every parameter per coordinate, as for Adam).
        spectral_lr_scale: ``f(param, group) -> float``, the factor between the group's
            ``lr`` and the nominal spectral step of ``param``. ``None`` follows the base:
            ``torch.optim.Muon``'s ``adjust_lr_fn`` (``"original"``:
            ``sqrt(max(1, rows/cols))``; ``"match_rms_adamw"``: ``0.2 sqrt(max(rows, cols))``),
            ``sqrt(max(1, shape[-2]/shape[-1]))`` for ``use_muon`` groups (Keller's rule), and 1
            otherwise.
        power_iters: Power-iteration steps per matrix and step for ``||delta W||_2``,
            warm-started from the previous step. The estimate approaches the norm from
            below. It is accurate when one direction dominates the step (the overshoot
            the default cap is for), but slow on Muon's nearly flat spectrum: with
            ``spectral_limit=1`` (inside Muon's own range) steps exceeded the bound by up
            to 13% at 3 iterations and 4% at 20. ``None`` computes the norm exactly (one
            SVD per matrix and step) for a strict cap; for bfloat16 or float16 parameters
            the cap is still approximate, because the result is rounded back onto the
            parameter's grid.
    """

    limit: float = 1.0
    mode: GuardMode = "refill"
    decoupled_weight_decay: Optional[bool] = None
    track_stats: bool = False
    spectral_limit: float = 1.25
    geometry: Geometry = "auto"
    spectral_lr_scale: Optional[Callable[[Tensor, dict], float]] = None
    power_iters: Optional[int] = 3

    def __post_init__(self) -> None:
        if not self.limit > 0:
            raise ValueError(f"limit must be positive, got {self.limit}")
        if self.mode not in ("cap", "refill"):
            raise ValueError(f"mode must be 'cap' or 'refill', got {self.mode!r}")
        if not self.spectral_limit > 0:
            raise ValueError(f"spectral_limit must be positive, got {self.spectral_limit}")
        if self.geometry not in ("auto", "coordinate", "spectral"):
            raise ValueError(f"geometry must be 'auto', 'coordinate' or 'spectral', got {self.geometry!r}")
        if self.power_iters is not None and self.power_iters < 1:
            raise ValueError(f"power_iters must be at least 1, got {self.power_iters}")


def _unwrap(optimizer: Any) -> Any:
    """The innermost optimizer of a delegating wrapper chain (``DoPr``, ``GuardedOptimizer``)."""
    seen = set()
    while hasattr(optimizer, "__dict__") and "base" in optimizer.__dict__ and id(optimizer) not in seen:
        seen.add(id(optimizer))
        optimizer = optimizer.__dict__["base"]
    return optimizer


def _adam_moments(state: dict, group: dict) -> Optional[tuple[Tensor, Tensor, float, float, float]]:
    """``(exp_avg, second_moment, bias_correction1, bias_correction2, eps)``, or None.

    ``second_moment`` is the tensor Adam divides by: ``max_exp_avg_sq`` under AMSGrad,
    else ``exp_avg_sq``.
    """
    if not all(k in state for k in ("exp_avg", "exp_avg_sq", "step")) or "betas" not in group:
        return None
    beta1, beta2 = group["betas"]
    t = float(state["step"])
    if t < 1:
        return None
    bc1, bc2 = 1.0 - float(beta1) ** t, 1.0 - float(beta2) ** t
    second = state["max_exp_avg_sq"] if group.get("amsgrad", False) and "max_exp_avg_sq" in state else state["exp_avg_sq"]
    return state["exp_avg"], second, bc1, bc2, float(group.get("eps", 1e-8))


def _is_orthogonalized(p: Tensor, group: dict, geometry: str) -> bool:
    """Whether the guard treats ``p`` in spectral norm (see the module docstring)."""
    if p.ndim < 2 or geometry == "coordinate":
        return False
    if geometry == "spectral":
        return True
    if "use_muon" in group:
        return bool(group["use_muon"])
    return p.ndim == 2 and ("ns_steps" in group or "muon_ns_steps" in group)


def _as_matrices(t: Tensor) -> Tensor:
    """``t`` as a batch of matrices: 2-D -> 1 matrix, 3-D -> a stack, more -> ``(out, rest)``."""
    if t.ndim == 3:
        return t
    return t.reshape(1, t.shape[0], -1)


def _default_spectral_lr_scale(p: Tensor, group: dict) -> float:
    """Factor between ``group["lr"]`` and the nominal spectral step of ``p``."""
    rows, cols = p.shape[-2], p.shape[-1]           # as torch Muon and Keller's muon_update
    if "adjust_lr_fn" in group:                   # torch.optim.Muon; None means "original"
        fn = group["adjust_lr_fn"] or "original"
    else:
        fn = "original" if "use_muon" in group else None
    if fn == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, cols))
    if fn == "original":
        return math.sqrt(max(1.0, rows / cols))
    return 1.0


@torch.no_grad()
def adam_update_overshoot(optimizer: Any) -> dict[Tensor, float]:
    """Per-parameter ``max |m_hat| / (sqrt(v_hat) + eps)`` read from Adam-style state.

    A free spike monitor: values well above 1 mean the second-moment memory is stale
    relative to the current first moment (study E9: no run whose largest value stayed
    below about 1.85 had a large spike, and every spike above severity 3 came with values
    of 3.7 or more). It is bounded by :func:`snr_grad.adam_reactivation_gain`. Parameters
    without Adam-style state are omitted.
    """
    base = _unwrap(optimizer)
    out: dict[Tensor, float] = {}
    for group in base.param_groups:
        for p in group["params"]:
            mom = _adam_moments(base.state.get(p, {}), group)
            if mom is None:
                continue
            exp_avg, second, bc1, bc2, eps = mom
            out[p] = float((exp_avg.abs() / bc1).div_((second / bc2).sqrt_().add_(eps)).max())
    return out


class MemoryGuard:
    """External helper bounding each coordinate's update ratio (see module docstring).

    Call :meth:`snapshot_` before the base optimizer's ``step()`` and :meth:`guard_`
    after it. The snapshot holds one copy of every parameter with a gradient.

    Args:
        optimizer: The optimizer whose steps are guarded (any object with
            ``param_groups`` and ``state``; delegating wrappers such as :class:`DoPr`
            are fine).
        config: A :class:`MemoryGuardConfig` (defaults: ``limit=1``, ``"refill"``).
    """

    def __init__(self, optimizer: Any, config: Optional[MemoryGuardConfig] = None) -> None:
        self.optimizer = optimizer
        self.config = config or MemoryGuardConfig()
        self._snapshot: dict[Tensor, Tensor] = {}
        self._armed = False          # a snapshot was taken (it may be empty)
        self._vectors: dict[Tensor, Tensor] = {}   # power-iteration warm starts
        self.last_stats: dict[str, float] = {}

    def _decoupled(self, group: dict) -> bool:
        if self.config.decoupled_weight_decay is not None:
            return self.config.decoupled_weight_decay
        if "decoupled_weight_decay" in group:
            return bool(group["decoupled_weight_decay"])
        base = _unwrap(self.optimizer)
        return not (isinstance(base, torch.optim.Adam) and not isinstance(base, torch.optim.AdamW))

    @torch.no_grad()
    def snapshot_(self) -> None:
        """Record the parameters (with a gradient) before the base step."""
        self._snapshot = {
            p: p.detach().clone()
            for group in self.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        }
        self._armed = True

    @torch.no_grad()
    def guard_(self) -> dict[str, float]:
        """Bound this step's updates per coordinate or in spectral norm (and refill stale memory).

        With ``track_stats``, returns and stores in :attr:`last_stats` ``max_ratio``
        (largest unclipped ``|delta theta| / lr`` over per-coordinate parameters),
        ``clipped`` (coordinates clipped), ``refilled`` (coordinates whose second moment
        was raised), ``max_spectral_ratio`` (largest unclipped ``||delta W||_2 / lr_eff``
        over orthogonalized matrices) and ``rescaled`` (matrices scaled down); otherwise
        returns an empty dict.
        """
        if not self._armed:
            raise RuntimeError("MemoryGuard.guard_() called without a preceding snapshot_()")
        self._armed = False
        cfg = self.config
        track = cfg.track_stats
        ratios: list[Tensor] = []
        clipped: list[Tensor] = []
        refilled: list[Tensor] = []
        spectral_ratios: list[Tensor] = []
        rescaled: list[Tensor] = []
        state_all = self.optimizer.state
        for group in self.optimizer.param_groups:
            lr = float(group["lr"])
            if lr <= 0:
                continue
            wd = float(group.get("weight_decay", 0.0)) if self._decoupled(group) else 0.0
            bound = cfg.limit * lr
            for p in group["params"]:
                anchor = self._snapshot.get(p)
                if anchor is None:
                    continue
                if wd:
                    anchor.mul_(1.0 - lr * wd)            # where decay alone would have moved it
                if _is_orthogonalized(p, group, cfg.geometry):
                    self._cap_spectral(p, anchor, group, lr, spectral_ratios if track else None, rescaled)
                elif math.isfinite(bound) or track:
                    delta = p.sub_(anchor)                # p now holds this step's update
                    if track:
                        ratios.append(delta.abs().amax() / lr)
                        if math.isfinite(bound):
                            clipped.append((delta.abs() > bound).sum())
                    delta.clamp_(-bound, bound).add_(anchor)
                if cfg.mode == "refill" and math.isfinite(cfg.limit):
                    count = self._refill(state_all.get(p, {}), group, track)
                    if count is not None:
                        refilled.append(count)
        self._snapshot = {}
        if not track:
            return {}

        def total(xs: list[Tensor], reduce: Any) -> float:
            if not xs:
                return 0.0
            dev = xs[0].device
            dtype = torch.float32 if dev.type == "mps" else torch.float64   # MPS has no float64
            return float(reduce(torch.stack([x.to(dev, dtype) for x in xs])))

        self.last_stats = {"max_ratio": total(ratios, torch.max), "clipped": total(clipped, torch.sum),
                           "refilled": total(refilled, torch.sum),
                           "max_spectral_ratio": total(spectral_ratios, torch.max),
                           "rescaled": total(rescaled, torch.sum)}
        return self.last_stats

    def _cap_spectral(self, p: Tensor, anchor: Tensor, group: dict, lr: float,
                      ratios: Optional[list[Tensor]], rescaled: list[Tensor]) -> None:
        """Scale this step's update of ``p`` so that ``||delta W||_2 <= spectral_limit * lr_eff`` per matrix."""
        scale_fn = self.config.spectral_lr_scale or _default_spectral_lr_scale
        nominal = lr * float(scale_fn(p, group))
        bound = self.config.spectral_limit * nominal
        if not math.isfinite(bound) and ratios is None:
            return
        delta = p.sub_(anchor)                            # p now holds this step's update
        sigma = self._spectral_norm(p, _as_matrices(delta))           # one norm per matrix
        if ratios is not None:
            ratios.append((sigma / nominal).max())
            if math.isfinite(bound):
                rescaled.append((sigma > bound).sum())
        if math.isfinite(bound):
            factor = (bound / sigma).clamp_(max=1.0)                   # sigma = 0 gives inf -> 1
            delta.mul_(factor.view(-1, 1, 1) if delta.ndim == 3 else factor.reshape(()))
        delta.add_(anchor)

    def _spectral_norm(self, p: Tensor, mats: Tensor) -> Tensor:
        """``||M||_2`` for each matrix of the batch ``mats`` (shape ``(batch,)``, no sync).

        Warm-started power iteration, or exact with ``power_iters=None``. A matrix with a
        non-finite entry gets a NaN norm, so a NaN step stays NaN as in the base.
        """
        dtype = mats.dtype if mats.dtype in (torch.float32, torch.float64) else torch.float32
        m = mats.to(dtype)
        if self.config.power_iters is None:
            finite = torch.isfinite(m)
            sigma = torch.linalg.matrix_norm(torch.where(finite, m, 0.0), ord=2)   # SVD rejects inf/NaN
            return torch.where(finite.flatten(1).all(1), sigma, float("nan"))
        v = self._vectors.get(p)
        if v is None or v.shape != (m.shape[0], m.shape[2]) or v.dtype != dtype or v.device != m.device:
            g = torch.Generator().manual_seed(0)
            v = torch.randn(m.shape[0], m.shape[2], generator=g, dtype=torch.float64).to(m.device, dtype)
            v = v / v.norm(dim=1, keepdim=True)
        tiny = torch.finfo(dtype).tiny
        sigma = torch.zeros(m.shape[0], dtype=dtype, device=m.device)
        for _ in range(self.config.power_iters):
            u = (m @ v.unsqueeze(-1)).squeeze(-1)
            u = u / u.norm(dim=1, keepdim=True).clamp_min(tiny)
            w = (m.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1)
            sigma = w.norm(dim=1)
            s = sigma.unsqueeze(1)
            v = torch.where(s > 0, w / s.clamp_min(tiny), v)            # keep the start on a zero step
        self._vectors[p] = v
        return sigma

    def _refill(self, state: dict, group: dict, track: bool) -> Optional[Tensor]:
        mom = _adam_moments(state, group)
        if mom is None:
            return None
        exp_avg, _, bc1, bc2, eps = mom
        # v needed for |m_hat| / (sqrt(v_hat) + eps) == limit, stored without bias correction.
        need = exp_avg.abs().mul_(1.0 / (bc1 * self.config.limit)).sub_(eps).clamp_min_(0.0).square_().mul_(bc2)
        count = (need > state["exp_avg_sq"]).sum() if track else None
        torch.maximum(state["exp_avg_sq"], need, out=state["exp_avg_sq"])
        if "max_exp_avg_sq" in state:
            torch.maximum(state["max_exp_avg_sq"], need, out=state["max_exp_avg_sq"])
        return count


class GuardedOptimizer:
    """Delegating wrapper: :meth:`MemoryGuard.snapshot_`, ``base.step()``, then :meth:`MemoryGuard.guard_`.

    Like :class:`DoPr`, this is **not** a ``torch.optim.Optimizer`` subclass. Attach
    learning-rate schedulers to the **base** optimizer; ``param_groups`` and ``state``
    are the base's, so the scheduler still drives the learning rate the guard uses.
    Unknown attributes are forwarded to the base optimizer. Wrappers compose, e.g.
    ``GuardedOptimizer(DoPr(SNRAdamW(...), model))``.

    Args:
        base_optimizer: The optimizer to guard (see the module docstring for scope).
        config: A :class:`MemoryGuardConfig`.

    Example::

        base = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
        opt = GuardedOptimizer(base)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=1000)
        loss.backward(); opt.step(); opt.zero_grad(); sched.step()
        opt.guard_stats   # {"max_ratio": ..., "clipped": ..., "refilled": ...}
    """

    def __init__(self, base_optimizer: Any, config: Optional[MemoryGuardConfig] = None) -> None:
        self.base = base_optimizer
        self.guard = MemoryGuard(base_optimizer, config)

    def step(self, closure: Any = None) -> Any:
        """Take one guarded base-optimizer step.

        A ``closure`` is evaluated here, once and with grad enabled, so the snapshot can
        cover every parameter that receives a gradient; the base is then stepped
        without it. Optimizers that need several closure evaluations per step (LBFGS)
        are therefore not supported.
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            self.guard.snapshot_()
            self.base.step()
        else:
            self.guard.snapshot_()
            loss = self.base.step()
        self.guard.guard_()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """The base optimizer's state dict (the guard keeps no persistent state)."""
        return self.base.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.base.load_state_dict(state)

    @property
    def param_groups(self) -> Any:
        return self.base.param_groups

    @property
    def state(self) -> Any:
        return self.base.state

    @property
    def config(self) -> MemoryGuardConfig:
        return self.guard.config

    @property
    def guard_stats(self) -> dict[str, float]:
        """Statistics of the last guarded step (see :meth:`MemoryGuard.guard_`).

        Named so as not to hide a base optimizer's own ``last_stats`` (``SNRAdamW``),
        which stays reachable through attribute forwarding.
        """
        return self.guard.last_stats

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails; delegate to the base optimizer.
        try:
            base = self.__dict__["base"]
        except KeyError:
            raise AttributeError(name)
        return getattr(base, name)
