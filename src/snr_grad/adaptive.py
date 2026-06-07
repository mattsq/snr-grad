"""
Adaptive thresholding for the SNR / population-risk gate.

A fixed gate threshold says "suppress gradients when m^2/s is below a manually
chosen boundary". An adaptive threshold instead *chooses* the boundary so the gate
maintains a desired behaviour as gradient statistics drift during training.

For the SNR gate the local signal-to-noise statistic is

    r = m_hat^2 / (s_hat + eps)

and the gate can be written as

    q = r / (r + alpha * lambda_pop)

so ``alpha * lambda_pop`` is the effective threshold scale. For the soft and hard
gates the pass condition ``m_hat^2 > alpha * s_hat`` is equivalent to ``r > alpha``,
so ``alpha`` is the direct pass/fail threshold.

Two control targets are implemented:

* ``target_mean_gate``: keep the average gate value near a target (robust, general).
* ``target_active_fraction``: keep a target fraction of coordinates "active"
  (``q >= active_gate_threshold``), which is more interpretable.

The controller lives in :class:`AdaptiveThresholdConfig` and the helper functions
in this module; the optimizers wire them into ``step()`` around the existing
``compute_gate`` call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, MutableMapping, Optional, Union

import torch
from torch import Tensor


AdaptiveMode = Literal[
    "off",
    "target_mean_gate",
    "target_active_fraction",
    "quantile_threshold",
    "shock_then_sparsify",
]


@dataclass
class AdaptiveThresholdConfig:
    """
    Configuration for self-tuning the SNR gate threshold.

    The controller adapts ``lambda_pop`` (default) and/or ``alpha`` so the gate
    maintains a target behaviour. Adaptation is per optimizer parameter group.
    """

    mode: AdaptiveMode = "off"
    # Which threshold to adapt.
    adapt: Literal["lambda_pop", "alpha", "both"] = "lambda_pop"
    # Main targets.
    target_mean_gate: float = 0.2
    target_active_fraction: float = 0.2
    active_gate_threshold: float = 0.5
    # Update dynamics.
    update_interval: int = 50
    warmup_steps: int = 100
    beta: float = 0.9
    adaptation_lr: float = 0.05
    # Safety clamps.
    min_lambda_pop: float = 1e-4
    max_lambda_pop: float = 1e3
    min_alpha: float = 1e-4
    max_alpha: float = 1e3
    # Granularity (only "param_group" is implemented for now).
    granularity: Literal["global", "param_group", "tensor"] = "param_group"
    # Statistic collection.
    max_sampled_elements: int = 100_000
    # Hysteresis: ignore deviations smaller than this, and cap per-update log moves.
    tolerance: float = 0.02
    max_log_change: float = 0.25
    # Staleness-aware thresholding (optional; requires grad_variances).
    staleness_detection: bool = False
    stale_gate_delta_threshold: float = 0.15
    stale_boost_steps: int = 50
    stale_update_interval: int = 5
    # "shock_then_sparsify": hold a sparse active-fraction target in steady state,
    # transiently raise it when a regime shift is detected, then decay back.
    sparse_target_active_fraction: float = 0.05
    shock_target_active_fraction: float = 0.2
    shock_steps: int = 100
    shock_fast_beta: float = 0.5
    shift_detect_threshold: float = 0.1

    def __post_init__(self) -> None:
        valid_modes = {
            "off",
            "target_mean_gate",
            "target_active_fraction",
            "quantile_threshold",
            "shock_then_sparsify",
        }
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid adaptive mode: {self.mode!r}. Expected one of {valid_modes}.")
        if self.adapt not in {"lambda_pop", "alpha", "both"}:
            raise ValueError(f"Invalid adapt target: {self.adapt!r}.")
        if not (0.0 <= self.target_mean_gate <= 1.0):
            raise ValueError(f"target_mean_gate must be in [0, 1], got {self.target_mean_gate}.")
        if not (0.0 < self.target_active_fraction < 1.0):
            raise ValueError(
                f"target_active_fraction must be in (0, 1), got {self.target_active_fraction}."
            )
        if not (0.0 < self.active_gate_threshold < 1.0):
            raise ValueError(
                f"active_gate_threshold must be in (0, 1), got {self.active_gate_threshold}."
            )
        if self.update_interval < 1:
            raise ValueError(f"update_interval must be >= 1, got {self.update_interval}.")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}.")
        if not (0.0 <= self.beta < 1.0):
            raise ValueError(f"beta must be in [0, 1), got {self.beta}.")
        if self.adaptation_lr < 0.0:
            raise ValueError(f"adaptation_lr must be >= 0, got {self.adaptation_lr}.")
        if self.min_lambda_pop <= 0 or self.max_lambda_pop <= 0:
            raise ValueError("lambda_pop clamps must be positive.")
        if self.min_lambda_pop > self.max_lambda_pop:
            raise ValueError("min_lambda_pop must be <= max_lambda_pop.")
        if self.min_alpha <= 0 or self.max_alpha <= 0:
            raise ValueError("alpha clamps must be positive.")
        if self.min_alpha > self.max_alpha:
            raise ValueError("min_alpha must be <= max_alpha.")
        if self.max_sampled_elements < 1:
            raise ValueError("max_sampled_elements must be >= 1.")
        if self.tolerance < 0.0:
            raise ValueError(f"tolerance must be >= 0, got {self.tolerance}.")
        if self.max_log_change <= 0.0:
            raise ValueError(f"max_log_change must be > 0, got {self.max_log_change}.")
        if self.stale_update_interval < 1:
            raise ValueError("stale_update_interval must be >= 1.")
        if self.stale_boost_steps < 0:
            raise ValueError("stale_boost_steps must be >= 0.")
        if not (0.0 < self.sparse_target_active_fraction < 1.0):
            raise ValueError(
                f"sparse_target_active_fraction must be in (0, 1), "
                f"got {self.sparse_target_active_fraction}."
            )
        if not (0.0 < self.shock_target_active_fraction < 1.0):
            raise ValueError(
                f"shock_target_active_fraction must be in (0, 1), "
                f"got {self.shock_target_active_fraction}."
            )
        if self.shock_steps < 0:
            raise ValueError("shock_steps must be >= 0.")
        if not (0.0 <= self.shock_fast_beta < 1.0):
            raise ValueError(f"shock_fast_beta must be in [0, 1), got {self.shock_fast_beta}.")
        if self.shift_detect_threshold < 0.0:
            raise ValueError("shift_detect_threshold must be >= 0.")


AdaptiveThresholdSpec = Union[AdaptiveThresholdConfig, Mapping[str, Any], None]


def coerce_adaptive_config(
    adaptive_threshold: AdaptiveThresholdSpec,
) -> Optional[AdaptiveThresholdConfig]:
    """Normalise the public ``adaptive_threshold`` argument to a config or None."""
    if adaptive_threshold is None:
        return None
    if isinstance(adaptive_threshold, AdaptiveThresholdConfig):
        return adaptive_threshold
    if isinstance(adaptive_threshold, Mapping):
        return AdaptiveThresholdConfig(**dict(adaptive_threshold))
    raise TypeError(
        "adaptive_threshold must be AdaptiveThresholdConfig, a dict, or None; "
        f"got {type(adaptive_threshold).__name__}."
    )


@dataclass
class AdaptiveObservation:
    """Statistics gathered over one param group during an update step."""

    mean_gate: Optional[float] = None
    active_fraction: Optional[float] = None
    r_samples: Optional[Tensor] = None


def sample_flat(tensor: Tensor, max_elements: int) -> Tensor:
    """Flatten ``tensor`` and uniformly subsample to at most ``max_elements`` entries."""
    flat = tensor.detach().reshape(-1)
    if flat.numel() <= max_elements:
        return flat
    idx = torch.randint(flat.numel(), (max_elements,), device=flat.device)
    return flat[idx]


def smooth_clamped_update(
    *,
    old: float,
    proposed: float,
    beta: float,
    min_value: float,
    max_value: float,
    max_log_change: float,
) -> float:
    """
    EMA-smooth ``proposed`` toward in log-space, cap the per-update log move, and clamp.

    Smoothing and the log-move cap together prevent threshold chatter when a quantile
    spikes, while still letting the threshold track sustained shifts.
    """
    old_v = max(float(old), 1e-30)
    proposed_v = max(float(proposed), 1e-30)
    log_old = math.log(old_v)
    log_proposed = math.log(proposed_v)
    # EMA in log space toward the proposal.
    log_target = beta * log_old + (1.0 - beta) * log_proposed
    delta = log_target - log_old
    delta = max(-max_log_change, min(max_log_change, delta))
    new_value = math.exp(log_old + delta)
    return min(max(new_value, min_value), max_value)


def lambda_for_target_active_fraction(
    r_samples: Tensor,
    *,
    target_active_fraction: float,
    active_gate_threshold: float,
    alpha: float,
    min_lambda: float,
    max_lambda: float,
) -> float:
    """
    Closed-form lambda_pop that makes the top ``target_active_fraction`` of the SNR
    gate active (``q >= active_gate_threshold``).

    Derived from ``q = r / (r + alpha * lambda)``:
        q >= q0  <=>  r >= alpha * lambda * q0 / (1 - q0)
    so for the boundary ``r_threshold = quantile(r, 1 - p)``:
        lambda = r_threshold * (1 - q0) / (alpha * q0)
    """
    p = target_active_fraction
    q0 = active_gate_threshold
    r_threshold = torch.quantile(r_samples.float(), 1.0 - p).item()
    lam = r_threshold * (1.0 - q0) / max(alpha * q0, 1e-12)
    return min(max(lam, min_lambda), max_lambda)


def init_adaptive_group(group: MutableMapping[str, Any]) -> None:
    """Initialise per-group adaptive state, preserving the user's base thresholds."""
    cfg: Optional[AdaptiveThresholdConfig] = group.get("adaptive_threshold")
    if cfg is None or cfg.mode == "off":
        return
    if "_adaptive_state" in group:
        return
    group["base_lambda_pop"] = group["lambda_pop"]
    group["base_alpha"] = group["alpha"]
    group["_adaptive_state"] = {
        "step": 0,
        "ema_mean_gate": None,
        "ema_active_fraction": None,
        "force_update_countdown": 0,
        "shock_countdown": 0,
    }


def _should_update(state: Mapping[str, Any], cfg: AdaptiveThresholdConfig) -> bool:
    """Whether the current (already-incremented) step is an adaptive update step."""
    step = state["step"]
    if step < cfg.warmup_steps:
        return False
    if state.get("force_update_countdown", 0) > 0:
        interval = cfg.stale_update_interval
    else:
        interval = cfg.update_interval
    return step % interval == 0


def adaptive_pre_step(
    group: MutableMapping[str, Any],
) -> tuple[Optional[AdaptiveThresholdConfig], bool]:
    """
    Increment the group's adaptive step counter and report whether to collect stats.

    Returns ``(cfg, collect)`` where ``cfg`` is None when adaptation is disabled.
    """
    cfg: Optional[AdaptiveThresholdConfig] = group.get("adaptive_threshold")
    if cfg is None or cfg.mode == "off":
        return None, False
    init_adaptive_group(group)
    state = group["_adaptive_state"]
    state["step"] += 1
    return cfg, _should_update(state, cfg)


def _update_by_mean_gate(
    group: MutableMapping[str, Any],
    cfg: AdaptiveThresholdConfig,
    observed: AdaptiveObservation,
    alpha_value: float,
) -> None:
    if observed.mean_gate is None:
        return
    state = group["_adaptive_state"]
    old = state.get("ema_mean_gate")
    ema = observed.mean_gate if old is None else cfg.beta * old + (1.0 - cfg.beta) * observed.mean_gate
    state["ema_mean_gate"] = ema

    error = ema - cfg.target_mean_gate
    if abs(error) < cfg.tolerance:
        return

    # error > 0 => gate too permissive => raise threshold => q decreases.
    delta = cfg.adaptation_lr * error
    delta = max(-cfg.max_log_change, min(cfg.max_log_change, delta))

    if cfg.adapt in {"lambda_pop", "both"}:
        lam = max(float(group["lambda_pop"]), 1e-30)
        new_lam = math.exp(math.log(lam) + delta)
        group["lambda_pop"] = min(max(new_lam, cfg.min_lambda_pop), cfg.max_lambda_pop)
    if cfg.adapt in {"alpha", "both"}:
        a = max(float(alpha_value), 1e-30)
        new_a = math.exp(math.log(a) + delta)
        group["alpha"] = min(max(new_a, cfg.min_alpha), cfg.max_alpha)


def _update_by_active_fraction(
    group: MutableMapping[str, Any],
    cfg: AdaptiveThresholdConfig,
    observed: AdaptiveObservation,
    alpha_value: float,
    target_active_fraction: Optional[float] = None,
) -> None:
    r_samples = observed.r_samples
    if r_samples is None or r_samples.numel() < 32:
        return

    p = cfg.target_active_fraction if target_active_fraction is None else target_active_fraction
    group["_adaptive_state"]["current_target_active_fraction"] = p

    state = group["_adaptive_state"]
    if observed.active_fraction is not None:
        old = state.get("ema_active_fraction")
        ema = (
            observed.active_fraction
            if old is None
            else cfg.beta * old + (1.0 - cfg.beta) * observed.active_fraction
        )
        state["ema_active_fraction"] = ema
        if abs(ema - p) < cfg.tolerance:
            return

    q0 = cfg.active_gate_threshold
    r_threshold = torch.quantile(r_samples.float(), 1.0 - p).item()

    def _set_lambda(proposed: float) -> None:
        group["lambda_pop"] = smooth_clamped_update(
            old=group["lambda_pop"],
            proposed=proposed,
            beta=cfg.beta,
            min_value=cfg.min_lambda_pop,
            max_value=cfg.max_lambda_pop,
            max_log_change=cfg.max_log_change,
        )

    def _set_alpha(proposed: float) -> None:
        group["alpha"] = smooth_clamped_update(
            old=alpha_value,
            proposed=proposed,
            beta=cfg.beta,
            min_value=cfg.min_alpha,
            max_value=cfg.max_alpha,
            max_log_change=cfg.max_log_change,
        )

    if group["gate"] == "snr":
        # active <=> r >= alpha * lambda_pop * q0 / (1 - q0), so the quantity the
        # controller targets is the product (alpha * lambda_pop):
        target_scale = r_threshold * (1.0 - q0) / max(q0, 1e-12)
        lam = max(float(group["lambda_pop"]), 1e-30)
        a = max(float(alpha_value), 1e-30)
        if cfg.adapt == "lambda_pop":
            _set_lambda(target_scale / a)
        elif cfg.adapt == "alpha":
            _set_alpha(target_scale / lam)
        else:  # "both": move both geometrically so their product hits the target.
            factor = math.sqrt(max(target_scale, 1e-30) / (a * lam))
            _set_lambda(lam * factor)
            _set_alpha(a * factor)
    else:
        # soft / hard gates: "active" means r > alpha, so the boundary IS alpha and
        # lambda_pop does not move it. We can only honor the target by adapting alpha;
        # if the user pinned alpha (adapt="lambda_pop") there is nothing sound to do.
        if cfg.adapt in {"alpha", "both"}:
            _set_alpha(r_threshold)


def _detect_regime_shift(state: MutableMapping[str, Any], cfg: AdaptiveThresholdConfig,
                         mean_gate: Optional[float]) -> bool:
    """
    Self-contained regime-shift detector based on a fast/slow EMA split of the
    observed mean gate. A shift drives the realized gate away from its operating
    point before the controller re-adapts, so the fast and slow EMAs diverge.
    """
    if mean_gate is None:
        return False
    fast = state.get("shock_fast_ema")
    slow = state.get("shock_slow_ema")
    fast = mean_gate if fast is None else cfg.shock_fast_beta * fast + (1.0 - cfg.shock_fast_beta) * mean_gate
    slow = mean_gate if slow is None else cfg.beta * slow + (1.0 - cfg.beta) * mean_gate
    state["shock_fast_ema"] = fast
    state["shock_slow_ema"] = slow
    return abs(fast - slow) > cfg.shift_detect_threshold


def _update_shock_then_sparsify(
    group: MutableMapping[str, Any],
    cfg: AdaptiveThresholdConfig,
    observed: AdaptiveObservation,
    alpha_value: float,
) -> None:
    """
    Hold a sparse active-fraction target; on a detected regime shift, transiently
    raise the target (broad exploration), then decay it back toward sparse.

    The elevated window lasts ``shock_steps`` controller updates; the effective
    target decays linearly from ``shock_target_active_fraction`` back to
    ``sparse_target_active_fraction`` across it.
    """
    state = group["_adaptive_state"]

    if _detect_regime_shift(state, cfg, observed.mean_gate) and state.get("shock_countdown", 0) == 0:
        state["shock_countdown"] = cfg.shock_steps

    countdown = state.get("shock_countdown", 0)
    if countdown > 0 and cfg.shock_steps > 0:
        frac = countdown / cfg.shock_steps  # 1.0 at shift -> 0.0 at end of window
        target = (
            cfg.sparse_target_active_fraction
            + frac * (cfg.shock_target_active_fraction - cfg.sparse_target_active_fraction)
        )
        state["shock_countdown"] = countdown - 1
    else:
        target = cfg.sparse_target_active_fraction

    _update_by_active_fraction(group, cfg, observed, alpha_value, target_active_fraction=target)


def apply_adaptive_update(
    group: MutableMapping[str, Any],
    cfg: AdaptiveThresholdConfig,
    observed: AdaptiveObservation,
    alpha_value: float,
) -> None:
    """Run the configured controller and mutate the group's live threshold(s)."""
    state = group["_adaptive_state"]
    if state.get("force_update_countdown", 0) > 0:
        state["force_update_countdown"] -= 1

    if cfg.mode == "shock_then_sparsify":
        _update_shock_then_sparsify(group, cfg, observed, alpha_value)
    elif cfg.mode == "target_mean_gate":
        _update_by_mean_gate(group, cfg, observed, alpha_value)
    elif cfg.mode in {"target_active_fraction", "quantile_threshold"}:
        _update_by_active_fraction(group, cfg, observed, alpha_value)


def finalize_adaptive_group(
    group: MutableMapping[str, Any],
    cfg: Optional[AdaptiveThresholdConfig],
    collect: bool,
    *,
    gate_sums: list,
    active_sums: list,
    r_samples: list,
    elem_count: int,
    delta_sums: list,
    delta_count: int,
    alpha_value: float,
) -> None:
    """
    Reduce one group's collected statistics and run the controller.

    No-op unless adaptation is enabled and this is an update step. Reductions move a
    handful of scalars to host memory; this only runs on update steps so the
    per-step cost is amortised.
    """
    if cfg is None or not collect or elem_count == 0:
        return

    device = gate_sums[0].device
    mean_gate = torch.stack([x.to(device) for x in gate_sums]).sum().item() / elem_count
    active_fraction = (
        torch.stack([x.to(device) for x in active_sums]).sum().item() / elem_count
    )
    r_cat = torch.cat([x.to(device) for x in r_samples]) if r_samples else None

    observed = AdaptiveObservation(
        mean_gate=mean_gate,
        active_fraction=active_fraction,
        r_samples=r_cat,
    )
    apply_adaptive_update(group, cfg, observed, alpha_value)

    # Staleness: trigger a temporary high-frequency recalibration window.
    if cfg.staleness_detection and delta_count > 0:
        gate_delta = (
            torch.stack([x.to(device) for x in delta_sums]).sum().item() / delta_count
        )
        if gate_delta > cfg.stale_gate_delta_threshold:
            group["_adaptive_state"]["force_update_countdown"] = cfg.stale_boost_steps


def get_threshold_state(optimizer: Any) -> dict:
    """Snapshot the live adaptive threshold state, keyed by ``group_<i>``."""
    out: dict = {}
    for i, group in enumerate(optimizer.param_groups):
        cfg: Optional[AdaptiveThresholdConfig] = group.get("adaptive_threshold")
        if cfg is None or cfg.mode == "off":
            continue
        state = group.get("_adaptive_state", {})
        alpha = group["alpha"]
        out[f"group_{i}"] = {
            "lambda_pop": float(group["lambda_pop"]),
            "alpha": alpha if isinstance(alpha, str) else float(alpha),
            "ema_mean_gate": state.get("ema_mean_gate"),
            "ema_active_fraction": state.get("ema_active_fraction"),
            "target_active_fraction": state.get("current_target_active_fraction"),
            "shock_countdown": state.get("shock_countdown", 0),
            "step": state.get("step", 0),
        }
    return out


def reset_threshold_state(optimizer: Any) -> None:
    """Restore every adaptive group to its base thresholds and clear controller state."""
    for group in optimizer.param_groups:
        cfg: Optional[AdaptiveThresholdConfig] = group.get("adaptive_threshold")
        if cfg is None or cfg.mode == "off":
            continue
        if "base_lambda_pop" in group:
            group["lambda_pop"] = group["base_lambda_pop"]
        if "base_alpha" in group:
            group["alpha"] = group["base_alpha"]
        group["_adaptive_state"] = {
            "step": 0,
            "ema_mean_gate": None,
            "ema_active_fraction": None,
            "force_update_countdown": 0,
        }
