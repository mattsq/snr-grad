"""
Benchmark: variance-estimation backends for the SNRAdamW gate.

Compares ways of supplying ``grad_variances`` to ``SNRAdamW.step`` on a synthetic
sparse linear-regression task with label noise (the same overparameterized regime
used by ``benchmark.py``):

  - EMA-only        : the optimizer's internal streaming variance (baseline)
  - exact-every-step: exact per-sample-gradient variance every step (reference)
  - exact-every-K   : exact probe every K steps, EMA otherwise (hybrid cadence)
  - microbatch K=2  : cheap split-batch estimator, 2 backward chunks
  - microbatch K=4  : cheap split-batch estimator, 4 backward chunks

Because the ground-truth signal coordinates are known, we report gate *quality*
(signal vs noise gate separation and an AUC for signal-vs-noise ranking by gate
value), not just loss. We also report wall-clock and the correlation between the
internal EMA variance and the exact variance.

Run:
    uv run python benchmark_variance_estimators.py
    uv run python benchmark_variance_estimators.py --quick
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_grad import (
    SNRAdamW,
    ExactVarianceEstimator,
    backward_with_microbatch_variance,
    compute_gate,
    resolve_alpha,
)
from torch.func import functional_call


# ---------------------------------------------------------------------------
# Config + data (mirrors benchmark.py)
# ---------------------------------------------------------------------------

@dataclass
class Config:
    task: str = "stationary"          # "stationary", "drift", or "switch"
    d: int = 200
    k: int = 5
    n_train: int = 100
    batch_size: int = 32
    sigma_noise: float = 3.0
    n_steps: int = 2000
    n_seeds: int = 3
    test_size: int = 5000
    lr: float = 3e-3
    weight_decay: float = 0.0
    signal_magnitude: float = 3.0
    rho: float = 0.99
    beta1: float = 0.9
    lambda_pop: float = 1.0
    gate: str = "snr"
    # Non-stationary ("drift") regime: signal coordinates oscillate so their true
    # gradient is large and time-varying. The temporal EMA conflates this drift with
    # noise and over-estimates variance on signal coords; within-batch estimators do not.
    drift_period: int = 40
    drift_strength: float = 0.9
    # "switch" regime: an abrupt task change (new signal support + magnitudes) at this
    # step. A clean single regime-change event for the staleness detector.
    switch_step: int = 300
    # Adaptive "triggered" controller: probe exact variance every `probe_every` steps and
    # track a running baseline of the gate divergence |q_exact - q_ema|. When the current
    # divergence *spikes* above that baseline (regime change), use exact variance for the
    # next `correction_steps` steps, then fall back to the EMA. A change-relative trigger
    # (rather than an absolute threshold) stays quiet when the divergence is merely high but
    # steady, and fires on transitions -- which is what "the EMA just went stale" means.
    probe_every: int = 20
    staleness_beta: float = 0.9      # EMA over observed gate divergence (the baseline)
    staleness_spike: float = 1.8     # fire when divergence > spike * baseline + floor
    staleness_floor: float = 0.02
    correction_steps: int = 60


# Tuned preset that surfaces the EMA's miscalibration under non-stationarity.
DRIFT_PRESET = dict(
    sigma_noise=0.5, n_train=200, n_steps=600, lr=1e-2, beta1=0.95,
    lambda_pop=2.0, drift_period=40, drift_strength=0.9,
)

# "switch" regime: train to convergence on task A, abruptly swap to task B halfway.
SWITCH_PRESET = dict(
    sigma_noise=0.5, n_train=200, n_steps=600, lr=1e-2, beta1=0.95,
    lambda_pop=2.0, switch_step=300,
)


def make_true_weights(d, k, magnitude, seed=0):
    gen = torch.Generator().manual_seed(seed)
    w = torch.zeros(d)
    idx = torch.randperm(d, generator=gen)[:k]
    w[idx] = torch.randn(k, generator=gen) * magnitude
    return w, idx


def make_dataset(w_true, n, sigma_noise, gen):
    d = w_true.shape[0]
    X = torch.randn(n, d, generator=gen)
    noise = torch.randn(n, generator=gen) * sigma_noise
    y = X @ w_true + noise
    return X, y.unsqueeze(1)


def make_signal_spec(cfg: Config, seed=0):
    """Fixed support + base magnitudes + phases for the drifting target."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(cfg.d, generator=gen)[: cfg.k]
    base = torch.randn(cfg.k, generator=gen) * cfg.signal_magnitude
    phase = torch.rand(cfg.k, generator=gen) * 2 * math.pi
    return idx, base, phase


def drifting_weights(step, cfg: Config, idx, base, phase):
    """Instantaneous target weights at a given step (signal magnitudes oscillate)."""
    w = torch.zeros(cfg.d)
    mod = 1.0 + cfg.drift_strength * torch.sin(2 * math.pi * step / cfg.drift_period + phase)
    w[idx] = base * mod
    return w


# ---------------------------------------------------------------------------
# Per-sample / microbatch loss closures for a linear regression head
# ---------------------------------------------------------------------------

def loss_one_sample(model):
    """Per-example summed squared error: its gradient is the per-example gradient."""
    def fn(params, buffers, sample):
        x, y = sample
        pred = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        return ((pred.squeeze(0) - y) ** 2).sum()
    return fn


def microbatch_loss(model, sub_batch):
    x, y = sub_batch  # y has shape [chunk]; squeeze the model's trailing dim to match.
    return ((model(x).squeeze(1) - y) ** 2).mean()


# ---------------------------------------------------------------------------
# Gate-quality metrics
# ---------------------------------------------------------------------------

def gate_auc(gate_vals: torch.Tensor, signal_mask: torch.Tensor) -> float:
    """AUC for separating signal vs noise coordinates by gate value (Mann-Whitney)."""
    pos = gate_vals[signal_mask]
    neg = gate_vals[~signal_mask]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    # Rank-based AUC.
    all_vals = torch.cat([pos, neg])
    ranks = all_vals.argsort().argsort().float() + 1.0
    r_pos = ranks[: pos.numel()].sum()
    auc = (r_pos - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel())
    return float(auc)


@dataclass
class RunResult:
    final_train: float = 0.0
    final_test: float = 0.0
    wall_clock: float = 0.0
    mean_signal_gate: float = 0.0
    mean_noise_gate: float = 0.0
    gate_auc: float = 0.0
    false_suppress_signal: float = 0.0  # fraction of signal coords with gate < 0.5
    false_pass_noise: float = 0.0       # fraction of noise coords with gate > 0.5
    ema_exact_corr: float = float("nan")
    signal_var_ratio: float = float("nan")  # median s_ema / s_exact on signal coords
    probe_count: int = 0                     # exact/microbatch probes used (cost proxy)
    history_test: list = field(default_factory=list)


def make_targets(cfg: Config, seed: int):
    """Return (current_target(step) -> weight vector, signal_mask) for the task."""
    if cfg.task == "drift":
        idx, base, phase = make_signal_spec(cfg, seed)
        fn = lambda step: drifting_weights(step, cfg, idx, base, phase)
        support = idx
    elif cfg.task == "switch":
        w_a, idx_a = make_true_weights(cfg.d, cfg.k, cfg.signal_magnitude, seed=seed)
        w_b, idx_b = make_true_weights(cfg.d, cfg.k, cfg.signal_magnitude, seed=seed + 777)
        fn = lambda step: (w_a if step < cfg.switch_step else w_b)
        support = torch.unique(torch.cat([idx_a, idx_b]))  # union of A/B supports
    else:
        w_true, idx = make_true_weights(cfg.d, cfg.k, cfg.signal_magnitude, seed=seed)
        fn = lambda step: w_true
        support = idx
    signal_mask = torch.zeros(cfg.d, dtype=torch.bool)
    signal_mask[support] = True
    return fn, signal_mask


# ---------------------------------------------------------------------------
# Single run for a given variance "mode"
# ---------------------------------------------------------------------------

def _final_s_for_mode(mode, model, opt, rep_batch, exact_est, cfg, probe_interval):
    """Variance estimate the mode actually applies to the gate, for a fresh batch.

    This makes the gate-quality panel reflect each method's own estimator rather than
    always reading the internal EMA.
    """
    state = opt.state[model.weight]
    if mode == "ema":
        return state["exp_grad_var"].squeeze() / (1 - cfg.rho ** state["step"])
    if mode in ("exact", "exact_k", "triggered"):
        return exact_est.estimate(model, loss_one_sample(model), rep_batch)[model.weight].squeeze()
    if mode.startswith("micro"):
        k = int(mode.split("_")[1])
        _, gv = backward_with_microbatch_variance(model, microbatch_loss, rep_batch, num_splits=k)
        opt.zero_grad(set_to_none=True)
        return gv[model.weight].squeeze()
    raise ValueError(mode)


def _gate_delta(model, opt, cfg, alpha_val, s_exact):
    """Mean |q_exact - q_ema| on the weight, using current m_hat and EMA s_hat."""
    st = opt.state.get(model.weight)
    if not st or st.get("step", 0) < 1:
        return 0.0
    t = st["step"]
    m_hat = st["exp_avg"].squeeze() / (1 - cfg.beta1 ** t)
    s_ema = st["exp_grad_var"].squeeze() / (1 - cfg.rho ** t)
    q_ema = compute_gate(m_hat, s_ema, gate=cfg.gate, alpha=alpha_val, lambda_pop=cfg.lambda_pop)
    q_ex = compute_gate(m_hat, s_exact, gate=cfg.gate, alpha=alpha_val, lambda_pop=cfg.lambda_pop)
    return float((q_ex - q_ema).abs().mean())


def run_one(mode: str, cfg: Config, seed: int, probe_interval: int = 10) -> RunResult:
    nonstationary = cfg.task in ("drift", "switch")
    current_target, signal_mask = make_targets(cfg, seed)

    train_gen = torch.Generator().manual_seed(seed)
    X_train = torch.randn(cfg.n_train, cfg.d, generator=train_gen)
    X_test = torch.randn(cfg.test_size, cfg.d, generator=torch.Generator().manual_seed(9999))
    label_gen = torch.Generator().manual_seed(seed + 5)
    if not nonstationary:
        w_true = current_target(0)
        y_train = (X_train @ w_true).unsqueeze(1) + torch.randn(cfg.n_train, 1, generator=train_gen) * cfg.sigma_noise
        y_test = (X_test @ w_true).unsqueeze(1) + torch.randn(cfg.test_size, 1, generator=torch.Generator().manual_seed(8888)) * cfg.sigma_noise

    def test_loss(step):
        with torch.no_grad():
            if nonstationary:  # noiseless instantaneous target -> tracking error
                yt = (X_test @ current_target(step)).unsqueeze(1)
            else:
                yt = y_test
            return ((model(X_test) - yt) ** 2).mean().item()

    torch.manual_seed(seed + 1000)
    model = nn.Linear(cfg.d, 1, bias=False)
    nn.init.zeros_(model.weight)
    if nonstationary:
        opt = SNRAdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, gate=cfg.gate,
            rho=cfg.rho, betas=(cfg.beta1, 0.999), alpha="online",
            lambda_pop=cfg.lambda_pop, track_stats=True,
        )
        alpha_val = resolve_alpha("online")
    else:
        opt = SNRAdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, gate=cfg.gate,
            rho=cfg.rho, betas=(cfg.beta1, 0.999), alpha="finite",
            batch_size=cfg.batch_size, dataset_size=cfg.n_train, lambda_pop=cfg.lambda_pop,
            track_stats=True,
        )
        alpha_val = resolve_alpha("finite", batch_size=cfg.batch_size, dataset_size=cfg.n_train)
    exact_est = ExactVarianceEstimator(exclude_norm=False)

    result = RunResult()
    probe_count = 0
    use_exact_until = -1
    gd_baseline = None  # running baseline of gate divergence for the triggered controller
    t0 = time.perf_counter()
    last_loss = 0.0
    for step in range(cfg.n_steps):
        idx = torch.randint(cfg.n_train, (cfg.batch_size,))
        X_b = X_train[idx]
        if nonstationary:
            y_b = X_b @ current_target(step) + torch.randn(cfg.batch_size, generator=label_gen) * cfg.sigma_noise
        else:
            y_b = y_train[idx].squeeze(1)
        batch = (X_b, y_b)

        opt.zero_grad(set_to_none=True)
        grad_variances = None

        if mode.startswith("micro"):
            k = int(mode.split("_")[1])
            last_loss, grad_variances = backward_with_microbatch_variance(
                model, microbatch_loss, batch, num_splits=k
            )
            probe_count += 1
        else:
            loss = ((model(X_b).squeeze(1) - y_b) ** 2).mean()
            loss.backward()
            last_loss = loss.item()
            if mode == "exact":
                grad_variances = exact_est.estimate(model, loss_one_sample(model), batch)
                probe_count += 1
            elif mode == "exact_k" and step % probe_interval == 0:
                grad_variances = exact_est.estimate(model, loss_one_sample(model), batch)
                probe_count += 1
            elif mode == "triggered":
                # Adaptive freshness: probe to detect staleness, correct for a short window.
                correcting = step < use_exact_until
                detecting = step % cfg.probe_every == 0
                if correcting or detecting:
                    s_dict = exact_est.estimate(model, loss_one_sample(model), batch)
                    probe_count += 1
                    if correcting:
                        grad_variances = s_dict
                    else:  # detection probe: spike of divergence above its running baseline?
                        gd = _gate_delta(model, opt, cfg, alpha_val,
                                         s_dict[model.weight].squeeze().detach())
                        if gd_baseline is None:
                            gd_baseline = gd
                        elif gd > cfg.staleness_spike * gd_baseline + cfg.staleness_floor:
                            use_exact_until = step + cfg.correction_steps
                            grad_variances = s_dict
                        gd_baseline = cfg.staleness_beta * gd_baseline + (1 - cfg.staleness_beta) * gd
            elif mode not in ("ema", "exact", "exact_k", "triggered"):
                raise ValueError(f"Unknown mode: {mode}")

        opt.step(grad_variances=grad_variances)

        if step % EVAL_EVERY == 0:
            result.history_test.append(test_loss(step))

    result.wall_clock = time.perf_counter() - t0
    result.final_train = last_loss
    result.probe_count = probe_count
    if cfg.task == "drift":
        # Average tracking error over the last ~2 drift periods (more stable than one phase).
        w = max(1, (2 * cfg.drift_period) // EVAL_EVERY)
        tail = result.history_test[-w:]
        result.final_test = sum(tail) / len(tail)
    else:
        result.final_test = test_loss(cfg.n_steps - 1)

    # Representative fresh batch for the final-gate diagnostics.
    ridx = torch.randint(cfg.n_train, (cfg.batch_size,))
    Xr = X_train[ridx]
    yr = (Xr @ current_target(cfg.n_steps - 1)
          + torch.randn(cfg.batch_size, generator=label_gen) * cfg.sigma_noise)
    rep_batch = (Xr, yr)

    state = opt.state[model.weight]
    step_num = state["step"]
    m_hat = state["exp_avg"].squeeze() / (1 - cfg.beta1 ** step_num)
    # Gate quality from the s this mode actually applies.
    s_used = _final_s_for_mode(mode, model, opt, rep_batch, exact_est, cfg, probe_interval).detach()
    gate_vals = compute_gate(m_hat, s_used, gate=cfg.gate, alpha=alpha_val, lambda_pop=cfg.lambda_pop)

    result.mean_signal_gate = float(gate_vals[signal_mask].mean())
    result.mean_noise_gate = float(gate_vals[~signal_mask].mean())
    result.gate_auc = gate_auc(gate_vals, signal_mask)
    result.false_suppress_signal = float((gate_vals[signal_mask] < 0.5).float().mean())
    result.false_pass_noise = float((gate_vals[~signal_mask] > 0.5).float().mean())

    # Mechanism metrics: how the internal EMA compares to a fresh exact estimate.
    s_ema = (state["exp_grad_var"].squeeze() / (1 - cfg.rho ** step_num)).detach()
    s_exact = exact_est.estimate(model, loss_one_sample(model), rep_batch)[model.weight].squeeze().detach()
    log_ema = torch.log(s_ema.clamp_min(1e-12))
    log_exact = torch.log(s_exact.clamp_min(1e-12))
    if torch.std(log_ema) > 0 and torch.std(log_exact) > 0:
        result.ema_exact_corr = float(torch.corrcoef(torch.stack([log_ema, log_exact]))[0, 1])
    result.signal_var_ratio = float(
        (s_ema[signal_mask] / s_exact[signal_mask].clamp_min(1e-12)).median()
    )

    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

MODES = [
    ("ema", "EMA-only"),
    ("exact", "exact/step"),
    ("exact_k", "exact/10"),
    ("micro_2", "microbatch K=2"),
    ("micro_4", "microbatch K=4"),
    ("triggered", "triggered"),
]

EVAL_EVERY = 10


def aggregate(results: list[RunResult]) -> dict:
    def mean(attr):
        vals = [getattr(r, attr) for r in results]
        vals = [v for v in vals if v == v]  # drop nan
        return sum(vals) / len(vals) if vals else float("nan")
    return {
        "train": mean("final_train"),
        "test": mean("final_test"),
        "wall": mean("wall_clock"),
        "sig_gate": mean("mean_signal_gate"),
        "noise_gate": mean("mean_noise_gate"),
        "auc": mean("gate_auc"),
        "false_suppress": mean("false_suppress_signal"),
        "false_pass": mean("false_pass_noise"),
        "corr": mean("ema_exact_corr"),
        "var_ratio": mean("signal_var_ratio"),
        "probes": mean("probe_count"),
    }


COLORS = {
    "EMA-only": "tab:gray",
    "exact/step": "tab:blue",
    "exact/10": "tab:cyan",
    "microbatch K=2": "tab:orange",
    "microbatch K=4": "tab:red",
    "triggered": "tab:purple",
}


def make_figures(all_runs: dict, summary: dict, cfg: Config, out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    labels = [label for _, label in MODES]
    steps = list(range(0, cfg.n_steps, EVAL_EVERY))
    nonstationary = cfg.task in ("drift", "switch")
    irreducible = 0.0 if nonstationary else cfg.sigma_noise ** 2
    drift = cfg.task == "drift"
    regime = {
        "drift": f"non-stationary drifting target (period={cfg.drift_period})",
        "switch": f"abrupt task switch at step {cfg.switch_step}",
        "stationary": "stationary target",
    }[cfg.task]
    n_eval = min(len(steps), min(len(all_runs[l][0].history_test) for l in labels))
    steps = steps[:n_eval]

    # ---- Figure 1: test-loss curves ----
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    fig.suptitle(
        f"Variance backends for the SNRAdamW gate: test loss\n"
        f"{regime}",
        fontsize=12, fontweight="bold",
    )
    ax.set_title(
        f"d={cfg.d}, k={cfg.k}, n={cfg.n_train}, noise={cfg.sigma_noise}, "
        f"lambda_pop={cfg.lambda_pop}, {cfg.n_seeds} seeds",
        fontsize=9,
    )
    # Under drift the instantaneous tracking error oscillates with the target; all variants
    # share that phase, so plotting each variant relative to EMA-only (per seed, per step)
    # cancels the oscillation and isolates the gate's effect.
    def smooth(t, w):
        if w <= 1 or t.shape[-1] < w:
            return t
        kernel = torch.ones(1, 1, w) / w
        pad = w // 2
        x = t.unsqueeze(1)
        x = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
        return torch.nn.functional.conv1d(x, kernel).squeeze(1)[:, : t.shape[-1]]

    # Absolute tracking error varies in scale across seeds (and oscillates under drift),
    # so for the non-stationary regimes plot each variant relative to EMA-only per seed and
    # per step (<1 = better). Under drift we smooth the absolute error over one full period
    # FIRST (a boxcar of one period nulls the oscillation) before taking the ratio.
    relative = nonstationary
    win = max(1, cfg.drift_period // EVAL_EVERY) if drift else (3 if cfg.task == "switch" else 1)
    ema_smooth = smooth(torch.tensor([r.history_test for r in all_runs["EMA-only"]])[:, :n_eval], win)
    for label in labels:
        hist = smooth(torch.tensor([r.history_test for r in all_runs[label]])[:, :n_eval], win)
        series = (hist / ema_smooth.clamp_min(1e-9)) if relative else hist
        mean = series.mean(dim=0)
        std = series.std(dim=0)
        ax.plot(steps, mean, label=label, color=COLORS[label], linewidth=1.6)
        ax.fill_between(steps, (mean - std).numpy(), (mean + std).numpy(),
                        color=COLORS[label], alpha=0.12)
    if relative:
        ax.axhline(1.0, ls="--", color="black", alpha=0.5, label="EMA-only baseline")
        ax.set_ylabel("Test MSE relative to EMA-only (<1 is better)")
        if drift:
            ax.set_ylim(0.5, 1.25)
        if cfg.task == "switch":
            ax.set_ylim(0.6, 1.4)
            ax.axvline(cfg.switch_step, ls="--", color="black", alpha=0.5, label="task switch")
    else:
        if irreducible > 0:
            ax.axhline(irreducible, ls="--", color="black", alpha=0.5,
                       label=f"irreducible ({irreducible:.0f})")
        ax.set_ylabel("Test MSE")
    ax.set_xlabel("Step")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path1 = os.path.join(out_dir, f"benchmark_variance_{tag}_curves.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)

    # ---- Figure 2: gate-quality + cost summary ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle(f"Variance backends: gate quality and cost  --  {regime}",
                 fontsize=12, fontweight="bold")
    x = range(len(labels))
    colors = [COLORS[l] for l in labels]

    # (a) signal vs noise gate.
    ax = axes[0]
    width = 0.38
    sig = [summary[l]["sig_gate"] for l in labels]
    noise = [summary[l]["noise_gate"] for l in labels]
    ax.bar([i - width / 2 for i in x], sig, width, label="signal coords", color="tab:green")
    ax.bar([i + width / 2 for i in x], noise, width, label="noise coords", color="tab:red")
    ax.set_ylabel("Mean gate")
    ax.set_title("(a) Gate on signal vs noise")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (b) AUC (signal-vs-noise ranking).
    ax = axes[1]
    auc = [summary[l]["auc"] for l in labels]
    ax.bar(list(x), auc, color=colors)
    ax.axhline(0.5, ls="--", color="gray", alpha=0.6, label="chance (0.5)")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Signal-vs-noise AUC")
    ax.set_title("(b) Gate ranking quality")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (c) probe budget (cost proxy: number of exact/microbatch estimates per run).
    ax = axes[2]
    probes = [summary[l]["probes"] for l in labels]
    ax.bar(list(x), probes, color=colors)
    ax.axhline(cfg.n_steps, ls="--", color="gray", alpha=0.6, label=f"every step ({cfg.n_steps})")
    ax.set_ylabel("Probes per run")
    ax.set_title("(c) Cost: variance probes")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path2 = os.path.join(out_dir, f"benchmark_variance_{tag}_summary.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)

    return [path1, path2]


# ---------------------------------------------------------------------------
# Staleness detector: "when is the EMA stale?"
# ---------------------------------------------------------------------------

# Shared hyperparameters so the three regimes differ only by their task.
STALENESS_BASE = dict(
    sigma_noise=0.5, n_train=200, n_steps=600, lr=1e-2, beta1=0.95,
    lambda_pop=2.0, drift_period=40, switch_step=300,
)


def staleness_trace(cfg: Config, seed: int, probe_every: int = 10):
    """Train EMA-only and, every `probe_every` steps, compare the internal EMA variance
    to a fresh exact probe.

    Returns (steps, d_t, gate_delta) where
        d_t        = median |log s_exact - log s_ema| over the weight coords,
        gate_delta = mean |q_exact - q_ema|.
    These spike when the recent-history EMA no longer describes the current batch.
    """
    nonstationary = cfg.task in ("drift", "switch")
    current_target, _ = make_targets(cfg, seed)
    train_gen = torch.Generator().manual_seed(seed)
    X_train = torch.randn(cfg.n_train, cfg.d, generator=train_gen)
    label_gen = torch.Generator().manual_seed(seed + 5)
    if not nonstationary:
        w_true = current_target(0)
        y_train = (X_train @ w_true) + torch.randn(cfg.n_train, generator=train_gen) * cfg.sigma_noise

    torch.manual_seed(seed + 1000)
    model = nn.Linear(cfg.d, 1, bias=False)
    nn.init.zeros_(model.weight)
    opt = SNRAdamW(
        model.parameters(), lr=cfg.lr, gate=cfg.gate, rho=cfg.rho, betas=(cfg.beta1, 0.999),
        alpha="online" if nonstationary else "finite",
        batch_size=None if nonstationary else cfg.batch_size,
        dataset_size=None if nonstationary else cfg.n_train,
        lambda_pop=cfg.lambda_pop, track_stats=True,
    )
    alpha_val = (resolve_alpha("online") if nonstationary
                 else resolve_alpha("finite", batch_size=cfg.batch_size, dataset_size=cfg.n_train))
    exact_est = ExactVarianceEstimator(exclude_norm=False)

    steps, d_list, gd_list = [], [], []
    for step in range(cfg.n_steps):
        idx = torch.randint(cfg.n_train, (cfg.batch_size,))
        X_b = X_train[idx]
        if nonstationary:
            y_b = X_b @ current_target(step) + torch.randn(cfg.batch_size, generator=label_gen) * cfg.sigma_noise
        else:
            y_b = y_train[idx]
        batch = (X_b, y_b)

        opt.zero_grad(set_to_none=True)
        ((model(X_b).squeeze(1) - y_b) ** 2).mean().backward()

        st = opt.state.get(model.weight)
        if step % probe_every == 0 and st is not None and st.get("step", 0) >= 1:
            t = st["step"]
            s_ema = (st["exp_grad_var"].squeeze() / (1 - cfg.rho ** t)).detach()
            m_hat = (st["exp_avg"].squeeze() / (1 - cfg.beta1 ** t)).detach()
            s_exact = exact_est.estimate(model, loss_one_sample(model), batch)[model.weight].squeeze().detach()
            d_t = float((torch.log(s_exact.clamp_min(1e-12)) - torch.log(s_ema.clamp_min(1e-12))).abs().median())
            q_ema = compute_gate(m_hat, s_ema, gate=cfg.gate, alpha=alpha_val, lambda_pop=cfg.lambda_pop)
            q_ex = compute_gate(m_hat, s_exact, gate=cfg.gate, alpha=alpha_val, lambda_pop=cfg.lambda_pop)
            steps.append(step)
            d_list.append(d_t)
            gd_list.append(float((q_ex - q_ema).abs().mean()))

        opt.step()  # EMA-only

    return steps, d_list, gd_list


def make_staleness_figure(out_dir: str, n_seeds: int = 5, n_steps: int | None = None):
    """One figure: the staleness signals across stationary / switch / drift regimes."""
    os.makedirs(out_dir, exist_ok=True)

    def smooth(v, w=5):
        if v.shape[-1] < w:
            return v
        kernel = torch.ones(1, 1, w) / w
        x = torch.nn.functional.pad(v.view(1, 1, -1), (w // 2, w // 2), mode="replicate")
        return torch.nn.functional.conv1d(x, kernel).view(-1)[: v.shape[-1]]

    regimes = [
        ("stationary", "stationary", "tab:gray"),
        ("switch", "abrupt switch", "tab:red"),
        ("drift", "drift", "tab:blue"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle(
        "Staleness detector: when does the recent-history EMA stop describing the current batch?",
        fontsize=12, fontweight="bold",
    )
    switch_step = None
    for task, label, color in regimes:
        cfg = Config(task=task, **STALENESS_BASE)
        if n_steps is not None:
            cfg.n_steps = n_steps
        if task == "switch":
            switch_step = cfg.switch_step
        traces = [staleness_trace(cfg, seed) for seed in range(n_seeds)]
        steps = traces[0][0]
        gd = smooth(torch.tensor([t[2] for t in traces]).mean(dim=0))
        d = smooth(torch.tensor([t[1] for t in traces]).mean(dim=0))
        axes[0].plot(steps, gd, label=label, color=color, linewidth=1.8)
        axes[1].plot(steps, d, label=label, color=color, linewidth=1.8)

    for ax, title, ylab in [
        (axes[0], "(a) Gate impact  |q_exact - q_ema|", "Mean gate delta"),
        (axes[1], "(b) Log-variance gap  |log s_exact - log s_ema|", "Median log-variance gap"),
    ]:
        if switch_step is not None:
            ax.axvline(switch_step, ls="--", color="black", alpha=0.5, label="task switch")
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out_dir, "benchmark_variance_staleness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


PRESETS = {"drift": DRIFT_PRESET, "switch": SWITCH_PRESET}
REGIME_NAMES = {
    "stationary": "stationary target",
    "drift": "non-stationary drifting target",
    "switch": "abrupt task switch",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["stationary", "drift", "switch", "all"], default="all",
                        help="Which regime(s) to run.")
    parser.add_argument("--quick", action="store_true", help="Fewer steps/seeds for a smoke run.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--out-dir", default="benchmarks", help="Directory for output PNGs.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    parser.add_argument("--no-staleness", action="store_true", help="Skip the staleness figure.")
    args = parser.parse_args()

    tasks = ["stationary", "switch", "drift"] if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, args)

    if args.task == "all" and not args.no_figures and not args.no_staleness:
        n_seeds = 2 if args.quick else 3
        n_steps = 200 if args.quick else None
        print("\n=== staleness detector across regimes ===")
        path = make_staleness_figure(args.out_dir, n_seeds=n_seeds, n_steps=n_steps)
        print(f"Saved figure:\n  {path}")


def run_task(task: str, args):
    cfg = Config(task=task)
    for k, v in PRESETS.get(task, {}).items():
        setattr(cfg, k, v)
    if args.quick:
        cfg.n_steps = min(cfg.n_steps, 300)
        cfg.n_seeds = 2
    if args.steps is not None:
        cfg.n_steps = args.steps
    if args.seeds is not None:
        cfg.n_seeds = args.seeds

    print(
        f"\n=== {REGIME_NAMES[task]} ===\n"
        f"d={cfg.d}, k={cfg.k}, n_train={cfg.n_train}, batch={cfg.batch_size}, "
        f"sigma={cfg.sigma_noise}, lr={cfg.lr}, beta1={cfg.beta1}, lambda_pop={cfg.lambda_pop}, "
        f"steps={cfg.n_steps}, seeds={cfg.n_seeds}\n"
    )

    summary, all_runs = {}, {}
    for mode, label in MODES:
        runs = [run_one(mode, cfg, seed) for seed in range(cfg.n_seeds)]
        all_runs[label] = runs
        summary[label] = aggregate(runs)

    ema_test = summary["EMA-only"]["test"]
    header = (
        f"{'variant':<16} {'test':>8} {'vs EMA':>8} {'probes':>7} {'wall(s)':>8} "
        f"{'sig_gate':>9} {'auc':>6} {'s_ema/s_ex':>11}"
    )
    print(header)
    print("-" * len(header))
    for _, label in MODES:
        a = summary[label]
        gain = 100.0 * (ema_test - a["test"]) / ema_test if ema_test else float("nan")
        print(
            f"{label:<16} {a['test']:>8.3f} {gain:>7.1f}% {a['probes']:>7.0f} {a['wall']:>8.1f} "
            f"{a['sig_gate']:>9.3f} {a['auc']:>6.3f} {a['var_ratio']:>11.2f}"
        )

    print(
        "\nNotes:"
        "\n  - vs EMA: test-loss improvement over the EMA-only baseline (higher is better)."
        "\n  - probes: exact/microbatch variance estimates used per run (cost proxy);"
        "\n    'triggered' uses exact only when its staleness detector fires."
        "\n  - sig_gate: mean gate this method applies to signal coords."
        "\n  - auc: ranking of signal vs noise coords by gate value (1.0 = perfect)."
        "\n  - s_ema/s_ex: median ratio of the internal EMA variance to a fresh exact"
        "\n    estimate on signal coords (>1 means the EMA over-estimates signal variance)."
    )

    if not args.no_figures:
        paths = make_figures(all_runs, summary, cfg, args.out_dir, tag=task)
        print("Saved figures:")
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
