"""
Adaptive thresholding benchmark for sparse regression under regime shifts.

This benchmark is built around a hard-won lesson. On a genuinely sparse problem
(here d=200 with only k=5 true signal coordinates, a 2.5% active fraction), a *low*
mean gate is not a pathology -- it is correct, because most coordinates should be
suppressed. A quota-based "make p fraction active" target forces some coordinates
through even when none deserve it, which admits noise and hurts generalization.

So the headline question is no longer "how many coordinates are active?" but:

    Does the gate allocate *update mass* to signal coordinates more efficiently
    than to noise coordinates?

Two ideas address the quota problem (both controllable from AdaptiveThresholdConfig):

  * Capped active fraction + absolute SNR floor (`min_snr_threshold`): allow *at
    most* p fraction active, but never lower the boundary below an absolute SNR
    floor, so the controller can keep almost nothing active when nothing clears it.
  * Conditional shock (`shock_then_sparsify`): stay sparse, and on a detected
    regime shift recalibrate faster but only *open the budget* if the top
    coordinates actually separate from the bulk.

Task: a nonstationary schedule (implementation plan, section 14):

    steps    0 - 999 : signal coords A, noise sigma = 1
    steps 1000 - 1999 : signal coords A, noise sigma = 5   (noise jumps)
    steps 2000 - 2999 : signal coords B, noise sigma = 2   (signal support moves)

Because the true signal coordinates are known, the benchmark measures update-mass
allocation and top-k precision, not just binary active fraction.

Figures:
  1. benchmark_adaptive_threshold.png       -- diagnostic curves
  2. benchmark_adaptive_threshold_rdist.png -- distribution of r=m^2/s, signal vs noise
  3. benchmark_adaptive_threshold_sweep.png -- target_active_fraction x q0 and
                                               target_active_fraction x min_snr_threshold

Run:
    uv run python benchmark_adaptive_threshold.py            # all figures
    uv run python benchmark_adaptive_threshold.py --no-sweep # curves + r-dist only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_grad import SNRAdamW, AdaptiveThresholdConfig, compute_gate, resolve_alpha


# Reference cutoff for the binary signal/noise diagnostics, shared across methods.
DIAG_Q0 = 0.5


@dataclass
class TaskConfig:
    d: int = 200            # input dimension
    k: int = 5              # active signal coordinates (true frac = k/d = 2.5%)
    n_test: int = 4000
    batch_size: int = 64
    signal_magnitude: float = 1.0
    regimes: list = field(default_factory=lambda: [
        (1000, 1.0, "A"),
        (2000, 5.0, "A"),
        (3000, 2.0, "B"),
    ])

    @property
    def true_active_fraction(self) -> float:
        return self.k / self.d


def _make_signal(d, k, magnitude, generator, exclude=None):
    excl = set() if exclude is None else set(exclude.tolist())
    candidates = [i for i in range(d) if i not in excl]
    perm = torch.randperm(len(candidates), generator=generator)
    idx = torch.tensor([candidates[j] for j in perm[:k].tolist()])
    w = torch.zeros(d)
    signs = torch.randint(0, 2, (k,), generator=generator).float() * 2 - 1
    w[idx] = signs * magnitude
    return w, idx


def _sample_batch(w_true, sigma, n, generator):
    d = w_true.numel()
    X = torch.randn(n, d, generator=generator)
    y = X @ w_true + sigma * torch.randn(n, generator=generator)
    return X, y


def _regime_for_step(step, cfg):
    for until, sigma, sig in cfg.regimes:
        if step < until:
            return sigma, sig
    return cfg.regimes[-1][1], cfg.regimes[-1][2]


def _moments(opt, p):
    """Return (m_hat, s_hat, group) reconstructed from optimizer state, or None."""
    group = next((g for g in opt.param_groups if p in g["params"]), None)
    st = opt.state.get(p)
    if group is None or st is None or "exp_grad_var" not in st:
        return None
    b1, _ = group["betas"]
    rho = group["rho"]
    t = st["step"]
    m_hat = st["exp_avg"] / (1.0 - b1 ** t)
    s_hat = st["exp_grad_var"] / (1.0 - rho ** t)
    return m_hat, s_hat, group


def _gate_and_r(opt, p):
    """Reconstruct per-coordinate (q, r=m^2/s) from optimizer state, or (None, None)."""
    res = _moments(opt, p)
    if res is None:
        return None, None
    m_hat, s_hat, group = res
    alpha = resolve_alpha(group["alpha"])
    q = compute_gate(m_hat, s_hat, gate=group["gate"], alpha=alpha,
                     lambda_pop=group["lambda_pop"], gate_eps=group["gate_eps"])
    r = m_hat.square() / (s_hat + group["gate_eps"])
    return q.detach().reshape(-1), r.detach().reshape(-1)


def run_one(opt_factory, cfg, n_steps, seed, log_every=20, snapshot_steps=()):
    """Train one optimizer over the nonstationary schedule, logging diagnostics."""
    gen = torch.Generator().manual_seed(seed)
    w_A, idx_A = _make_signal(cfg.d, cfg.k, cfg.signal_magnitude, gen)
    w_B, idx_B = _make_signal(cfg.d, cfg.k, cfg.signal_magnitude, gen, exclude=idx_A)
    signals = {"A": (w_A, idx_A), "B": (w_B, idx_B)}

    model = torch.nn.Linear(cfg.d, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    p = model.weight
    opt = opt_factory(model.parameters())

    log = {k: [] for k in (
        "step", "excess_test", "lambda_pop", "target_af",
        "signal_gate", "noise_gate", "snr_ratio",
        "false_passthrough", "false_suppression",
        "mass_ratio", "precision_at_k",
    )}
    cum_signal_mass = 0.0
    cum_noise_mass = 0.0
    snapshots = {}

    for step in range(n_steps):
        sigma, sig = _regime_for_step(step, cfg)
        w_true, idx = signals[sig]
        mask = torch.zeros(cfg.d, dtype=torch.bool)
        mask[idx] = True

        X, y = _sample_batch(w_true, sigma, cfg.batch_size, gen)
        loss = ((model(X).squeeze(-1) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()

        w_before = p.detach().clone()
        opt.step()
        delta = (p.detach() - w_before).abs().reshape(-1)
        cum_signal_mass += delta[mask].sum().item()
        cum_noise_mass += delta[~mask].sum().item()

        if step % log_every == 0:
            with torch.no_grad():
                Xt, yt = _sample_batch(w_true, sigma, cfg.n_test, gen)
                test_loss = ((model(Xt).squeeze(-1) - yt) ** 2).mean().item()
            log["step"].append(step)
            log["excess_test"].append(max(test_loss - sigma ** 2, 1e-6))
            log["mass_ratio"].append(cum_signal_mass / max(cum_noise_mass, 1e-12))

            ts = opt.get_threshold_state() if hasattr(opt, "get_threshold_state") else {}
            g0 = ts.get("group_0") if ts else None
            if g0 is not None:
                log["lambda_pop"].append(g0["lambda_pop"])
                log["target_af"].append(g0.get("target_active_fraction"))
            elif "lambda_pop" in opt.param_groups[0]:
                log["lambda_pop"].append(opt.param_groups[0]["lambda_pop"])
                log["target_af"].append(None)
            else:
                log["lambda_pop"].append(float("nan"))
                log["target_af"].append(None)

            q, r = _gate_and_r(opt, p)
            if q is None:
                for key in ("signal_gate", "noise_gate", "snr_ratio",
                            "false_passthrough", "false_suppression", "precision_at_k"):
                    log[key].append(float("nan"))
            else:
                sig_q, noi_q = q[mask], q[~mask]
                sg, ng = sig_q.mean().item(), noi_q.mean().item()
                log["signal_gate"].append(sg)
                log["noise_gate"].append(ng)
                log["snr_ratio"].append(sg / max(ng, 1e-9))
                log["false_passthrough"].append((noi_q >= DIAG_Q0).float().mean().item())
                log["false_suppression"].append((sig_q < DIAG_Q0).float().mean().item())
                topk = torch.topk(r, cfg.k).indices
                log["precision_at_k"].append(mask[topk].float().mean().item())
                if step in snapshot_steps:
                    snapshots[step] = (r.clone(), mask.clone())

    return log, snapshots


def _mean_after(log, key, min_step):
    vals = [v for s, v in zip(log["step"], log[key]) if s >= min_step and v == v]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def build_methods(cfg, lr):
    common = dict(lr=lr, gate="snr", rho=0.99, alpha="online", track_stats=True)

    def af(target, q0=0.5, floor=0.0):
        return lambda params: SNRAdamW(
            params, lambda_pop=1.0, **common,
            adaptive_threshold=AdaptiveThresholdConfig(
                mode="target_active_fraction", target_active_fraction=target,
                active_gate_threshold=q0, min_snr_threshold=floor,
                warmup_steps=100, update_interval=25,
            ),
        )

    return {
        "AdamW": (lambda params: torch.optim.AdamW(params, lr=lr), "gray"),
        "SNR static": (lambda params: SNRAdamW(params, lambda_pop=1.0, **common), "C0"),
        "AF=0.20 (too high)": (af(0.20), "C3"),
        "AF=0.025 (true)": (af(0.025, q0=0.25), "C2"),
        "capped (floor=2)": (af(0.05, q0=0.25, floor=2.0), "C5"),
        "shock_then_sparsify": (
            lambda params: SNRAdamW(
                params, lambda_pop=1.0, **common,
                adaptive_threshold=AdaptiveThresholdConfig(
                    mode="shock_then_sparsify",
                    sparse_target_active_fraction=0.025,
                    shock_target_active_fraction=0.2,
                    shock_steps=20, shift_detect_threshold=0.04,
                    shock_separation_threshold=2.0,
                    active_gate_threshold=0.25, warmup_steps=100, update_interval=10,
                ),
            ),
            "C4",
        ),
    }


# ---------------------------------------------------------------------------
# Main diagnostic comparison
# ---------------------------------------------------------------------------

def run_main(cfg, n_steps, seed, out, rdist_out):
    lr = 5e-2
    methods = build_methods(cfg, lr)
    snap_steps = tuple(min(b - cfg.k, n_steps - n_steps % 20) for b in
                       [r[0] for r in cfg.regimes])  # near each regime end
    snap_steps = tuple((s // 20) * 20 for s in snap_steps)

    logs, snaps = {}, {}
    for name, (factory, _) in methods.items():
        print(f"Running {name} ...")
        logs[name], snaps[name] = run_one(factory, cfg, n_steps, seed, snapshot_steps=snap_steps)

    shift_steps = [until for until, _, _ in cfg.regimes[:-1]]

    print("\n==== Summary (means over steps >= 200) ====")
    print(f"{'method':<22}{'excess test':>12}{'sig gate':>10}{'noise gate':>11}"
          f"{'mass ratio':>11}{'prec@k':>8}")
    for name, log in logs.items():
        print(f"{name:<22}{_mean_after(log,'excess_test',200):>12.4f}"
              f"{_mean_after(log,'signal_gate',200):>10.4f}"
              f"{_mean_after(log,'noise_gate',200):>11.4f}"
              f"{_mean_after(log,'mass_ratio',200):>11.2f}"
              f"{_mean_after(log,'precision_at_k',200):>8.2f}")

    panels = [
        ("excess_test", "excess test loss", True),
        ("lambda_pop", "lambda_pop", True),
        ("mass_ratio", "cumulative signal/noise update-mass ratio", True),
        ("precision_at_k", "precision@k (top-k r are true signal)", False),
        ("signal_gate", "mean gate (signal coords)", False),
        ("noise_gate", "mean gate (noise coords)", False),
        ("snr_ratio", "signal / noise gate ratio", True),
        ("target_af", "effective active-fraction target", False),
        ("false_passthrough", "false pass-through (noise active)", False),
        ("false_suppression", "false suppression (signal inactive)", False),
    ]
    fig, axes = plt.subplots(5, 2, figsize=(13, 20))
    axes = axes.ravel()
    for ax, (key, ylabel, logy) in zip(axes, panels):
        for name, (_, color) in methods.items():
            ys = logs[name][key]
            if all(v != v for v in ys):
                continue
            ax.plot(logs[name]["step"], ys, label=name, color=color, lw=1.4)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        for s in shift_steps:
            ax.axvline(s, color="k", ls="--", alpha=0.3)
        ax.grid(True, alpha=0.3)
    axes[7].axhline(cfg.true_active_fraction, ls=":", color="k", alpha=0.6)
    axes[2].axhline(1.0, ls=":", color="k", alpha=0.6)
    axes[0].set_title("Adaptive SNR thresholding on sparse regression under regime shifts")
    axes[0].legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("step")
    axes[-2].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nSaved curves to {out}")

    _plot_rdist(cfg, snaps, snap_steps, rdist_out)


def _plot_rdist(cfg, snaps, snap_steps, out):
    shown = ["SNR static", "AF=0.20 (too high)", "AF=0.025 (true)", "capped (floor=2)"]
    shown = [m for m in shown if any(snaps.get(m, {}))]
    if not shown:
        return
    fig, axes = plt.subplots(len(snap_steps), len(shown),
                             figsize=(3.4 * len(shown), 3.0 * len(snap_steps)),
                             squeeze=False)
    for i, step in enumerate(snap_steps):
        for j, name in enumerate(shown):
            ax = axes[i][j]
            snap = snaps.get(name, {}).get(step)
            if snap is None:
                ax.axis("off")
                continue
            r, mask = snap
            logr = torch.log10(r.clamp_min(1e-12)).numpy()
            ax.hist(logr[(~mask).numpy()], bins=40, color="gray", alpha=0.6,
                    density=True, label="noise")
            ax.hist(logr[mask.numpy()], bins=10, color="C1", alpha=0.7,
                    density=True, label="signal")
            if i == 0:
                ax.set_title(name, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"step {step}\ndensity", fontsize=9)
            ax.set_xlabel("log10 r = log10(m^2/s)", fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Distribution of r = m^2/s, signal vs noise coordinates "
                 "(separation = exploitable signal)", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved r-distributions to {out}")


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def _heatmap(ax, grid, row_vals, col_vals, row_label, col_label, title):
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto", origin="lower")
    ax.set_xticks(range(len(col_vals)), [str(c) for c in col_vals])
    ax.set_yticks(range(len(row_vals)), [str(r) for r in row_vals])
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(title, fontsize=10)
    for i in range(len(row_vals)):
        for j in range(len(col_vals)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
    return im


def run_sweep(cfg, n_steps, seed, out):
    lr = 5e-2
    common = dict(lr=lr, gate="snr", rho=0.99, alpha="online", track_stats=True)

    def excess(**adaptive):
        factory = lambda params: SNRAdamW(
            params, lambda_pop=1.0, **common,
            adaptive_threshold=AdaptiveThresholdConfig(
                mode="target_active_fraction", warmup_steps=100, update_interval=25, **adaptive),
        )
        log, _ = run_one(factory, cfg, n_steps, seed)
        return _mean_after(log, "excess_test", 200)

    # Sweep 1: target_active_fraction x active_gate_threshold (no floor).
    afs, q0s = [0.025, 0.05, 0.10, 0.20], [0.25, 0.5, 0.75]
    g1 = np.full((len(afs), len(q0s)), np.nan)
    print("\n==== Sweep 1: target_active_fraction x q0 (mean excess test loss) ====")
    for i, a in enumerate(afs):
        for j, q in enumerate(q0s):
            g1[i, j] = excess(target_active_fraction=a, active_gate_threshold=q)
            print(f"  af={a:<6} q0={q:<5} -> {g1[i, j]:.4f}")

    # Sweep 2: target_active_fraction x min_snr_threshold (floor) at q0=0.25.
    afs2, floors = [0.025, 0.05], [0.0, 0.5, 1.0, 2.0, 5.0]
    g2 = np.full((len(afs2), len(floors)), np.nan)
    print("\n==== Sweep 2: target_active_fraction x min_snr_threshold, q0=0.25 ====")
    for i, a in enumerate(afs2):
        for j, fl in enumerate(floors):
            g2[i, j] = excess(target_active_fraction=a, active_gate_threshold=0.25,
                              min_snr_threshold=fl)
            print(f"  af={a:<6} floor={fl:<5} -> {g2[i, j]:.4f}")

    static_log, _ = run_one(lambda params: SNRAdamW(params, lambda_pop=1.0, **common),
                            cfg, n_steps, seed)
    static = _mean_after(static_log, "excess_test", 200)
    adamw_log, _ = run_one(lambda params: torch.optim.AdamW(params, lr=lr), cfg, n_steps, seed)
    adamw = _mean_after(adamw_log, "excess_test", 200)
    print(f"  reference: SNR static = {static:.4f}, AdamW = {adamw:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = _heatmap(axes[0], g1, afs, q0s, "target_active_fraction",
                   "active_gate_threshold (q0)",
                   f"Quota target (no floor)\nSNR static={static:.3f}, AdamW={adamw:.3f}")
    fig.colorbar(im0, ax=axes[0], label="mean excess test loss")
    im1 = _heatmap(axes[1], g2, afs2, floors, "target_active_fraction",
                   "min_snr_threshold (absolute SNR floor)",
                   "Capped target + SNR floor (q0=0.25)\nlower is better")
    fig.colorbar(im1, ax=axes[1], label="mean excess test loss")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved sweep to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="benchmarks/benchmark_adaptive_threshold.png")
    parser.add_argument("--rdist-out", type=str,
                        default="benchmarks/benchmark_adaptive_threshold_rdist.png")
    parser.add_argument("--sweep-out", type=str,
                        default="benchmarks/benchmark_adaptive_threshold_sweep.png")
    parser.add_argument("--no-sweep", action="store_true", help="Skip the sparsity sweeps.")
    args = parser.parse_args()

    cfg = TaskConfig()
    run_main(cfg, args.steps, args.seed, args.out, args.rdist_out)
    if not args.no_sweep:
        run_sweep(cfg, args.steps, args.seed, args.sweep_out)


if __name__ == "__main__":
    main()
