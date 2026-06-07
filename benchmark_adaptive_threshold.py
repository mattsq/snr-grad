"""
Adaptive thresholding benchmark for sparse regression under regime shifts.

This benchmark is built around a specific lesson: on a genuinely sparse problem
(here d=200 with only k=5 true signal coordinates, a 2.5% active fraction), a *low*
mean gate is not a pathology -- it is the correct behaviour. Forcing the gate to
pass a fixed high fraction of update mass (e.g. mean gate ~0.3, or 20% active) is
over-permissive and hurts generalization. Adaptive thresholding earns its keep as a
*regime-shift response mechanism*, not as a permanent controller that forces high
update density.

The task uses a nonstationary schedule (see the implementation plan, section 14):

    steps    0 - 999 : signal coords A, noise sigma = 1
    steps 1000 - 1999 : signal coords A, noise sigma = 5   (noise jumps)
    steps 2000 - 2999 : signal coords B, noise sigma = 2   (signal support moves)

Because the true signal coordinates are known, we can measure what mean gate alone
cannot distinguish -- "correctly sparse" vs "wrongly suppressed":

    * mean gate on true signal coordinates
    * mean gate on noise coordinates
    * signal / noise gate ratio
    * false pass-through rate (noise coords gated active)
    * false suppression rate (signal coords gated inactive)

Two figures are produced:

  1. benchmark_adaptive_threshold.png -- diagnostic curves comparing AdamW, static
     SNR, fixed-active-fraction controllers at several targets, and the
     "shock_then_sparsify" controller.
  2. benchmark_adaptive_threshold_sweep.png -- a sweep over target_active_fraction
     x active_gate_threshold, reporting mean excess test loss (lower is better),
     expected to favour sparse targets near the true 2.5%.

Run:
    uv run python benchmark_adaptive_threshold.py            # both figures
    uv run python benchmark_adaptive_threshold.py --no-sweep # curves only
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


# Reference cutoff for the signal/noise diagnostics, shared across all methods so
# the false-rate curves are comparable regardless of each controller's own q0.
DIAG_Q0 = 0.5


@dataclass
class TaskConfig:
    d: int = 200            # input dimension
    k: int = 5              # number of active signal coordinates (true frac = k/d = 2.5%)
    n_test: int = 4000
    batch_size: int = 64
    signal_magnitude: float = 1.0
    # Regime schedule: (until_step, noise_sigma, signal_set in {"A", "B"}).
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


def _gate_vector(opt, p):
    """Reconstruct the per-coordinate gate q from the optimizer's live state."""
    group = next((g for g in opt.param_groups if p in g["params"]), None)
    st = opt.state.get(p)
    if group is None or st is None or "exp_grad_var" not in st:
        return None
    b1, _ = group["betas"]
    rho = group["rho"]
    t = st["step"]
    m_hat = st["exp_avg"] / (1.0 - b1 ** t)
    s_hat = st["exp_grad_var"] / (1.0 - rho ** t)
    alpha = resolve_alpha(group["alpha"])
    q = compute_gate(
        m_hat, s_hat, gate=group["gate"], alpha=alpha,
        lambda_pop=group["lambda_pop"], gate_eps=group["gate_eps"],
    )
    return q.detach().reshape(-1)


def run_one(opt_factory, cfg, n_steps, seed, log_every=20):
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
    )}

    for step in range(n_steps):
        sigma, sig = _regime_for_step(step, cfg)
        w_true, idx = signals[sig]
        X, y = _sample_batch(w_true, sigma, cfg.batch_size, gen)
        loss = ((model(X).squeeze(-1) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % log_every == 0:
            with torch.no_grad():
                Xt, yt = _sample_batch(w_true, sigma, cfg.n_test, gen)
                test_loss = ((model(Xt).squeeze(-1) - yt) ** 2).mean().item()
            log["step"].append(step)
            log["excess_test"].append(max(test_loss - sigma ** 2, 1e-6))

            ts = opt.get_threshold_state() if hasattr(opt, "get_threshold_state") else {}
            g0 = ts.get("group_0") if ts else None
            if g0 is not None:
                log["lambda_pop"].append(g0["lambda_pop"])
                log["target_af"].append(g0.get("target_active_fraction"))
            elif hasattr(opt, "param_groups") and "lambda_pop" in opt.param_groups[0]:
                log["lambda_pop"].append(opt.param_groups[0]["lambda_pop"])
                log["target_af"].append(None)
            else:
                log["lambda_pop"].append(float("nan"))
                log["target_af"].append(None)

            q = _gate_vector(opt, p)
            if q is None:
                for key in ("signal_gate", "noise_gate", "snr_ratio",
                            "false_passthrough", "false_suppression"):
                    log[key].append(float("nan"))
            else:
                mask = torch.zeros(cfg.d, dtype=torch.bool)
                mask[idx] = True
                sig_q = q[mask]
                noi_q = q[~mask]
                sg = sig_q.mean().item()
                ng = noi_q.mean().item()
                log["signal_gate"].append(sg)
                log["noise_gate"].append(ng)
                log["snr_ratio"].append(sg / max(ng, 1e-9))
                log["false_passthrough"].append((noi_q >= DIAG_Q0).float().mean().item())
                log["false_suppression"].append((sig_q < DIAG_Q0).float().mean().item())

    return log


def _mean_after(log, key, min_step):
    vals = [v for s, v in zip(log["step"], log[key]) if s >= min_step and v == v]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Main diagnostic comparison
# ---------------------------------------------------------------------------

def build_methods(cfg, lr):
    common = dict(lr=lr, gate="snr", rho=0.99, alpha="online", track_stats=True)

    def af_method(target):
        return lambda params: SNRAdamW(
            params, lambda_pop=1.0, **common,
            adaptive_threshold=AdaptiveThresholdConfig(
                mode="target_active_fraction", target_active_fraction=target,
                active_gate_threshold=0.5, warmup_steps=100, update_interval=25,
            ),
        )

    return {
        "AdamW": (lambda params: torch.optim.AdamW(params, lr=lr), "gray"),
        "SNR static": (lambda params: SNRAdamW(params, lambda_pop=1.0, **common), "C0"),
        "AF=0.20 (too high)": (af_method(0.20), "C3"),
        "AF=0.05": (af_method(0.05), "C1"),
        "AF=0.025 (true)": (af_method(0.025), "C2"),
        "shock_then_sparsify": (
            lambda params: SNRAdamW(
                params, lambda_pop=1.0, **common,
                adaptive_threshold=AdaptiveThresholdConfig(
                    mode="shock_then_sparsify",
                    sparse_target_active_fraction=0.025,
                    shock_target_active_fraction=0.2,
                    shock_steps=20, shift_detect_threshold=0.04,
                    active_gate_threshold=0.5, warmup_steps=100, update_interval=10,
                ),
            ),
            "C4",
        ),
    }


def run_main(cfg, n_steps, seed, out):
    lr = 5e-2
    methods = build_methods(cfg, lr)
    logs = {}
    for name, (factory, _) in methods.items():
        print(f"Running {name} ...")
        logs[name] = run_one(factory, cfg, n_steps, seed)

    shift_steps = [until for until, _, _ in cfg.regimes[:-1]]

    print("\n==== Summary (means over steps >= 200) ====")
    hdr = f"{'method':<22}{'excess test':>13}{'signal gate':>13}{'noise gate':>12}{'SNR ratio':>11}"
    print(hdr)
    for name, log in logs.items():
        print(f"{name:<22}{_mean_after(log,'excess_test',200):>13.4f}"
              f"{_mean_after(log,'signal_gate',200):>13.4f}"
              f"{_mean_after(log,'noise_gate',200):>12.4f}"
              f"{_mean_after(log,'snr_ratio',200):>11.2f}")

    panels = [
        ("excess_test", "excess test loss", True),
        ("lambda_pop", "lambda_pop", True),
        ("signal_gate", "mean gate (signal coords)", False),
        ("noise_gate", "mean gate (noise coords)", False),
        ("snr_ratio", "signal / noise gate ratio", True),
        ("target_af", "effective active-fraction target", False),
        ("false_passthrough", "false pass-through (noise active)", False),
        ("false_suppression", "false suppression (signal inactive)", False),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13, 17))
    axes = axes.ravel()
    for ax, (key, ylabel, logy) in zip(axes, panels):
        for name, (_, color) in methods.items():
            xs = logs[name]["step"]
            ys = logs[name][key]
            if all(v != v for v in ys):  # all NaN (e.g. AdamW gate stats)
                continue
            ax.plot(xs, ys, label=name, color=color, lw=1.4)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        for s in shift_steps:
            ax.axvline(s, color="k", ls="--", alpha=0.3)
        ax.grid(True, alpha=0.3)
    axes[5].axhline(cfg.true_active_fraction, ls=":", color="k", alpha=0.6,
                    label=f"true frac = {cfg.true_active_fraction:.3f}")
    axes[0].set_title("Adaptive SNR thresholding on sparse regression under regime shifts")
    axes[0].legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("step")
    axes[-2].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nSaved curves to {out}")


# ---------------------------------------------------------------------------
# Sparsity sweep: target_active_fraction x active_gate_threshold
# ---------------------------------------------------------------------------

def run_sweep(cfg, n_steps, seed, out):
    lr = 5e-2
    common = dict(lr=lr, gate="snr", rho=0.99, alpha="online", track_stats=True)
    afs = [0.025, 0.05, 0.10, 0.20]
    q0s = [0.25, 0.5, 0.75]

    grid = np.full((len(afs), len(q0s)), np.nan)
    print("\n==== Sparsity sweep (mean excess test loss, lower is better) ====")
    for i, af in enumerate(afs):
        for j, q0 in enumerate(q0s):
            factory = lambda params, af=af, q0=q0: SNRAdamW(
                params, lambda_pop=1.0, **common,
                adaptive_threshold=AdaptiveThresholdConfig(
                    mode="target_active_fraction", target_active_fraction=af,
                    active_gate_threshold=q0, warmup_steps=100, update_interval=25,
                ),
            )
            log = run_one(factory, cfg, n_steps, seed)
            grid[i, j] = _mean_after(log, "excess_test", 200)
            print(f"  af={af:<6} q0={q0:<5} -> {grid[i, j]:.4f}")

    # Reference rows.
    static_log = run_one(lambda params: SNRAdamW(params, lambda_pop=1.0, **common),
                         cfg, n_steps, seed)
    static_excess = _mean_after(static_log, "excess_test", 200)
    adamw_log = run_one(lambda params: torch.optim.AdamW(params, lr=lr), cfg, n_steps, seed)
    adamw_excess = _mean_after(adamw_log, "excess_test", 200)
    print(f"  reference: SNR static = {static_excess:.4f}, AdamW = {adamw_excess:.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto", origin="lower")
    ax.set_xticks(range(len(q0s)), [str(q) for q in q0s])
    ax.set_yticks(range(len(afs)), [str(a) for a in afs])
    ax.set_xlabel("active_gate_threshold (q0)")
    ax.set_ylabel("target_active_fraction")
    ax.set_title("Mean excess test loss vs sparsity target\n"
                 f"(SNR static = {static_excess:.3f}, AdamW = {adamw_excess:.3f}; lower is better)")
    for i in range(len(afs)):
        for j in range(len(q0s)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                    color="w", fontsize=9)
    fig.colorbar(im, ax=ax, label="mean excess test loss")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved sweep to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="benchmarks/benchmark_adaptive_threshold.png")
    parser.add_argument("--sweep-out", type=str,
                        default="benchmarks/benchmark_adaptive_threshold_sweep.png")
    parser.add_argument("--no-sweep", action="store_true", help="Skip the sparsity sweep.")
    args = parser.parse_args()

    cfg = TaskConfig()
    run_main(cfg, args.steps, args.seed, args.out)
    if not args.no_sweep:
        run_sweep(cfg, args.steps, args.seed, args.sweep_out)


if __name__ == "__main__":
    main()
