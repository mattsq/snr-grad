"""
Adaptive thresholding benchmark: does a self-tuning SNR gate recalibrate under
changing gradient statistics better than a static lambda_pop?

The task is a sparse linear regression with a *nonstationary* schedule designed to
expose threshold staleness (see the implementation plan, section 14):

    steps   0 - 999 : signal coords A, noise sigma = 1
    steps 1000 -1999 : signal coords A, noise sigma = 5   (noise jumps)
    steps 2000 -2999 : signal coords B, noise sigma = 2   (signal support moves)

A *static* SNR gate is tuned for the first regime; once the statistics shift, its
mean gate / active fraction drift away from where they started. The adaptive gates
re-tune lambda_pop on the fly to hold their target.

Outputs a 3-panel plot (lambda_pop, mean gate, test loss over training) and prints
a summary table including how quickly each method recovers its target after a shift.

Run:
    uv run python benchmark_adaptive_threshold.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_grad import SNRAdamW, AdaptiveThresholdConfig


@dataclass
class TaskConfig:
    d: int = 200            # input dimension
    k: int = 10             # number of active signal coordinates
    n_test: int = 2000
    batch_size: int = 64
    signal_magnitude: float = 1.0
    # Regime schedule: (until_step, noise_sigma, signal_set) where signal_set in {"A", "B"}.
    regimes: list = field(default_factory=lambda: [
        (1000, 1.0, "A"),
        (2000, 5.0, "A"),
        (3000, 2.0, "B"),
    ])


def _make_signal(d: int, k: int, magnitude: float, generator: torch.Generator, exclude=None):
    """Pick k random coordinates (disjoint from `exclude`) and assign signed weights."""
    candidates = [i for i in range(d) if exclude is None or i not in set(exclude.tolist())]
    perm = torch.randperm(len(candidates), generator=generator)
    idx = torch.tensor([candidates[j] for j in perm[:k].tolist()])
    w = torch.zeros(d)
    signs = torch.randint(0, 2, (k,), generator=generator).float() * 2 - 1
    w[idx] = signs * magnitude
    return w, idx


def _sample_batch(w_true, sigma, n, generator):
    d = w_true.numel()
    X = torch.randn(n, d, generator=generator)
    noise = sigma * torch.randn(n, generator=generator)
    y = X @ w_true + noise
    return X, y


def _regime_for_step(step, cfg: TaskConfig):
    for until, sigma, sig in cfg.regimes:
        if step < until:
            return sigma, sig
    return cfg.regimes[-1][1], cfg.regimes[-1][2]


def run_one(name, opt_factory, cfg: TaskConfig, n_steps, seed, track_target=None):
    """Train one optimizer over the nonstationary schedule, logging diagnostics."""
    gen = torch.Generator().manual_seed(seed)
    w_A, idx_A = _make_signal(cfg.d, cfg.k, cfg.signal_magnitude, gen)
    w_B, idx_B = _make_signal(cfg.d, cfg.k, cfg.signal_magnitude, gen, exclude=idx_A)
    signals = {"A": w_A, "B": w_B}

    model = torch.nn.Linear(cfg.d, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    opt = opt_factory(model.parameters())

    # Fixed test set per regime is rebuilt as signal/noise change.
    log = {"step": [], "train_loss": [], "test_loss": [], "mean_gate": [],
           "active_fraction": [], "lambda_pop": []}

    for step in range(n_steps):
        sigma, sig = _regime_for_step(step, cfg)
        w_true = signals[sig]
        X, y = _sample_batch(w_true, sigma, cfg.batch_size, gen)

        pred = model(X).squeeze(-1)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0:
            with torch.no_grad():
                Xt, yt = _sample_batch(w_true, sigma, cfg.n_test, gen)
                test_loss = ((model(Xt).squeeze(-1) - yt) ** 2).mean().item()
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["test_loss"].append(test_loss)
            stats = getattr(opt, "last_stats", None)
            log["mean_gate"].append(stats.mean_gate if stats is not None else float("nan"))
            ts = opt.get_threshold_state() if hasattr(opt, "get_threshold_state") else {}
            if ts:
                g0 = ts["group_0"]
                log["lambda_pop"].append(g0["lambda_pop"])
                af = g0.get("ema_active_fraction")
                log["active_fraction"].append(af if af is not None else float("nan"))
            else:
                lp = opt.param_groups[0].get("lambda_pop", float("nan"))
                log["lambda_pop"].append(lp)
                log["active_fraction"].append(float("nan"))

    return log


def _recovery_steps(log, target, shift_step, tol=0.05, key="mean_gate"):
    """Steps after `shift_step` until `key` returns within `tol` of `target`."""
    if target is None:
        return None
    after = [(s, v) for s, v in zip(log["step"], log[key]) if s >= shift_step]
    for s, v in after:
        if v == v and abs(v - target) <= tol:  # v==v filters NaN
            return s - shift_step
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="benchmarks/benchmark_adaptive_threshold.png")
    args = parser.parse_args()

    cfg = TaskConfig()
    lr = 5e-2
    common = dict(lr=lr, gate="snr", rho=0.99, alpha="online", track_stats=True)
    target_mean = 0.3
    target_af = 0.2

    methods = {
        "AdamW": lambda params: torch.optim.AdamW(params, lr=lr),
        "SNR static": lambda params: SNRAdamW(params, lambda_pop=1.0, **common),
        "SNR target_mean_gate": lambda params: SNRAdamW(
            params, lambda_pop=1.0, **common,
            adaptive_threshold=AdaptiveThresholdConfig(
                mode="target_mean_gate", target_mean_gate=target_mean,
                warmup_steps=100, update_interval=25, adaptation_lr=0.2,
            ),
        ),
        "SNR target_active_fraction": lambda params: SNRAdamW(
            params, lambda_pop=1.0, **common,
            adaptive_threshold=AdaptiveThresholdConfig(
                mode="target_active_fraction", target_active_fraction=target_af,
                active_gate_threshold=0.5, warmup_steps=100, update_interval=25,
            ),
        ),
    }

    logs = {}
    for name, factory in methods.items():
        print(f"Running {name} ...")
        logs[name] = run_one(name, factory, cfg, args.steps, args.seed)

    shift_steps = [until for until, _, _ in cfg.regimes[:-1]]

    print("\n==== Summary ====")
    print(f"{'method':<28} {'final test':>12} {'final mean_gate':>16} {'final lambda':>14}")
    for name, log in logs.items():
        print(f"{name:<28} {log['test_loss'][-1]:>12.4f} "
              f"{log['mean_gate'][-1]:>16.4f} {log['lambda_pop'][-1]:>14.4f}")

    # Recovery after the second shift (signal support moves at step 2000).
    print("\n==== Target recovery after signal shift (step 2000) ====")
    rec = _recovery_steps(logs["SNR target_mean_gate"], target_mean, shift_steps[1],
                          tol=0.08, key="mean_gate")
    print(f"target_mean_gate recovered mean gate to within 0.08 after {rec} steps")

    # ---- Plot ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    colors = {"AdamW": "gray", "SNR static": "C0",
              "SNR target_mean_gate": "C1", "SNR target_active_fraction": "C2"}

    for name, log in logs.items():
        axes[0].plot(log["step"], log["lambda_pop"], label=name, color=colors[name])
        axes[1].plot(log["step"], log["mean_gate"], label=name, color=colors[name])
        axes[2].plot(log["step"], log["test_loss"], label=name, color=colors[name])

    axes[0].set_ylabel("lambda_pop")
    axes[0].set_yscale("log")
    axes[0].set_title("Adaptive thresholding under a nonstationary sparse-regression schedule")
    axes[1].set_ylabel("mean gate")
    axes[1].axhline(target_mean, ls=":", color="C1", alpha=0.6, label="mean-gate target")
    axes[2].set_ylabel("test loss")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("step")

    for ax in axes:
        for s in shift_steps:
            ax.axvline(s, color="k", ls="--", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nSaved plot to {args.out}")


if __name__ == "__main__":
    main()
