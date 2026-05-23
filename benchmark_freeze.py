"""
Benchmark: persistent-gate `requires_grad` freezing.

Compares each of the four SNR-gated optimizers in two arms:
  - baseline: freeze_low_snr=False
  - freeze:   freeze_low_snr=True

on an overparameterized MLP regressing onto a small noisy sparse target. Most
of the network is excess capacity and its gates should collapse to ~0 over
training, allowing the freeze arm to disable backward for those subgraphs.

Reports per (optimizer, arm):
  - peak_memory_mb_delta: process RSS peak above the pre-train baseline (psutil)
  - total_backward_time_sec: wall-clock summed across loss.backward() calls
  - final_test_loss: regression check that gating doesn't hurt quality
  - frozen_fraction_over_time: # elements frozen / total trainable, per log step

Outputs:
  benchmarks/freeze_memory_time.png  (multi-panel figure)
  optional CSV row via --sweep-out

Usage:
  uv run python benchmark_freeze.py --seeds 3 --steps 2000
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil
import torch
import torch.nn as nn

from snr_grad import RotatedSNRAdamW, SNRAdamW, SNRMuon, SpectralSNRMuon


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FreezeBenchConfig:
    d: int = 64                # input dim
    k: int = 4                 # active signal features
    n_train: int = 200
    test_size: int = 4000
    batch_size: int = 32
    sigma_noise: float = 1.5
    n_steps: int = 2000
    n_seeds: int = 3
    hidden: int = 256          # MLP width (deliberately overparameterized)
    n_hidden_layers: int = 2
    lr: float = 3e-4
    weight_decay: float = 0.0
    log_every: int = 50

    # Freeze hyperparameters
    freeze_threshold: float = 0.05
    freeze_patience: int = 100
    freeze_recheck_interval: int = 500
    freeze_beta: float = 0.99


OPTIMIZERS = [
    ("SNRAdamW", SNRAdamW),
    ("SNRMuon", SNRMuon),
    ("RotatedSNRAdamW", RotatedSNRAdamW),
    ("SpectralSNRMuon", SpectralSNRMuon),
]


# ---------------------------------------------------------------------------
# Data + model
# ---------------------------------------------------------------------------

def make_true_weights(d: int, k: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    w = torch.zeros(d)
    idx = torch.randperm(d, generator=gen)[:k]
    w[idx] = torch.randn(k, generator=gen) * 3.0
    return w


def make_dataset(w: torch.Tensor, n: int, sigma: float, gen: torch.Generator):
    d = w.shape[0]
    X = torch.randn(n, d, generator=gen)
    y = X @ w + torch.randn(n, generator=gen) * sigma
    return X, y.unsqueeze(1)


def make_model(cfg: FreezeBenchConfig, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    in_dim = cfg.d
    for _ in range(cfg.n_hidden_layers):
        layers.append(nn.Linear(in_dim, cfg.hidden))
        layers.append(nn.ReLU())
        in_dim = cfg.hidden
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    name: str
    arm: str  # "baseline" or "freeze"
    peak_memory_mb_delta: float = 0.0
    total_backward_time_sec: float = 0.0
    final_test_loss: float = 0.0
    frozen_fraction_over_time: list[float] = field(default_factory=list)
    log_steps: list[int] = field(default_factory=list)


def _trainable_elements(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_one_seed(
    opt_cls: type,
    arm: str,
    cfg: FreezeBenchConfig,
    seed: int,
) -> RunResult:
    process = psutil.Process(os.getpid())
    rss_baseline_bytes = process.memory_info().rss

    w_true = make_true_weights(cfg.d, cfg.k)
    train_gen = torch.Generator().manual_seed(seed)
    X_train, y_train = make_dataset(w_true, cfg.n_train, cfg.sigma_noise, train_gen)
    test_gen = torch.Generator().manual_seed(9999)
    X_test, y_test = make_dataset(w_true, cfg.test_size, cfg.sigma_noise, test_gen)

    model = make_model(cfg, seed=seed + 1000)
    total_elems = _trainable_elements(model)

    opt_kwargs: dict = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    if arm == "freeze":
        opt_kwargs.update(
            freeze_low_snr=True,
            freeze_threshold=cfg.freeze_threshold,
            freeze_patience=cfg.freeze_patience,
            freeze_recheck_interval=cfg.freeze_recheck_interval,
            freeze_beta=cfg.freeze_beta,
        )
    optimizer = opt_cls(model.parameters(), **opt_kwargs)

    result = RunResult(name=opt_cls.__name__, arm=arm)
    rss_peak_bytes = rss_baseline_bytes

    for step in range(cfg.n_steps):
        idx = torch.randint(cfg.n_train, (cfg.batch_size,))
        X_b, y_b = X_train[idx], y_train[idx]

        optimizer.zero_grad()
        loss = ((model(X_b) - y_b) ** 2).mean()

        # When every param is frozen, the loss has no grad_fn and backward
        # would raise. Skip backward in that case; optimizer.step() is still
        # called so the recheck cadence keeps advancing _global_step and will
        # eventually unfreeze the params for re-evaluation.
        if loss.requires_grad:
            t0 = time.perf_counter()
            loss.backward()
            result.total_backward_time_sec += time.perf_counter() - t0

        optimizer.step()

        if step % cfg.log_every == 0:
            n_p, n_e = optimizer.count_frozen()
            result.frozen_fraction_over_time.append(n_e / max(total_elems, 1))
            result.log_steps.append(step)
            rss_now = process.memory_info().rss
            if rss_now > rss_peak_bytes:
                rss_peak_bytes = rss_now

    with torch.no_grad():
        result.final_test_loss = float(((model(X_test) - y_test) ** 2).mean().item())

    result.peak_memory_mb_delta = (rss_peak_bytes - rss_baseline_bytes) / 1e6
    return result


# ---------------------------------------------------------------------------
# Multi-seed orchestration
# ---------------------------------------------------------------------------

def aggregate(results: list[RunResult]) -> dict:
    """Mean across seeds."""
    mem = [r.peak_memory_mb_delta for r in results]
    back = [r.total_backward_time_sec for r in results]
    loss = [r.final_test_loss for r in results]
    # Frozen-fraction curves: align by step index, mean across seeds
    if results:
        log_steps = results[0].log_steps
        # Assume all seeds log at the same step grid
        frozen_curves = torch.tensor([r.frozen_fraction_over_time for r in results])
        frozen_mean = frozen_curves.mean(dim=0).tolist()
    else:
        log_steps = []
        frozen_mean = []
    return {
        "peak_memory_mb_delta_mean": sum(mem) / len(mem),
        "peak_memory_mb_delta_std": float(torch.tensor(mem).std().item()) if len(mem) > 1 else 0.0,
        "total_backward_time_sec_mean": sum(back) / len(back),
        "total_backward_time_sec_std": float(torch.tensor(back).std().item()) if len(back) > 1 else 0.0,
        "final_test_loss_mean": sum(loss) / len(loss),
        "final_test_loss_std": float(torch.tensor(loss).std().item()) if len(loss) > 1 else 0.0,
        "log_steps": log_steps,
        "frozen_fraction_mean": frozen_mean,
    }


def run_experiment(cfg: FreezeBenchConfig) -> dict:
    out: dict = {}
    for name, opt_cls in OPTIMIZERS:
        out[name] = {}
        for arm in ("baseline", "freeze"):
            print(f"  {name} [{arm}]:", end=" ", flush=True)
            seed_results = []
            for seed in range(cfg.n_seeds):
                r = run_one_seed(opt_cls, arm, cfg, seed)
                seed_results.append(r)
                print(
                    f"seed={seed} mem={r.peak_memory_mb_delta:.1f}MB "
                    f"backward={r.total_backward_time_sec:.2f}s "
                    f"test={r.final_test_loss:.3f}",
                    end="  ", flush=True,
                )
            print()
            out[name][arm] = aggregate(seed_results)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(results: dict, cfg: FreezeBenchConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Freeze-low-SNR vs baseline (MLP {cfg.d}->{cfg.hidden}x{cfg.n_hidden_layers}->1, "
        f"{cfg.n_steps} steps, {cfg.n_seeds} seeds)",
        fontsize=13, fontweight="bold",
    )

    # (a) Frozen fraction over training, freeze arm only
    ax = axes[0, 0]
    for name in names:
        agg = results[name]["freeze"]
        ax.plot(agg["log_steps"], agg["frozen_fraction_mean"], label=name, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction of params frozen")
    ax.set_title("(a) Frozen fraction over training (freeze arm)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # (b) Peak RSS delta: baseline vs freeze, grouped bars
    ax = axes[0, 1]
    x = list(range(len(names)))
    w = 0.35
    bl = [results[n]["baseline"]["peak_memory_mb_delta_mean"] for n in names]
    fr = [results[n]["freeze"]["peak_memory_mb_delta_mean"] for n in names]
    bl_err = [results[n]["baseline"]["peak_memory_mb_delta_std"] for n in names]
    fr_err = [results[n]["freeze"]["peak_memory_mb_delta_std"] for n in names]
    ax.bar([xi - w/2 for xi in x], bl, w, yerr=bl_err, label="baseline", color="tab:orange", capsize=4)
    ax.bar([xi + w/2 for xi in x], fr, w, yerr=fr_err, label="freeze", color="tab:blue", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Peak RSS delta (MB)")
    ax.set_title("(b) Peak process RSS over baseline")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (c) Backward time
    ax = axes[1, 0]
    bl = [results[n]["baseline"]["total_backward_time_sec_mean"] for n in names]
    fr = [results[n]["freeze"]["total_backward_time_sec_mean"] for n in names]
    bl_err = [results[n]["baseline"]["total_backward_time_sec_std"] for n in names]
    fr_err = [results[n]["freeze"]["total_backward_time_sec_std"] for n in names]
    ax.bar([xi - w/2 for xi in x], bl, w, yerr=bl_err, label="baseline", color="tab:orange", capsize=4)
    ax.bar([xi + w/2 for xi in x], fr, w, yerr=fr_err, label="freeze", color="tab:blue", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Total backward time (s)")
    ax.set_title("(c) Cumulative loss.backward() wall-clock")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (d) Final test loss
    ax = axes[1, 1]
    bl = [results[n]["baseline"]["final_test_loss_mean"] for n in names]
    fr = [results[n]["freeze"]["final_test_loss_mean"] for n in names]
    bl_err = [results[n]["baseline"]["final_test_loss_std"] for n in names]
    fr_err = [results[n]["freeze"]["final_test_loss_std"] for n in names]
    ax.bar([xi - w/2 for xi in x], bl, w, yerr=bl_err, label="baseline", color="tab:orange", capsize=4)
    ax.bar([xi + w/2 for xi in x], fr, w, yerr=fr_err, label="freeze", color="tab:blue", capsize=4)
    irreducible = cfg.sigma_noise ** 2
    ax.axhline(irreducible, ls="--", color="gray", alpha=0.5, label=f"Irreducible ({irreducible:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Final test MSE")
    ax.set_title("(d) Final test loss (quality check)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = out_dir / "freeze_memory_time.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# CSV emission (for sweep compatibility)
# ---------------------------------------------------------------------------

def write_csv(results: dict, path: Path) -> None:
    rows = []
    for name, arms in results.items():
        for arm, agg in arms.items():
            rows.append({
                "optimizer": name,
                "arm": arm,
                "peak_memory_mb_delta": agg["peak_memory_mb_delta_mean"],
                "total_backward_time_sec": agg["total_backward_time_sec_mean"],
                "final_test_loss": agg["final_test_loss_mean"],
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["optimizer", "arm", "peak_memory_mb_delta",
                        "total_backward_time_sec", "final_test_loss"],
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--sweep-out", type=str, default=None)
    args = ap.parse_args()

    cfg = FreezeBenchConfig()
    if args.seeds is not None:
        cfg.n_seeds = args.seeds
    if args.steps is not None:
        cfg.n_steps = args.steps

    print(
        f"Config: d={cfg.d} hidden={cfg.hidden}x{cfg.n_hidden_layers} "
        f"steps={cfg.n_steps} seeds={cfg.n_seeds} "
        f"freeze_threshold={cfg.freeze_threshold} freeze_patience={cfg.freeze_patience} "
        f"freeze_recheck_interval={cfg.freeze_recheck_interval}"
    )
    results = run_experiment(cfg)

    out_dir = Path(__file__).resolve().parent / "benchmarks"
    make_figure(results, cfg, out_dir)

    # Summary table to stdout
    print("\nSummary (mean across seeds):")
    print(f"{'optimizer':<20} {'arm':<10} {'mem_MB':>10} {'backward_s':>12} {'test_loss':>12}")
    for name, arms in results.items():
        for arm, agg in arms.items():
            print(
                f"{name:<20} {arm:<10} "
                f"{agg['peak_memory_mb_delta_mean']:>10.2f} "
                f"{agg['total_backward_time_sec_mean']:>12.3f} "
                f"{agg['final_test_loss_mean']:>12.4f}"
            )

    if args.sweep_out:
        write_csv(results, Path(args.sweep_out))
        print(f"  Wrote CSV: {args.sweep_out}")
