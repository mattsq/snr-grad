"""
Does Double Preconditioning (DoPr) help on the EXISTING sparse-regression task?

`benchmark_dopr.py` evaluates DoPr on a bespoke test-time-feedback (TTF) task that
activation preconditioning (AP) was designed for. This script instead asks the
honest question: what does AP do to the *existing* flagship benchmark
(`benchmark.py` -- sparse linear regression with label noise, d=200, k=5, fixed
n=100 training set), which it was NOT designed for?

A null or backwards result is informative. Note this task is a genuine stress test
for AP: the model is a single `nn.Linear(d, 1)` and the per-minibatch input
covariance S_z = X_b^T X_b / b is heavily rank-deficient (batch=32 << d=200), so
AP is far from the identity -- it strongly reweights gradient directions using a
rank-32 estimate of a 200-dim covariance. We compare each base optimizer to its
DoPr-wrapped version on excess test MSE and parameter-recovery error.

Outputs benchmarks/dopr_existing_curves.png and benchmarks/dopr_existing_summary.png.
"""

import os
import argparse
from dataclasses import replace

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_grad import SNRAdamW, DoPr, ActivationPrecondConfig
from benchmark import BenchmarkConfig, make_true_weights, make_dataset, plot_band


EVAL_EVERY = 10


def run_one(cfg, seed, optimizer_cls, opt_kwargs, use_dopr, damping):
    """One training run on the sparse-regression task; returns per-eval metrics.

    Mirrors ``benchmark.run_one_seed`` (same data, init, loop) but optionally wraps
    the optimizer in :class:`DoPr` so the gradient is activation-preconditioned.
    """
    w_true, signal_idx = make_true_weights(cfg.d, cfg.k, cfg.signal_magnitude)

    train_gen = torch.Generator().manual_seed(seed)
    X_train, y_train = make_dataset(w_true, cfg.n_train, cfg.sigma_noise, train_gen)
    test_gen = torch.Generator().manual_seed(9999)
    X_test, y_test = make_dataset(w_true, cfg.test_size, cfg.sigma_noise, test_gen)

    torch.manual_seed(seed + 1000)
    model = nn.Linear(cfg.d, 1, bias=False)
    nn.init.zeros_(model.weight)

    base = optimizer_cls(model.parameters(), **opt_kwargs)
    opt = DoPr(base, model, ActivationPrecondConfig(damping=damping)) if use_dopr else base

    test_losses, param_errors = [], []
    for step in range(cfg.n_steps):
        idx = torch.randint(cfg.n_train, (cfg.batch_size,))
        X_b, y_b = X_train[idx], y_train[idx]

        opt.zero_grad()
        loss = ((model(X_b) - y_b) ** 2).mean()
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                test_losses.append(((model(X_test) - y_test) ** 2).mean().item())
            param_errors.append((model.weight.data.squeeze() - w_true).norm().item())

    return {"test_losses": test_losses, "param_errors": param_errors}


def arms(cfg):
    """(label, color, optimizer_cls, opt_kwargs, use_dopr)."""
    snr_kwargs = dict(
        lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", rho=cfg.rho,
        alpha=cfg.alpha, batch_size=cfg.batch_size, dataset_size=cfg.n_train,
        lambda_pop=cfg.lambda_pop, track_stats=False,
    )
    adam_kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    return [
        ("AdamW", "tab:orange", torch.optim.AdamW, adam_kwargs, False),
        ("DoPr(AdamW)", "tab:red", torch.optim.AdamW, dict(adam_kwargs), True),
        ("SNRAdamW", "tab:blue", SNRAdamW, snr_kwargs, False),
        ("DoPr(SNRAdamW)", "tab:green", SNRAdamW, dict(snr_kwargs), True),
    ]


def run_experiment(cfg, damping):
    results = {}
    for label, color, cls, kwargs, use_dopr in arms(cfg):
        runs = []
        for seed in range(cfg.n_seeds):
            runs.append(run_one(cfg, seed, cls, dict(kwargs), use_dopr, damping))
        test = torch.tensor([r["test_losses"] for r in runs])
        perr = torch.tensor([r["param_errors"] for r in runs])
        results[label] = {"color": color, "test": test, "perr": perr}
        print(f"  {label:16s} final excess test MSE = "
              f"{(test[:, -1].mean() - cfg.sigma_noise ** 2):7.3f} "
              f"(+/- {test[:, -1].std():.3f}),  ||w-w*|| = {perr[:, -1].mean():.3f}")
    return results


def make_figures(results, cfg, damping, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    steps = list(range(0, cfg.n_steps, EVAL_EVERY))
    irreducible = cfg.sigma_noise ** 2
    title_tail = (f"sparse regression (d={cfg.d}, k={cfg.k}, n={cfg.n_train}, "
                  f"batch={cfg.batch_size}, AP damping={damping}, {cfg.n_seeds} seeds)")

    # ---- Curves: excess test MSE + parameter error ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"DoPr vs base optimizer on the EXISTING {title_tail}",
                 fontsize=12, fontweight="bold")
    for label, r in results.items():
        plot_band(axes[0], steps, r["test"] - irreducible, label, r["color"])
        plot_band(axes[1], steps, r["perr"], label, r["color"])
    axes[0].axhline(0, ls="--", color="gray", alpha=0.4)
    axes[0].set_title("(a) Excess test MSE (lower is better)")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Excess test MSE")
    axes[0].legend(fontsize=9)
    axes[1].set_title("(b) Parameter recovery error")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("||w - w*||")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "dopr_existing_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Summary: final excess test MSE, paired base vs DoPr ----
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(results.keys())
    means = [(results[l]["test"][:, -1].mean() - irreducible).item() for l in labels]
    stds = [results[l]["test"][:, -1].std().item() for l in labels]
    colors = [results[l]["color"] for l in labels]
    ax.bar(labels, means, yerr=stds, color=colors, capsize=8)
    ax.axhline(0, ls="--", color="gray", alpha=0.4)
    ax.set_ylabel("Final excess test MSE")
    ax.set_title(f"DoPr effect on the existing task -- {title_tail}", fontsize=10)
    ax.tick_params(axis="x", labelrotation=15)
    plt.tight_layout()
    path = os.path.join(out_dir, "dopr_existing_summary.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Fewer seeds/steps for a fast smoke run.")
    p.add_argument("--damping", type=float, default=0.1,
                   help="AP damping gamma (relative to trace(S_z)).")
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    args = p.parse_args()

    cfg = BenchmarkConfig()
    if args.quick:
        cfg = replace(cfg, n_seeds=3, n_steps=1500)
    if args.seeds is not None:
        cfg = replace(cfg, n_seeds=args.seeds)
    if args.steps is not None:
        cfg = replace(cfg, n_steps=args.steps)

    print(f"Running DoPr-on-existing-task benchmark "
          f"(seeds={cfg.n_seeds}, steps={cfg.n_steps}, damping={args.damping})...")
    results = run_experiment(cfg, args.damping)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
    make_figures(results, cfg, args.damping, out_dir)
