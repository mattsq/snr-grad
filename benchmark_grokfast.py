"""
Synthetic benchmark: Grokfast-SNR vs SNRAdamW vs Grokfast vs AdamW on sparse linear regression with label noise.

Problem setup:
  - d=200 features, only k=5 have nonzero true weights
  - Fixed training set of n=100 samples with high label noise
  - Minibatch size 32, trained for 3000 steps
  - Compares:
    1. AdamW (Standard baseline)
    2. SNRAdamW (SNR gating only)
    3. Grokfast (Slow-gradient amplification only)
    4. Grokfast-SNR (Our proposed synergy)

Outputs PNG figures in the benchmarks/ directory.
"""

import os
import argparse
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_grad import SNRAdamW


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    d: int = 200
    k: int = 5
    n_train: int = 100
    batch_size: int = 32
    sigma_noise: float = 3.0
    n_steps: int = 3000
    n_seeds: int = 5
    test_size: int = 10000
    lr: float = 3e-3
    weight_decay: float = 0.0  # no WD so gate must do the work
    signal_magnitude: float = 3.0
    rho: float = 0.99
    alpha: object = "finite"
    lambda_pop: float = 1.0
    
    # Grokfast parameters
    grokfast_alpha: float = 0.98
    grokfast_lamb: float = 2.0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    train_losses: list = field(default_factory=list)
    test_losses: list = field(default_factory=list)
    param_errors: list = field(default_factory=list)
    final_weights: torch.Tensor = None


def run_one_seed(optimizer_cls, opt_kwargs, cfg, seed):
    w_true, signal_idx = make_true_weights(cfg.d, cfg.k, cfg.signal_magnitude)

    train_gen = torch.Generator().manual_seed(seed)
    X_train, y_train = make_dataset(w_true, cfg.n_train, cfg.sigma_noise, train_gen)
    test_gen = torch.Generator().manual_seed(9999)
    X_test, y_test = make_dataset(w_true, cfg.test_size, cfg.sigma_noise, test_gen)

    torch.manual_seed(seed + 1000)
    model = nn.Linear(cfg.d, 1, bias=False)
    nn.init.zeros_(model.weight)
    optimizer = optimizer_cls(model.parameters(), **opt_kwargs)

    result = RunResult()
    eval_every = 10

    for step in range(cfg.n_steps):
        idx = torch.randint(cfg.n_train, (cfg.batch_size,))
        X_b, y_b = X_train[idx], y_train[idx]

        optimizer.zero_grad()
        loss = ((model(X_b) - y_b) ** 2).mean()
        loss.backward()
        optimizer.step()

        if step % eval_every == 0:
            result.train_losses.append(loss.item())
            with torch.no_grad():
                result.test_losses.append(((model(X_test) - y_test) ** 2).mean().item())
            result.param_errors.append(
                (model.weight.data.squeeze() - w_true).norm().item()
            )

    result.final_weights = model.weight.data.squeeze().clone()
    return result


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_experiment(cfg):
    # 1. AdamW Baseline
    adamw_kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # 2. SNRAdamW
    snr_kwargs = dict(
        lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", rho=cfg.rho,
        alpha=cfg.alpha, batch_size=cfg.batch_size, dataset_size=cfg.n_train,
        lambda_pop=cfg.lambda_pop,
    )
    
    # 3. Grokfast (disable SNR gating by setting lambda_pop=0)
    grokfast_kwargs = dict(
        lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", rho=cfg.rho,
        alpha=cfg.alpha, batch_size=cfg.batch_size, dataset_size=cfg.n_train,
        lambda_pop=0.0, grokfast_alpha=cfg.grokfast_alpha, grokfast_lamb=cfg.grokfast_lamb,
    )
    
    # 4. Grokfast-SNR Synergy
    grokfast_snr_kwargs = dict(
        lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", rho=cfg.rho,
        alpha=cfg.alpha, batch_size=cfg.batch_size, dataset_size=cfg.n_train,
        lambda_pop=cfg.lambda_pop, grokfast_alpha=cfg.grokfast_alpha, grokfast_lamb=cfg.grokfast_lamb,
    )

    baselines = {
        "AdamW": (torch.optim.AdamW, adamw_kwargs),
        "SNRAdamW": (SNRAdamW, snr_kwargs),
        "Grokfast": (SNRAdamW, grokfast_kwargs),
        "Grokfast-SNR": (SNRAdamW, grokfast_snr_kwargs),
    }
    
    results = {k: [] for k in baselines}
    
    for seed in range(cfg.n_seeds):
        print(f"Seed {seed+1}/{cfg.n_seeds}...")
        for name, (cls, kwargs) in baselines.items():
            res = run_one_seed(cls, kwargs, cfg, seed)
            results[name].append(res)
            print(f"  {name:15} final excess test MSE: {res.test_losses[-1] - cfg.sigma_noise**2:.4f}")
            
    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def to_stack(results, attr):
    return torch.tensor([getattr(r, attr) for r in results])


def plot_band(ax, steps, data, label, color, alpha=0.2):
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    ax.plot(steps, mean, label=label, color=color, linewidth=2.0)
    ax.fill_between(steps, (mean - std).numpy(), (mean + std).numpy(),
                    color=color, alpha=alpha)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(results, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    eval_every = 10
    steps = list(range(0, cfg.n_steps, eval_every))
    irreducible = cfg.sigma_noise ** 2
    
    colors = {
        "AdamW": "tab:orange",
        "SNRAdamW": "tab:blue",
        "Grokfast": "tab:purple",
        "Grokfast-SNR": "tab:green",
    }

    # ---- Figure 1: Convergence curves ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Grokfast-SNR vs Baselines: Noisy Sparse Regression  "
        f"(d={cfg.d}, k={cfg.k}, n={cfg.n_train}, noise={cfg.sigma_noise}, "
        f"{cfg.n_seeds} seeds)",
        fontsize=14, fontweight="bold",
    )

    # Train MSE
    ax = axes[0]
    for name, runs in results.items():
        data = to_stack(runs, "train_losses")
        plot_band(ax, steps, data, name, colors[name])
    ax.axhline(irreducible, ls="--", color="gray", alpha=0.6, label="Irreducible noise limit")
    ax.set_ylabel("Train MSE")
    ax.set_xlabel("Step")
    ax.set_title("(a) Training Loss")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Excess Test MSE
    ax = axes[1]
    for name, runs in results.items():
        data = to_stack(runs, "test_losses") - irreducible
        plot_band(ax, steps, data, name, colors[name])
    ax.axhline(0, ls="--", color="gray", alpha=0.4)
    ax.set_ylabel("Excess Test MSE (lower is better)")
    ax.set_xlabel("Step")
    ax.set_title("(b) Generalization Error")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Parameter Recovery Error
    ax = axes[2]
    for name, runs in results.items():
        data = to_stack(runs, "param_errors")
        plot_band(ax, steps, data, name, colors[name])
    ax.set_ylabel("||w - w*||")
    ax.set_xlabel("Step")
    ax.set_title("(c) Parameter Recovery Error")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_grokfast_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Figure 2: Summary bars ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Final Step Performance (mean +/- std)", fontsize=14, fontweight="bold")
    
    names = list(results.keys())
    x_positions = range(len(names))
    
    # Excess Test MSE comparison
    ax = axes[0]
    final_test_mse = [to_stack(results[name], "test_losses")[:, -1] - irreducible for name in names]
    means = [data.mean().item() for data in final_test_mse]
    stds = [data.std().item() for data in final_test_mse]
    
    ax.bar(x_positions, means, yerr=stds, color=[colors[n] for n in names], capsize=8, alpha=0.85)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Final Excess Test MSE")
    ax.set_title("(a) Final Generalization Error")
    ax.grid(True, linestyle=":", alpha=0.4, axis="y")
    
    # Parameter error comparison
    ax = axes[1]
    final_param_err = [to_stack(results[name], "param_errors")[:, -1] for name in names]
    p_means = [data.mean().item() for data in final_param_err]
    p_stds = [data.std().item() for data in final_param_err]
    
    ax.bar(x_positions, p_means, yerr=p_stds, color=[colors[n] for n in names], capsize=8, alpha=0.85)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("||w - w*||")
    ax.set_title("(b) Final Parameter Error")
    ax.grid(True, linestyle=":", alpha=0.4, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_grokfast_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = BenchmarkConfig()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
    
    print("Running Grokfast-SNR Benchmark experiments...")
    results = run_experiment(cfg)
    
    print("Generating plots...")
    make_figures(results, cfg, out_dir)
    print("Benchmark complete!")
