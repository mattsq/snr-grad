"""
Benchmark comparing MARSSNRAdamW, SNRAdamW, and standard AdamW.
Plots are saved in the benchmarks/ directory.
"""

import os
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snr_grad import MARSSNRAdamW, SNRAdamW, compute_gate, resolve_alpha


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
    n_steps: int = 2000
    n_seeds: int = 5
    test_size: int = 5000
    lr: float = 1e-1
    weight_decay: float = 0.1
    signal_magnitude: float = 3.0
    rho: float = 0.99
    alpha: str = "online"
    lambda_pop: float = 1.0
    gamma: float = 0.025
    mars_clip: float = 1.0


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
    signal_gates: list = field(default_factory=list)
    noise_gates: list = field(default_factory=list)
    final_weights: torch.Tensor = None


def run_one_seed(optimizer_cls, opt_kwargs, cfg, seed, track_gates=False):
    w_true, signal_idx = make_true_weights(cfg.d, cfg.k, cfg.signal_magnitude)
    noise_mask = torch.ones(cfg.d, dtype=torch.bool)
    noise_mask[signal_idx] = False

    train_gen = torch.Generator().manual_seed(seed)
    X_train, y_train = make_dataset(w_true, cfg.n_train, cfg.sigma_noise, train_gen)
    test_gen = torch.Generator().manual_seed(9999)
    X_test, y_test = make_dataset(w_true, cfg.test_size, cfg.sigma_noise, test_gen)

    torch.manual_seed(seed + 1000)
    model = nn.Linear(cfg.d, 1, bias=False)
    nn.init.zeros_(model.weight)
    optimizer = optimizer_cls(model.parameters(), **opt_kwargs)

    result = RunResult()
    eval_every = 20

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

            if track_gates and hasattr(optimizer, "last_stats") and optimizer.state[model.weight]:
                state = optimizer.state[model.weight]
                step_num = state["step"]
                betas = opt_kwargs.get("betas", (0.9, 0.999))
                rho = opt_kwargs.get("rho", 0.99)
                m_hat = state["exp_avg"].squeeze() / (1 - betas[0] ** step_num)
                s_hat = state["exp_grad_var"].squeeze() / (1 - rho ** step_num)

                alpha_spec = opt_kwargs.get("alpha", "online")
                if isinstance(alpha_spec, str):
                    alpha_val = resolve_alpha(
                        alpha_spec,
                        batch_size=opt_kwargs.get("batch_size"),
                        dataset_size=opt_kwargs.get("dataset_size"),
                    )
                else:
                    alpha_val = float(alpha_spec)

                gate_vals = compute_gate(
                    m_hat, s_hat,
                    gate=opt_kwargs.get("gate", "snr"),
                    alpha=alpha_val,
                    lambda_pop=opt_kwargs.get("lambda_pop", 1.0),
                )
                result.signal_gates.append(gate_vals[signal_idx].mean().item())
                result.noise_gates.append(gate_vals[noise_mask].mean().item())

    result.final_weights = model.weight.data.squeeze().clone()
    return result


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_experiment(cfg):
    # Setup hyperparameters
    adam_kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    snr_kwargs = dict(
        lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", rho=cfg.rho,
        alpha=cfg.alpha, batch_size=cfg.batch_size, dataset_size=cfg.n_train,
        lambda_pop=cfg.lambda_pop, track_stats=True
    )
    
    mars_kwargs = dict(
        lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", rho=cfg.rho,
        alpha=cfg.alpha, batch_size=cfg.batch_size, dataset_size=cfg.n_train,
        lambda_pop=cfg.lambda_pop, gamma=cfg.gamma, mars_clip=cfg.mars_clip,
        optimize_1d=False, caution=False, track_stats=True
    )
    
    mars_caution_kwargs = dict(
        lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", rho=cfg.rho,
        alpha=cfg.alpha, batch_size=cfg.batch_size, dataset_size=cfg.n_train,
        lambda_pop=cfg.lambda_pop, gamma=cfg.gamma, mars_clip=cfg.mars_clip,
        optimize_1d=False, caution=True, track_stats=True
    )

    results = {}
    optimizers = [
        ("AdamW", torch.optim.AdamW, adam_kwargs, False),
        ("SNRAdamW", SNRAdamW, snr_kwargs, True),
        ("MARSSNR", MARSSNRAdamW, mars_kwargs, True),
        ("MARSSNR+Caution", MARSSNRAdamW, mars_caution_kwargs, True),
    ]

    for name, opt_cls, kwargs, track in optimizers:
        print(f"Running {name}...")
        res_list = []
        for seed in range(cfg.n_seeds):
            print(f"  Seed {seed+1}/{cfg.n_seeds}...", end=" ", flush=True)
            res = run_one_seed(opt_cls, kwargs, cfg, seed, track_gates=track)
            res_list.append(res)
            print(f"Test MSE={res.test_losses[-1]:.2f}")
        results[name] = res_list

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def to_stack(results, attr):
    return torch.tensor([getattr(r, attr) for r in results])


def plot_band(ax, steps, data, label, color, alpha=0.2):
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    ax.plot(steps, mean, label=label, color=color, linewidth=1.8)
    ax.fill_between(steps, (mean - std).numpy(), (mean + std).numpy(),
                    color=color, alpha=alpha)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(results, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    eval_every = 20
    steps = list(range(0, cfg.n_steps, eval_every))
    irreducible = cfg.sigma_noise ** 2
    w_true, signal_idx = make_true_weights(cfg.d, cfg.k, cfg.signal_magnitude)
    noise_mask = torch.ones(cfg.d, dtype=torch.bool)
    noise_mask[signal_idx] = False

    colors = {
        "AdamW": "tab:orange",
        "SNRAdamW": "tab:purple",
        "MARSSNR": "tab:red",
        "MARSSNR+Caution": "tab:blue"
    }

    # ---- Figure 1: Main 2x2 comparison ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"MARSSNRAdamW vs Baselines: High-Noise Sparse Regression\n"
        f"(d={cfg.d}, k={cfg.k}, n={cfg.n_train}, noise_std={cfg.sigma_noise}, "
        f"{cfg.n_seeds} seeds)",
        fontsize=14, fontweight="bold",
    )

    # Train Loss
    ax = axes[0, 0]
    for name, res_list in results.items():
        train_data = to_stack(res_list, "train_losses")
        plot_band(ax, steps, train_data, name, colors[name])
    ax.axhline(irreducible, ls="--", color="gray", alpha=0.6,
               label=f"Irreducible ({irreducible:.0f})")
    ax.set_ylabel("Train MSE", fontsize=11)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_title("(a) Train Loss Convergence", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Excess Test MSE
    ax = axes[0, 1]
    for name, res_list in results.items():
        test_data = to_stack(res_list, "test_losses")
        excess_test = test_data - irreducible
        plot_band(ax, steps, excess_test, name, colors[name])
    ax.axhline(0, ls="--", color="gray", alpha=0.4)
    ax.set_ylabel("Excess Test MSE", fontsize=11)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_title("(b) Excess Test MSE (Generalization)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Parameter recovery error
    ax = axes[1, 0]
    for name, res_list in results.items():
        perr_data = to_stack(res_list, "param_errors")
        plot_band(ax, steps, perr_data, name, colors[name])
    ax.set_ylabel("||w - w*||", fontsize=11)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_title("(c) Parameter Recovery Error", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Gate Dynamics (MARSSNR+Caution vs SNRAdamW)
    ax = axes[1, 1]
    snr_sg = to_stack(results["SNRAdamW"], "signal_gates")
    snr_ng = to_stack(results["SNRAdamW"], "noise_gates")
    mars_sg = to_stack(results["MARSSNR+Caution"], "signal_gates")
    mars_ng = to_stack(results["MARSSNR+Caution"], "noise_gates")
    
    plot_band(ax, steps, snr_sg, "SNRAdamW: Signal", "tab:purple")
    plot_band(ax, steps, snr_ng, "SNRAdamW: Noise", "tab:purple", alpha=0.08)
    
    plot_band(ax, steps, mars_sg, "MARS+Caution: Signal", "tab:blue")
    plot_band(ax, steps, mars_ng, "MARS+Caution: Noise", "tab:blue", alpha=0.08)
    
    ax.set_ylabel("Mean Gate Value", fontsize=11)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_title("(d) SNR Gate Dynamics (Signal vs Noise)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_mars_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Figure 2: Weight scatter comparison (seed 0) ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Weight Recovery Scatter Plot (Seed 0)", fontsize=14, fontweight="bold")
    
    plot_order = [
        ("AdamW", axes[0, 0]),
        ("SNRAdamW", axes[0, 1]),
        ("MARSSNR", axes[1, 0]),
        ("MARSSNR+Caution", axes[1, 1]),
    ]
    
    for name, ax in plot_order:
        w = results[name][0].final_weights
        ax.scatter(w_true[noise_mask].numpy(), w[noise_mask].numpy(),
                   alpha=0.3, s=12, color="gray", label="Noise features")
        ax.scatter(w_true[signal_idx].numpy(), w[signal_idx].numpy(),
                   alpha=0.9, s=50, color=colors[name], label="Signal features", zorder=5)
        
        lim = max(w_true.abs().max(), w.abs().max()) * 1.15
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="Ideal (y=x)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("True weight", fontsize=10)
        ax.set_ylabel("Learned weight", fontsize=10)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.set_aspect("equal")
        
        noise_norm = w[noise_mask].norm().item()
        signal_err = (w[signal_idx] - w_true[signal_idx]).norm().item()
        ax.text(0.04, 0.96,
                f"Noise ||w|| = {noise_norm:.3f}\nSignal error = {signal_err:.3f}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_mars_weights.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Figure 3: Summary bar charts ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Final Step Performance Summary (Mean +/- Std Dev)", fontsize=14, fontweight="bold")

    names = ["AdamW", "SNRAdamW", "MARSSNR", "MARSSNR+Caution"]
    bar_colors = [colors[n] for n in names]

    # 1. Final Excess Test MSE
    ax = axes[0]
    final_ex_vals = []
    final_ex_stds = []
    for name in names:
        test_stack = to_stack(results[name], "test_losses")[:, -1]
        ex_stack = test_stack - irreducible
        final_ex_vals.append(ex_stack.mean().item())
        final_ex_stds.append(ex_stack.std().item())
    
    ax.bar(names, final_ex_vals, yerr=final_ex_stds, color=bar_colors, capsize=8, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Excess Test MSE", fontsize=11)
    ax.set_title("(a) Final Excess Test MSE", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_xticklabels(names, rotation=15)

    # 2. Final Parameter Error
    ax = axes[1]
    final_pe_vals = []
    final_pe_stds = []
    for name in names:
        pe_stack = to_stack(results[name], "param_errors")[:, -1]
        final_pe_vals.append(pe_stack.mean().item())
        final_pe_stds.append(pe_stack.std().item())
        
    ax.bar(names, final_pe_vals, yerr=final_pe_stds, color=bar_colors, capsize=8, edgecolor="black", alpha=0.85)
    ax.set_ylabel("||w - w*||", fontsize=11)
    ax.set_title("(b) Final Parameter Error", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_xticklabels(names, rotation=15)

    # 3. Final Noise Feature Mass
    ax = axes[2]
    final_nm_vals = []
    final_nm_stds = []
    for name in names:
        nm_seeds = []
        for r in results[name]:
            nm_seeds.append(r.final_weights[noise_mask].norm().item())
        nm_tensor = torch.tensor(nm_seeds)
        final_nm_vals.append(nm_tensor.mean().item())
        final_nm_stds.append(nm_tensor.std().item())

    ax.bar(names, final_nm_vals, yerr=final_nm_stds, color=bar_colors, capsize=8, edgecolor="black", alpha=0.85)
    ax.set_ylabel("||w_noise||", fontsize=11)
    ax.set_title("(c) Stored Noise Feature Mass", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_xticklabels(names, rotation=15)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_mars_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = BenchmarkConfig()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
    results = run_experiment(cfg)
    make_figures(results, cfg, out_dir)
    print("All benchmark visualizations created and saved successfully.")
