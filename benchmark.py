"""
Synthetic benchmark: SNRAdamW vs AdamW on sparse linear regression with label noise.

Problem setup (finite-dataset, overparameterized regime):
  - d=200 features, only k=5 have nonzero true weights
  - Fixed training set of n=100 samples with high label noise
  - Minibatch size 32, trained for 5000 steps (~1600 epochs)
  - AdamW can memorize noise on the 195 irrelevant features
  - SNRAdamW with alpha=b/(n-b) should gate those parameters

Outputs three PNG figures in the benchmarks/ directory.
"""

import os
import argparse
import json
import csv
from dataclasses import dataclass, field
from pathlib import Path

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
    n_steps: int = 5000
    n_seeds: int = 10
    test_size: int = 10000
    lr: float = 3e-3
    weight_decay: float = 0.0  # no WD so the gate must do the work
    signal_magnitude: float = 3.0
    rho: float = 0.99
    alpha: object = "finite"
    lambda_pop: float = 1.0


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

            if track_gates and hasattr(optimizer, "last_stats") and optimizer.last_stats is not None:
                state = optimizer.state[model.weight]
                step_num = state["step"]
                betas = opt_kwargs.get("betas", (0.9, 0.999))
                rho = opt_kwargs.get("rho", 0.99)
                m_hat = state["exp_avg"].squeeze() / (1 - betas[0] ** step_num)
                s_hat = state["exp_grad_var"].squeeze() / (1 - rho ** step_num)

                from snr_grad import compute_gate, resolve_alpha
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
                    gate=opt_kwargs.get("gate", "soft"),
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

def run_experiment(cfg, gate_type="snr"):
    snr_kwargs = dict(
        lr=cfg.lr, weight_decay=cfg.weight_decay, gate=gate_type, rho=cfg.rho,
        alpha=cfg.alpha, batch_size=cfg.batch_size, dataset_size=cfg.n_train,
        lambda_pop=cfg.lambda_pop,
    )
    adam_kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)

    snr_results, adam_results = [], []
    for seed in range(cfg.n_seeds):
        print(f"  Seed {seed+1}/{cfg.n_seeds}...", end=" ", flush=True)
        snr = run_one_seed(SNRAdamW, snr_kwargs, cfg, seed, track_gates=True)
        adam = run_one_seed(torch.optim.AdamW, adam_kwargs, cfg, seed)
        snr_results.append(snr)
        adam_results.append(adam)
        print(f"SNR test={snr.test_losses[-1]:.2f}  Adam test={adam.test_losses[-1]:.2f}")
    return snr_results, adam_results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def to_stack(results, attr):
    return torch.tensor([getattr(r, attr) for r in results])


def plot_band(ax, steps, data, label, color, alpha=0.25):
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    ax.plot(steps, mean, label=label, color=color, linewidth=1.5)
    ax.fill_between(steps, (mean - std).numpy(), (mean + std).numpy(),
                    color=color, alpha=alpha)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(snr_results, adam_results, cfg, out_dir, gate_type="snr"):
    os.makedirs(out_dir, exist_ok=True)
    eval_every = 10
    steps = list(range(0, cfg.n_steps, eval_every))
    irreducible = cfg.sigma_noise ** 2
    w_true, signal_idx = make_true_weights(cfg.d, cfg.k, cfg.signal_magnitude)
    noise_mask = torch.ones(cfg.d, dtype=torch.bool)
    noise_mask[signal_idx] = False

    snr_test = to_stack(snr_results, "test_losses")
    adam_test = to_stack(adam_results, "test_losses")
    snr_train = to_stack(snr_results, "train_losses")
    adam_train = to_stack(adam_results, "train_losses")
    snr_perr = to_stack(snr_results, "param_errors")
    adam_perr = to_stack(adam_results, "param_errors")

    gate_label = gate_type.upper()
    snr_name = f"SNRAdamW ({gate_label})"
    suffix = f"_{gate_type}"

    # ---- Figure 1: Main 2x2 comparison ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"{snr_name} vs AdamW: Sparse Regression  "
        f"(d={cfg.d}, k={cfg.k}, n={cfg.n_train}, noise={cfg.sigma_noise}, "
        f"{cfg.n_seeds} seeds)",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0, 0]
    plot_band(ax, steps, snr_train, snr_name, "tab:blue")
    plot_band(ax, steps, adam_train, "AdamW", "tab:orange")
    ax.axhline(irreducible, ls="--", color="gray", alpha=0.6,
               label=f"Irreducible ({irreducible:.0f})")
    ax.set_ylabel("Train MSE")
    ax.set_xlabel("Step")
    ax.set_title("(a) Train Loss")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    snr_excess = snr_test - irreducible
    adam_excess = adam_test - irreducible
    plot_band(ax, steps, snr_excess, snr_name, "tab:blue")
    plot_band(ax, steps, adam_excess, "AdamW", "tab:orange")
    ax.axhline(0, ls="--", color="gray", alpha=0.4)
    ax.set_ylabel("Excess Test MSE")
    ax.set_xlabel("Step")
    ax.set_title("(b) Excess Test MSE (lower is better)")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    plot_band(ax, steps, snr_perr, snr_name, "tab:blue")
    plot_band(ax, steps, adam_perr, "AdamW", "tab:orange")
    ax.set_ylabel("||w - w*||")
    ax.set_xlabel("Step")
    ax.set_title("(c) Parameter Recovery Error")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    snr_sg = to_stack(snr_results, "signal_gates")
    snr_ng = to_stack(snr_results, "noise_gates")
    plot_band(ax, steps, snr_sg, f"Signal params (k={cfg.k})", "tab:green")
    plot_band(ax, steps, snr_ng, f"Noise params ({cfg.d - cfg.k})", "tab:red")
    ax.set_ylabel("Mean Gate Value")
    ax.set_xlabel("Step")
    ax.set_title(f"(d) {snr_name} Gate Values")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(out_dir, f"benchmark_main{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 2: Weight scatter (seed 0) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Weight Recovery - {snr_name} vs AdamW (seed 0)",
                 fontsize=13, fontweight="bold")
    for ax, res, name, color in [
        (axes[0], snr_results[0], snr_name, "tab:blue"),
        (axes[1], adam_results[0], "AdamW", "tab:orange"),
    ]:
        w = res.final_weights
        ax.scatter(w_true[noise_mask].numpy(), w[noise_mask].numpy(),
                   alpha=0.3, s=8, color="gray", label="Noise features")
        ax.scatter(w_true[signal_idx].numpy(), w[signal_idx].numpy(),
                   alpha=0.9, s=40, color=color, label="Signal features", zorder=5)
        lim = max(w_true.abs().max(), w.abs().max()) * 1.15
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y=x")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("True weight"); ax.set_ylabel("Learned weight")
        ax.set_title(name); ax.legend(fontsize=9); ax.set_aspect("equal")
        noise_norm = w[noise_mask].norm().item()
        signal_err = (w[signal_idx] - w_true[signal_idx]).norm().item()
        ax.text(0.02, 0.98,
                f"Noise ||w||={noise_norm:.3f}\nSignal err={signal_err:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = os.path.join(out_dir, f"benchmark_weights{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 3: Summary bars ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(f"Final Metrics - {snr_name} vs AdamW (mean +/- std, {cfg.n_seeds} seeds)",
                 fontsize=13, fontweight="bold")

    snr_final_ex = snr_excess[:, -1]
    adam_final_ex = adam_excess[:, -1]
    ax = axes[0]
    ax.bar([snr_name, "AdamW"],
           [snr_final_ex.mean(), adam_final_ex.mean()],
           yerr=[snr_final_ex.std(), adam_final_ex.std()],
           color=["tab:blue", "tab:orange"], capsize=8)
    ax.set_ylabel("Excess Test MSE")
    ax.set_title("(a) Excess Test MSE")

    snr_pe = snr_perr[:, -1]; adam_pe = adam_perr[:, -1]
    ax = axes[1]
    ax.bar([snr_name, "AdamW"],
           [snr_pe.mean(), adam_pe.mean()],
           yerr=[snr_pe.std(), adam_pe.std()],
           color=["tab:blue", "tab:orange"], capsize=8)
    ax.set_ylabel("||w - w*||")
    ax.set_title("(b) Parameter Error")

    snr_nm = torch.tensor([r.final_weights[noise_mask].norm().item() for r in snr_results])
    adam_nm = torch.tensor([r.final_weights[noise_mask].norm().item() for r in adam_results])
    ax = axes[2]
    ax.bar([snr_name, "AdamW"],
           [snr_nm.mean(), adam_nm.mean()],
           yerr=[snr_nm.std(), adam_nm.std()],
           color=["tab:blue", "tab:orange"], capsize=8)
    ax.set_ylabel("||w_noise||")
    ax.set_title("(c) Noise Feature Mass")

    plt.tight_layout()
    path = os.path.join(out_dir, f"benchmark_summary{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-config", type=str, default=None)
    ap.add_argument("--sweep-out", type=str, default=None)
    args = ap.parse_args()

    cfg = BenchmarkConfig()
    sweep_cfg = {}
    if args.sweep_config:
        sweep_cfg = json.loads(Path(args.sweep_config).read_text())
        cfg.n_seeds = 1
        cfg.n_steps = int(sweep_cfg.get("n_steps", cfg.n_steps))
        cfg.lr = float(sweep_cfg.get("lr", cfg.lr))
        cfg.weight_decay = float(sweep_cfg.get("weight_decay", cfg.weight_decay))
        cfg.batch_size = int(sweep_cfg.get("batch_size", cfg.batch_size))
        cfg.rho = float(sweep_cfg.get("rho", cfg.rho))
        cfg.lambda_pop = float(sweep_cfg.get("lambda_pop", cfg.lambda_pop))
        if "alpha" in sweep_cfg:
            a = sweep_cfg["alpha"]
            cfg.alpha = a if isinstance(a, str) else float(a)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
    gate_list = [sweep_cfg.get("gate", "snr")] if sweep_cfg else ["snr", "soft"]
    irreducible = cfg.sigma_noise ** 2
    summary_rows = []

    for gate_type in gate_list:
        snr_results, adam_results = run_experiment(cfg, gate_type=gate_type)
        if not sweep_cfg:
            make_figures(snr_results, adam_results, cfg, out_dir, gate_type=gate_type)
        snr_ex = to_stack(snr_results, "test_losses")[:, -1] - irreducible
        adam_ex = to_stack(adam_results, "test_losses")[:, -1] - irreducible
        summary_rows.append({"optimizer": f"SNRAdamW-{gate_type}", "final_excess_test_mse": float(snr_ex.mean().item())})
        summary_rows.append({"optimizer": "AdamW", "final_excess_test_mse": float(adam_ex.mean().item())})

    if args.sweep_out:
        with open(args.sweep_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["optimizer", "final_excess_test_mse"])
            w.writeheader(); w.writerows(summary_rows)
