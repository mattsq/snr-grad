"""
Synthetic benchmark: RotatedSNRAdamW & SpectralSNRMuon vs SNRAdamW vs AdamW.

Evaluates the new matrix-basis optimizers introduced in this branch:
  - RotatedSNRAdamW: SOAP-style eigenbasis rotation with SNR gating
  - SpectralSNRMuon (diag): SVD-basis SNR gating with diagonal coefficients
  - SpectralSNRMuon (full): SVD-basis SNR gating with full spectral coefficients

Problem setup mirrors benchmark_muon.py: a two-layer linear network on sparse
regression so 2D weight matrices activate the spectral/rotated code paths.

Outputs PNG figures in the benchmarks/ directory.
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

from snr_grad import SNRAdamW, RotatedSNRAdamW, SpectralSNRMuon


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SpectralBenchmarkConfig:
    d_in: int = 50
    d_hidden: int = 50
    d_out: int = 1
    k: int = 5
    n_train: int = 100
    batch_size: int = 32
    sigma_noise: float = 3.0
    n_steps: int = 5000
    n_seeds: int = 10
    test_size: int = 10000
    lr: float = 3e-3
    weight_decay: float = 0.0
    signal_magnitude: float = 3.0


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
# Model: two-layer linear (2D weight matrices for spectral paths)
# ---------------------------------------------------------------------------

class TwoLayerLinear(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.layer1 = nn.Linear(d_in, d_hidden, bias=False)
        self.layer2 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        return self.layer2(self.layer1(x))


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    train_losses: list = field(default_factory=list)
    test_losses: list = field(default_factory=list)
    label: str = ""


def run_one_seed(make_optimizer_fn, cfg, seed):
    w_true, _ = make_true_weights(cfg.d_in, cfg.k, cfg.signal_magnitude)

    train_gen = torch.Generator().manual_seed(seed)
    X_train, y_train = make_dataset(w_true, cfg.n_train, cfg.sigma_noise, train_gen)
    test_gen = torch.Generator().manual_seed(9999)
    X_test, y_test = make_dataset(w_true, cfg.test_size, cfg.sigma_noise, test_gen)

    torch.manual_seed(seed + 1000)
    model = TwoLayerLinear(cfg.d_in, cfg.d_hidden, cfg.d_out)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.1)

    optimizer = make_optimizer_fn(model.parameters())
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

    return result


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_all(cfg):
    optimizers = {
        "RotatedSNRAdamW": lambda params: RotatedSNRAdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
            gate="snr", rho=0.99, alpha="online",
            basis_update_interval=50,
        ),
        "Spectral (diag)": lambda params: SpectralSNRMuon(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
            gate="snr", rho=0.99, alpha="online",
            variant="adam_spectral_gate", mode="diag",
        ),
        "Spectral (full)": lambda params: SpectralSNRMuon(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
            gate="snr", rho=0.99, alpha="online",
            variant="adam_spectral_gate", mode="full",
        ),
        "Spectral Muon (diag)": lambda params: SpectralSNRMuon(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
            gate="snr", rho=0.99, alpha="online",
            variant="muon_spectral_gate", mode="diag",
        ),
        "SNRAdamW": lambda params: SNRAdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
            gate="snr", rho=0.99, alpha="finite",
            batch_size=cfg.batch_size, dataset_size=cfg.n_train,
        ),
        "AdamW": lambda params: torch.optim.AdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
        ),
    }

    all_results = {name: [] for name in optimizers}

    for seed in range(cfg.n_seeds):
        print(f"  Seed {seed+1}/{cfg.n_seeds}...", end=" ", flush=True)
        final_tests = []
        for name, make_opt in optimizers.items():
            r = run_one_seed(make_opt, cfg, seed)
            r.label = name
            all_results[name].append(r)
            final_tests.append(f"{name}={r.test_losses[-1]:.2f}")
        print("  ".join(final_tests))

    return all_results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = {
    "RotatedSNRAdamW": "tab:red",
    "Spectral (diag)": "tab:green",
    "Spectral (full)": "tab:purple",
    "Spectral Muon (diag)": "tab:cyan",
    "SNRAdamW": "tab:blue",
    "AdamW": "tab:orange",
}


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

def make_figures(all_results, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    eval_every = 10
    steps = list(range(0, cfg.n_steps, eval_every))
    irreducible = cfg.sigma_noise ** 2
    names = list(all_results.keys())

    # ---- Figure 1: Training & test curves (2x1) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "RotatedSNRAdamW & SpectralSNRMuon vs Baselines: Two-Layer Linear\n"
        f"(d_in={cfg.d_in}, d_hidden={cfg.d_hidden}, k={cfg.k}, "
        f"n={cfg.n_train}, noise={cfg.sigma_noise}, {cfg.n_seeds} seeds)",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    for name in names:
        data = to_stack(all_results[name], "train_losses")
        plot_band(ax, steps, data, name, COLORS[name])
    ax.axhline(irreducible, ls="--", color="gray", alpha=0.6,
               label=f"Irreducible ({irreducible:.0f})")
    ax.set_ylabel("Train MSE")
    ax.set_xlabel("Step")
    ax.set_title("(a) Train Loss")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for name in names:
        data = to_stack(all_results[name], "test_losses") - irreducible
        plot_band(ax, steps, data, name, COLORS[name])
    ax.axhline(0, ls="--", color="gray", alpha=0.4)
    ax.set_ylabel("Excess Test MSE")
    ax.set_xlabel("Step")
    ax.set_title("(b) Excess Test MSE (lower is better)")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_spectral_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 2: Summary bars (final metrics) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Final Metrics - Spectral/Rotated Variants vs Baselines\n"
        f"(mean +/- std, {cfg.n_seeds} seeds)",
        fontsize=12, fontweight="bold",
    )

    colors = [COLORS[n] for n in names]

    ax = axes[0]
    means, stds = [], []
    for name in names:
        final_excess = to_stack(all_results[name], "test_losses")[:, -1] - irreducible
        means.append(final_excess.mean().item())
        stds.append(final_excess.std().item())
    ax.bar(names, means, yerr=stds, color=colors, capsize=6)
    ax.set_ylabel("Excess Test MSE")
    ax.set_title("(a) Final Excess Test MSE")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1]
    means, stds = [], []
    for name in names:
        final_train = to_stack(all_results[name], "train_losses")[:, -1]
        means.append(final_train.mean().item())
        stds.append(final_train.std().item())
    ax.bar(names, means, yerr=stds, color=colors, capsize=6)
    ax.axhline(irreducible, ls="--", color="gray", alpha=0.6,
               label=f"Irreducible ({irreducible:.0f})")
    ax.set_ylabel("Train MSE")
    ax.set_title("(b) Final Train MSE")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_spectral_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 3: Convergence speed ----
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    fig.suptitle(
        "Convergence Speed: Steps to Reach 2x Irreducible Test MSE",
        fontsize=12, fontweight="bold",
    )

    threshold = 2.0 * irreducible
    convergence_means, convergence_stds = [], []
    for name in names:
        test_data = to_stack(all_results[name], "test_losses")
        steps_to_converge = []
        for i in range(test_data.shape[0]):
            reached = (test_data[i] <= threshold).nonzero(as_tuple=True)[0]
            if len(reached) > 0:
                steps_to_converge.append(reached[0].item() * eval_every)
            else:
                steps_to_converge.append(cfg.n_steps)
        conv = torch.tensor(steps_to_converge, dtype=torch.float)
        convergence_means.append(conv.mean().item())
        convergence_stds.append(conv.std().item())

    ax.bar(names, convergence_means, yerr=convergence_stds, color=colors, capsize=6)
    ax.set_ylabel("Steps")
    ax.set_title(f"Steps to reach test MSE <= {threshold:.0f} (2x irreducible)")
    ax.tick_params(axis="x", rotation=25)
    ax.axhline(cfg.n_steps, ls="--", color="red", alpha=0.4,
               label=f"Max steps ({cfg.n_steps})")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_spectral_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 4: Relative improvement heatmap ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(
        "Relative Excess Test MSE vs AdamW Baseline\n"
        f"(per-seed ratio, {cfg.n_seeds} seeds)",
        fontsize=12, fontweight="bold",
    )

    adamw_final = to_stack(all_results["AdamW"], "test_losses")[:, -1] - irreducible
    comparison_names = [n for n in names if n != "AdamW"]
    ratios = []
    for name in comparison_names:
        final_excess = to_stack(all_results[name], "test_losses")[:, -1] - irreducible
        # ratio < 1 means better than AdamW
        ratio = final_excess / (adamw_final + 1e-8)
        ratios.append(ratio)

    ratio_matrix = torch.stack(ratios)
    im = ax.imshow(ratio_matrix.numpy(), aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=2.0)
    ax.set_yticks(range(len(comparison_names)))
    ax.set_yticklabels(comparison_names)
    ax.set_xticks(range(cfg.n_seeds))
    ax.set_xticklabels([f"Seed {i}" for i in range(cfg.n_seeds)], fontsize=8)
    ax.set_xlabel("Seed")
    ax.set_title("Excess Test MSE / AdamW Excess (green < 1 = better)")
    fig.colorbar(im, ax=ax, label="Ratio vs AdamW")

    for i in range(len(comparison_names)):
        for j in range(cfg.n_seeds):
            val = ratio_matrix[i, j].item()
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if val > 1.3 or val < 0.3 else "black")

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_spectral_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sweep-config", type=str, default=None); ap.add_argument("--sweep-out", type=str, default=None); args = ap.parse_args()
    cfg = SpectralBenchmarkConfig(); sweep_cfg = {}
    if args.sweep_config:
        sweep_cfg = json.loads(Path(args.sweep_config).read_text()); cfg.n_seeds=1; cfg.n_steps=int(sweep_cfg.get("n_steps", cfg.n_steps)); cfg.lr=float(sweep_cfg.get("lr", cfg.lr)); cfg.weight_decay=float(sweep_cfg.get("weight_decay", cfg.weight_decay)); cfg.batch_size=int(sweep_cfg.get("batch_size", cfg.batch_size))
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
    all_results = run_all(cfg)
    if not sweep_cfg: make_figures(all_results, cfg, out_dir)
    irreducible = cfg.sigma_noise ** 2
    rows=[]
    for name, results in all_results.items():
        final_ex = to_stack(results, "test_losses")[:, -1] - irreducible
        rows.append({"optimizer":name, "final_excess_test_mse": float(final_ex.mean().item())})
    if args.sweep_out:
        with open(args.sweep_out, "w", newline="") as f:
            w=csv.DictWriter(f, fieldnames=["optimizer","final_excess_test_mse"]); w.writeheader(); w.writerows(rows)
