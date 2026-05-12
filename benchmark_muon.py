"""
Synthetic benchmark: SNRMuon vs SNRAdamW vs AdamW on sparse regression.

Demonstrates the SNR+Muon hybrid optimizer on 2D weight matrices,
comparing Newton-Schulz orthogonalization modes (pre/post gating)
against the standard SNRAdamW and plain AdamW baselines.

Problem setup mirrors benchmark.py but uses a two-layer linear network
so the 2D weight matrices trigger Muon's orthogonalization path.

Outputs PNG figures in the benchmarks/ directory alongside existing plots.
"""

import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_grad import SNRAdamW, SNRMuon, compute_gate, resolve_alpha


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MuonBenchmarkConfig:
    d_in: int = 50
    d_hidden: int = 50
    d_out: int = 1
    k: int = 5          # sparse signal features
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
# Model: two-layer linear (so hidden layer is a 2D matrix for Muon)
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
    w_true, signal_idx = make_true_weights(cfg.d_in, cfg.k, cfg.signal_magnitude)

    train_gen = torch.Generator().manual_seed(seed)
    X_train, y_train = make_dataset(w_true, cfg.n_train, cfg.sigma_noise, train_gen)
    test_gen = torch.Generator().manual_seed(9999)
    X_test, y_test = make_dataset(w_true, cfg.test_size, cfg.sigma_noise, test_gen)

    torch.manual_seed(seed + 1000)
    model = TwoLayerLinear(cfg.d_in, cfg.d_hidden, cfg.d_out)
    # Small init for stability
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
        "SNRMuon (post)": lambda params: SNRMuon(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
            gate="snr", rho=0.99, alpha="finite",
            batch_size=cfg.batch_size, dataset_size=cfg.n_train,
            muon_mode="post",
        ),
        "SNRMuon (pre)": lambda params: SNRMuon(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
            gate="snr", rho=0.99, alpha="finite",
            batch_size=cfg.batch_size, dataset_size=cfg.n_train,
            muon_mode="pre",
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
    "SNRMuon (post)": "tab:green",
    "SNRMuon (pre)": "tab:purple",
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

    # ---- Figure 1: Training curves comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"SNRMuon vs SNRAdamW vs AdamW: Two-Layer Linear Network\n"
        f"(d_in={cfg.d_in}, d_hidden={cfg.d_hidden}, k={cfg.k}, "
        f"n={cfg.n_train}, noise={cfg.sigma_noise}, {cfg.n_seeds} seeds)",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    for name, results in all_results.items():
        data = to_stack(results, "train_losses")
        plot_band(ax, steps, data, name, COLORS[name])
    ax.axhline(irreducible, ls="--", color="gray", alpha=0.6,
               label=f"Irreducible ({irreducible:.0f})")
    ax.set_ylabel("Train MSE")
    ax.set_xlabel("Step")
    ax.set_title("(a) Train Loss")
    ax.legend(fontsize=9)

    ax = axes[1]
    for name, results in all_results.items():
        data = to_stack(results, "test_losses") - irreducible
        plot_band(ax, steps, data, name, COLORS[name])
    ax.axhline(0, ls="--", color="gray", alpha=0.4)
    ax.set_ylabel("Excess Test MSE")
    ax.set_xlabel("Step")
    ax.set_title("(b) Excess Test MSE (lower is better)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_muon_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 2: Summary bars (final metrics) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Final Metrics - SNRMuon Variants vs Baselines\n"
        f"(mean +/- std, {cfg.n_seeds} seeds)",
        fontsize=12, fontweight="bold",
    )

    names = list(all_results.keys())
    colors = [COLORS[n] for n in names]

    # Final excess test MSE
    ax = axes[0]
    means = []
    stds = []
    for name in names:
        final_excess = to_stack(all_results[name], "test_losses")[:, -1] - irreducible
        means.append(final_excess.mean().item())
        stds.append(final_excess.std().item())
    ax.bar(names, means, yerr=stds, color=colors, capsize=8)
    ax.set_ylabel("Excess Test MSE")
    ax.set_title("(a) Final Excess Test MSE")
    ax.tick_params(axis='x', rotation=15)

    # Final train loss
    ax = axes[1]
    means = []
    stds = []
    for name in names:
        final_train = to_stack(all_results[name], "train_losses")[:, -1]
        means.append(final_train.mean().item())
        stds.append(final_train.std().item())
    ax.bar(names, means, yerr=stds, color=colors, capsize=8)
    ax.axhline(irreducible, ls="--", color="gray", alpha=0.6,
               label=f"Irreducible ({irreducible:.0f})")
    ax.set_ylabel("Train MSE")
    ax.set_title("(b) Final Train MSE")
    ax.tick_params(axis='x', rotation=15)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_muon_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 3: Convergence speed (steps to reach threshold) ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle(
        "Convergence Speed: Steps to Reach 2x Irreducible Test MSE",
        fontsize=12, fontweight="bold",
    )

    threshold = 2.0 * irreducible
    convergence_steps = {}
    for name in names:
        test_data = to_stack(all_results[name], "test_losses")
        steps_to_converge = []
        for i in range(test_data.shape[0]):
            reached = (test_data[i] <= threshold).nonzero(as_tuple=True)[0]
            if len(reached) > 0:
                steps_to_converge.append(reached[0].item() * eval_every)
            else:
                steps_to_converge.append(cfg.n_steps)
        convergence_steps[name] = torch.tensor(steps_to_converge, dtype=torch.float)

    means = [convergence_steps[n].mean().item() for n in names]
    stds = [convergence_steps[n].std().item() for n in names]
    ax.bar(names, means, yerr=stds, color=colors, capsize=8)
    ax.set_ylabel("Steps")
    ax.set_title(f"Steps to reach test MSE <= {threshold:.0f} (2x irreducible)")
    ax.tick_params(axis='x', rotation=15)
    ax.axhline(cfg.n_steps, ls="--", color="red", alpha=0.4,
               label=f"Max steps ({cfg.n_steps})")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_muon_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = MuonBenchmarkConfig()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")

    print("SNRMuon benchmark: Two-layer linear network (sparse regression)")
    print(f"  d_in={cfg.d_in}, d_hidden={cfg.d_hidden}, k={cfg.k}, "
          f"n_train={cfg.n_train}, noise={cfg.sigma_noise}, "
          f"batch={cfg.batch_size}, steps={cfg.n_steps}, seeds={cfg.n_seeds}")
    print()

    print("Running experiments...")
    all_results = run_all(cfg)
    print()

    print("Generating figures...")
    make_figures(all_results, cfg, out_dir)
    print()

    irreducible = cfg.sigma_noise ** 2
    print("Final excess test MSE:")
    for name, results in all_results.items():
        final_ex = to_stack(results, "test_losses")[:, -1] - irreducible
        print(f"  {name:20s}: {final_ex.mean():.4f} +/- {final_ex.std():.4f}")
