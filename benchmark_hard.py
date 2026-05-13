"""
Stress-test benchmark: Low-rank matrix recovery with anisotropic inputs.

Compares spectral/rotated SNR optimizers against per-coordinate baselines
on a task designed to reveal *when* matrix-basis gating helps vs hurts.

Setup:
  - Single linear layer: y = W* @ x + noise, W* is rank-k (k=5) in R^{dxd}
  - W* = U_k @ diag(s) @ V_k^T with random orthogonal U_k, V_k
  - Inputs x ~ N(0, Sigma) where Sigma = Q @ diag(lambda) @ Q^T
    with condition number ~100 and random rotation Q
  - Two conditions: "aligned" (axis-aligned W*) vs "rotated" (random W*)

Key findings:
  Aligned + anisotropic inputs: RotatedSNRAdamW >> SNRAdamW >> AdamW.
    The eigenbasis rotation compensates for the input covariance mismatch,
    correctly preconditioning gradient noise that is non-uniform across
    coordinates due to the anisotropic Sigma.

  Rotated (dense signal): AdamW ~ SNRAdamW >> RotatedSNRAdamW >> Spectral.
    When the rotation makes W* dense (all entries nonzero), per-coordinate
    methods correctly treat all entries as having signal. Spectral gating
    is too aggressive in suppressing the (d-k) noise singular directions
    that, in the standard basis, contribute to all matrix entries.

This delineates the regime where matrix-basis methods add value: problems
with structured sparsity in the gradient covariance eigenbasis, not
problems where signal is uniformly distributed across parameters.

Outputs PNG figures in the benchmarks/ directory.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snr_grad import SNRAdamW, RotatedSNRAdamW, SpectralSNRMuon


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HardBenchmarkConfig:
    d: int = 100            # matrix dimension (d x d weight)
    rank: int = 5           # true rank of W*
    n_train: int = 500      # training samples
    batch_size: int = 64
    sigma_noise: float = 1.0
    n_steps: int = 8000
    n_seeds: int = 8
    test_size: int = 5000
    lr: float = 1e-3
    weight_decay: float = 0.0
    cond_number: float = 100.0   # condition number of input covariance
    singular_values: tuple = (10.0, 7.0, 5.0, 3.0, 2.0)  # true singular values
    eval_every: int = 20


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def random_orthogonal(n, k, gen):
    """Sample k orthonormal columns in R^n."""
    Z = torch.randn(n, k, generator=gen)
    Q, _ = torch.linalg.qr(Z)
    return Q


def make_true_weight(d, rank, singular_values, aligned=False, seed=0):
    """Create rank-k ground truth weight matrix."""
    gen = torch.Generator().manual_seed(seed)
    s = torch.tensor(singular_values[:rank], dtype=torch.float32)

    if aligned:
        # Axis-aligned: singular vectors are standard basis vectors
        U = torch.eye(d)[:, :rank]
        V = torch.eye(d)[:, :rank]
    else:
        U = random_orthogonal(d, rank, gen)
        V = random_orthogonal(d, rank, gen)

    W_star = U @ torch.diag(s) @ V.t()
    return W_star, U, V, s


def make_input_covariance(d, cond_number, seed=42):
    """Create anisotropic input covariance Sigma = Q @ diag(lam) @ Q^T."""
    gen = torch.Generator().manual_seed(seed)
    # Log-spaced eigenvalues from 1 to cond_number
    lam = torch.logspace(0, np.log10(cond_number), d)
    # Random rotation
    Q = random_orthogonal(d, d, gen)
    # Sigma = Q @ diag(lam) @ Q^T, but we store the sqrt for sampling
    sqrt_lam = lam.sqrt()
    # x = Q @ diag(sqrt_lam) @ z, z ~ N(0, I)
    transform = Q @ torch.diag(sqrt_lam)  # d x d
    return transform, lam, Q


def make_dataset(W_star, transform, n, sigma_noise, gen):
    """Generate y = W* @ x + noise with x ~ N(0, Sigma)."""
    d = W_star.shape[0]
    Z = torch.randn(n, d, generator=gen)
    X = Z @ transform.t()  # X[i] = transform @ z[i], so Cov(x) = transform @ transform^T
    Y = X @ W_star.t()  # (n, d) @ (d, d)^T -> (n, d)
    noise = torch.randn_like(Y) * sigma_noise
    return X, Y + noise


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def frobenius_error(W_learned, W_star):
    """Relative Frobenius error."""
    return (W_learned - W_star).norm() / (W_star.norm() + 1e-12)


def subspace_alignment(W_learned, U_true, V_true, k):
    """
    Measure alignment between top-k singular subspaces.
    Returns (left_align, right_align) in [0, 1] where 1 = perfect alignment.
    """
    try:
        U_l, _, Vh_l = torch.linalg.svd(W_learned, full_matrices=False)
    except Exception:
        return 0.0, 0.0

    U_hat = U_l[:, :k]
    V_hat = Vh_l[:k, :].t()

    # Alignment = ||U_true^T @ U_hat||_F^2 / k (fraction of subspace captured)
    left = (U_true.t() @ U_hat).norm() ** 2 / k
    right = (V_true.t() @ V_hat).norm() ** 2 / k
    return left.item(), right.item()


def effective_rank(W):
    """Stable rank: ||W||_F^2 / ||W||_2^2."""
    s = torch.linalg.svdvals(W)
    if s[0] < 1e-12:
        return 0.0
    return ((s ** 2).sum() / (s[0] ** 2)).item()


def singular_value_error(W_learned, s_true, k):
    """L2 error between top-k singular values."""
    s_l = torch.linalg.svdvals(W_learned)[:k]
    return (s_l - s_true[:k]).norm().item()


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    train_losses: list = field(default_factory=list)
    test_losses: list = field(default_factory=list)
    frob_errors: list = field(default_factory=list)
    left_alignments: list = field(default_factory=list)
    right_alignments: list = field(default_factory=list)
    eff_ranks: list = field(default_factory=list)
    sv_errors: list = field(default_factory=list)
    final_singular_values: torch.Tensor = None
    label: str = ""


def run_one_seed(make_optimizer_fn, cfg, W_star, U_true, V_true, s_true, transform, seed):
    gen_train = torch.Generator().manual_seed(seed)
    X_train, Y_train = make_dataset(W_star, transform, cfg.n_train, cfg.sigma_noise, gen_train)
    gen_test = torch.Generator().manual_seed(9999)
    X_test, Y_test = make_dataset(W_star, transform, cfg.test_size, cfg.sigma_noise, gen_test)

    torch.manual_seed(seed + 2000)
    model = nn.Linear(cfg.d, cfg.d, bias=False)
    nn.init.zeros_(model.weight)

    optimizer = make_optimizer_fn(model.parameters())
    result = RunResult()

    for step in range(cfg.n_steps):
        idx = torch.randint(cfg.n_train, (cfg.batch_size,))
        X_b, Y_b = X_train[idx], Y_train[idx]

        optimizer.zero_grad()
        pred = model(X_b)
        loss = ((pred - Y_b) ** 2).mean()
        loss.backward()
        optimizer.step()

        if step % cfg.eval_every == 0:
            with torch.no_grad():
                result.train_losses.append(loss.item())
                test_pred = model(X_test)
                result.test_losses.append(((test_pred - Y_test) ** 2).mean().item())

                W = model.weight.data
                result.frob_errors.append(frobenius_error(W, W_star).item())
                la, ra = subspace_alignment(W, U_true, V_true, cfg.rank)
                result.left_alignments.append(la)
                result.right_alignments.append(ra)
                result.eff_ranks.append(effective_rank(W))
                result.sv_errors.append(singular_value_error(W, s_true, cfg.rank))

    with torch.no_grad():
        result.final_singular_values = torch.linalg.svdvals(model.weight.data)

    return result


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def get_optimizers(cfg):
    return {
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
        "SNRAdamW": lambda params: SNRAdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
            gate="snr", rho=0.99, alpha="online",
        ),
        "AdamW": lambda params: torch.optim.AdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
        ),
    }


def run_condition(cfg, aligned=False):
    """Run all optimizers on either aligned or rotated condition."""
    W_star, U_true, V_true, s_true = make_true_weight(
        cfg.d, cfg.rank, cfg.singular_values, aligned=aligned, seed=0
    )
    transform, _, _ = make_input_covariance(cfg.d, cfg.cond_number, seed=42)

    optimizers = get_optimizers(cfg)
    all_results = {name: [] for name in optimizers}

    condition_name = "aligned" if aligned else "rotated"
    for seed in range(cfg.n_seeds):
        print(f"  [{condition_name}] Seed {seed+1}/{cfg.n_seeds}...", end=" ", flush=True)
        summaries = []
        for name, make_opt in optimizers.items():
            r = run_one_seed(make_opt, cfg, W_star, U_true, V_true, s_true, transform, seed)
            r.label = name
            all_results[name].append(r)
            summaries.append(f"{name}={r.frob_errors[-1]:.3f}")
        print("  ".join(summaries))

    return all_results, W_star, U_true, V_true, s_true


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = {
    "RotatedSNRAdamW": "tab:red",
    "Spectral (diag)": "tab:green",
    "Spectral (full)": "tab:purple",
    "SNRAdamW": "tab:blue",
    "AdamW": "tab:orange",
}


def to_stack(results, attr):
    return torch.tensor([getattr(r, attr) for r in results])


def plot_band(ax, steps, data, label, color, alpha=0.2):
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    ax.plot(steps, mean, label=label, color=color, linewidth=1.5)
    ax.fill_between(steps, (mean - std).numpy(), (mean + std).numpy(),
                    color=color, alpha=alpha)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(results_rotated, results_aligned, cfg, W_star, U_true, V_true, s_true, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    steps = list(range(0, cfg.n_steps, cfg.eval_every))
    names = list(results_rotated.keys())

    # ---- Figure 1: Rotated condition - main curves (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Low-Rank Matrix Recovery (ROTATED subspace, anisotropic inputs)\n"
        f"d={cfg.d}, rank={cfg.rank}, n={cfg.n_train}, "
        f"cond={cfg.cond_number:.0f}, noise={cfg.sigma_noise}, {cfg.n_seeds} seeds",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0, 0]
    for name in names:
        data = to_stack(results_rotated[name], "test_losses")
        plot_band(ax, steps, data, name, COLORS[name])
    ax.set_ylabel("Test MSE")
    ax.set_xlabel("Step")
    ax.set_title("(a) Test Loss")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    ax = axes[0, 1]
    for name in names:
        data = to_stack(results_rotated[name], "frob_errors")
        plot_band(ax, steps, data, name, COLORS[name])
    ax.set_ylabel("||W - W*||_F / ||W*||_F")
    ax.set_xlabel("Step")
    ax.set_title("(b) Relative Frobenius Error")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name in names:
        data = to_stack(results_rotated[name], "left_alignments")
        plot_band(ax, steps, data, name, COLORS[name])
    ax.set_ylabel("Subspace Alignment")
    ax.set_xlabel("Step")
    ax.set_title("(c) Left Singular Subspace Alignment (1=perfect)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1, 1]
    for name in names:
        data = to_stack(results_rotated[name], "eff_ranks")
        plot_band(ax, steps, data, name, COLORS[name])
    ax.axhline(cfg.rank, ls="--", color="gray", alpha=0.6, label=f"True rank ({cfg.rank})")
    ax.set_ylabel("Effective Rank")
    ax.set_xlabel("Step")
    ax.set_title("(d) Effective (Stable) Rank of Learned W")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_hardrot_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 2: Aligned vs Rotated comparison (bar chart) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Axis-Aligned vs Rotated Subspace: Final Recovery Metrics\n"
        f"(mean +/- std, {cfg.n_seeds} seeds)",
        fontsize=12, fontweight="bold",
    )

    colors = [COLORS[n] for n in names]
    x = np.arange(len(names))
    width = 0.35

    # Frobenius error
    ax = axes[0]
    aligned_means = [to_stack(results_aligned[n], "frob_errors")[:, -1].mean().item() for n in names]
    aligned_stds = [to_stack(results_aligned[n], "frob_errors")[:, -1].std().item() for n in names]
    rotated_means = [to_stack(results_rotated[n], "frob_errors")[:, -1].mean().item() for n in names]
    rotated_stds = [to_stack(results_rotated[n], "frob_errors")[:, -1].std().item() for n in names]
    bars1 = ax.bar(x - width/2, aligned_means, width, yerr=aligned_stds,
                   label="Aligned", color=colors, alpha=0.5, capsize=4, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, rotated_means, width, yerr=rotated_stds,
                   label="Rotated", color=colors, alpha=1.0, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("(a) ||W - W*||_F / ||W*||_F")
    ax.legend(["Aligned (faded)", "Rotated (solid)"], fontsize=8)

    # Subspace alignment
    ax = axes[1]
    aligned_means = [to_stack(results_aligned[n], "left_alignments")[:, -1].mean().item() for n in names]
    aligned_stds = [to_stack(results_aligned[n], "left_alignments")[:, -1].std().item() for n in names]
    rotated_means = [to_stack(results_rotated[n], "left_alignments")[:, -1].mean().item() for n in names]
    rotated_stds = [to_stack(results_rotated[n], "left_alignments")[:, -1].std().item() for n in names]
    ax.bar(x - width/2, aligned_means, width, yerr=aligned_stds,
           label="Aligned", color=colors, alpha=0.5, capsize=4, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, rotated_means, width, yerr=rotated_stds,
           label="Rotated", color=colors, alpha=1.0, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Left Subspace Alignment")
    ax.set_title("(b) Subspace Recovery (1=perfect)")
    ax.legend(["Aligned (faded)", "Rotated (solid)"], fontsize=8)
    ax.set_ylim(0, 1.1)

    # Effective rank
    ax = axes[2]
    aligned_means = [to_stack(results_aligned[n], "eff_ranks")[:, -1].mean().item() for n in names]
    aligned_stds = [to_stack(results_aligned[n], "eff_ranks")[:, -1].std().item() for n in names]
    rotated_means = [to_stack(results_rotated[n], "eff_ranks")[:, -1].mean().item() for n in names]
    rotated_stds = [to_stack(results_rotated[n], "eff_ranks")[:, -1].std().item() for n in names]
    ax.bar(x - width/2, aligned_means, width, yerr=aligned_stds,
           label="Aligned", color=colors, alpha=0.5, capsize=4, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, rotated_means, width, yerr=rotated_stds,
           label="Rotated", color=colors, alpha=1.0, capsize=4)
    ax.axhline(cfg.rank, ls="--", color="gray", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Effective Rank")
    ax.set_title(f"(c) Effective Rank (true={cfg.rank})")
    ax.legend(["Aligned (faded)", "Rotated (solid)"], fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_hardrot_aligned_vs_rotated.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 3: Singular value spectrum (rotated, seed 0) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Learned Singular Value Spectrum (rotated condition, seed 0)\n"
        f"True rank={cfg.rank}, singular values={list(cfg.singular_values[:cfg.rank])}",
        fontsize=12, fontweight="bold",
    )

    true_sv = torch.zeros(cfg.d)
    true_sv[:cfg.rank] = torch.tensor(cfg.singular_values[:cfg.rank])

    ax = axes[0]
    for name in names:
        sv = results_rotated[name][0].final_singular_values
        ax.plot(range(min(20, len(sv))), sv[:20].numpy(), "o-",
                label=name, color=COLORS[name], markersize=4, linewidth=1.5)
    ax.plot(range(min(20, cfg.d)), true_sv[:20].numpy(), "k--",
            label="True W*", linewidth=2, alpha=0.7)
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value")
    ax.set_title("(a) Top-20 Singular Values")
    ax.legend(fontsize=8)

    ax = axes[1]
    for name in names:
        sv = results_rotated[name][0].final_singular_values
        ax.semilogy(range(len(sv)), sv.numpy() + 1e-10,
                    label=name, color=COLORS[name], linewidth=1.2, alpha=0.8)
    ax.semilogy(range(cfg.d), true_sv.numpy() + 1e-10, "k--",
                label="True W*", linewidth=2, alpha=0.7)
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value (log scale)")
    ax.set_title("(b) Full Spectrum (log scale)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_hardrot_spectrum.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 4: Subspace alignment over training (rotated) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Subspace Alignment Over Training (rotated condition)\n"
        "How quickly does each optimizer discover the true singular subspace?",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    for name in names:
        data = to_stack(results_rotated[name], "left_alignments")
        plot_band(ax, steps, data, name, COLORS[name])
    ax.set_ylabel("Left Subspace Alignment")
    ax.set_xlabel("Step")
    ax.set_title("(a) Left (column) Subspace")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    for name in names:
        data = to_stack(results_rotated[name], "right_alignments")
        plot_band(ax, steps, data, name, COLORS[name])
    ax.set_ylabel("Right Subspace Alignment")
    ax.set_xlabel("Step")
    ax.set_title("(b) Right (row) Subspace")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_hardrot_subspace.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 5: Convergence advantage ratio ----
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    fig.suptitle(
        "Improvement Over AdamW: Final Frobenius Error Ratio\n"
        "(< 1 means better than AdamW, both conditions)",
        fontsize=12, fontweight="bold",
    )

    comparison_names = [n for n in names if n != "AdamW"]
    adamw_aligned = to_stack(results_aligned["AdamW"], "frob_errors")[:, -1]
    adamw_rotated = to_stack(results_rotated["AdamW"], "frob_errors")[:, -1]

    x = np.arange(len(comparison_names))
    width = 0.35

    aligned_ratios = []
    rotated_ratios = []
    for name in comparison_names:
        a = to_stack(results_aligned[name], "frob_errors")[:, -1]
        r = to_stack(results_rotated[name], "frob_errors")[:, -1]
        aligned_ratios.append((a / (adamw_aligned + 1e-8)).mean().item())
        rotated_ratios.append((r / (adamw_rotated + 1e-8)).mean().item())

    bars1 = ax.bar(x - width/2, aligned_ratios, width, label="Aligned",
                   color=[COLORS[n] for n in comparison_names], alpha=0.5,
                   edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, rotated_ratios, width, label="Rotated",
                   color=[COLORS[n] for n in comparison_names], alpha=1.0)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.6, label="AdamW baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Frobenius Error / AdamW Error")
    ax.set_title("Relative improvement (lower is better)")
    ax.legend(["Aligned", "Rotated", "AdamW = 1.0"], fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_hardrot_advantage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = HardBenchmarkConfig()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")

    print("=" * 70)
    print("HARD BENCHMARK: Low-Rank Matrix Recovery with Anisotropic Inputs")
    print("=" * 70)
    print(f"  d={cfg.d}, rank={cfg.rank}, n_train={cfg.n_train}, "
          f"cond_number={cfg.cond_number}")
    print(f"  noise={cfg.sigma_noise}, batch={cfg.batch_size}, "
          f"steps={cfg.n_steps}, seeds={cfg.n_seeds}")
    print(f"  true singular values: {cfg.singular_values[:cfg.rank]}")
    print()

    print("--- CONDITION 1: Rotated (random) subspace ---")
    results_rotated, W_star, U_true, V_true, s_true = run_condition(cfg, aligned=False)
    print()

    print("--- CONDITION 2: Axis-aligned subspace ---")
    results_aligned, _, _, _, _ = run_condition(cfg, aligned=True)
    print()

    print("Generating figures...")
    make_figures(results_rotated, results_aligned, cfg, W_star, U_true, V_true, s_true, out_dir)
    print()

    # Print summary table
    print("=" * 70)
    print("SUMMARY: Final Relative Frobenius Error (||W-W*||/||W*||)")
    print("=" * 70)
    print(f"{'Optimizer':<25s} {'Aligned':>18s} {'Rotated':>18s} {'Gap':>10s}")
    print("-" * 70)
    for name in results_rotated.keys():
        a = to_stack(results_aligned[name], "frob_errors")[:, -1]
        r = to_stack(results_rotated[name], "frob_errors")[:, -1]
        gap = r.mean() - a.mean()
        print(f"  {name:<23s} {a.mean():.4f} +/- {a.std():.4f}  "
              f"{r.mean():.4f} +/- {r.std():.4f}  {gap:+.4f}")
    print()

    print("SUMMARY: Final Left Subspace Alignment")
    print("-" * 70)
    for name in results_rotated.keys():
        a = to_stack(results_aligned[name], "left_alignments")[:, -1]
        r = to_stack(results_rotated[name], "left_alignments")[:, -1]
        print(f"  {name:<23s} {a.mean():.4f} +/- {a.std():.4f}  "
              f"{r.mean():.4f} +/- {r.std():.4f}")
