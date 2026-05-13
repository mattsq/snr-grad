"""
Phase 1: Mathematical Analysis of SNR Gate Hyperparameters.

Derives and plots closed-form properties of the gate functions without running
any optimizer. Produces figures in studies/hyperparameter_study/results/.

Covers:
  1a. Gate response surfaces (heatmaps)
  1b. Sensitivity analysis: dq/d(lambda_pop) and dq/d(alpha)
  1c. Effective threshold analysis
  1d. Rho bias-variance tradeoff

Usage:
    python studies/hyperparameter_study/gate_analysis.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Ensure snr_grad is importable from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from snr_grad import compute_gate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gate_grid(
    gate: str,
    m_range: np.ndarray,
    s_range: np.ndarray,
    alpha: float = 1.0,
    lambda_pop: float = 1.0,
) -> np.ndarray:
    """Evaluate gate on a meshgrid of (m_hat, s_hat) and return 2D array."""
    M, S = np.meshgrid(m_range, s_range, indexing="ij")
    m_flat = torch.tensor(M.ravel(), dtype=torch.float32)
    s_flat = torch.tensor(S.ravel(), dtype=torch.float32)
    q = compute_gate(m_flat, s_flat, gate=gate, alpha=alpha, lambda_pop=lambda_pop)
    return q.numpy().reshape(M.shape)


def _save(fig, name: str):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# 1a. Gate Response Surfaces
# ===================================================================

def plot_gate_surfaces():
    """2D heatmaps of q(m_hat, s_hat) for each gate type and varying lambda_pop."""
    print("1a. Gate response surfaces...")
    m_range = np.linspace(-5, 5, 200)
    s_range = np.linspace(0.01, 5, 200)

    # --- Comparison across gate types (fixed lambda_pop=1, alpha=1) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Gate Response Surface: q(m_hat, s_hat)", fontsize=14, fontweight="bold")
    for ax, gate_name in zip(axes, ["snr", "soft", "hard"]):
        Z = _gate_grid(gate_name, m_range, s_range)
        im = ax.imshow(
            Z.T, origin="lower", aspect="auto",
            extent=[m_range[0], m_range[-1], s_range[0], s_range[-1]],
            cmap="viridis", vmin=0, vmax=1,
        )
        ax.contour(
            m_range, s_range, Z.T, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
            colors="white", linewidths=0.8, linestyles="--",
        )
        ax.set_xlabel("m_hat (signal)")
        ax.set_ylabel("s_hat (noise)")
        ax.set_title(f"gate='{gate_name}'")
    fig.colorbar(im, ax=axes, label="Gate value q", shrink=0.8)
    _save(fig, "1a_gate_surfaces.png")

    # --- lambda_pop sweep for SNR gate ---
    lambdas = [0.1, 0.5, 1.0, 5.0, 10.0]
    fig, axes = plt.subplots(1, len(lambdas), figsize=(4 * len(lambdas), 4))
    fig.suptitle("SNR Gate Surface: varying lambda_pop", fontsize=14, fontweight="bold")
    for ax, lam in zip(axes, lambdas):
        Z = _gate_grid("snr", m_range, s_range, lambda_pop=lam)
        im = ax.imshow(
            Z.T, origin="lower", aspect="auto",
            extent=[m_range[0], m_range[-1], s_range[0], s_range[-1]],
            cmap="viridis", vmin=0, vmax=1,
        )
        ax.contour(
            m_range, s_range, Z.T, levels=[0.5],
            colors="red", linewidths=1.5,
        )
        ax.set_title(f"lambda_pop={lam}")
        ax.set_xlabel("m_hat")
        if lam == lambdas[0]:
            ax.set_ylabel("s_hat")
    fig.colorbar(im, ax=axes, label="q", shrink=0.8)
    _save(fig, "1a_snr_lambda_sweep.png")

    # --- alpha sweep for soft gate ---
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 4))
    fig.suptitle("Soft Gate Surface: varying alpha", fontsize=14, fontweight="bold")
    for ax, a in zip(axes, alphas):
        Z = _gate_grid("soft", m_range, s_range, alpha=a)
        im = ax.imshow(
            Z.T, origin="lower", aspect="auto",
            extent=[m_range[0], m_range[-1], s_range[0], s_range[-1]],
            cmap="viridis", vmin=0, vmax=1,
        )
        # Show the hard threshold line m^2 = alpha * s
        threshold_s = np.linspace(0.01, 5, 100)
        threshold_m = np.sqrt(a * threshold_s)
        ax.plot(threshold_m, threshold_s, "r-", linewidth=1.5, label=f"m^2=alpha*s")
        ax.plot(-threshold_m, threshold_s, "r-", linewidth=1.5)
        ax.set_title(f"alpha={a}")
        ax.set_xlabel("m_hat")
        if a == alphas[0]:
            ax.set_ylabel("s_hat")
    fig.colorbar(im, ax=axes, label="q", shrink=0.8)
    _save(fig, "1a_soft_alpha_sweep.png")


# ===================================================================
# 1b. Sensitivity Analysis
# ===================================================================

def plot_sensitivity():
    """Closed-form derivatives dq/d(lambda_pop) and dq/d(alpha)."""
    print("1b. Sensitivity analysis...")
    snr_ratio = np.linspace(0.01, 20, 500)  # m^2 / s
    s_val = 1.0  # fix s=1 so m^2 = snr_ratio

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Gate Sensitivity to Hyperparameters", fontsize=14, fontweight="bold")

    # --- dq/d(lambda) for SNR gate ---
    # q = m^2 / (m^2 + lam*s + eps), dq/dlam = -m^2 * s / (m^2 + lam*s + eps)^2
    ax = axes[0]
    for lam in [0.1, 1.0, 5.0, 10.0]:
        m2 = snr_ratio * s_val
        denom = (m2 + lam * s_val + 1e-12) ** 2
        dq_dlam = -m2 * s_val / denom
        ax.plot(snr_ratio, dq_dlam, label=f"lambda={lam}")
    ax.set_xlabel("SNR ratio (m^2 / s)")
    ax.set_ylabel("dq / d(lambda_pop)")
    ax.set_title("SNR gate: sensitivity to lambda_pop")
    ax.legend()
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.annotate("Most sensitive\nat moderate SNR", xy=(2, -0.2), fontsize=9,
                ha="center", style="italic")

    # --- dq/d(alpha) for soft gate ---
    # Above threshold (m^2 > alpha*s): delta = m^2 - alpha*s
    # q = delta / (delta + lam*s + eps)
    # dq/d(alpha) = -s * lam * s / (delta + lam*s + eps)^2  (for delta > 0)
    ax = axes[1]
    for alpha in [0.5, 1.0, 2.0]:
        lam = 1.0
        m2 = snr_ratio * s_val
        delta = np.maximum(m2 - alpha * s_val, 0)
        denom = (delta + lam * s_val + 1e-12) ** 2
        # dq/d(alpha) = d/d(alpha) [delta / (delta + C)]
        # = -s / (delta + C) - delta * (-s) / (delta + C)^2
        # = -s * C / (delta + C)^2  where C = lam*s + eps
        C = lam * s_val + 1e-12
        dq_dalpha = np.where(delta > 0, -s_val * C / (delta + C) ** 2, 0)
        ax.plot(snr_ratio, dq_dalpha, label=f"alpha={alpha}")
    ax.set_xlabel("SNR ratio (m^2 / s)")
    ax.set_ylabel("dq / d(alpha)")
    ax.set_title("Soft gate: sensitivity to alpha")
    ax.legend()
    ax.axhline(0, color="gray", ls="--", alpha=0.3)

    # --- Gate value vs SNR ratio for different lambda_pop ---
    ax = axes[2]
    for lam in [0.1, 0.5, 1.0, 5.0, 10.0]:
        m2 = snr_ratio * s_val
        q_snr = m2 / (m2 + lam * s_val + 1e-12)
        ax.plot(snr_ratio, q_snr, label=f"lambda={lam}")
    ax.set_xlabel("SNR ratio (m^2 / s)")
    ax.set_ylabel("Gate value q")
    ax.set_title("SNR gate value vs signal-to-noise")
    ax.legend()
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.annotate("q=0.5 at m^2/s = lambda", xy=(5, 0.5), fontsize=9,
                ha="left", va="bottom", style="italic")

    plt.tight_layout()
    _save(fig, "1b_sensitivity.png")


# ===================================================================
# 1c. Effective Threshold Analysis
# ===================================================================

def plot_threshold_analysis():
    """What fraction of parameters get gated under different settings."""
    print("1c. Effective threshold analysis...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Effective Gating Threshold Analysis", fontsize=14, fontweight="bold")

    # Simulate a population of parameters with varying SNR
    torch.manual_seed(42)
    n_params = 10000

    # --- Fraction gated to q < threshold vs alpha (soft gate) ---
    ax = axes[0]
    mean_snr_values = [0.5, 1.0, 2.0, 5.0]
    alphas = np.linspace(0.01, 5.0, 100)

    for mean_snr in mean_snr_values:
        # m^2 ~ Exponential with mean = mean_snr, s = 1
        m2_samples = torch.distributions.Exponential(1.0 / mean_snr).sample((n_params,))
        s_samples = torch.ones(n_params)
        fracs = []
        for a in alphas:
            q = compute_gate(m2_samples.sqrt(), s_samples, gate="soft", alpha=a)
            fracs.append((q < 0.01).float().mean().item())
        ax.plot(alphas, fracs, label=f"mean SNR={mean_snr}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Fraction of params with q < 0.01")
    ax.set_title("Soft gate: fraction gated off vs alpha")
    ax.legend()

    # --- Effective q < 0.1 threshold for SNR gate vs lambda_pop ---
    ax = axes[1]
    lambdas = np.linspace(0.01, 20, 100)
    for mean_snr in mean_snr_values:
        m2_samples = torch.distributions.Exponential(1.0 / mean_snr).sample((n_params,))
        s_samples = torch.ones(n_params)
        fracs = []
        for lam in lambdas:
            q = compute_gate(m2_samples.sqrt(), s_samples, gate="snr", lambda_pop=lam)
            fracs.append((q < 0.1).float().mean().item())
        ax.plot(lambdas, fracs, label=f"mean SNR={mean_snr}")
    ax.set_xlabel("lambda_pop")
    ax.set_ylabel("Fraction of params with q < 0.1")
    ax.set_title("SNR gate: fraction effectively suppressed vs lambda_pop")
    ax.legend()

    # --- Mean gate value comparison: SNR vs soft across lambda_pop ---
    ax = axes[2]
    mean_snr = 1.0
    m2_samples = torch.distributions.Exponential(1.0 / mean_snr).sample((n_params,))
    s_samples = torch.ones(n_params)
    lambdas_fine = np.linspace(0.01, 10, 100)
    for gate_type, ls in [("snr", "-"), ("soft", "--")]:
        mean_gates = []
        for lam in lambdas_fine:
            q = compute_gate(m2_samples.sqrt(), s_samples, gate=gate_type, lambda_pop=lam)
            mean_gates.append(q.mean().item())
        ax.plot(lambdas_fine, mean_gates, ls=ls, label=f"{gate_type} gate")
    ax.set_xlabel("lambda_pop")
    ax.set_ylabel("Mean gate value")
    ax.set_title(f"Mean gate vs lambda_pop (mean SNR={mean_snr})")
    ax.legend()

    plt.tight_layout()
    _save(fig, "1c_threshold_analysis.png")


# ===================================================================
# 1d. Rho Bias-Variance Tradeoff
# ===================================================================

def plot_rho_analysis():
    """Visualize rho's effect on variance estimation dynamics."""
    print("1d. Rho bias-variance tradeoff...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Rho: Variance EMA Dynamics", fontsize=14, fontweight="bold")

    rho_values = [0.9, 0.95, 0.99, 0.995, 0.999]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rho_values)))

    # --- Bias correction factor 1/(1 - rho^t) ---
    ax = axes[0]
    steps = np.arange(1, 1001)
    for rho, c in zip(rho_values, colors):
        correction = 1.0 / (1.0 - rho ** steps)
        ax.plot(steps, correction, label=f"rho={rho}", color=c)
    ax.set_xlabel("Step t")
    ax.set_ylabel("Bias correction factor")
    ax.set_title("Bias correction: 1/(1 - rho^t)")
    ax.set_yscale("log")
    ax.legend()
    ax.axhline(1.0, color="gray", ls="--", alpha=0.3)

    # --- Effective window size (half-life) ---
    ax = axes[1]
    rho_range = np.linspace(0.8, 0.999, 100)
    half_life = -np.log(2) / np.log(rho_range)
    effective_window = 1.0 / (1.0 - rho_range)
    ax.plot(rho_range, effective_window, "b-", label="Effective window 1/(1-rho)")
    ax.plot(rho_range, half_life, "r--", label="Half-life")
    ax.set_xlabel("rho")
    ax.set_ylabel("Steps")
    ax.set_title("EMA Memory: window size vs rho")
    ax.set_yscale("log")
    ax.legend()

    # --- Simulated variance tracking with different rho ---
    ax = axes[2]
    torch.manual_seed(7)
    n_steps = 500
    true_var_1 = 1.0
    true_var_2 = 5.0  # shift at step 250

    for rho, c in zip(rho_values, colors):
        s = 0.0
        tracked = []
        for t in range(1, n_steps + 1):
            true_var = true_var_1 if t <= 250 else true_var_2
            g = torch.randn(1).item() * (true_var ** 0.5)
            # Simplified: treating m_prev as 0 for illustration
            s = rho * s + (1 - rho) * g ** 2
            s_corrected = s / (1 - rho ** t)
            tracked.append(s_corrected)
        ax.plot(range(1, n_steps + 1), tracked, label=f"rho={rho}", color=c, alpha=0.8)
    ax.axhline(true_var_1, color="gray", ls="--", alpha=0.5)
    ax.axhline(true_var_2, color="gray", ls="--", alpha=0.5)
    ax.axvline(250, color="red", ls=":", alpha=0.5, label="Distribution shift")
    ax.set_xlabel("Step")
    ax.set_ylabel("Estimated variance (bias-corrected)")
    ax.set_title("Variance tracking: stationary then shift at t=250")
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "1d_rho_analysis.png")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: Mathematical Analysis of SNR Gate Hyperparameters")
    print("=" * 60)
    print()
    plot_gate_surfaces()
    plot_sensitivity()
    plot_threshold_analysis()
    plot_rho_analysis()
    print()
    print("Done. All figures saved to:", RESULTS_DIR)
