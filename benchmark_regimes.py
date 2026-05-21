"""
Scientific regime benchmark: Sweeps training dataset size (n) from underdetermined to overdetermined,
for both low-noise and high-noise regimes.

Generates a beautiful dual-panel visualization showing:
1. Low Noise (sigma=0.2): Excess Test MSE vs Dataset Size (n)
2. High Noise (sigma=3.0): Excess Test MSE vs Dataset Size (n)

This clearly illustrates the crossover point where Grokfast becomes highly effective (overdetermined regime)
versus where it amplifies spurious noise correlations (underdetermined regime).
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snr_grad import SNRAdamW
from benchmark_grokfast import BenchmarkConfig, run_one_seed

def run_regime_sweep():
    # Setup sweep parameters
    n_sizes = [50, 100, 150, 200, 300, 500]
    noises = [0.2, 3.0]
    n_seeds = 3
    n_steps = 2000  # 2000 steps is plenty for convergence and keeps runs fast
    
    # Track results
    # Structure: results[noise_level][optimizer_name][n_size] = list of excess MSEs
    opt_names = ["AdamW", "SNRAdamW", "Grokfast", "Grokfast-SNR"]
    results = {
        noise: {name: {n: [] for n in n_sizes} for name in opt_names}
        for noise in noises
    }
    
    # Base config values
    base_cfg = BenchmarkConfig(n_steps=n_steps)
    
    print("Starting regime sweep...")
    for noise in noises:
        print(f"\n================ NOISE LEVEL: {noise} ================")
        for n in n_sizes:
            print(f"\n--- Training Dataset Size n = {n} ---")
            
            # Setup configs for this specific size and noise
            cfg = BenchmarkConfig(
                d=base_cfg.d,
                k=base_cfg.k,
                n_train=n,
                batch_size=base_cfg.batch_size,
                sigma_noise=noise,
                n_steps=n_steps,
                lr=base_cfg.lr,
                weight_decay=base_cfg.weight_decay,
                signal_magnitude=base_cfg.signal_magnitude,
                rho=base_cfg.rho,
                alpha=base_cfg.alpha,
                lambda_pop=base_cfg.lambda_pop,
                grokfast_alpha=base_cfg.grokfast_alpha,
                grokfast_lamb=base_cfg.grokfast_lamb
            )
            
            # Setup optimizer arguments
            adamw_kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
            
            snr_kwargs = dict(
                lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", rho=cfg.rho,
                alpha=cfg.alpha, batch_size=cfg.batch_size, dataset_size=cfg.n_train,
                lambda_pop=cfg.lambda_pop,
            )
            
            grokfast_kwargs = dict(
                lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", rho=cfg.rho,
                alpha=cfg.alpha, batch_size=cfg.batch_size, dataset_size=cfg.n_train,
                lambda_pop=0.0, grokfast_alpha=cfg.grokfast_alpha, grokfast_lamb=cfg.grokfast_lamb,
            )
            
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
            
            for seed in range(n_seeds):
                for name, (cls, kwargs) in baselines.items():
                    res = run_one_seed(cls, kwargs, cfg, seed)
                    # Test loss is MSE; compute excess test MSE
                    excess_mse = res.test_losses[-1] - noise**2
                    results[noise][name][n].append(excess_mse)
            
            # Print intermediate summary for sanity check
            print(f"Summary for n={n}:")
            for name in opt_names:
                vals = results[noise][name][n]
                mean_val = np.mean(vals)
                print(f"  {name:15}: Mean Excess MSE = {mean_val:.4f}")
                
    return results, n_sizes, noises

def plot_regimes(results, n_sizes, noises, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Setup premium scientific plotting style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.alpha": 0.4,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Generalization Performance across Dataset Sizes: Underdetermined to Overdetermined\n"
        "(Linear Regression, d=200 features, k=5 sparse signal variables)",
        fontsize=15, fontweight="bold", y=0.98
    )
    
    colors = {
        "AdamW": "tab:orange",
        "SNRAdamW": "tab:blue",
        "Grokfast": "tab:purple",
        "Grokfast-SNR": "tab:green",
    }
    
    markers = {
        "AdamW": "o",
        "SNRAdamW": "s",
        "Grokfast": "^",
        "Grokfast-SNR": "d",
    }
    
    for idx, noise in enumerate(noises):
        ax = axes[idx]
        title_str = f"Low Noise Regime ($\sigma$ = {noise})" if noise < 1.0 else f"High Noise Regime ($\sigma$ = {noise})"
        ax.set_title(title_str, pad=12, fontweight="semibold")
        
        # Draw vertical line representing n=d=200 boundary
        ax.axvline(200, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="Boundary (n=d=200)")
        
        for name in ["AdamW", "SNRAdamW", "Grokfast", "Grokfast-SNR"]:
            means = []
            stds = []
            for n in n_sizes:
                vals = results[noise][name][n]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                
            means = np.array(means)
            stds = np.array(stds)
            
            # Plot line and error bands
            ax.plot(n_sizes, means, label=name, color=colors[name], marker=markers[name], linewidth=2.0, markersize=6)
            ax.fill_between(n_sizes, means - stds, means + stds, color=colors[name], alpha=0.15)
            
        ax.set_xlabel("Training Set Size ($n$)")
        ax.set_ylabel("Excess Test MSE (lower is better)")
        ax.grid(True, linestyle=":")
        
        # Shade the underdetermined region (n < d) to highlight it
        ax.axvspan(0, 200, color="gray", alpha=0.08, label="Underdetermined (n < d)")
        
        ax.legend(fontsize=10, loc="upper right")
        ax.set_xlim(min(n_sizes) - 10, max(n_sizes) + 10)
        
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "benchmark_regimes_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"\nSaved regimes comparison figure to: {path}")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
    results, n_sizes, noises = run_regime_sweep()
    plot_regimes(results, n_sizes, noises, out_dir)
    print("All done!")
