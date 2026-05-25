"""
Scientific regime benchmark: Sweeps training dataset size (n) from underdetermined to overdetermined,
for both low-noise and high-noise regimes, comparing Constant LR vs Cosine LR Decay.

Generates two beautiful scientific visualizations:
1. `benchmarks/benchmark_regimes_detail.png` - Comparative 2x2 grid showing final Excess Test MSE
   vs dataset size (n) under Constant LR vs Cosine LR Decay.
2. `benchmarks/benchmark_regimes_curves.png` - 2x2 learning curves detail showing Excess Test MSE
   over steps for underdetermined (n=100) vs overdetermined (n=300) regimes.
"""

from __future__ import annotations

import os
import argparse
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snr_grad import SNRAdamW
from benchmark_grokfast import BenchmarkConfig, run_one_seed

def run_regime_sweep(n_sizes, noises, n_seeds, n_steps, use_schedulers):
    # Track results
    # Structure: results[use_scheduler][noise_level][optimizer_name][n_size] = list of final excess MSEs
    opt_names = ["AdamW", "SNRAdamW", "Grokfast", "Grokfast-SNR"]
    
    results = {
        sched: {
            noise: {name: {n: [] for n in n_sizes} for name in opt_names}
            for noise in noises
        }
        for sched in use_schedulers
    }
    
    # Track learning curves for specific sizes (n=100 and n=300)
    curve_sizes = [100, 300]
    # Structure: curves[use_scheduler][noise_level][optimizer_name][n_size] = list of test loss lists
    curves = {
        sched: {
            noise: {name: {n: [] for n in curve_sizes if n in n_sizes} for name in opt_names}
            for noise in noises
        }
        for sched in use_schedulers
    }
    
    # Base config values
    base_cfg = BenchmarkConfig(n_steps=n_steps)
    
    print("Starting regime sweep...")
    for sched in use_schedulers:
        sched_name = "Cosine Decay LR" if sched else "Constant LR"
        print(f"\n================ SCHEDULER: {sched_name} ================")
        
        for noise in noises:
            print(f"\n--- Noise Level: {noise} ---")
            for n in n_sizes:
                print(f"  Training Dataset Size n = {n}...", end=" ", flush=True)
                
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
                        res = run_one_seed(cls, kwargs, cfg, seed, use_scheduler=sched)
                        # Test loss is MSE; compute excess test MSE
                        excess_mse = res.test_losses[-1] - noise**2
                        results[sched][noise][name][n].append(excess_mse)
                        
                        # Store curves for representative sizes
                        if n in curve_sizes:
                            # Convert history to excess MSE
                            excess_curve = [loss - noise**2 for loss in res.test_losses]
                            curves[sched][noise][name][n].append(excess_curve)
                
                # Print intermediate summary for sanity check
                summary_parts = []
                for name in opt_names:
                    vals = results[sched][noise][name][n]
                    mean_val = np.mean(vals)
                    summary_parts.append(f"{name}: {mean_val:.4f}")
                print(" | ".join(summary_parts))
                
    return results, curves

def plot_regimes_detail(results, n_sizes, noises, use_schedulers, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Premium scientific plotting style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.alpha": 0.4,
    })
    
    # 2x2 grid: Rows are schedulers (Constant LR vs Cosine LR), Columns are noises (0.2 vs 3.0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Final Generalization Performance across Dataset Sizes: Underdetermined to Overdetermined\n"
        "(Linear Regression, d=200 features, k=5 sparse signal variables, mean +/- std over seeds)",
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
    
    for row_idx, sched in enumerate(use_schedulers):
        sched_label = "Cosine LR Decay" if sched else "Constant Learning Rate"
        
        for col_idx, noise in enumerate(noises):
            ax = axes[row_idx, col_idx]
            
            noise_label = f"Low Noise (sigma = {noise})" if noise < 1.0 else f"High Noise (sigma = {noise})"
            ax.set_title(f"{sched_label} | {noise_label}", pad=10, fontweight="semibold")
            
            # Draw vertical line representing n=d=200 boundary
            ax.axvline(200, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="Boundary (n=d=200)")
            
            # Shade underdetermined region
            ax.axvspan(0, 200, color="gray", alpha=0.08, label="Underdetermined (n < d)")
            
            for name in ["AdamW", "SNRAdamW", "Grokfast", "Grokfast-SNR"]:
                means = []
                stds = []
                for n in n_sizes:
                    vals = results[sched][noise][name][n]
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    
                means = np.array(means)
                stds = np.array(stds)
                
                # Plot line and error bands
                ax.plot(n_sizes, means, label=name, color=colors[name], marker=markers[name], linewidth=2.0, markersize=6)
                ax.fill_between(n_sizes, means - stds, means + stds, color=colors[name], alpha=0.15)
                
            ax.set_xlabel("Training Set Size (n)")
            ax.set_ylabel("Excess Test MSE")
            ax.grid(True, linestyle=":")
            ax.legend(fontsize=9, loc="upper right")
            ax.set_xlim(min(n_sizes) - 10, max(n_sizes) + 10)
            
            # Use log scale if high noise final values span a wide range
            if noise > 1.0:
                ax.set_yscale("log")
                
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "benchmark_regimes_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"\nSaved regimes comparison figure to: {path}")

def plot_learning_curves(curves, noises, use_schedulers, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # 2x2 grid: Rows are noises, Columns are dataset sizes (n=100 vs n=300)
    # We plot the Cosine LR Decay case (row_idx) to show how it suppresses steady state noise
    # Or we can plot the Constant LR case, which clearly shows the steady-state noise plateau.
    # Let's show Constant LR because it is where the noise-amplification trade-off is most dramatically proven!
    sched_to_plot = False # Constant LR
    sched_label = "Constant Learning Rate"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Generalization Dynamics over Training Steps ({sched_label})\n"
        "Illustrating Early-Stage Fast Convergence vs. Late-Stage Steady-State Noise Amplification",
        fontsize=15, fontweight="bold", y=0.98
    )
    
    colors = {
        "AdamW": "tab:orange",
        "SNRAdamW": "tab:blue",
        "Grokfast": "tab:purple",
        "Grokfast-SNR": "tab:green",
    }
    
    sizes_to_plot = [100, 300]
    eval_every = 10
    
    for row_idx, noise in enumerate(noises):
        noise_label = "Low Noise (sigma = 0.2)" if noise < 1.0 else "High Noise (sigma = 3.0)"
        
        for col_idx, n in enumerate(sizes_to_plot):
            ax = axes[row_idx, col_idx]
            
            regime_label = "Underdetermined (n=100)" if n == 100 else "Overdetermined (n=300)"
            ax.set_title(f"{noise_label} | {regime_label}", pad=10, fontweight="semibold")
            
            for name in ["AdamW", "SNRAdamW", "Grokfast", "Grokfast-SNR"]:
                if n not in curves[sched_to_plot][noise][name]:
                    continue
                # curves list has shape [seeds, steps]
                runs = np.array(curves[sched_to_plot][noise][name][n])
                steps = np.arange(0, runs.shape[1] * eval_every, eval_every)
                
                mean_curve = np.mean(runs, axis=0)
                std_curve = np.std(runs, axis=0)
                
                ax.plot(steps, mean_curve, label=name, color=colors[name], linewidth=2.0)
                ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, color=colors[name], alpha=0.15)
                
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Excess Test MSE (lower is better)")
            ax.grid(True, linestyle=":")
            ax.legend(fontsize=9, loc="upper right")
            
            # Use log scale to see final convergence differences clearly
            ax.set_yscale("log")
            
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "benchmark_regimes_curves.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved learning curves detail figure to: {path}")

def main():
    parser = argparse.ArgumentParser(description="Grokfast Scientific Regime Sweep")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to run per sweep (default: 3)")
    parser.add_argument("--steps", type=int, default=2000, help="Number of training steps (default: 2000)")
    parser.add_argument("--quick", action="store_true", help="Run a quick sweep over fewer dataset sizes")
    args = parser.parse_args()
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
    
    if args.quick:
        n_sizes = [100, 300]
    else:
        n_sizes = [50, 100, 150, 200, 300, 500]
        
    noises = [0.2, 3.0]
    use_schedulers = [False, True]
    
    print(f"Sweep Config: sizes={n_sizes}, noises={noises}, seeds={args.seeds}, steps={args.steps}")
    
    results, curves = run_regime_sweep(
        n_sizes=n_sizes,
        noises=noises,
        n_seeds=args.seeds,
        n_steps=args.steps,
        use_schedulers=use_schedulers
    )
    
    print("\nGenerating regime sweeps comparative visualization...")
    plot_regimes_detail(results, n_sizes, noises, use_schedulers, out_dir)
    
    print("\nGenerating training dynamics learning curves visualization...")
    plot_learning_curves(curves, noises, use_schedulers, out_dir)
    
    print("\nScientific regimes sweep completed successfully!")

if __name__ == "__main__":
    main()
