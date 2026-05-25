"""
Benchmark comparing SNRScheduleFreeAdamW against standard AdamW (Flat), AdamW (Cosine), and SNRAdamW
on a heavy-duty Deep ResNet-MLP learning nested concentric hyper-spherical shells.
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

from snr_grad import SNRScheduleFreeAdamW, SNRAdamW


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    d_signal: int = 50           # Active input dimension
    d_noise: int = 50            # Noise padding input dimension
    hidden_features: int = 256   # Width of MLP layers
    depth: int = 4               # ResNet block depth
    num_classes: int = 5         # Number of concentric shell classes
    n_train: int = 1000          # Train dataset size
    test_size: int = 4000        # Test dataset size
    batch_size: int = 64
    n_steps: int = 2500
    n_seeds: int = 5
    lr: float = 3e-3
    weight_decay: float = 1e-4   # Small L2 regularization
    rho: float = 0.99
    alpha: str = "online"
    lambda_pop: float = 1.0
    sf_beta: float = 0.9


# ---------------------------------------------------------------------------
# Deep ResNet-MLP Architecture
# ---------------------------------------------------------------------------

class ResNetMLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, depth=4):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_features)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.GELU(),
                nn.Linear(hidden_features, hidden_features),
                nn.LayerNorm(hidden_features)
            ) for _ in range(depth)
        ])
        
        self.gelu = nn.GELU()
        self.out_proj = nn.Linear(hidden_features, num_classes)
        
    def forward(self, x):
        h = self.gelu(self.input_proj(x))
        for block in self.blocks:
            h = h + block(h)  # Residual connection
        return self.out_proj(self.gelu(h))


# ---------------------------------------------------------------------------
# Synthetic Concentric Hyper-Spherical Shells Dataset Generator
# ---------------------------------------------------------------------------

def make_dataset(n_samples, d_signal, d_noise, num_classes, seed=0):
    gen = torch.Generator().manual_seed(seed)
    
    # Generate signal features (concentric coordinate space)
    X_signal = torch.randn(n_samples, d_signal, generator=gen)
    
    # Class assignment based on distance to center
    norms = X_signal.norm(dim=1)
    
    # Partition norms into equal-sized classes using quantiles
    quantiles = torch.tensor([torch.quantile(norms, i / num_classes) for i in range(1, num_classes)])
    y = torch.bucketize(norms, quantiles)
    
    # Add random noise padding features (completely uninformative)
    X_noise = torch.randn(n_samples, d_noise, generator=gen)
    X = torch.cat([X_signal, X_noise], dim=1)
    
    # Introduce label noise (corrupt 10% of training labels)
    flip_mask = torch.rand(n_samples, generator=gen) < 0.1
    random_labels = torch.randint(0, num_classes, (n_samples,), generator=gen)
    y[flip_mask] = random_labels[flip_mask]
    
    return X, y


# ---------------------------------------------------------------------------
# Single run execution
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    train_losses: list = field(default_factory=list)
    test_losses: list = field(default_factory=list)
    test_accuracies: list = field(default_factory=list)
    signal_feature_norms: list = field(default_factory=list)
    noise_feature_norms: list = field(default_factory=list)
    # Dynamics tracking for seed 0
    z_vals: list = field(default_factory=list)
    y_vals: list = field(default_factory=list)
    x_vals: list = field(default_factory=list)


def run_one_seed(optimizer_cls, opt_kwargs, cfg, seed, use_cosine_scheduler=False, is_schedulefree=False, track_dynamics=False):
    # Test set is fixed to evaluate true generalization
    X_test, y_test = make_dataset(cfg.test_size, cfg.d_signal, cfg.d_noise, cfg.num_classes, seed=9999)

    torch.manual_seed(seed + 1000)
    model = ResNetMLP(
        in_features=cfg.d_signal + cfg.d_noise,
        hidden_features=cfg.hidden_features,
        num_classes=cfg.num_classes,
        depth=cfg.depth
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_cls(model.parameters(), **opt_kwargs)
    
    scheduler = None
    if use_cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_steps, eta_min=0.0)

    # Initialize ScheduleFree mode
    if is_schedulefree:
        optimizer.train()

    result = RunResult()
    eval_every = 50

    # Pick a specific weight parameter inside input projection to track averaging dynamics (seed 0)
    target_row = 0
    target_col = 0

    # Generator for online train data streaming
    train_data_gen = torch.Generator().manual_seed(seed)

    for step in range(cfg.n_steps):
        model.train()
        
        # ONLINE DATA STREAMING: Generate a fresh, never-before-seen minibatch at each step!
        # This prevents instant memorization, keeping the gradients active and meaningful.
        X_b, y_b = make_dataset(cfg.batch_size, cfg.d_signal, cfg.d_noise, cfg.num_classes, seed=int(torch.randint(0, 100000, (1,), generator=train_data_gen).item()))

        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        # Step-level tracking of dynamics for ScheduleFree (only for seed 0)
        if track_dynamics and is_schedulefree and step % 10 == 0:
            p = model.input_proj.weight
            st = optimizer.state[p]
            if "z" in st:
                y_val = p.data[target_row, target_col].item()
                z_val = st["z"][target_row, target_col].item()
                
                optimizer.eval()
                x_val = p.data[target_row, target_col].item()
                optimizer.train()
                
                result.z_vals.append(z_val)
                result.y_vals.append(y_val)
                result.x_vals.append(x_val)

        if step % eval_every == 0:
            result.train_losses.append(loss.item())
            
            # Switch to eval mode
            model.eval()
            if is_schedulefree:
                optimizer.eval()
                
            with torch.no_grad():
                test_logits = model(X_test)
                t_loss = criterion(test_logits, y_test).item()
                result.test_losses.append(t_loss)
                
                # Accuracy
                preds = test_logits.argmax(dim=1)
                acc = (preds == y_test).float().mean().item() * 100.0
                result.test_accuracies.append(acc)
                
                # Analyze first-layer input weight column norms
                w = model.input_proj.weight.data
                col_norms = w.norm(dim=0)  # shape (in_features,)
                sig_norm = col_norms[:cfg.d_signal].mean().item()
                noi_norm = col_norms[cfg.d_signal:].mean().item()
                
                result.signal_feature_norms.append(sig_norm)
                result.noise_feature_norms.append(noi_norm)
                
            # Restore train mode
            if is_schedulefree:
                optimizer.train()
                
    return result


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_experiment(cfg):
    adam_flat_kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    adam_cosine_kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    snr_kwargs = dict(
        lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", rho=cfg.rho,
        alpha=cfg.alpha, batch_size=cfg.batch_size, dataset_size=cfg.n_train,
        lambda_pop=cfg.lambda_pop, track_stats=True
    )
    
    sf_kwargs = dict(
        lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", rho=cfg.rho,
        alpha=cfg.alpha, batch_size=cfg.batch_size, dataset_size=cfg.n_train,
        lambda_pop=cfg.lambda_pop, sf_beta=cfg.sf_beta
    )

    results = {}
    arms = [
        ("AdamW (Flat)", torch.optim.AdamW, adam_flat_kwargs, False, False),
        ("AdamW (Cosine)", torch.optim.AdamW, adam_cosine_kwargs, True, False),
        ("SNRAdamW (Flat)", SNRAdamW, snr_kwargs, False, False),
        ("SNRScheduleFree", SNRScheduleFreeAdamW, sf_kwargs, False, True),
    ]

    for name, opt_cls, kwargs, use_sched, is_sf in arms:
        print(f"Running {name}...")
        res_list = []
        for seed in range(cfg.n_seeds):
            print(f"  Seed {seed+1}/{cfg.n_seeds}...", end=" ", flush=True)
            track_dyn = (seed == 0)
            res = run_one_seed(
                opt_cls, kwargs, cfg, seed,
                use_cosine_scheduler=use_sched,
                is_schedulefree=is_sf,
                track_dynamics=track_dyn
            )
            res_list.append(res)
            print(f"Test Acc={res.test_accuracies[-1]:.2f}%")
        results[name] = res_list

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def to_stack(results, attr):
    return torch.tensor([getattr(r, attr) for r in results])


def plot_band(ax, steps, data, label, color, alpha=0.15):
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
    eval_every = 50
    steps = list(range(0, cfg.n_steps, eval_every))

    # Sleek modern layout styling
    colors = {
        "AdamW (Flat)": "#94A3B8",       # Slate Grey
        "AdamW (Cosine)": "#F59E0B",     # Amber Orange
        "SNRAdamW (Flat)": "#8B5CF6",    # Purple
        "SNRScheduleFree": "#06B6D4"     # Cyan / Teal
    }

    # ---- Figure 1: Main 2x2 Curves Dashboard ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f"Gated ScheduleFree vs Schedulers on Deep ResNet-MLP\n"
        f"Nested Concentric Hyper-Spherical Shells (d_in={cfg.d_signal+cfg.d_noise}, {cfg.n_seeds} seeds)",
        fontsize=15, fontweight="bold", y=0.97, color="#0F172A"
    )

    # 1. Train Loss (Cross Entropy)
    ax = axes[0, 0]
    for name, res_list in results.items():
        train_data = to_stack(res_list, "train_losses")
        plot_band(ax, steps, train_data, name, colors[name])
    ax.set_ylabel("Train Cross-Entropy Loss", fontsize=11, fontweight="semibold", color="#334155")
    ax.set_xlabel("Optimization Step", fontsize=11, color="#334155")
    ax.set_title("(a) Training Loss Convergence", fontsize=12, fontweight="bold", pad=10, color="#0F172A")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9, edgecolor="#E2E8F0")
    ax.grid(True, linestyle="--", color="#E2E8F0", alpha=0.7)
    ax.tick_params(colors="#475569")

    # 2. Test Accuracy (%)
    ax = axes[0, 1]
    for name, res_list in results.items():
        acc_data = to_stack(res_list, "test_accuracies")
        plot_band(ax, steps, acc_data, name, colors[name])
    ax.set_ylabel("Test Set Accuracy (%)", fontsize=11, fontweight="semibold", color="#334155")
    ax.set_xlabel("Optimization Step", fontsize=11, color="#334155")
    ax.set_title("(b) Generalization Performance (Accuracy)", fontsize=12, fontweight="bold", pad=10, color="#0F172A")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9, edgecolor="#E2E8F0")
    ax.grid(True, linestyle="--", color="#E2E8F0", alpha=0.7)
    ax.tick_params(colors="#475569")

    # 3. Input Projection Weight Suppression (SNR Gating Visual)
    ax = axes[1, 0]
    # We display SNRAdamW's signal vs noise feature weights suppression
    snr_sig = to_stack(results["SNRAdamW (Flat)"], "signal_feature_norms")
    snr_noi = to_stack(results["SNRAdamW (Flat)"], "noise_feature_norms")
    sf_sig = to_stack(results["SNRScheduleFree"], "signal_feature_norms")
    sf_noi = to_stack(results["SNRScheduleFree"], "noise_feature_norms")
    
    plot_band(ax, steps, snr_sig, "SNRAdamW: Active Signal Weights", "#8B5CF6")
    plot_band(ax, steps, snr_noi, "SNRAdamW: Gated Noise Weights", "#D8B4FE", alpha=0.08)
    plot_band(ax, steps, sf_sig, "SNRScheduleFree: Active Signal Weights", "#06B6D4")
    plot_band(ax, steps, sf_noi, "SNRScheduleFree: Gated Noise Weights", "#99F6E4", alpha=0.08)
    
    ax.set_ylabel("Mean Weight Column L2 Norm", fontsize=11, fontweight="semibold", color="#334155")
    ax.set_xlabel("Optimization Step", fontsize=11, color="#334155")
    ax.set_title("(c) Feature Weight Column Norms (Gating Suppression)", fontsize=12, fontweight="bold", pad=10, color="#0F172A")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9, edgecolor="#E2E8F0")
    ax.grid(True, linestyle="--", color="#E2E8F0", alpha=0.7)
    ax.tick_params(colors="#475569")

    # 4. Polyak-Ruppert Iterate Averaging Dynamics
    ax = axes[1, 1]
    sf_seed0 = results["SNRScheduleFree"][0]
    dyn_steps = list(range(0, cfg.n_steps, 10))
    
    # Zoomed initial phase to clearly resolve averaging smoothing
    zoom_steps = min(150, len(sf_seed0.z_vals))
    ax.plot(dyn_steps[:zoom_steps], sf_seed0.z_vals[:zoom_steps], label="Base Sequence z_t", color="#F43F5E", alpha=0.6, linewidth=1.2)
    ax.plot(dyn_steps[:zoom_steps], sf_seed0.y_vals[:zoom_steps], label="Evaluation Point y_t", color="#F59E0B", alpha=0.5, linewidth=1.2)
    ax.plot(dyn_steps[:zoom_steps], sf_seed0.x_vals[:zoom_steps], label="Averaged Iterate x_t", color="#06B6D4", alpha=1.0, linewidth=2.2)
    
    ax.set_ylabel("First-Layer Weight Value", fontsize=11, fontweight="semibold", color="#334155")
    ax.set_xlabel("Optimization Step (Zoomed Initial Phase)", fontsize=11, color="#334155")
    ax.set_title("(d) Polyak-Ruppert Iterate Averaging Dynamics", fontsize=12, fontweight="bold", pad=10, color="#0F172A")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9, edgecolor="#E2E8F0")
    ax.grid(True, linestyle="--", color="#E2E8F0", alpha=0.7)
    ax.tick_params(colors="#475569")

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_sf_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Figure 2: Final Summary Metrics Bar Chart ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    fig.suptitle("Final Optimization Performance Metrics Summary", fontsize=15, fontweight="bold", y=0.97, color="#0F172A")

    names = ["AdamW (Flat)", "AdamW (Cosine)", "SNRAdamW (Flat)", "SNRScheduleFree"]
    bar_colors = [colors[n] for n in names]

    # 1. Final Classification Test Accuracy
    ax = axes[0]
    final_acc_vals = []
    final_acc_stds = []
    for name in names:
        acc_stack = to_stack(results[name], "test_accuracies")[:, -1]
        final_acc_vals.append(acc_stack.mean().item())
        final_acc_stds.append(acc_stack.std().item())
    
    ax.bar(names, final_acc_vals, yerr=final_acc_stds, color=bar_colors, capsize=8, edgecolor="#1E293B", alpha=0.85, error_kw={'ecolor': '#1E293B', 'linewidth': 1.5})
    ax.set_ylabel("Final Test Accuracy (%)", fontsize=11, fontweight="semibold", color="#334155")
    ax.set_title("(a) Final Generalization Accuracy", fontsize=12, fontweight="bold", pad=10, color="#0F172A")
    ax.grid(True, axis="y", linestyle="--", color="#E2E8F0", alpha=0.7)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.tick_params(colors="#475569")

    # 2. Final active signal column L2 norms
    ax = axes[1]
    final_sig_vals = []
    final_sig_stds = []
    for name in names:
        sig_stack = to_stack(results[name], "signal_feature_norms")[:, -1]
        final_sig_vals.append(sig_stack.mean().item())
        final_sig_stds.append(sig_stack.std().item())
        
    ax.bar(names, final_sig_vals, yerr=final_sig_stds, color=bar_colors, capsize=8, edgecolor="#1E293B", alpha=0.85, error_kw={'ecolor': '#1E293B', 'linewidth': 1.5})
    ax.set_ylabel("Final Signal Weights L2 Norm", fontsize=11, fontweight="semibold", color="#334155")
    ax.set_title("(b) Recovered Signal Column Norms", fontsize=12, fontweight="bold", pad=10, color="#0F172A")
    ax.grid(True, axis="y", linestyle="--", color="#E2E8F0", alpha=0.7)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.tick_params(colors="#475569")

    # 3. Final noise column L2 norms
    ax = axes[2]
    final_noi_vals = []
    final_noi_stds = []
    for name in names:
        noi_stack = to_stack(results[name], "noise_feature_norms")[:, -1]
        final_noi_vals.append(noi_stack.mean().item())
        final_noi_stds.append(noi_stack.std().item())

    ax.bar(names, final_noi_vals, yerr=final_noi_stds, color=bar_colors, capsize=8, edgecolor="#1E293B", alpha=0.85, error_kw={'ecolor': '#1E293B', 'linewidth': 1.5})
    ax.set_ylabel("Final Noise Weights L2 Norm", fontsize=11, fontweight="semibold", color="#334155")
    ax.set_title("(c) Memorized Noise Feature suppression", fontsize=12, fontweight="bold", pad=10, color="#0F172A")
    ax.grid(True, axis="y", linestyle="--", color="#E2E8F0", alpha=0.7)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.tick_params(colors="#475569")

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_sf_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SNRScheduleFreeAdamW against standard baselines.")
    parser.add_argument("--seeds", type=int, default=None, help="Number of random seeds to run.")
    parser.add_argument("--steps", type=int, default=None, help="Number of optimization steps to run.")
    args = parser.parse_args()

    cfg = BenchmarkConfig()
    if args.seeds is not None:
        cfg.n_seeds = args.seeds
    if args.steps is not None:
        cfg.n_steps = args.steps

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
    
    print(f"Starting deep learning experiment with seeds={cfg.n_seeds}, steps={cfg.n_steps}, lr={cfg.lr}...")
    results = run_experiment(cfg)
    make_figures(results, cfg, out_dir)
    print("All heavy-duty benchmark visualizations created and saved successfully.")
