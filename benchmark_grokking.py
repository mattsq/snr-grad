"""
Canonical Grokking Benchmark: Modular Addition (a + b) % p
Demonstrates how Grokfast (slow-gradient pre-amplification) and Grokfast-SNR
dramatically accelerate generalization (grokking) in non-convex representation learning.

Problem setup:
  - Prime p = 97, total 9409 pairs of equations.
  - Split: 80% train / 20% validation.
  - Model: 2-layer MLP with trainable embedding layers of size 128, hidden size 256.
  - Compares:
    1. AdamW (standard baseline - slow grokking)
    2. SNRAdamW (SNR gating only)
    3. Grokfast (Slow-gradient amplification only)
    4. Grokfast-SNR (Synergistic combination)
"""

import os
import argparse
import time
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snr_grad import SNRAdamW

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GrokkingConfig:
    p: int = 97                 # modulo prime
    d_embed: int = 128          # embedding size
    d_hidden: int = 256         # hidden layer size
    train_fraction: float = 0.8 # split fraction
    batch_size: int = 512
    n_steps: int = 5000
    lr: float = 1e-3
    weight_decay: float = 1e-3  # L2 regularization helps generalization
    seed: int = 42
    
    # Grokfast parameters
    grokfast_alpha: float = 0.98
    grokfast_lamb: float = 2.0
    lambda_pop: float = 1.0     # SNR population gating scaling

# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------

def make_modular_dataset(p, train_fraction, seed=0):
    # Generate all pairs
    x = torch.arange(p)
    grid = torch.cartesian_prod(x, x)
    y = (grid[:, 0] + grid[:, 1]) % p
    
    # Split train/validation
    gen = torch.Generator().manual_seed(seed)
    n_total = len(grid)
    indices = torch.randperm(n_total, generator=gen)
    n_train = int(n_total * train_fraction)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    return grid[train_idx], y[train_idx], grid[val_idx], y[val_idx]

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class GrokkingMLP(nn.Module):
    def __init__(self, p, d_embed, d_hidden):
        super().__init__()
        self.embed_a = nn.Embedding(p, d_embed)
        self.embed_b = nn.Embedding(p, d_embed)
        self.fc1 = nn.Linear(d_embed, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, p)
        
        # Initialize embeddings uniformly
        nn.init.uniform_(self.embed_a.weight, -0.1, 0.1)
        nn.init.uniform_(self.embed_b.weight, -0.1, 0.1)

    def forward(self, x):
        # x shape: [batch, 2]
        a = x[:, 0]
        b = x[:, 1]
        e_a = self.embed_a(a)
        e_b = self.embed_b(b)
        h = e_a + e_b
        h = self.relu(self.fc1(h))
        out = self.fc2(h)
        return out

# ---------------------------------------------------------------------------
# Run one optimization seed
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    steps: list = field(default_factory=list)
    train_losses: list = field(default_factory=list)
    train_accs: list = field(default_factory=list)
    val_accs: list = field(default_factory=list)

def run_one_baseline(optimizer_cls, opt_kwargs, cfg, X_train, y_train, X_val, y_val):
    torch.manual_seed(cfg.seed)
    model = GrokkingMLP(cfg.p, cfg.d_embed, cfg.d_hidden)
    optimizer = optimizer_cls(model.parameters(), **opt_kwargs)
    criterion = nn.CrossEntropyLoss()
    
    result = RunResult()
    eval_every = 50
    
    for step in range(cfg.n_steps):
        # Train step
        idx = torch.randint(0, len(X_train), (cfg.batch_size,))
        X_b, y_b = X_train[idx], y_train[idx]
        
        model.train()
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        
        # Eval step
        if step % eval_every == 0 or step == cfg.n_steps - 1:
            model.eval()
            with torch.no_grad():
                # Train accuracy (on subset for speed)
                train_sample_idx = torch.randint(0, len(X_train), (1000,))
                train_logits = model(X_train[train_sample_idx])
                train_acc = (train_logits.argmax(dim=-1) == y_train[train_sample_idx]).float().mean().item()
                
                # Validation accuracy
                val_logits = model(X_val)
                val_acc = (val_logits.argmax(dim=-1) == y_val).float().mean().item()
                
            result.steps.append(step)
            result.train_losses.append(loss.item())
            result.train_accs.append(train_acc)
            result.val_accs.append(val_acc)
            
    return result

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grokking_results(results, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.alpha": 0.4,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"Grokfast Modular Addition Generalization Acceleration  "
        f"(p={cfg.p}, split={int(cfg.train_fraction*100)}/{int((1-cfg.train_fraction)*100)})",
        fontsize=15, fontweight="bold", y=0.98
    )
    
    colors = {
        "AdamW": "tab:orange",
        "SNRAdamW": "tab:blue",
        "Grokfast": "tab:purple",
        "Grokfast-SNR": "tab:green",
    }
    
    # Panel 1: Training Loss (Cross Entropy)
    ax = axes[0]
    ax.set_title("Training Loss Convergence", pad=10, fontweight="semibold")
    for name, res in results.items():
        ax.plot(res.steps, res.train_losses, label=name, color=colors[name], linewidth=2.0)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=10)
    
    # Panel 2: Validation Accuracy (Grokking Dynamics)
    ax = axes[1]
    ax.set_title("Validation Generalization Accuracy", pad=10, fontweight="semibold")
    for name, res in results.items():
        ax.plot(res.steps, res.val_accs, label=name, color=colors[name], linewidth=2.0)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Accuracy (fraction)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "benchmark_grokking_curves.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"\nSaved modular grokking curves figure to: {path}")

# ---------------------------------------------------------------------------
# Main Sweep Orchesrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Canonical Modular Addition Grokking Sweep")
    parser.add_argument("--steps", type=int, default=5000, help="Number of steps to train (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split and initialization")
    args = parser.parse_args()
    
    cfg = GrokkingConfig(n_steps=args.steps, seed=args.seed)
    
    print("Preparing modular addition dataset...")
    X_train, y_train, X_val, y_val = make_modular_dataset(cfg.p, cfg.train_fraction, cfg.seed)
    
    # Configure optimizers to sweep
    adamw_kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    snr_kwargs = dict(
        lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", lambda_pop=cfg.lambda_pop,
    )
    
    grokfast_kwargs = dict(
        lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", lambda_pop=0.0,
        grokfast_alpha=cfg.grokfast_alpha, grokfast_lamb=cfg.grokfast_lamb,
    )
    
    grokfast_snr_kwargs = dict(
        lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr", lambda_pop=cfg.lambda_pop,
        grokfast_alpha=cfg.grokfast_alpha, grokfast_lamb=cfg.grokfast_lamb,
    )
    
    baselines = {
        "AdamW": (torch.optim.AdamW, adamw_kwargs),
        "SNRAdamW": (SNRAdamW, snr_kwargs),
        "Grokfast": (SNRAdamW, grokfast_kwargs),
        "Grokfast-SNR": (SNRAdamW, grokfast_snr_kwargs),
    }
    
    results = {}
    
    print("\nStarting optimization sweeps on modular addition...")
    for name, (cls, kwargs) in baselines.items():
        print(f"  Running {name}...", end=" ", flush=True)
        t0 = time.time()
        res = run_one_baseline(cls, kwargs, cfg, X_train, y_train, X_val, y_val)
        results[name] = res
        print(f"done in {time.time()-t0:.1f}s | Final Train Loss: {res.train_losses[-1]:.4f} | Final Val Acc: {res.val_accs[-1]*100:.1f}%")
        
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")
    plot_grokking_results(results, cfg, out_dir)
    print("\nModular grokking benchmark completed successfully!")

if __name__ == "__main__":
    main()
