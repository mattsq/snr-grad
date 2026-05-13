"""
Phase 2: Hyperparameter Sweep Experiments on Synthetic Problems.

Runs controlled experiments on sparse linear regression with known ground truth
to understand the effect of each SNR hyperparameter.

Sweeps:
  2a. lambda_pop, alpha, rho, gate type on fixed problem
  2b. Varying signal-to-noise ratio (noise level, sparsity)
  2c. Varying dataset size (finite alpha study)
  2d. Non-stationary gradients (rho robustness)

Results saved as .pt files in studies/hyperparameter_study/results/.

Usage:
    python studies/hyperparameter_study/hyperparameter_sweep.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from snr_grad import SNRAdamW, compute_gate, resolve_alpha

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SEEDS = 5
EVAL_EVERY = 50


# ---------------------------------------------------------------------------
# Data helpers (shared with benchmark.py)
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
    mean_gates: list = field(default_factory=list)
    signal_gates: list = field(default_factory=list)
    noise_gates: list = field(default_factory=list)


def run_single(
    d: int, k: int, n_train: int, batch_size: int, sigma_noise: float,
    n_steps: int, lr: float, seed: int,
    gate: str = "snr", lambda_pop: float = 1.0,
    alpha: str = "online", rho: float = 0.99,
    signal_magnitude: float = 3.0,
    shift_target_at: Optional[int] = None,
) -> RunResult:
    """Run one training seed and collect metrics."""
    w_true, signal_idx = make_true_weights(d, k, signal_magnitude)
    noise_mask = torch.ones(d, dtype=torch.bool)
    noise_mask[signal_idx] = False

    train_gen = torch.Generator().manual_seed(seed)
    X_train, y_train = make_dataset(w_true, n_train, sigma_noise, train_gen)
    test_gen = torch.Generator().manual_seed(9999)
    X_test, y_test = make_dataset(w_true, 5000, sigma_noise, test_gen)

    # If we shift target mid-training, prepare alternate target
    if shift_target_at is not None:
        w_true_alt, _ = make_true_weights(d, k, signal_magnitude, seed=42)
        y_train_alt = X_train @ w_true_alt + torch.randn(n_train, generator=torch.Generator().manual_seed(seed + 500)) * sigma_noise
        y_train_alt = y_train_alt.unsqueeze(1)
        y_test_alt = X_test @ w_true_alt + torch.randn(5000, generator=torch.Generator().manual_seed(9998)) * sigma_noise
        y_test_alt = y_test_alt.unsqueeze(1)

    torch.manual_seed(seed + 1000)
    model = nn.Linear(d, 1, bias=False)
    nn.init.zeros_(model.weight)

    opt_kwargs = dict(lr=lr, gate=gate, lambda_pop=lambda_pop, rho=rho, track_stats=True)
    if alpha == "finite":
        opt_kwargs.update(alpha="finite", batch_size=batch_size, dataset_size=n_train)
    elif isinstance(alpha, (int, float)):
        opt_kwargs["alpha"] = float(alpha)
    else:
        opt_kwargs["alpha"] = alpha

    optimizer = SNRAdamW(model.parameters(), **opt_kwargs)
    result = RunResult()

    for step in range(n_steps):
        # Handle target shift
        if shift_target_at is not None and step >= shift_target_at:
            y_tr_cur, y_te_cur = y_train_alt, y_test_alt
        else:
            y_tr_cur, y_te_cur = y_train, y_test

        idx = torch.randint(n_train, (batch_size,))
        X_b, y_b = X_train[idx], y_tr_cur[idx]

        optimizer.zero_grad()
        loss = ((model(X_b) - y_b) ** 2).mean()
        loss.backward()
        optimizer.step()

        if step % EVAL_EVERY == 0:
            result.train_losses.append(loss.item())
            with torch.no_grad():
                result.test_losses.append(((model(X_test) - y_te_cur) ** 2).mean().item())
            result.param_errors.append(
                (model.weight.data.squeeze() - w_true).norm().item()
            )
            if optimizer.last_stats is not None:
                result.mean_gates.append(optimizer.last_stats.mean_gate)

                # Per-feature gate values
                state = optimizer.state[model.weight]
                step_num = state["step"]
                betas = (0.9, 0.999)
                m_hat = state["exp_avg"].squeeze() / (1 - betas[0] ** step_num)
                s_hat = state["exp_grad_var"].squeeze() / (1 - rho ** step_num)
                alpha_val = resolve_alpha(
                    opt_kwargs["alpha"],
                    batch_size=opt_kwargs.get("batch_size"),
                    dataset_size=opt_kwargs.get("dataset_size"),
                )
                gate_vals = compute_gate(m_hat, s_hat, gate=gate, alpha=alpha_val, lambda_pop=lambda_pop)
                result.signal_gates.append(gate_vals[signal_idx].mean().item())
                result.noise_gates.append(gate_vals[noise_mask].mean().item())

    return result


def run_multi_seed(n_seeds: int = N_SEEDS, **kwargs) -> List[RunResult]:
    """Run experiment over multiple seeds."""
    return [run_single(seed=s, **kwargs) for s in range(n_seeds)]


def aggregate(results: List[RunResult], attr: str) -> torch.Tensor:
    """Stack results across seeds into a 2D tensor [seeds, steps]."""
    data = [getattr(r, attr) for r in results]
    min_len = min(len(d) for d in data)
    return torch.tensor([d[:min_len] for d in data])


# ---------------------------------------------------------------------------
# 2a. Core hyperparameter sweeps
# ---------------------------------------------------------------------------

def sweep_lambda_pop():
    """Sweep lambda_pop across gate types."""
    print("2a. Lambda_pop sweep...")
    lambdas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
    gates = ["snr", "soft"]
    results = {}

    for gate_type, lam in product(gates, lambdas):
        key = f"{gate_type}_lam{lam}"
        print(f"  {key}...", end=" ", flush=True)
        t0 = time.time()
        res = run_multi_seed(
            d=200, k=5, n_train=100, batch_size=32, sigma_noise=3.0,
            n_steps=2000, lr=3e-3, gate=gate_type, lambda_pop=lam,
        )
        results[key] = {
            "test_mse": aggregate(res, "test_losses"),
            "param_error": aggregate(res, "param_errors"),
            "mean_gates": aggregate(res, "mean_gates"),
            "signal_gates": aggregate(res, "signal_gates"),
            "noise_gates": aggregate(res, "noise_gates"),
        }
        final = results[key]["test_mse"][:, -1]
        print(f"{time.time()-t0:.1f}s  test={final.mean():.2f}+/-{final.std():.2f}")

    torch.save(results, os.path.join(RESULTS_DIR, "2a_lambda_sweep.pt"))
    return results


def sweep_alpha():
    """Sweep alpha across gate types."""
    print("2a. Alpha sweep...")
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, "finite"]
    gates = ["snr", "soft"]
    results = {}

    for gate_type, alpha in product(gates, alphas):
        key = f"{gate_type}_alpha{alpha}"
        print(f"  {key}...", end=" ", flush=True)
        t0 = time.time()
        res = run_multi_seed(
            d=200, k=5, n_train=100, batch_size=32, sigma_noise=3.0,
            n_steps=2000, lr=3e-3, gate=gate_type, alpha=alpha,
        )
        results[key] = {
            "test_mse": aggregate(res, "test_losses"),
            "param_error": aggregate(res, "param_errors"),
            "mean_gates": aggregate(res, "mean_gates"),
        }
        final = results[key]["test_mse"][:, -1]
        print(f"{time.time()-t0:.1f}s  test={final.mean():.2f}+/-{final.std():.2f}")

    torch.save(results, os.path.join(RESULTS_DIR, "2a_alpha_sweep.pt"))
    return results


def sweep_rho():
    """Sweep rho across gate types."""
    print("2a. Rho sweep...")
    rhos = [0.9, 0.95, 0.99, 0.995, 0.999]
    gates = ["snr", "soft"]
    results = {}

    for gate_type, rho in product(gates, rhos):
        key = f"{gate_type}_rho{rho}"
        print(f"  {key}...", end=" ", flush=True)
        t0 = time.time()
        res = run_multi_seed(
            d=200, k=5, n_train=100, batch_size=32, sigma_noise=3.0,
            n_steps=2000, lr=3e-3, gate=gate_type, rho=rho,
        )
        results[key] = {
            "test_mse": aggregate(res, "test_losses"),
            "param_error": aggregate(res, "param_errors"),
            "mean_gates": aggregate(res, "mean_gates"),
        }
        final = results[key]["test_mse"][:, -1]
        print(f"{time.time()-t0:.1f}s  test={final.mean():.2f}+/-{final.std():.2f}")

    torch.save(results, os.path.join(RESULTS_DIR, "2a_rho_sweep.pt"))
    return results


# ---------------------------------------------------------------------------
# 2b. Varying signal-to-noise ratio
# ---------------------------------------------------------------------------

def sweep_problem_snr():
    """Vary problem difficulty and find best lambda_pop for each regime."""
    print("2b. Problem SNR sweep...")
    sigma_values = [0.5, 1.0, 3.0, 10.0]
    k_values = [1, 5, 20, 50]
    lambdas = [0.1, 1.0, 5.0, 10.0]
    results = {}

    for sigma, k in product(sigma_values, k_values):
        for lam in lambdas:
            key = f"sigma{sigma}_k{k}_lam{lam}"
            print(f"  {key}...", end=" ", flush=True)
            t0 = time.time()
            res = run_multi_seed(
                d=200, k=k, n_train=100, batch_size=32, sigma_noise=sigma,
                n_steps=2000, lr=3e-3, gate="snr", lambda_pop=lam, n_seeds=3,
            )
            results[key] = {
                "test_mse": aggregate(res, "test_losses"),
                "param_error": aggregate(res, "param_errors"),
            }
            final = results[key]["test_mse"][:, -1]
            print(f"{time.time()-t0:.1f}s  test={final.mean():.2f}")

    torch.save(results, os.path.join(RESULTS_DIR, "2b_problem_snr.pt"))
    return results


# ---------------------------------------------------------------------------
# 2c. Varying dataset size
# ---------------------------------------------------------------------------

def sweep_dataset_size():
    """Compare online vs finite alpha across dataset sizes."""
    print("2c. Dataset size sweep...")
    n_values = [50, 100, 500, 2000, 10000]
    results = {}

    for n_train in n_values:
        for alpha_mode in ["online", "finite"]:
            key = f"n{n_train}_{alpha_mode}"
            print(f"  {key}...", end=" ", flush=True)
            t0 = time.time()
            steps = min(2000, max(1000, n_train * 3))
            res = run_multi_seed(
                d=200, k=5, n_train=n_train, batch_size=min(32, n_train // 2),
                sigma_noise=3.0, n_steps=steps, lr=3e-3, gate="snr", alpha=alpha_mode,
            )
            results[key] = {
                "test_mse": aggregate(res, "test_losses"),
                "param_error": aggregate(res, "param_errors"),
                "mean_gates": aggregate(res, "mean_gates"),
            }
            final = results[key]["test_mse"][:, -1]
            print(f"{time.time()-t0:.1f}s  test={final.mean():.2f}+/-{final.std():.2f}")

    torch.save(results, os.path.join(RESULTS_DIR, "2c_dataset_size.pt"))
    return results


# ---------------------------------------------------------------------------
# 2d. Non-stationary gradients
# ---------------------------------------------------------------------------

def sweep_nonstationary():
    """Test rho robustness under distribution shift."""
    print("2d. Non-stationary gradient sweep...")
    rhos = [0.9, 0.95, 0.99, 0.995, 0.999]
    results = {}

    for rho in rhos:
        key = f"rho{rho}"
        print(f"  {key}...", end=" ", flush=True)
        t0 = time.time()
        res = run_multi_seed(
            d=200, k=5, n_train=100, batch_size=32, sigma_noise=3.0,
            n_steps=3000, lr=3e-3, gate="snr", rho=rho,
            shift_target_at=1500,
        )
        results[key] = {
            "test_mse": aggregate(res, "test_losses"),
            "param_error": aggregate(res, "param_errors"),
            "mean_gates": aggregate(res, "mean_gates"),
        }
        final = results[key]["test_mse"][:, -1]
        print(f"{time.time()-t0:.1f}s  test={final.mean():.2f}")

    torch.save(results, os.path.join(RESULTS_DIR, "2d_nonstationary.pt"))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2: Hyperparameter Sweep Experiments")
    print("=" * 60)
    print()

    t_start = time.time()
    sweep_lambda_pop()
    print()
    sweep_alpha()
    print()
    sweep_rho()
    print()
    sweep_problem_snr()
    print()
    sweep_dataset_size()
    print()
    sweep_nonstationary()
    print()
    elapsed = time.time() - t_start
    print(f"All sweeps complete in {elapsed:.0f}s. Results in: {RESULTS_DIR}")
