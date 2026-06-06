"""
Benchmark: variance-estimation backends for the SNRAdamW gate.

Compares ways of supplying ``grad_variances`` to ``SNRAdamW.step`` on a synthetic
sparse linear-regression task with label noise (the same overparameterized regime
used by ``benchmark.py``):

  - EMA-only        : the optimizer's internal streaming variance (baseline)
  - exact-every-step: exact per-sample-gradient variance every step (reference)
  - exact-every-K   : exact probe every K steps, EMA otherwise (hybrid cadence)
  - microbatch K=2  : cheap split-batch estimator, 2 backward chunks
  - microbatch K=4  : cheap split-batch estimator, 4 backward chunks

Because the ground-truth signal coordinates are known, we report gate *quality*
(signal vs noise gate separation and an AUC for signal-vs-noise ranking by gate
value), not just loss. We also report wall-clock and the correlation between the
internal EMA variance and the exact variance.

Run:
    uv run python benchmark_variance_estimators.py
    uv run python benchmark_variance_estimators.py --quick
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_grad import (
    SNRAdamW,
    ExactVarianceEstimator,
    backward_with_microbatch_variance,
    compute_gate,
    resolve_alpha,
)
from torch.func import functional_call


# ---------------------------------------------------------------------------
# Config + data (mirrors benchmark.py)
# ---------------------------------------------------------------------------

@dataclass
class Config:
    d: int = 200
    k: int = 5
    n_train: int = 100
    batch_size: int = 32
    sigma_noise: float = 3.0
    n_steps: int = 2000
    n_seeds: int = 3
    test_size: int = 5000
    lr: float = 3e-3
    weight_decay: float = 0.0
    signal_magnitude: float = 3.0
    rho: float = 0.99
    lambda_pop: float = 1.0
    gate: str = "snr"


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
# Per-sample / microbatch loss closures for a linear regression head
# ---------------------------------------------------------------------------

def loss_one_sample(model):
    """Per-example summed squared error: its gradient is the per-example gradient."""
    def fn(params, buffers, sample):
        x, y = sample
        pred = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        return ((pred.squeeze(0) - y) ** 2).sum()
    return fn


def microbatch_loss(model, sub_batch):
    x, y = sub_batch
    return ((model(x) - y) ** 2).mean()


# ---------------------------------------------------------------------------
# Gate-quality metrics
# ---------------------------------------------------------------------------

def gate_auc(gate_vals: torch.Tensor, signal_mask: torch.Tensor) -> float:
    """AUC for separating signal vs noise coordinates by gate value (Mann-Whitney)."""
    pos = gate_vals[signal_mask]
    neg = gate_vals[~signal_mask]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    # Rank-based AUC.
    all_vals = torch.cat([pos, neg])
    ranks = all_vals.argsort().argsort().float() + 1.0
    r_pos = ranks[: pos.numel()].sum()
    auc = (r_pos - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel())
    return float(auc)


@dataclass
class RunResult:
    final_train: float = 0.0
    final_test: float = 0.0
    wall_clock: float = 0.0
    mean_signal_gate: float = 0.0
    mean_noise_gate: float = 0.0
    gate_auc: float = 0.0
    false_suppress_signal: float = 0.0  # fraction of signal coords with gate < 0.5
    false_pass_noise: float = 0.0       # fraction of noise coords with gate > 0.5
    ema_exact_corr: float = float("nan")
    history_test: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Single run for a given variance "mode"
# ---------------------------------------------------------------------------

def run_one(mode: str, cfg: Config, seed: int, probe_interval: int = 10) -> RunResult:
    w_true, signal_idx = make_true_weights(cfg.d, cfg.k, cfg.signal_magnitude)
    signal_mask = torch.zeros(cfg.d, dtype=torch.bool)
    signal_mask[signal_idx] = True

    train_gen = torch.Generator().manual_seed(seed)
    X_train, y_train = make_dataset(w_true, cfg.n_train, cfg.sigma_noise, train_gen)
    test_gen = torch.Generator().manual_seed(9999)
    X_test, y_test = make_dataset(w_true, cfg.test_size, cfg.sigma_noise, test_gen)

    torch.manual_seed(seed + 1000)
    model = nn.Linear(cfg.d, 1, bias=False)
    nn.init.zeros_(model.weight)
    opt = SNRAdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, gate=cfg.gate,
        rho=cfg.rho, alpha="finite", batch_size=cfg.batch_size, dataset_size=cfg.n_train,
        lambda_pop=cfg.lambda_pop, track_stats=True,
    )
    exact_est = ExactVarianceEstimator(exclude_norm=False)

    result = RunResult()
    t0 = time.perf_counter()
    last_loss = 0.0
    for step in range(cfg.n_steps):
        idx = torch.randint(cfg.n_train, (cfg.batch_size,))
        X_b, y_b = X_train[idx], y_train[idx]
        batch = (X_b, y_b.squeeze(1))

        opt.zero_grad(set_to_none=True)
        grad_variances = None

        if mode == "ema":
            loss = ((model(X_b) - y_b) ** 2).mean()
            loss.backward()
            last_loss = loss.item()
        elif mode == "exact":
            loss = ((model(X_b) - y_b) ** 2).mean()
            loss.backward()
            last_loss = loss.item()
            grad_variances = exact_est.estimate(model, loss_one_sample(model), batch)
        elif mode == "exact_k":
            loss = ((model(X_b) - y_b) ** 2).mean()
            loss.backward()
            last_loss = loss.item()
            if step % probe_interval == 0:
                grad_variances = exact_est.estimate(model, loss_one_sample(model), batch)
        elif mode.startswith("micro"):
            k = int(mode.split("_")[1])
            loss_val, grad_variances = backward_with_microbatch_variance(
                model, microbatch_loss, batch, num_splits=k
            )
            last_loss = loss_val
        else:
            raise ValueError(f"Unknown mode: {mode}")

        opt.step(grad_variances=grad_variances)

        if step % 50 == 0:
            with torch.no_grad():
                result.history_test.append(((model(X_test) - y_test) ** 2).mean().item())

    result.wall_clock = time.perf_counter() - t0
    result.final_train = last_loss
    with torch.no_grad():
        result.final_test = ((model(X_test) - y_test) ** 2).mean().item()

    # Final gate quality, computed from the EMA-corrected s_hat actually used.
    state = opt.state[model.weight]
    step_num = state["step"]
    m_hat = (state["exp_avg"].squeeze() / (1 - 0.9 ** step_num))
    s_hat = state["exp_grad_var"].squeeze() / (1 - cfg.rho ** step_num)
    alpha_val = resolve_alpha("finite", batch_size=cfg.batch_size, dataset_size=cfg.n_train)
    gate_vals = compute_gate(m_hat, s_hat, gate=cfg.gate, alpha=alpha_val, lambda_pop=cfg.lambda_pop)

    result.mean_signal_gate = float(gate_vals[signal_mask].mean())
    result.mean_noise_gate = float(gate_vals[~signal_mask].mean())
    result.gate_auc = gate_auc(gate_vals, signal_mask)
    result.false_suppress_signal = float((gate_vals[signal_mask] < 0.5).float().mean())
    result.false_pass_noise = float((gate_vals[~signal_mask] > 0.5).float().mean())

    # Correlation between the internal EMA variance and a fresh exact variance.
    full_batch = (X_train, y_train.squeeze(1))
    exact_full = exact_est.estimate(model, loss_one_sample(model), full_batch)
    s_exact = exact_full[model.weight].squeeze().detach()
    log_ema = torch.log(s_hat.detach().clamp_min(1e-12))
    log_exact = torch.log(s_exact.clamp_min(1e-12))
    if torch.std(log_ema) > 0 and torch.std(log_exact) > 0:
        result.ema_exact_corr = float(torch.corrcoef(torch.stack([log_ema, log_exact]))[0, 1])

    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

MODES = [
    ("ema", "EMA-only"),
    ("exact", "exact/step"),
    ("exact_k", "exact/10"),
    ("micro_2", "microbatch K=2"),
    ("micro_4", "microbatch K=4"),
]


def aggregate(results: list[RunResult]) -> dict:
    def mean(attr):
        vals = [getattr(r, attr) for r in results]
        vals = [v for v in vals if v == v]  # drop nan
        return sum(vals) / len(vals) if vals else float("nan")
    return {
        "train": mean("final_train"),
        "test": mean("final_test"),
        "wall": mean("wall_clock"),
        "sig_gate": mean("mean_signal_gate"),
        "noise_gate": mean("mean_noise_gate"),
        "auc": mean("gate_auc"),
        "false_suppress": mean("false_suppress_signal"),
        "false_pass": mean("false_pass_noise"),
        "corr": mean("ema_exact_corr"),
    }


COLORS = {
    "EMA-only": "tab:gray",
    "exact/step": "tab:blue",
    "exact/10": "tab:cyan",
    "microbatch K=2": "tab:orange",
    "microbatch K=4": "tab:red",
}


def make_figures(all_runs: dict, summary: dict, cfg: Config, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    labels = [label for _, label in MODES]
    eval_every = 50
    steps = list(range(0, cfg.n_steps, eval_every))
    irreducible = cfg.sigma_noise ** 2

    # ---- Figure 1: test-loss curves ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.suptitle(
        f"Variance backends for the SNRAdamW gate: test loss\n"
        f"sparse regression (d={cfg.d}, k={cfg.k}, n={cfg.n_train}, "
        f"noise={cfg.sigma_noise}, {cfg.n_seeds} seeds)",
        fontsize=12, fontweight="bold",
    )
    for label in labels:
        hist = torch.tensor([r.history_test for r in all_runs[label]])  # [seeds, T]
        n = min(len(steps), hist.shape[1])
        mean = hist[:, :n].mean(dim=0)
        std = hist[:, :n].std(dim=0)
        ax.plot(steps[:n], mean, label=label, color=COLORS[label], linewidth=1.6)
        ax.fill_between(steps[:n], (mean - std).numpy(), (mean + std).numpy(),
                        color=COLORS[label], alpha=0.15)
    ax.axhline(irreducible, ls="--", color="black", alpha=0.5,
               label=f"irreducible ({irreducible:.0f})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Test MSE")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path1 = os.path.join(out_dir, "benchmark_variance_curves.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)

    # ---- Figure 2: gate-quality + cost summary ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("Variance backends: gate quality and cost", fontsize=12, fontweight="bold")
    x = range(len(labels))
    colors = [COLORS[l] for l in labels]

    # (a) signal vs noise gate.
    ax = axes[0]
    width = 0.38
    sig = [summary[l]["sig_gate"] for l in labels]
    noise = [summary[l]["noise_gate"] for l in labels]
    ax.bar([i - width / 2 for i in x], sig, width, label="signal coords", color="tab:green")
    ax.bar([i + width / 2 for i in x], noise, width, label="noise coords", color="tab:red")
    ax.set_ylabel("Mean gate")
    ax.set_title("(a) Gate on signal vs noise")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (b) AUC (signal-vs-noise ranking).
    ax = axes[1]
    auc = [summary[l]["auc"] for l in labels]
    ax.bar(list(x), auc, color=colors)
    ax.axhline(0.5, ls="--", color="gray", alpha=0.6, label="chance (0.5)")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Signal-vs-noise AUC")
    ax.set_title("(b) Gate ranking quality")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (c) wall-clock cost.
    ax = axes[2]
    wall = [summary[l]["wall"] for l in labels]
    ax.bar(list(x), wall, color=colors)
    ax.set_ylabel("Wall-clock (s)")
    ax.set_title(f"(c) Cost ({cfg.n_steps} steps x {cfg.n_seeds} seeds)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path2 = os.path.join(out_dir, "benchmark_variance_summary.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)

    return [path1, path2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fewer steps/seeds for a smoke run.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--out-dir", default="benchmarks", help="Directory for output PNGs.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    args = parser.parse_args()

    cfg = Config()
    if args.quick:
        cfg.n_steps, cfg.n_seeds = 300, 2
    if args.steps is not None:
        cfg.n_steps = args.steps
    if args.seeds is not None:
        cfg.n_seeds = args.seeds

    print(
        f"Sparse regression: d={cfg.d}, k={cfg.k}, n_train={cfg.n_train}, "
        f"batch={cfg.batch_size}, sigma={cfg.sigma_noise}, steps={cfg.n_steps}, "
        f"seeds={cfg.n_seeds}\n"
    )

    summary = {}
    all_runs = {}
    for mode, label in MODES:
        runs = []
        for seed in range(cfg.n_seeds):
            runs.append(run_one(mode, cfg, seed))
        all_runs[label] = runs
        summary[label] = aggregate(runs)
        agg = summary[label]
        print(
            f"  {label:<16} test={agg['test']:.3f}  wall={agg['wall']:.1f}s  "
            f"sig_gate={agg['sig_gate']:.3f}  noise_gate={agg['noise_gate']:.3f}  "
            f"AUC={agg['auc']:.3f}"
        )

    # Table.
    header = (
        f"\n{'variant':<16} {'test':>8} {'train':>8} {'wall(s)':>8} "
        f"{'sig_gate':>9} {'noise_gt':>9} {'auc':>6} {'f_supp':>7} {'f_pass':>7} {'corr':>6}"
    )
    print(header)
    print("-" * len(header))
    for _, label in MODES:
        a = summary[label]
        print(
            f"{label:<16} {a['test']:>8.3f} {a['train']:>8.3f} {a['wall']:>8.1f} "
            f"{a['sig_gate']:>9.3f} {a['noise_gate']:>9.3f} {a['auc']:>6.3f} "
            f"{a['false_suppress']:>7.3f} {a['false_pass']:>7.3f} {a['corr']:>6.3f}"
        )

    print(
        "\nNotes:"
        "\n  - sig_gate/noise_gate: mean gate on true signal vs noise coordinates"
        "\n    (higher signal, lower noise is better)."
        "\n  - auc: ranking of signal vs noise coords by gate value (1.0 = perfect)."
        "\n  - f_supp: fraction of signal coords wrongly suppressed (gate < 0.5)."
        "\n  - f_pass: fraction of noise coords wrongly passed (gate > 0.5)."
        "\n  - corr: correlation of log EMA-variance vs log exact-variance (full batch)."
    )

    if not args.no_figures:
        paths = make_figures(all_runs, summary, cfg, args.out_dir)
        print("\nSaved figures:")
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
