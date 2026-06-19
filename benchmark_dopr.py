"""
Benchmark: Double Preconditioning (DoPr) on a test-time-feedback (TTF) task.

Reproduces the paper's minimal example (arXiv:2606.06418, Section 3.1): behavior
cloning in a linear dynamical system (LDS). A demonstrator policy ``a = K* s`` is
imitated by an *overparameterized* linear policy ``K_theta = F G`` trained with a
one-step L2 behavior-cloning loss on states drawn from an **anisotropic**
demonstrator distribution. The learned policy is then *rolled out* in the system,
where one-step errors compound (TTF).

The point of the paper: a gradient preconditioner (GP, e.g. Adam / SNRAdamW)
accelerates validation loss but, under non-isotropic activations, learns the
feature subspace poorly, which the rollout amplifies. Activation preconditioning
(AP) debiases the gradient by the input-activation covariance, learning the
feature subspace more uniformly and reducing rollout error -- often *without*
improving validation loss.

We compare, per training step, three metrics averaged over seeds:
  1. L_val       : held-out one-step BC loss (what is optimized).
  2. subspace    : dist(G_theta, G*) -- feature-learning quality (Eq. 3.4).
  3. rollout cost: closed-loop cost of the learned policy over a horizon (TTF).

Arms: SNRAdamW vs DoPr(SNRAdamW), and Adam vs DoPr(Adam) (AP is base-agnostic).

Run:
    uv run python benchmark_dopr.py
    uv run python benchmark_dopr.py --quick
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_grad import SNRAdamW, DoPr, ActivationPrecondConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    n_state: int = 8          # state dimension
    m_action: int = 4         # action dimension
    d_feat: int = 4           # rank of K* / feature dimension (d < n_state)
    cond: float = 100.0       # condition number of the state covariance (anisotropy)
    horizon: int = 20         # rollout horizon for the TTF metric
    n_steps: int = 800
    batch_size: int = 64
    n_seeds: int = 5
    lr: float = 3e-3
    damping: float = 1e-2
    process_noise: float = 0.05
    rho_A: float = 0.5        # spectral radius of the (stable) open-loop dynamics A
    b_scale: float = 0.2      # input-matrix scale (feedback strength)
    k_scale: float = 0.3      # demonstrator-policy scale
    state_clip: float = 1e4   # guard rollouts against numerical blow-up


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

def _make_system(cfg: Config, gen: torch.Generator):
    """Return (A, B, Kstar, Gstar, Sigma_s_sqrt) for one seed."""
    n, m, d = cfg.n_state, cfg.m_action, cfg.d_feat
    # Anisotropic state covariance: eigenvalues spaced log-linearly over [1, 1/cond].
    eig = torch.logspace(0, -torch.log10(torch.tensor(cfg.cond)), n, base=10.0)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen))
    Sigma_s = Q @ torch.diag(eig) @ Q.t()
    Sigma_s_sqrt = Q @ torch.diag(eig.sqrt()) @ Q.t()

    # Low-rank demonstrator K* = F* G*.
    Gstar = torch.randn(d, n, generator=gen)
    Fstar = torch.randn(m, d, generator=gen)
    Kstar = (Fstar @ Gstar) * cfg.k_scale

    # Open-loop dynamics A rescaled to a fixed spectral radius (stable). With a
    # modest input scale B, the closed loop A + B K stays bounded for the policies
    # encountered during training, so the rollout metric measures tracking error
    # (TTF) rather than divergence.
    A = torch.randn(n, n, generator=gen)
    radius = torch.linalg.eigvals(A).abs().max().real
    A = A * (cfg.rho_A / radius)
    B = torch.randn(n, m, generator=gen) * cfg.b_scale
    return A, B, Kstar, Gstar, Sigma_s_sqrt


def _policy(cfg: Config, gen: torch.Generator) -> nn.Sequential:
    """Overparameterized linear policy K_theta = F G (two linear layers, no bias)."""
    torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
    return nn.Sequential(
        nn.Linear(cfg.n_state, cfg.d_feat, bias=False),   # G
        nn.Linear(cfg.d_feat, cfg.m_action, bias=False),  # F
    )


def _effective_K(policy: nn.Sequential) -> torch.Tensor:
    G = policy[0].weight.detach()  # [d, n]
    F = policy[1].weight.detach()  # [m, d]
    return F @ G                    # [m, n]


def _subspace_distance(G: torch.Tensor, Gstar: torch.Tensor) -> float:
    """dist(G, G*) = ||P_G (I - P_{G*})||_op via principal angles (Eq. 3.4)."""
    QG = torch.linalg.qr(G.t())[0]        # [n, d] orthonormal basis of rowspace(G)
    Qs = torch.linalg.qr(Gstar.t())[0]    # [n, d]
    sv = torch.linalg.svdvals(QG.t() @ Qs)
    sv = sv.clamp(max=1.0)
    return float((1.0 - sv.min() ** 2).clamp_min(0.0).sqrt())


def _rollout_cost(K: torch.Tensor, A, B, Kstar, Sigma_s_sqrt, cfg, gen) -> float:
    """Closed-loop cost of policy K rolled out in the LDS (where TTF compounds)."""
    n_traj = 64
    s = torch.randn(n_traj, cfg.n_state, generator=gen) @ Sigma_s_sqrt.t()
    cost = 0.0
    for _ in range(cfg.horizon):
        a = s @ K.t()
        err = s @ (Kstar - K).t()          # action error vs demonstrator
        cost += float((err ** 2).sum(dim=1).mean())
        w = torch.randn(n_traj, cfg.n_state, generator=gen) * cfg.process_noise
        s = (s @ A.t() + a @ B.t() + w).clamp(-cfg.state_clip, cfg.state_clip)
    return cost / cfg.horizon


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_optimizer(name: str, policy: nn.Sequential, cfg: Config):
    if name == "SNRAdamW":
        return SNRAdamW(policy.parameters(), lr=cfg.lr)
    if name == "DoPr(SNRAdamW)":
        return DoPr(SNRAdamW(policy.parameters(), lr=cfg.lr), policy,
                    ActivationPrecondConfig(damping=cfg.damping))
    if name == "Adam":
        return torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    if name == "DoPr(Adam)":
        return DoPr(torch.optim.Adam(policy.parameters(), lr=cfg.lr), policy,
                    ActivationPrecondConfig(damping=cfg.damping))
    raise ValueError(name)


def _run_arm(name: str, cfg: Config, seed: int, record_every: int):
    gen = torch.Generator().manual_seed(seed)
    A, B, Kstar, Gstar, Ss = _make_system(cfg, gen)
    policy = _policy(cfg, gen)
    opt = _build_optimizer(name, policy, cfg)

    # Held-out validation states (fixed across steps).
    val_s = torch.randn(2048, cfg.n_state, generator=gen) @ Ss.t()
    val_a = val_s @ Kstar.t()

    steps, lvals, subs, rolls = [], [], [], []
    for step in range(cfg.n_steps):
        s = torch.randn(cfg.batch_size, cfg.n_state, generator=gen) @ Ss.t()
        a_star = s @ Kstar.t()
        opt.zero_grad(set_to_none=True)
        pred = policy(s)
        loss = ((pred - a_star) ** 2).sum(dim=1).mean()
        loss.backward()
        opt.step()

        if step % record_every == 0 or step == cfg.n_steps - 1:
            with torch.no_grad():
                K = _effective_K(policy)
                lval = float(((policy(val_s) - val_a) ** 2).sum(dim=1).mean())
                sub = _subspace_distance(policy[0].weight.detach(), Gstar)
                roll = _rollout_cost(K, A, B, Kstar, Ss, cfg, gen)
            steps.append(step)
            lvals.append(lval)
            subs.append(sub)
            rolls.append(roll)
    return steps, lvals, subs, rolls


def run(cfg: Config):
    record_every = max(1, cfg.n_steps // 40)
    arms = ["SNRAdamW", "DoPr(SNRAdamW)", "Adam", "DoPr(Adam)"]
    results: dict[str, dict] = {}
    for name in arms:
        L, S, R = [], [], []
        steps = None
        for seed in range(cfg.n_seeds):
            steps, lv, sb, rl = _run_arm(name, cfg, seed, record_every)
            L.append(lv); S.append(sb); R.append(rl)
        results[name] = {
            "steps": steps,
            "lval": torch.tensor(L),
            "sub": torch.tensor(S),
            "roll": torch.tensor(R),
        }
        print(f"{name:16s}  final L_val={results[name]['lval'][:, -1].mean():.4f}  "
              f"subspace={results[name]['sub'][:, -1].mean():.4f}  "
              f"rollout={results[name]['roll'][:, -1].mean():.4f}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(results: dict, cfg: Config, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    colors = {"SNRAdamW": "C0", "DoPr(SNRAdamW)": "C1", "Adam": "C2", "DoPr(Adam)": "C3"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [("lval", "Validation loss (one-step BC)", True),
              ("sub", "Feature subspace distance dist(G, G*)", False),
              ("roll", "Rollout cost (TTF, lower is better)", True)]
    for ax, (key, title, logy) in zip(axes, panels):
        for name, res in results.items():
            mean = res[key].mean(dim=0)
            std = res[key].std(dim=0)
            x = res["steps"]
            ax.plot(x, mean, label=name, color=colors[name])
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=colors[name])
        ax.set_title(title)
        ax.set_xlabel("step")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("DoPr (activation preconditioning) on a linear-dynamics TTF task")
    fig.tight_layout()
    path = os.path.join(out_dir, "dopr_ttf.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    # Bar summary of final rollout cost.
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    names = list(results.keys())
    means = [float(results[n]["roll"][:, -1].mean()) for n in names]
    stds = [float(results[n]["roll"][:, -1].std()) for n in names]
    ax2.bar(names, means, yerr=stds, color=[colors[n] for n in names], capsize=4)
    ax2.set_ylabel("final rollout cost")
    ax2.set_title("Final rollout cost (lower = better downstream performance)")
    plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")
    fig2.tight_layout()
    path2 = os.path.join(out_dir, "dopr_ttf_rollout_bar.png")
    fig2.savefig(path2, dpi=120)
    plt.close(fig2)
    print(f"\nSaved figures to {path} and {path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--out-dir", default="benchmarks")
    args = parser.parse_args()

    cfg = Config()
    if args.quick:
        cfg.n_steps = 150
        cfg.n_seeds = 2
        cfg.horizon = 15

    torch.manual_seed(0)
    results = run(cfg)
    plot(results, cfg, args.out_dir)


if __name__ == "__main__":
    main()
