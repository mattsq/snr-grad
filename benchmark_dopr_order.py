"""
Experiment: does the ORDER of the SNR gate (GP) relative to the DoPr activation
preconditioner (AP) change or improve performance?

DoPr as shipped applies AP *before* the base gradient preconditioner (GP), i.e.
the base optimizer (SNRAdamW / Adam) consumes the activation-preconditioned
gradient (see README, ``DoPr.step``)::

    AP->GP (the shipped order, "pre"):
        G = dL/dW
        M = G @ (S_z + tau I)^-1      # activation precondition the gradient
        D = GP(M)                     # SNR gate + Adam-normalize the AP'd gradient
        W <- W - eta * D

This script adds the *swapped* order, where the SNR gate / Adam normalization runs
first on the raw gradient and AP conditions the resulting **update direction**::

    GP->AP (swapped, "post"):
        G = dL/dW
        D = GP(G)                     # SNR gate + Adam-normalize the raw gradient
        M = D @ (S_z + tau I)^-1      # activation precondition the update
        W <- W - eta * M

Why the order can matter: Adam's per-coordinate ``1/sqrt(v)`` normalization in the
GP largely *undoes* AP's magnitude reweighting when AP runs first (AP->GP), because
both act coordinate-wise on the same gradient. When AP runs *after* the GP
(GP->AP), it reshapes an already-normalized, isotropic-magnitude update, so the
activation geometry survives into the step. The undamped maps commute up to the
GP's nonlinearity; with damping and Adam's nonlinear normalization they do not, so
this is an empirical question.

We measure both orders against the no-AP baseline on two tasks:

  * ``--task ttf`` (default): the DoPr canonical task -- the LDS behavior-cloning
    test-time-feedback problem from ``benchmark_dopr.py``. Primary metric is
    closed-loop **rollout cost** (TTF), where AP is supposed to help; we also report
    one-step validation loss and feature-subspace distance.
  * ``--task sparse``: the repo's canonical task -- sparse linear regression with
    label noise from ``benchmark.py``. Primary metric is **excess test MSE**.

Implementation note: the "post" order is realized by snapshotting the weights,
taking the ordinary base step (which moves ``W`` by the GP update ``delta``),
restoring ``W``, stashing ``delta`` into ``.grad``, running the *same*
``ActivationPreconditioner.precondition_`` on it (so ``delta <- delta @ S_z^-1``),
and re-applying. Because AP is linear this conditions exactly the GP update
direction. Both tasks use ``weight_decay=0``, so ``delta`` is purely the gradient
update and AP is not applied to a decoupled-weight-decay term (which would be a
confound under nonzero WD -- see the caveat printed at the end).

Run:
    uv run python benchmark_dopr_order.py            # TTF task
    uv run python benchmark_dopr_order.py --quick
    uv run python benchmark_dopr_order.py --task sparse
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_grad import SNRAdamW, ActivationPreconditioner, ActivationPrecondConfig

# Reuse the DoPr canonical task setup verbatim from benchmark_dopr.py.
from benchmark_dopr import (
    Config as TTFConfig,
    _make_system,
    _policy,
    _effective_K,
    _subspace_distance,
    _rollout_cost,
)
# Reuse the repo canonical task setup from benchmark.py / benchmark_dopr_existing.py.
from benchmark import BenchmarkConfig, make_true_weights, make_dataset


# ---------------------------------------------------------------------------
# Ordered DoPr wrapper: AP before ("pre") or after ("post") the base GP step.
# ---------------------------------------------------------------------------

class OrderedDoPr:
    """DoPr with a selectable AP/GP ordering.

    mode="pre"       -> AP then GP (the shipped ``DoPr`` order): precondition the
                        gradient, then base.step() consumes it.
    mode="post"      -> GP then AP: base.step() on the raw gradient to form the
                        update, then activation-precondition that update direction.
    mode="post_norm" -> like "post", but rescale each conditioned update back to the
                        base GP update's norm. This isolates the *direction* change
                        of AP from its *magnitude* change (the GP otherwise fixes the
                        update norm, so a bare "post" also changes the effective step
                        size -- a confound when judging whether geometry alone helps).
    mode="off"       -> no AP (the bare base optimizer); the baseline arm.
    """

    def __init__(self, base, model, config, mode):
        if mode not in {"pre", "post", "post_norm", "off"}:
            raise ValueError(mode)
        self.base = base
        self.mode = mode
        # Always build the preconditioner (registers hooks) so the activation
        # capture cost is identical across arms; "off" simply never solves.
        self.ap = ActivationPreconditioner(model, config)

    @torch.no_grad()
    def step(self):
        if self.mode == "off":
            # Drain the per-step activation cache so it cannot leak forward, but
            # do not advance AP's step/warmup counter (it never preconditions).
            self.base.step()
            self.ap.zero_grad()
            return

        if self.mode == "pre":
            self.ap.precondition_()
            self.base.step()
            return

        # mode in {"post", "post_norm"}: GP first, then AP the update direction.
        params = [
            p for g in self.base.param_groups for p in g["params"]
            if p.grad is not None
        ]
        prev = [p.detach().clone() for p in params]
        self.base.step()  # W <- W - eta * D  (delta = -eta * D, wd=0 in these tasks)
        for p, p0 in zip(params, prev):
            p.grad.copy_(p.detach() - p0)  # stash the update direction into .grad
            p.data.copy_(p0)               # roll the weights back
        pre_norms = [float(p.grad.norm()) for p in params] if self.mode == "post_norm" else None
        self.ap.precondition_()            # .grad <- delta @ S_z^-1 (registered layers)
        if self.mode == "post_norm":
            for p, n0 in zip(params, pre_norms):
                n1 = float(p.grad.norm())
                if n1 > 0.0:
                    p.grad.mul_(n0 / n1)   # preserve the base GP update norm
        for p in params:
            p.data.add_(p.grad)            # re-apply the (norm-matched) update

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)
        self.ap.zero_grad()


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

# (label, base optimizer factory keyed by name, AP mode)
def _arms(damping):
    cfg_ap = ActivationPrecondConfig(damping=damping)
    return [
        ("SNRAdamW",                "SNRAdamW", "off",       cfg_ap),
        ("AP->GP (pre)  SNR",       "SNRAdamW", "pre",       cfg_ap),
        ("GP->AP (post) SNR",       "SNRAdamW", "post",      cfg_ap),
        ("GP->AP (post,norm) SNR",  "SNRAdamW", "post_norm", cfg_ap),
        ("Adam",                    "Adam",     "off",       cfg_ap),
        ("AP->GP (pre)  Adam",      "Adam",     "pre",       cfg_ap),
        ("GP->AP (post) Adam",      "Adam",     "post",      cfg_ap),
        ("GP->AP (post,norm) Adam", "Adam",     "post_norm", cfg_ap),
    ]


_COLORS = {
    "SNRAdamW": "C0", "AP->GP (pre)  SNR": "C1", "GP->AP (post) SNR": "C3",
    "GP->AP (post,norm) SNR": "C6",
    "Adam": "C2", "AP->GP (pre)  Adam": "C4", "GP->AP (post) Adam": "C5",
    "GP->AP (post,norm) Adam": "C7",
}


# ---------------------------------------------------------------------------
# Task 1: DoPr canonical task (LDS behavior cloning, test-time feedback)
# ---------------------------------------------------------------------------

def _build_base_ttf(name, policy, cfg):
    if name == "SNRAdamW":
        return SNRAdamW(policy.parameters(), lr=cfg.lr)
    if name == "Adam":
        return torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    raise ValueError(name)


def _run_ttf_arm(base_name, mode, ap_cfg, cfg, seed, record_every):
    gen = torch.Generator().manual_seed(seed)
    A, B, Kstar, Gstar, Ss = _make_system(cfg, gen)
    policy = _policy(cfg, gen)
    base = _build_base_ttf(base_name, policy, cfg)
    opt = OrderedDoPr(base, policy, ap_cfg, mode)

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


def run_ttf(cfg, damping):
    record_every = max(1, cfg.n_steps // 40)
    results = {}
    for label, base_name, mode, ap_cfg in _arms(damping):
        L, S, R = [], [], []
        steps = None
        for seed in range(cfg.n_seeds):
            steps, lv, sb, rl = _run_ttf_arm(base_name, mode, ap_cfg, cfg, seed, record_every)
            L.append(lv); S.append(sb); R.append(rl)
        results[label] = {
            "steps": steps,
            "lval": torch.tensor(L),
            "sub": torch.tensor(S),
            "roll": torch.tensor(R),
        }
        print(f"  {label:22s}  L_val={results[label]['lval'][:, -1].mean():.4f}  "
              f"subspace={results[label]['sub'][:, -1].mean():.4f}  "
              f"rollout={results[label]['roll'][:, -1].mean():.4f} "
              f"(+/-{results[label]['roll'][:, -1].std():.4f})")
    return results


def plot_ttf(results, cfg, damping, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [("lval", "Validation loss (one-step BC)", True),
              ("sub", "Feature subspace distance dist(G, G*)", False),
              ("roll", "Rollout cost (TTF, lower is better)", True)]
    for ax, (key, title, logy) in zip(axes, panels):
        for name, res in results.items():
            mean = res[key].mean(dim=0)
            std = res[key].std(dim=0)
            x = res["steps"]
            ls = "--" if name.startswith("GP->AP") else "-"
            ax.plot(x, mean, label=name, color=_COLORS[name], linestyle=ls)
            ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=_COLORS[name])
        ax.set_title(title)
        ax.set_xlabel("step")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"SNR/DoPr operation order on the TTF task (damping={damping}, "
                 f"{cfg.n_seeds} seeds)")
    fig.tight_layout()
    path = os.path.join(out_dir, "dopr_order_ttf.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    # Bar summary of final rollout cost.
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    names = list(results.keys())
    means = [float(results[n]["roll"][:, -1].mean()) for n in names]
    stds = [float(results[n]["roll"][:, -1].std()) for n in names]
    ax2.bar(names, means, yerr=stds, color=[_COLORS[n] for n in names], capsize=4)
    ax2.set_ylabel("final rollout cost (TTF)")
    ax2.set_title("Final rollout cost by AP/GP order (lower = better)")
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
    fig2.tight_layout()
    path2 = os.path.join(out_dir, "dopr_order_ttf_bar.png")
    fig2.savefig(path2, dpi=120)
    plt.close(fig2)
    print(f"\n  Saved {path} and {path2}")


# ---------------------------------------------------------------------------
# Task 2: repo canonical task (sparse linear regression with label noise)
# ---------------------------------------------------------------------------

EVAL_EVERY = 10


def _build_base_sparse(name, model, cfg):
    if name == "SNRAdamW":
        return SNRAdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, gate="snr",
            rho=cfg.rho, alpha=cfg.alpha, batch_size=cfg.batch_size,
            dataset_size=cfg.n_train, lambda_pop=cfg.lambda_pop,
        )
    if name == "Adam":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(name)


def _run_sparse_arm(base_name, mode, ap_cfg, cfg, seed):
    w_true, _ = make_true_weights(cfg.d, cfg.k, cfg.signal_magnitude)
    train_gen = torch.Generator().manual_seed(seed)
    X_train, y_train = make_dataset(w_true, cfg.n_train, cfg.sigma_noise, train_gen)
    test_gen = torch.Generator().manual_seed(9999)
    X_test, y_test = make_dataset(w_true, cfg.test_size, cfg.sigma_noise, test_gen)

    torch.manual_seed(seed + 1000)
    model = nn.Linear(cfg.d, 1, bias=False)
    nn.init.zeros_(model.weight)
    base = _build_base_sparse(base_name, model, cfg)
    opt = OrderedDoPr(base, model, ap_cfg, mode)

    test_losses, param_errors, steps = [], [], []
    for step in range(cfg.n_steps):
        idx = torch.randint(cfg.n_train, (cfg.batch_size,))
        X_b, y_b = X_train[idx], y_train[idx]
        opt.zero_grad()
        loss = ((model(X_b) - y_b) ** 2).mean()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                test_losses.append(((model(X_test) - y_test) ** 2).mean().item())
            param_errors.append((model.weight.data.squeeze() - w_true).norm().item())
            steps.append(step)
    return steps, test_losses, param_errors


def run_sparse(cfg, damping):
    results = {}
    irreducible = cfg.sigma_noise ** 2
    for label, base_name, mode, ap_cfg in _arms(damping):
        T, P = [], []
        steps = None
        for seed in range(cfg.n_seeds):
            steps, tl, pe = _run_sparse_arm(base_name, mode, ap_cfg, cfg, seed)
            T.append(tl); P.append(pe)
        results[label] = {
            "steps": steps,
            "test": torch.tensor(T),
            "perr": torch.tensor(P),
        }
        excess = results[label]["test"][:, -1].mean() - irreducible
        print(f"  {label:22s}  excess test MSE={excess:7.3f} "
              f"(+/-{results[label]['test'][:, -1].std():.3f})  "
              f"||w-w*||={results[label]['perr'][:, -1].mean():.3f}")
    return results


def plot_sparse(results, cfg, damping, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    irreducible = cfg.sigma_noise ** 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, res in results.items():
        ls = "--" if name.startswith("GP->AP") else "-"
        x = res["steps"]
        for ax, key, transform in [(axes[0], "test", lambda t: t - irreducible),
                                    (axes[1], "perr", lambda t: t)]:
            mean = transform(res[key]).mean(dim=0)
            std = res[key].std(dim=0)
            ax.plot(x, mean, label=name, color=_COLORS[name], linestyle=ls)
            ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=_COLORS[name])
    axes[0].axhline(0, ls="--", color="gray", alpha=0.4)
    axes[0].set_title("(a) Excess test MSE (lower is better)")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("excess test MSE")
    axes[0].legend(fontsize=7)
    axes[1].set_title("(b) Parameter recovery error")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("||w - w*||")
    fig.suptitle(f"SNR/DoPr operation order on the sparse-regression task "
                 f"(damping={damping}, {cfg.n_seeds} seeds)")
    fig.tight_layout()
    path = os.path.join(out_dir, "dopr_order_sparse.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["ttf", "sparse"], default="ttf")
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--out-dir", default="benchmarks")
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.task == "ttf":
        cfg = TTFConfig()
        # The TTF benchmark's own default damping is 1e-2; keep AP comparable to it
        # unless overridden, but default this script to 0.1 (the README AP default).
        if args.quick:
            cfg = replace(cfg, n_steps=150, n_seeds=2, horizon=15)
        print(f"TTF task (DoPr canonical): seeds={cfg.n_seeds}, steps={cfg.n_steps}, "
              f"damping={args.damping}")
        results = run_ttf(cfg, args.damping)
        plot_ttf(results, cfg, args.damping, args.out_dir)
    else:
        cfg = BenchmarkConfig()
        if args.quick:
            cfg = replace(cfg, n_seeds=3, n_steps=1500)
        print(f"Sparse-regression task (repo canonical): seeds={cfg.n_seeds}, "
              f"steps={cfg.n_steps}, damping={args.damping}")
        results = run_sparse(cfg, args.damping)
        plot_sparse(results, cfg, args.damping, args.out_dir)

    print("\nCaveat: the 'post' (GP->AP) order applies AP to the realized weight "
          "delta. Both tasks use weight_decay=0, so the delta is purely the GP "
          "gradient update; under nonzero decoupled weight decay the 'post' order "
          "would also condition the WD term and the comparison would need care.")


if __name__ == "__main__":
    main()
