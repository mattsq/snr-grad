"""
Benchmark: stability-sensor fidelity and simulator checks on a small MLP.

Stage 3 of ``docs/proposals/curvature-aware-stability-control.md``. A
teacher-student regression MLP with power-law (anisotropic) inputs is trained
with several optimizers and learning rates. At checkpoints we measure:

1. **Curvature-proxy fidelity.** For probe update deviations ``d_k = P(g_k - g_bar)``
   (real microbatches, frozen optimizer map), compare ``d^T H d`` under
     * the activation-correlation proxy ``I_out (x) C_h`` (DoPr statistics),
     * the GGN on a fixed held-out batch,
   against the *actual* finite-difference curvature of the held-out loss,
   ``(L(theta + e d) + L(theta - e d) - 2 L(theta)) / e^2``.
   Reported: median log10 ratio to finite difference, and Spearman across
   directions/checkpoints. The activation proxy assumes identity output
   curvature, so a systematic scale error for hidden layers is expected; the
   point is to measure it rather than assume it away.
2. **Simulator vs real continuations.** From the same checkpoint, run real
   short continuations (fresh data, the real optimizer, the real network) and
   simulated ones (``QuadraticStabilitySimulator``, GGN or activation curvature,
   probe-fitted noise). Score whether simulated excursion severity ranks
   configurations like the real ones, and report what each costs.
3. **Cost.** Seconds for probes, curvature evaluations, simulations, and real
   continuations.

Run:
    uv run python benchmark_stability_layers.py
    uv run python benchmark_stability_layers.py --quick
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_grad import (
    ActivationCurvature,
    GGNCurvature,
    QuadraticStabilitySimulator,
    SNRAdamW,
    StabilityProbe,
    collect_microbatch_gradients,
    detect_excursions,
)

OUT_DIR = "benchmarks"


@dataclass
class Config:
    d_in: int = 32
    hidden: int = 64
    d_out: int = 8
    alpha_input: float = 0.9      # input covariance eigenvalues j^(-2 alpha)
    batch_size: int = 16
    steps: int = 1500
    checkpoints: tuple = (500, 1000)
    seeds: int = 3
    heldout: int = 512
    probe_k: int = 32
    continuations: int = 8
    horizon: int = 100
    sim_seeds: int = 8
    fd_eps: float = 1e-3
    arms: tuple = (
        ("sgd", 0.05), ("sgd", 0.3), ("sgd", 1.0), ("sgd", 2.0),
        ("adamw", 3e-3), ("adamw", 3e-2), ("adamw", 0.1),
        ("snr_adamw", 3e-2), ("snr_adamw", 0.1),
    )


def make_opt(name, params, lr):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    if name == "snr_adamw":
        return SNRAdamW(params, lr=lr, betas=(0.9, 0.999), gate="snr")
    raise ValueError(name)


class Task:
    def __init__(self, cfg: Config):
        gen = torch.Generator().manual_seed(0)
        self.cfg = cfg
        self.scale = torch.arange(1, cfg.d_in + 1, dtype=torch.float32) ** (-cfg.alpha_input)
        torch.manual_seed(123)
        self.teacher = nn.Sequential(nn.Linear(cfg.d_in, 32), nn.Tanh(), nn.Linear(32, cfg.d_out))
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.x_val, self.y_val = self.sample(cfg.heldout, gen)

    def sample(self, n, gen):
        x = torch.randn(n, self.cfg.d_in, generator=gen) * self.scale
        with torch.no_grad():
            y = self.teacher(x) * 3.0 + 0.1 * torch.randn(n, self.cfg.d_out, generator=gen)
        return x, y

    def student(self, seed):
        torch.manual_seed(seed)
        c = self.cfg
        return nn.Sequential(nn.Linear(c.d_in, c.hidden), nn.Tanh(),
                             nn.Linear(c.hidden, c.hidden), nn.Tanh(),
                             nn.Linear(c.hidden, c.d_out))


def mse(model, batch):
    x, y = batch
    return 0.5 * ((model(x) - y) ** 2).sum(1).mean()


def val_loss(task, model):
    with torch.no_grad():
        return float(mse(model, (task.x_val, task.y_val)))


def run_real(task, model, opt, steps, gen):
    losses = [val_loss(task, model)]
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        mse(model, task.sample(task.cfg.batch_size, gen)).backward()
        opt.step()
        losses.append(val_loss(task, model))
        if not math.isfinite(losses[-1]) or losses[-1] > 1e6:
            losses.extend([float("inf")] * (steps + 1 - len(losses)))
            break
    return losses


def checkpoint_analysis(task, model, opt, name, lr, cfg, seed):
    out = {}
    gen = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    # Probe from K real microbatches.
    batch = task.sample(cfg.probe_k * cfg.batch_size, gen)
    t0 = time.perf_counter()
    grads = collect_microbatch_gradients(model, mse, batch, num_splits=cfg.probe_k)
    out["probe_grad_seconds"] = time.perf_counter() - t0

    out_loss = lambda o: 0.5 * ((o - task.y_val) ** 2).sum(1).mean()  # noqa: E731
    t0 = time.perf_counter()
    act = ActivationCurvature.from_model(model, lambda m: m(task.x_val))
    out["act_build_seconds"] = time.perf_counter() - t0
    ggn = GGNCurvature(model, task.x_val, out_loss, params=params)

    probe = StabilityProbe.from_microbatches(grads, cfg.batch_size, opt, ggn)
    m = probe.measure(lr=lr, batch_size=cfg.batch_size)
    out["probe_seconds_ggn"] = m.cost["seconds"]
    out["risk_trace_ggn"] = m.risk_trace_per_example
    out["noise_penalty_ggn"] = m.noise_penalty
    out["concentration_status"] = m.concentration_status
    out["concentration_ggn"] = m.concentration
    out["gns_optimizer"] = m.gns_optimizer
    m_act = StabilityProbe.from_microbatches(grads, cfg.batch_size, opt, act).measure(
        lr=lr, batch_size=cfg.batch_size)
    out["risk_trace_act"] = m_act.risk_trace_per_example
    out["probe_seconds_act"] = m_act.cost["seconds"]

    # Fidelity: per-direction quad values vs finite-difference curvature.
    pmap = probe.update_map
    fid = []
    base = val_loss(task, model)
    for dev in probe.deviations()[:8]:
        d = pmap(dev)
        norm = math.sqrt(sum(float((v ** 2).sum()) for v in d.values()))
        if norm == 0:
            continue
        d = {p: v / norm for p, v in d.items()}
        e = cfg.fd_eps
        saved = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p in params:
                p.add_(d[p], alpha=e)
            lp = val_loss(task, model)
            for p, s in zip(params, saved):
                p.copy_(s)
                p.sub_(d[p], alpha=e)
            lm = val_loss(task, model)
            for p, s in zip(params, saved):
                p.copy_(s)
        fd = (lp + lm - 2 * base) / (e * e)
        fid.append({"fd": fd, "ggn": ggn.quad(d), "act": act.quad(d)})
    out["fidelity"] = fid

    # Real continuations from this checkpoint.
    t0 = time.perf_counter()
    reals = []
    for i in range(cfg.continuations):
        m2 = copy.deepcopy(model)
        o2 = make_opt(name, m2.parameters(), lr)
        o2.load_state_dict(copy.deepcopy(opt.state_dict()))
        reals.append(run_real(task, m2, o2, cfg.horizon, torch.Generator().manual_seed(seed + 100 + i)))
    out["real_seconds"] = time.perf_counter() - t0
    reps = [detect_excursions(r, window=20, warmup=10) for r in reals]
    out["real_severity"] = sum(min(r.max_excursion, 100.0) for r in reps) / len(reps)
    out["real_event_fraction"] = sum(r.spiked for r in reps) / len(reps)
    out["real_final_change"] = float(torch.tensor([r[-1] - r[0] for r in reals]).median())

    for key, curv in (("ggn", ggn), ("act", act)):
        t0 = time.perf_counter()
        sim = QuadraticStabilitySimulator.from_checkpoint(
            opt, probe, curv, model=model, batch_size=cfg.batch_size, loss0=base,
            noise_scaling="loss_proportional",
        )
        dist = sim.run(cfg.sim_seeds, cfg.horizon, seed=seed + 7,
                       excursion_kwargs={"window": 20, "warmup": 10})
        sm = dist.summary()
        out[f"sim_{key}_severity"] = sm["excursion_severity_mean"]
        out[f"sim_{key}_event_fraction"] = sm["spike_fraction"]
        out[f"sim_{key}_final_change"] = sm["final_loss_median"] - base
        out[f"sim_{key}_seconds"] = time.perf_counter() - t0
    return out


def spearman(x, y):
    def rank(v):
        t = torch.tensor(v, dtype=torch.float64)
        r = torch.empty_like(t)
        r[torch.argsort(t)] = torch.arange(len(v), dtype=torch.float64)
        return r
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = float(rx.norm() * ry.norm())
    return float((rx * ry).sum()) / den if den > 0 else float("nan")


def _finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default=OUT_DIR)
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads (small ops are faster single-threaded)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = Config()
    if args.quick:
        cfg = Config(steps=400, checkpoints=(200,), seeds=2, probe_k=16, continuations=4,
                     horizon=40, sim_seeds=4,
                     arms=(("sgd", 0.05), ("sgd", 0.5), ("adamw", 3e-3), ("adamw", 3e-2)))
    os.makedirs(args.out, exist_ok=True)
    task = Task(cfg)
    t_start = time.perf_counter()
    rows = []
    for name, lr in cfg.arms:
        for seed in range(cfg.seeds):
            model = task.student(seed)
            opt = make_opt(name, model.parameters(), lr)
            gen = torch.Generator().manual_seed(1000 + seed)
            losses = [val_loss(task, model)]
            for t in range(1, cfg.steps + 1):
                opt.zero_grad(set_to_none=True)
                mse(model, task.sample(cfg.batch_size, gen)).backward()
                opt.step()
                losses.append(val_loss(task, model))
                if not math.isfinite(losses[-1]):
                    break
                if t in cfg.checkpoints:
                    res = checkpoint_analysis(task, model, opt, name, lr, cfg, 50_000 + 31 * seed + t)
                    rows.append({"optimizer": name, "lr": lr, "seed": seed, "step": t, **res})
            rep = detect_excursions(losses)
            print(f"{name:>10} lr={lr:<7g} seed={seed} final val={losses[-1]:.4g} "
                  f"events={rep.num_events} diverged={rep.diverged}")

    # Fidelity summary.
    fid = [f for r in rows for f in r["fidelity"] if f["fd"] > 0]
    def log_ratio(key):
        vals = [math.log10(max(f[key], 1e-30) / f["fd"]) for f in fid]
        return float(torch.tensor(vals).median()) if vals else float("nan")
    fidelity = {
        "directions": len(fid),
        "negative_fd_fraction": 1 - len(fid) / max(1, sum(len(r["fidelity"]) for r in rows)),
        "ggn_log10_ratio_median": log_ratio("ggn"),
        "act_log10_ratio_median": log_ratio("act"),
        "ggn_spearman": spearman([f["ggn"] for f in fid], [f["fd"] for f in fid]),
        "act_spearman": spearman([f["act"] for f in fid], [f["fd"] for f in fid]),
    }
    print("\n=== Curvature-proxy fidelity vs finite-difference held-out curvature ===")
    for k, v in fidelity.items():
        print(f"   {k:<28} {v:.3f}" if isinstance(v, float) else f"   {k:<28} {v}")

    print("\n=== Ranking real continuation severity across checkpoints ===")
    ranking = {}
    for key in ("sim_ggn_severity", "sim_act_severity", "noise_penalty_ggn", "gns_optimizer",
                "risk_trace_ggn", "risk_trace_act"):
        rs = [r for r in rows if _finite(r.get(key)) and _finite(r["real_severity"])]
        ranking[key] = spearman([r[key] for r in rs], [r["real_severity"] for r in rs])
        print(f"   {key:<24} spearman={ranking[key]:+.3f} (n={len(rs)})")
    rs = [r for r in rows if _finite(r.get("sim_ggn_final_change"))]
    ranking["sim_ggn_final_change_vs_real"] = spearman(
        [r["sim_ggn_final_change"] for r in rs], [r["real_final_change"] for r in rs])
    print(f"   sim_ggn final-change vs real final-change spearman="
          f"{ranking['sim_ggn_final_change_vs_real']:+.3f}")

    cost = {k: float(torch.tensor([r[k] for r in rows]).median())
            for k in ("probe_grad_seconds", "probe_seconds_ggn", "probe_seconds_act",
                      "sim_ggn_seconds", "sim_act_seconds", "real_seconds")}
    print("\n=== Median cost per checkpoint (s) ===")
    for k, v in cost.items():
        print(f"   {k:<24} {v:.3f}")

    # Plot: fidelity scatter and sim-vs-real severity.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog([f["fd"] for f in fid], [f["ggn"] for f in fid], "o", ms=3, label="GGN")
    axes[0].loglog([f["fd"] for f in fid], [f["act"] for f in fid], "x", ms=3, label="activation proxy")
    lim = [min(f["fd"] for f in fid), max(f["fd"] for f in fid)] if fid else [1e-3, 1]
    axes[0].loglog(lim, lim, "k--", lw=0.8)
    axes[0].set_xlabel("finite-difference d^T H d (held-out)")
    axes[0].set_ylabel("proxy d^T H d")
    axes[0].legend(fontsize=8)
    for name in sorted({r["optimizer"] for r in rows}):
        rr = [r for r in rows if r["optimizer"] == name]
        axes[1].scatter([r["sim_ggn_severity"] for r in rr], [r["real_severity"] for r in rr],
                        s=14, label=name)
    axes[1].set_xlabel("simulated severity (GGN)")
    axes[1].set_ylabel("real continuation severity")
    axes[1].set_xscale("symlog", linthresh=0.01)
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "stability_layers.png"), dpi=120)
    plt.close(fig)

    results = {"config": asdict(cfg), "fidelity": fidelity, "ranking": ranking, "cost": cost,
               "checkpoints": rows, "seconds": time.perf_counter() - t_start}
    suffix = "_quick" if args.quick else ""
    path = os.path.join(args.out, f"stability_layers_results{suffix}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nWrote {path} ({results['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
