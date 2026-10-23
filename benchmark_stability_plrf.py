"""
Benchmark: curvature-aware stability sensing on power-law random features (PLRF).

Stages 0-2 (and the simulator check, M3) of
``docs/proposals/curvature-aware-stability-control.md``, on the convex PLRF
quadratic studied by Tian & Paquette (arXiv:2610.13524):

    x ~ N(0, K),  K = diag(j^(-2 alpha_spectrum)),  j = 1..V
    target y = <x, b>,  b_j = j^(-beta)
    model    f(x) = <W^T x, theta>,  W in R^{V x D} fixed, W_ij ~ N(0, 1/D)
    loss     0.5 (f(x) - y)^2

Everything is exact in the D-dimensional feature space: ``z = W^T x`` and ``y``
are jointly Gaussian, so minibatches are sampled exactly from that
``(D+1)``-dimensional Gaussian, and

    population loss  L(theta) = 0.5 (theta^T H theta - 2 theta^T c + y2)
    Hessian          H = W^T K W,  c = W^T K b,  y2 = b^T K b
    mean gradient    mu = H theta - c
    per-example cov  Sigma = 2 L(theta) H + mu mu^T

so the probe, the frozen-update sensor and the simulator can all be checked
against ground truth. The "validation loss" is the exact population loss (no
evaluation noise).

Stages:
  0. For each spectrum exponent and optimizer, measure the stability edge (the
     largest learning rate that still makes progress) by bisection, then sweep
     learning rates as fractions of it and classify trajectories as smooth,
     intermittent excursions, or divergence (shared, preregistered definition:
     ``snr_grad.detect_excursions``).
  1. At checkpoints of those runs, compute baseline and curvature-aware sensors
     (exact and probe-estimated) and compare them with the excursion severity of
     independent *real* short continuations from the same checkpoint. Even data
     seeds are "dev", odd seeds are held out; sensors have no fitted parameters
     and only held-out checkpoints are scored.
  2. Optimizer ablations ride along (SNRAdamW, a slow beta2) via ``--optimizers``.
  M3. Checkpoint-local simulator (``QuadraticStabilitySimulator``) with probe-fitted
     or exact noise, scored against the same real continuations.

Run:
    uv run python benchmark_stability_plrf.py            # default (D=256)
    uv run python benchmark_stability_plrf.py --quick    # smoke test
    uv run python benchmark_stability_plrf.py --paper    # D=1024, V=20000, B=8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_grad import (
    CovarianceFactorNoise,
    ExactQuadraticCurvature,
    GaussianProbeNoise,
    QuadraticStabilitySimulator,
    SNRAdamW,
    StabilityProbe,
    detect_excursions,
    exact_noise_geometry,
    frozen_update_map,
    spectrum_metrics,
)

DTYPE = torch.float64
OUT_DIR = "benchmarks"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    D: int = 256
    V: int = 4096
    batch_size: int = 8
    alphas: tuple = (0.3, 0.5, 0.7, 0.9)
    beta: float = 0.7
    steps: int = 4000
    seeds: int = 8
    optimizers: tuple = ("sgd", "adamw", "snr_adamw")
    lr_mults: tuple = (0.3, 0.5, 0.7, 0.85, 0.95)
    edge_iters: int = 10
    checkpoint_fracs: tuple = (0.25, 0.5)
    checkpoint_seeds: int = 4         # seeds per config used for checkpoints
    continuations: int = 16
    horizon: int = 200
    probe_k: int = 32
    sim_seeds: int = 16
    excess_threshold: float = 0.5     # preregistered "spiky continuation" threshold
    spiky_fraction: float = 0.25      # checkpoint label: >= this fraction spiky
    feature_seed: int = 0
    # Preregistered excursion definition for full trajectories.
    window: int = 50
    k_mad: float = 6.0
    rel_floor: float = 0.25
    divergence_factor: float = 100.0


# ---------------------------------------------------------------------------
# PLRF problem (exact, in feature space)
# ---------------------------------------------------------------------------

class PLRF:
    def __init__(self, cfg: Config, alpha: float):
        gen = torch.Generator().manual_seed(cfg.feature_seed)
        j = torch.arange(1, cfg.V + 1, dtype=DTYPE)
        k = j ** (-2.0 * alpha)
        b = j ** (-cfg.beta)
        W = torch.randn(cfg.V, cfg.D, generator=gen, dtype=DTYPE) / math.sqrt(cfg.D)
        kw = k[:, None] * W
        self.H = W.t() @ kw
        self.H = 0.5 * (self.H + self.H.t())
        self.c = kw.t() @ b
        self.y2 = float((k * b * b).sum())
        joint = torch.zeros(cfg.D + 1, cfg.D + 1, dtype=DTYPE)
        joint[: cfg.D, : cfg.D] = self.H
        joint[: cfg.D, cfg.D] = self.c
        joint[cfg.D, : cfg.D] = self.c
        joint[cfg.D, cfg.D] = self.y2
        ev, evec = torch.linalg.eigh(joint)
        self.F = evec * ev.clamp_min(0.0).sqrt()       # joint = F F^T
        hev, hvec = torch.linalg.eigh(self.H)
        self.H_sqrt = (hvec * hev.clamp_min(0.0).sqrt()) @ hvec.t()
        self.lambda_max = float(hev.max())
        self.alpha = alpha
        self.D = cfg.D
        self.B = cfg.batch_size
        theta_star = torch.linalg.lstsq(self.H, self.c[:, None]).solution[:, 0]
        self.loss_star = float(self.loss(theta_star[None])[0])
        self.spectrum = spectrum_metrics(hev)

    def loss(self, theta: torch.Tensor) -> torch.Tensor:
        """Population loss for a [S, D] batch of parameter vectors."""
        q = ((theta @ self.H) * theta).sum(-1)
        return 0.5 * (q - 2.0 * theta @ self.c + self.y2)

    def pop_grad(self, theta: torch.Tensor) -> torch.Tensor:
        return theta @ self.H - self.c

    def minibatch_grad(self, theta: torch.Tensor, gen: torch.Generator, b: int | None = None):
        b = self.B if b is None else b
        s = theta.shape[0]
        u = torch.randn(s, b, self.D + 1, generator=gen, dtype=DTYPE)
        zy = u @ self.F.t()
        z, y = zy[..., : self.D], zy[..., self.D]
        r = (z * theta[:, None, :]).sum(-1) - y
        return (z * r[..., None]).mean(1)

    def sigma(self, theta: torch.Tensor) -> torch.Tensor:
        """Exact per-example gradient covariance at a single [D] theta."""
        L = float(self.loss(theta[None])[0])
        mu = self.pop_grad(theta[None])[0]
        return 2.0 * L * self.H + torch.outer(mu, mu)

    def sigma_factor(self, theta: torch.Tensor) -> torch.Tensor:
        """F with Sigma = F F^T (exact, [D, D+1])."""
        L = float(self.loss(theta[None])[0])
        mu = self.pop_grad(theta[None])[0]
        return torch.cat([math.sqrt(2.0 * L) * self.H_sqrt, mu[:, None]], dim=1)


# ---------------------------------------------------------------------------
# Optimizers on stacked (elementwise-independent) seeds
# ---------------------------------------------------------------------------

def make_opt(name: str, params, lr: float):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    if name == "adamw_b2_095":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    if name == "snr_adamw":
        return SNRAdamW(params, lr=lr, betas=(0.9, 0.999), gate="snr", rho=0.99)
    raise ValueError(name)


def _map_state(opt, fn):
    """Return a state_dict with every per-coordinate state tensor mapped by fn."""
    sd = opt.state_dict()
    new_state = {}
    for idx, st in sd["state"].items():
        new_state[idx] = {
            k: (fn(v) if torch.is_tensor(v) and v.ndim == 2 else
                (v.clone() if torch.is_tensor(v) else v))
            for k, v in st.items()
        }
    return {**sd, "state": new_state}


def slice_optimizer(name, opt, theta_stack, s, lr, *, repeat: int | None = None):
    """Single-seed (``repeat=None``, param shape [D]) or replicated ([repeat, D]) copy."""
    if repeat is None:
        p = nn.Parameter(theta_stack[s].detach().clone())
        sd = _map_state(opt, lambda v: v[s].clone())
    else:
        p = nn.Parameter(theta_stack[s].detach().clone()[None].repeat(repeat, 1))
        sd = _map_state(opt, lambda v: v[s].clone()[None].repeat(repeat, 1))
    new = make_opt(name, [p], lr)
    if opt.state:
        new.load_state_dict(sd)
    return p, new


def train(prob: PLRF, name: str, lr: float, seeds: int, steps: int, *,
          data_seed: int = 1000, checkpoints=(), divergence_factor=100.0):
    """Train ``seeds`` independent runs stacked into one [S, D] parameter.

    Valid for elementwise optimizers (SGD, Adam, SNRAdamW without adaptive
    thresholds): each row evolves independently. Returns the population-loss
    curves and the stacked (theta, optimizer) at the requested checkpoint steps.
    """
    theta = nn.Parameter(torch.zeros(seeds, prob.D, dtype=DTYPE))
    opt = make_opt(name, [theta], lr)
    gen = torch.Generator().manual_seed(data_seed)
    losses = torch.empty(seeds, steps + 1, dtype=DTYPE)
    with torch.no_grad():
        losses[:, 0] = prob.loss(theta)
    l0 = float(losses[0, 0])
    dead = torch.zeros(seeds, dtype=torch.bool)
    snaps = {}
    for t in range(steps):
        if t in checkpoints:
            # Frozen copy of (theta, optimizer state) at this step.
            p_copy = nn.Parameter(theta.detach().clone())
            o_copy = make_opt(name, [p_copy], lr)
            if opt.state:
                o_copy.load_state_dict(_map_state(opt, lambda v: v.clone()))
            snaps[t] = (p_copy.detach().clone(), o_copy)
        with torch.no_grad():
            g = prob.minibatch_grad(theta.detach(), gen)
            g[dead] = 0.0
            theta.grad = g
        opt.step()
        with torch.no_grad():
            lt = prob.loss(theta)
            dead |= ~torch.isfinite(lt) | (lt > divergence_factor * 1e4 * l0)
            # Park diverged rows at a finite value so they cannot poison the others.
            theta.data[dead] = 0.0
            lt[dead] = float("inf")
            losses[:, t + 1] = lt
    return losses, snaps


def _seed_failed(curve: torch.Tensor, l0: float, factor: float) -> bool:
    if not bool(torch.isfinite(curve).all()) or float(curve.max()) > factor * l0:
        return True
    tail = curve[-max(1, curve.numel() // 10):]
    return float(tail.median()) >= l0


def find_edge(prob: PLRF, name: str, cfg: Config) -> float:
    """Largest lr at which a majority of seeds make progress without blow-up."""
    steps = max(200, cfg.steps // 2)
    seeds = max(2, cfg.seeds // 2)

    def ok(lr):
        losses, _ = train(prob, name, lr, seeds, steps, data_seed=777)
        l0 = float(losses[0, 0])
        fails = sum(_seed_failed(losses[s], l0, cfg.divergence_factor) for s in range(seeds))
        return fails < seeds / 2

    if name.startswith("sgd"):
        lo, hi = 0.01 / prob.lambda_max, 8.0 / prob.lambda_max
    else:
        lo, hi = 1e-5, 2.0
    while not ok(lo):
        lo /= 4
    while ok(hi):
        hi *= 4
    for _ in range(cfg.edge_iters):
        mid = math.sqrt(lo * hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Stage 1: checkpoint sensors vs real continuations
# ---------------------------------------------------------------------------

def real_continuations(prob, name, opt, theta_stack, s, lr, cfg, seed):
    p, o = slice_optimizer(name, opt, theta_stack, s, lr, repeat=cfg.continuations)
    pd, od = slice_optimizer(name, opt, theta_stack, s, lr, repeat=1)
    gen = torch.Generator().manual_seed(seed)
    T = cfg.horizon
    noisy = torch.empty(cfg.continuations, T + 1, dtype=DTYPE)
    det = torch.empty(T + 1, dtype=DTYPE)
    with torch.no_grad():
        noisy[:, 0] = prob.loss(p)
        det[0] = prob.loss(pd)[0]
    for t in range(T):
        with torch.no_grad():
            p.grad = prob.minibatch_grad(p.detach(), gen)
            pd.grad = prob.pop_grad(pd.detach())
        o.step()
        od.step()
        with torch.no_grad():
            lt = prob.loss(p)
            lt[~torch.isfinite(lt)] = float("inf")
            noisy[:, t + 1] = lt
            det[t + 1] = prob.loss(pd)[0]
    excess = (noisy[:, 1:] / det[1:].clamp_min(1e-300) - 1.0).max(1).values
    excess[~torch.isfinite(excess)] = float("inf")
    reports = [detect_excursions(noisy[i], window=20, warmup=10,
                                 divergence_factor=cfg.divergence_factor)
               for i in range(cfg.continuations)]
    return {
        "severity": sum(min(r.max_excursion, 100.0) for r in reports) / len(reports),
        "excess_median": float(excess.median()),
        "excess_mean_clipped": float(excess.clamp_max(100.0).mean()),
        "spiky_fraction": float((excess > cfg.excess_threshold).to(DTYPE).mean()),
        "event_fraction": sum(r.spiked for r in reports) / len(reports),
        "diverged_fraction": sum(r.diverged for r in reports) / len(reports),
    }


def checkpoint_sensors(prob, name, opt, theta_stack, s, lr, cfg, seed):
    p, o = slice_optimizer(name, opt, theta_stack, s, lr)
    theta = p.detach()
    L = float(prob.loss(theta[None])[0])
    mu = prob.pop_grad(theta[None])[0]
    sigma = prob.sigma(theta)
    B = cfg.batch_size
    out = {"loss": L, "excess_loss": L - prob.loss_star}

    # Exact frozen update map (exact Adam Jacobian at mu) and its diagonal.
    pmap = frozen_update_map(o, at_gradient={p: mu})
    pdiag = pmap({p: torch.ones_like(mu)})[p] if pmap.supported else None
    pfroz = frozen_update_map(o)
    pfdiag = pfroz({p: torch.ones_like(mu)})[p] if pfroz.supported else torch.ones_like(mu)

    out["gns_euclidean"] = float(sigma.diagonal().sum() / (mu @ mu))
    pm = pdiag * mu
    out["gns_optimizer"] = float((pdiag ** 2 * sigma.diagonal()).sum() / (pm @ pm))
    out["lambda_max"] = prob.lambda_max
    ph = pfdiag.clamp_min(0).sqrt()
    out["lr_sharpness"] = lr * float(torch.linalg.eigvalsh(ph[:, None] * prob.H * ph[None, :]).max())
    out["activation_participation_ratio"] = prob.spectrum["participation_ratio"]
    out["activation_slope"] = prob.spectrum["powerlaw_slope"]

    raw = exact_noise_geometry(prob.H, sigma)
    geo = exact_noise_geometry(prob.H, sigma, pdiag)
    progress = float(mu @ pm)
    out["noise_penalty_raw"] = lr * lr * raw["risk_trace"] / (2 * B)
    out["noise_penalty"] = lr * lr * geo["risk_trace"] / (2 * B)
    out["noise_to_progress"] = out["noise_penalty"] / (lr * abs(progress))
    out["noise_to_loss"] = out["noise_penalty"] / out["excess_loss"]
    out["concentration"] = geo["concentration"]
    out["rel_sd"] = math.sqrt(2.0 / geo["concentration"])
    out["upper_tail_ratio"] = out["noise_to_loss"] * (1.0 + out["rel_sd"])
    out["risk_trace_exact"] = geo["risk_trace"]

    # Probe estimate from K real microbatches at the checkpoint.
    gen = torch.Generator().manual_seed(seed)
    grads = [{p: prob.minibatch_grad(theta[None], gen)[0]} for _ in range(cfg.probe_k)]
    curv = ExactQuadraticCurvature([p], prob.H)
    probe = StabilityProbe.from_microbatches(grads, B, o, curv)
    m = probe.measure(lr=lr, batch_size=B)
    out["probe_risk_trace"] = m.risk_trace_per_example
    out["probe_risk_trace_se"] = m.risk_trace_se
    out["probe_noise_to_loss"] = (m.noise_penalty / out["excess_loss"]) if m.noise_penalty else None
    out["probe_gns_euclidean"] = m.gns_euclidean
    out["probe_concentration"] = m.concentration
    out["probe_concentration_status"] = m.concentration_status
    out["probe_seconds"] = m.cost["seconds"]

    # M3: simulator, probe-fitted noise (fixed / loss-proportional) and exact noise.
    devs = torch.stack([g[p] for g in probe.deviations()])
    variants = {
        "sim_probe_fixed": GaussianProbeNoise(devs, B, B),
        "sim_probe_lossprop": GaussianProbeNoise(devs, B, B, scaling="loss_proportional"),
        "sim_exact_lossprop": CovarianceFactorNoise(prob.sigma_factor(theta), B,
                                                    scaling="loss_proportional"),
    }
    for key, noise in variants.items():
        t0 = time.perf_counter()
        sim = QuadraticStabilitySimulator([p], o, curv, mu, noise, loss0=L)
        dist = sim.run(cfg.sim_seeds, cfg.horizon, seed=seed + 1,
                       thresholds=(cfg.excess_threshold,),
                       divergence_factor=cfg.divergence_factor,
                       excursion_kwargs={"window": 20, "warmup": 10})
        sm = dist.summary()
        out[f"{key}_severity"] = sm["excursion_severity_mean"]
        out[f"{key}_event_fraction"] = sm["spike_fraction"]
        out[f"{key}_excess_median"] = sm["max_rel_excess_median"]
        out[f"{key}_seconds"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _rank(x: list[float]) -> torch.Tensor:
    t = torch.tensor(x, dtype=DTYPE)
    order = torch.argsort(t)
    r = torch.empty_like(t)
    r[order] = torch.arange(len(x), dtype=DTYPE)
    # Average ties.
    for v in torch.unique(t):
        m = t == v
        if int(m.sum()) > 1:
            r[m] = r[m].mean()
    return r


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = float(rx.norm() * ry.norm())
    return float((rx * ry).sum()) / den if den > 0 else float("nan")


def auroc(score: list[float], label: list[bool]) -> float:
    pos = [s for s, l in zip(score, label) if l]
    neg = [s for s, l in zip(score, label) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


SENSORS = [
    "gns_euclidean", "gns_optimizer", "lr_sharpness", "activation_participation_ratio",
    "noise_penalty_raw", "noise_penalty", "noise_to_progress", "noise_to_loss",
    "rel_sd", "upper_tail_ratio", "probe_noise_to_loss",
    "sim_probe_fixed_severity", "sim_probe_lossprop_severity", "sim_exact_lossprop_severity",
]


def _finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def score_sensors(rows: list[dict], target="severity", spiky_fraction=0.25) -> dict:
    """Spearman vs excursion severity; AUROC for 'spiky' checkpoints.

    A checkpoint is spiky when at least ``spiky_fraction`` of its real
    continuations contain an excursion event (preregistered definition).
    """
    out = {}
    for sensor in SENSORS:
        pairs = [(r[sensor], r[target], r["event_fraction"] >= spiky_fraction)
                 for r in rows if _finite(r.get(sensor)) and _finite(r.get(target))]
        if len(pairs) < 3:
            continue
        xs, ys, ls = zip(*pairs)
        out[sensor] = {
            "n": len(pairs),
            "spearman": spearman(list(xs), list(ys)),
            "auroc_spiky": auroc(list(xs), list(ls)),
        }
    return out


def stratified_scores(rows: list[dict], target="severity") -> dict:
    """Mean within-(optimizer, alpha) Spearman: removes between-spectrum confounding.

    Sensors that are constant within a stratum (spectrum-only proxies such as the
    activation participation ratio) get no credit here by construction.
    """
    out = {}
    strata = sorted({(r["optimizer"], r["alpha"]) for r in rows})
    for sensor in SENSORS:
        vals = []
        for key in strata:
            rs = [r for r in rows if (r["optimizer"], r["alpha"]) == key
                  and _finite(r.get(sensor)) and _finite(r.get(target))]
            if len(rs) < 4 or len({r[sensor] for r in rs}) < 2 or len({r[target] for r in rs}) < 2:
                continue
            rho = spearman([r[sensor] for r in rs], [r[target] for r in rs])
            if math.isfinite(rho):
                vals.append(rho)
        if vals:
            out[sensor] = {"strata": len(vals), "mean_spearman": sum(vals) / len(vals)}
    return out


def matched_gns_pairs(rows: list[dict], tol=1.25, gns_key="gns_optimizer") -> dict:
    """Pairs at similar GNS (same optimizer) but different spectra."""
    result = {}
    for sensor in ("noise_to_loss", "upper_tail_ratio", "rel_sd", "lr_sharpness",
                   "sim_exact_lossprop_severity", "gns_euclidean"):
        agree = total = 0
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if a["optimizer"] != b["optimizer"] or a["alpha"] == b["alpha"]:
                    continue
                if not (_finite(a[gns_key]) and _finite(b[gns_key])):
                    continue
                if abs(math.log(a[gns_key] / b[gns_key])) > math.log(tol):
                    continue
                da = a["severity"] - b["severity"]
                if abs(da) < 0.05:
                    continue
                if not (_finite(a.get(sensor)) and _finite(b.get(sensor))):
                    continue
                total += 1
                agree += (a[sensor] - b[sensor]) * da > 0
        result[sensor] = {"pairs": total, "concordance": agree / total if total else float("nan")}
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--paper", action="store_true", help="D=1024, V=20000, B=8")
    ap.add_argument("--optimizers", type=str, default=None,
                    help="comma list from sgd,sgd_momentum,adamw,adamw_b2_095,snr_adamw")
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()
    cfg = Config()
    if args.quick:
        cfg = Config(D=96, V=1024, alphas=(0.3, 0.9), steps=1200, seeds=4, optimizers=("sgd", "adamw"),
                     lr_mults=(0.5, 0.9), edge_iters=6, checkpoint_seeds=2,
                     continuations=8, horizon=100, probe_k=24, sim_seeds=8)
    if args.paper:
        cfg.D, cfg.V = 1024, 20000
    if args.optimizers:
        cfg.optimizers = tuple(args.optimizers.split(","))
    os.makedirs(args.out, exist_ok=True)
    t_start = time.perf_counter()
    print(f"Config: {cfg}")

    problems = {a: PLRF(cfg, a) for a in cfg.alphas}
    for a, prob in problems.items():
        print(f"alpha_spectrum={a}: lambda_max={prob.lambda_max:.4g} "
              f"PR(H)={prob.spectrum['participation_ratio']:.2f} "
              f"slope={prob.spectrum['powerlaw_slope']:.2f} L*={prob.loss_star:.4g}")

    # ---------------- Stage 0 ----------------
    stage0 = []
    curves = {}
    ckpt_steps = tuple(int(f * cfg.steps) for f in cfg.checkpoint_fracs)
    ckpt_store = {}
    for name in cfg.optimizers:
        for a, prob in problems.items():
            edge = find_edge(prob, name, cfg)
            print(f"[stage0] {name:>12} alpha={a}: edge lr = {edge:.4g}")
            for mult in cfg.lr_mults:
                lr = mult * edge
                losses, snaps = train(prob, name, lr, cfg.seeds, cfg.steps,
                                      checkpoints=ckpt_steps)
                reps = [detect_excursions(losses[s], window=cfg.window, k_mad=cfg.k_mad,
                                          rel_floor=cfg.rel_floor,
                                          divergence_factor=cfg.divergence_factor)
                        for s in range(cfg.seeds)]
                n_div = sum(r.diverged for r in reps)
                n_spk = sum(r.spiked and not r.diverged for r in reps)
                row = {
                    "optimizer": name, "alpha": a, "edge_lr": edge, "lr_mult": mult, "lr": lr,
                    "spike_fraction": n_spk / cfg.seeds,
                    "divergence_fraction": n_div / cfg.seeds,
                    "mean_events": sum(r.num_events for r in reps) / cfg.seeds,
                    "max_excursion_median": float(torch.tensor(
                        [r.max_excursion for r in reps]).median()),
                    "final_excess_loss_median": float(
                        (losses[:, -1] - prob.loss_star).median()),
                }
                row["regime"] = ("divergent" if n_div > cfg.seeds / 2 else
                                 "intermittent" if n_spk > 0 else "smooth")
                stage0.append(row)
                curves[(name, a, mult)] = losses
                ckpt_store[(name, a, mult)] = (snaps, lr)
                print(f"   lr={mult:.2f}*edge  regime={row['regime']:<12} "
                      f"spike={row['spike_fraction']:.2f} div={row['divergence_fraction']:.2f} "
                      f"events={row['mean_events']:.2f} maxexc={row['max_excursion_median']:.3f}")

    # ---------------- Stage 1 / M3 ----------------
    rows = []
    for (name, a, mult), (snaps, lr) in ckpt_store.items():
        prob = problems[a]
        for t, (theta_stack, opt) in snaps.items():
            for s in range(min(cfg.checkpoint_seeds, cfg.seeds)):
                if not bool(torch.isfinite(theta_stack[s]).all()):
                    continue
                if float(prob.loss(theta_stack[s][None])[0]) > cfg.divergence_factor * prob.y2:
                    continue
                seed = 10_000 + 97 * s + t
                real = real_continuations(prob, name, opt, theta_stack, s, lr, cfg, seed)
                sens = checkpoint_sensors(prob, name, opt, theta_stack, s, lr, cfg, seed + 5)
                rows.append({"optimizer": name, "alpha": a, "lr_mult": mult, "step": t,
                             "seed": s, "split": "dev" if s % 2 == 0 else "heldout",
                             **real, **sens})
        print(f"[stage1] {name} alpha={a} lr={mult:.2f}*edge: {len(rows)} checkpoints so far")

    held = [r for r in rows if r["split"] == "heldout"]
    scores = {"pooled": score_sensors(held, spiky_fraction=cfg.spiky_fraction)}
    for name in cfg.optimizers:
        scores[name] = score_sensors([r for r in held if r["optimizer"] == name],
                                     spiky_fraction=cfg.spiky_fraction)
    strat = stratified_scores(held)
    pairs = matched_gns_pairs(held)
    probe_err = [abs(r["probe_risk_trace"] / r["risk_trace_exact"] - 1.0) for r in rows
                 if _finite(r.get("probe_risk_trace"))]
    probe_cov = [abs(r["probe_risk_trace"] - r["risk_trace_exact"]) <= 2 * r["probe_risk_trace_se"]
                 for r in rows if _finite(r.get("probe_risk_trace_se"))]
    statuses = {}
    for r in rows:
        statuses[r["probe_concentration_status"]] = statuses.get(r["probe_concentration_status"], 0) + 1

    print("\n=== Stage 1: held-out sensor scores "
          "(target = excursion severity of real continuations) ===")
    for group, sc in scores.items():
        print(f"-- {group}")
        for sensor, v in sorted(sc.items(), key=lambda kv: -abs(kv[1]['spearman'])):
            print(f"   {sensor:<36} n={v['n']:>3} spearman={v['spearman']:+.3f} "
                  f"auroc={v['auroc_spiky']:.3f}")
    print("\n=== Within-(optimizer, alpha) mean Spearman (held-out) ===")
    for sensor, v in sorted(strat.items(), key=lambda kv: -kv[1]["mean_spearman"]):
        print(f"   {sensor:<36} strata={v['strata']:>2} mean_spearman={v['mean_spearman']:+.3f}")
    print("\n=== Matched-GNS pairs (same optimizer, |log GNS ratio| < log 1.25, different alpha) ===")
    for sensor, v in pairs.items():
        print(f"   {sensor:<36} pairs={v['pairs']:>3} concordance={v['concordance']:.3f}")
    if probe_err:
        print(f"\nProbe R (K={cfg.probe_k}) relative error: median "
              f"{float(torch.tensor(probe_err).median()):.3f}; "
              f"|err| <= 2 SE in {sum(probe_cov)}/{len(probe_cov)} checkpoints")
    print(f"Probe concentration status counts: {statuses}")

    # ---------------- Plots ----------------
    _plot_curves(curves, cfg, args.out)
    _plot_regimes(stage0, cfg, args.out)
    _plot_scatter(held, args.out)

    results = {
        "config": asdict(cfg),
        "problems": {str(a): {"lambda_max": p.lambda_max, "loss_star": p.loss_star,
                              **{k: v for k, v in p.spectrum.items()}}
                     for a, p in problems.items()},
        "stage0": stage0,
        "checkpoints": rows,
        "scores_heldout": scores,
        "stratified_scores_heldout": strat,
        "matched_gns_pairs": pairs,
        "probe_risk_relative_error_median": (float(torch.tensor(probe_err).median())
                                             if probe_err else None),
        "probe_risk_2se_coverage": (sum(probe_cov) / len(probe_cov)) if probe_cov else None,
        "probe_concentration_status": statuses,
        "seconds": time.perf_counter() - t_start,
    }
    suffix = "_quick" if args.quick else "_paper" if args.paper else ""
    path = os.path.join(args.out, f"stability_plrf_results{suffix}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nWrote {path} ({results['seconds']:.0f}s)")


def _plot_curves(curves, cfg, out):
    opts = cfg.optimizers
    fig, axes = plt.subplots(len(opts), len(cfg.alphas), figsize=(5 * len(cfg.alphas), 3.2 * len(opts)),
                             squeeze=False)
    mult = cfg.lr_mults[-1]
    for i, name in enumerate(opts):
        for j, a in enumerate(cfg.alphas):
            ax = axes[i][j]
            losses = curves[(name, a, mult)]
            for s in range(min(4, losses.shape[0])):
                ax.semilogy(losses[s].clamp_max(1e6).numpy(), lw=0.8)
            ax.set_title(f"{name}, alpha={a}, lr={mult}*edge", fontsize=9)
            ax.set_xlabel("step")
            ax.set_ylabel("population loss")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stability_plrf_curves.png"), dpi=120)
    plt.close(fig)


def _plot_regimes(stage0, cfg, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name in cfg.optimizers:
        for a in cfg.alphas:
            rs = [r for r in stage0 if r["optimizer"] == name and r["alpha"] == a]
            ls = "-" if a == cfg.alphas[0] else "--"
            axes[0].plot([r["lr_mult"] for r in rs], [r["mean_events"] for r in rs], ls,
                         marker="o", label=f"{name} a={a}")
            axes[1].plot([r["lr_mult"] for r in rs], [r["divergence_fraction"] for r in rs], ls,
                         marker="o", label=f"{name} a={a}")
    axes[0].set_ylabel("mean excursion events / run")
    axes[1].set_ylabel("divergence fraction")
    for ax in axes:
        ax.set_xlabel("lr / measured edge")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stability_plrf_regimes.png"), dpi=120)
    plt.close(fig)


def _plot_scatter(rows, out):
    keys = ["gns_optimizer", "lr_sharpness", "noise_to_loss", "sim_exact_lossprop_severity"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.4))
    for ax, k in zip(axes, keys):
        for name in sorted({r["optimizer"] for r in rows}):
            rs = [r for r in rows if r["optimizer"] == name and _finite(r.get(k))]
            ax.scatter([r[k] for r in rs], [r["severity"] for r in rs], s=12, label=name)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel(k)
        ax.set_ylabel("real continuation severity")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stability_plrf_sensors.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
