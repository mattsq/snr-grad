"""
E3: simulator ablations and calibration on PLRF.

Checkpoints: alpha in {0.3, 0.5, 0.7, 0.9} x {sgd, adamw, snr_adamw} x lr in
{0.3, 0.7, 0.95} x edge, step 1000, 2 seeds. For each checkpoint, 32 real
continuations of 800 steps are scored on prefixes of 50 / 200 / 800 steps with
the shared excursion definition. Simulator variants:

  probe_fixed_K32       Gaussian noise fitted from 32 microbatches, fixed scale
  probe_lossprop_K32    ... std scaled by sqrt(L/L0)
  probe_lossprop_K8     ... from only 8 microbatches
  probe_t5_lossprop     ... multivariate Student-t (dof 5), same covariance
  resample_*            ResampledProbeNoise: resample observed deviations (keeps
                        their non-Gaussian shape), K = 32 or 128
  exact_fixed           exact Sigma at the checkpoint, fixed
  exact_lossprop        exact Sigma, std scaled by sqrt(L/L0)
  oracle                true PLRF minibatch noise at the current point (the
                        surrogate is exact for PLRF, so this checks replay)

Scores per (variant, horizon): Spearman with real severity, Spearman and mean
absolute error of the simulated vs real event fraction (calibration), median
log10 ratio of severities, and seconds. Seed-count ablation: exact_lossprop with
4 / 16 / 64 simulated seeds at horizon 200.

Run: uv run python studies/stability_study/e3_simulator.py [--quick]
"""

import argparse
import math
import time

import torch
import torch.nn as nn

from common import bp, load_edges, save
from snr_grad import (
    CovarianceFactorNoise,
    ExactQuadraticCurvature,
    GaussianProbeNoise,
    QuadraticStabilitySimulator,
    ResampledProbeNoise,
    StabilityProbe,
    detect_excursions,
)

EXC = {"window": 20, "warmup": 10}


class OracleNoise:
    """True PLRF minibatch-gradient noise at theta0 + delta."""

    def __init__(self, prob, theta0):
        self.prob, self.theta0 = prob, theta0

    def __call__(self, delta, loss_ratio, step, gen):
        th = (self.theta0 + delta)[None]
        return self.prob.minibatch_grad(th, gen)[0] - self.prob.pop_grad(th)[0]


def severity_stats(losses, horizons):
    """losses: [n, T+1]. Returns {h: (severity, event_fraction)}."""
    out = {}
    for h in horizons:
        reps = [detect_excursions(losses[i, : h + 1], divergence_factor=100.0, **EXC)
                for i in range(losses.shape[0])]
        out[h] = (sum(min(r.max_excursion, 100.0) for r in reps) / len(reps),
                  sum(r.spiked for r in reps) / len(reps))
    return out


def real_runs(prob, name, opt, theta_stack, s, lr, n, T, seed):
    p, o = bp.slice_optimizer(name, opt, theta_stack, s, lr, repeat=n)
    gen = torch.Generator().manual_seed(seed)
    L = torch.empty(n, T + 1, dtype=bp.DTYPE)
    with torch.no_grad():
        L[:, 0] = prob.loss(p)
    for t in range(T):
        with torch.no_grad():
            p.grad = prob.minibatch_grad(p.detach(), gen)
        o.step()
        with torch.no_grad():
            lt = prob.loss(p)
            lt[~torch.isfinite(lt)] = float("inf")
            L[:, t + 1] = lt
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    edges = load_edges()
    cfg = bp.Config()
    B = cfg.batch_size
    alphas, opts, mults, seeds = (0.3, 0.5, 0.7, 0.9), ("sgd", "adamw", "snr_adamw"), (0.3, 0.7, 0.95), 2
    n_real, horizons, sim_seeds = 32, (50, 200, 800), 16
    if args.quick:
        alphas, opts, mults, seeds, n_real, horizons, sim_seeds = (0.3, 0.9), ("sgd",), (0.7,), 1, 8, (50, 100), 4
    T = max(horizons)
    t0 = time.perf_counter()
    rows = []
    for a in alphas:
        prob = bp.PLRF(cfg, a)
        for name in opts:
            for mult in mults:
                lr = mult * edges[(name, a)]
                _, snaps = bp.train(prob, name, lr, seeds, 1001, checkpoints=(1000,))
                theta_stack, opt = snaps[1000]
                for s in range(seeds):
                    if not bool(torch.isfinite(theta_stack[s]).all()):
                        continue
                    seed = 7000 + 13 * s
                    real = severity_stats(real_runs(prob, name, opt, theta_stack, s, lr, n_real, T, seed),
                                          horizons)
                    p, o = bp.slice_optimizer(name, opt, theta_stack, s, lr)
                    theta = p.detach()
                    L0 = float(prob.loss(theta[None])[0])
                    mu = prob.pop_grad(theta[None])[0]
                    curv = ExactQuadraticCurvature([p], prob.H)
                    gen = torch.Generator().manual_seed(seed + 1)
                    devs = {}
                    for K in (8, 32, 128):
                        g = prob.minibatch_grad(theta[None].expand(K, -1), gen)
                        devs[K] = g - g.mean(0)
                    factor = prob.sigma_factor(theta)
                    variants = {
                        "probe_fixed_K32": GaussianProbeNoise(devs[32], B, B),
                        "probe_lossprop_K32": GaussianProbeNoise(devs[32], B, B, scaling="loss_proportional"),
                        "probe_lossprop_K8": GaussianProbeNoise(devs[8], B, B, scaling="loss_proportional"),
                        "probe_t5_lossprop": GaussianProbeNoise(devs[32], B, B, heavy_tail_dof=5.0,
                                                                scaling="loss_proportional"),
                        "resample_fixed_K32": ResampledProbeNoise(devs[32], B, B),
                        "resample_lossprop_K32": ResampledProbeNoise(devs[32], B, B, scaling="loss_proportional"),
                        "resample_lossprop_K128": ResampledProbeNoise(devs[128], B, B, scaling="loss_proportional"),
                        "exact_fixed": CovarianceFactorNoise(factor, B),
                        "exact_lossprop": CovarianceFactorNoise(factor, B, scaling="loss_proportional"),
                        "oracle": OracleNoise(prob, theta),
                    }
                    row = {"alpha": a, "optimizer": name, "lr_mult": mult, "seed": s,
                           "split": "dev" if s % 2 == 0 else "heldout",
                           **{f"real_sev_{h}": real[h][0] for h in horizons},
                           **{f"real_evt_{h}": real[h][1] for h in horizons}}
                    for key, noise in variants.items():
                        ts = time.perf_counter()
                        sim = QuadraticStabilitySimulator([p], o, curv, mu, noise, loss0=L0)
                        dist = sim.run(sim_seeds, T, seed=seed + 2, excursion_kwargs=EXC)
                        st = severity_stats(dist.losses, horizons)
                        row[f"{key}_seconds"] = time.perf_counter() - ts
                        for h in horizons:
                            row[f"{key}_sev_{h}"], row[f"{key}_evt_{h}"] = st[h]
                    h_mid = horizons[1]
                    for ns in (4, 64):
                        sim = QuadraticStabilitySimulator([p], o, curv, mu, variants["exact_lossprop"], loss0=L0)
                        dist = sim.run(ns, h_mid, seed=seed + 3, excursion_kwargs=EXC)
                        row[f"exact_lossprop_n{ns}_sev_{h_mid}"], row[f"exact_lossprop_n{ns}_evt_{h_mid}"] = \
                            severity_stats(dist.losses, (h_mid,))[h_mid]
                    rows.append(row)
                print(f"alpha={a} {name:>9} mult={mult}: {len(rows)} checkpoints "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    keys = ["probe_fixed_K32", "probe_lossprop_K32", "probe_lossprop_K8", "probe_t5_lossprop",
            "resample_fixed_K32", "resample_lossprop_K32", "resample_lossprop_K128",
            "exact_fixed", "exact_lossprop", "oracle"]
    scores = {}
    for key in keys + [f"exact_lossprop_n{ns}" for ns in (4, 64)]:
        for h in horizons:
            col = f"{key}_sev_{h}"
            rs = [r for r in rows if col in r]
            if len(rs) < 3:
                continue
            sev_s = [r[col] for r in rs]
            sev_r = [r[f"real_sev_{h}"] for r in rs]
            evt_s = [r[f"{key}_evt_{h}"] for r in rs]
            evt_r = [r[f"real_evt_{h}"] for r in rs]
            lr_ = [math.log10((x + 1e-3) / (y + 1e-3)) for x, y in zip(sev_s, sev_r)]
            scores[f"{key}@{h}"] = {
                "n": len(rs),
                "spearman_severity": bp.spearman(sev_s, sev_r),
                "spearman_event_fraction": bp.spearman(evt_s, evt_r),
                "event_fraction_mae": sum(abs(x - y) for x, y in zip(evt_s, evt_r)) / len(rs),
                "event_fraction_bias": sum(x - y for x, y in zip(evt_s, evt_r)) / len(rs),
                "severity_log10_ratio_median": float(torch.tensor(lr_).median()),
                "seconds_median": float(torch.tensor([r.get(f"{key}_seconds", float("nan")) for r in rs]).median()),
            }
    print(f"\n{'variant@horizon':<28}{'n':>4}{'rho_sev':>9}{'rho_evt':>9}{'evt_MAE':>9}"
          f"{'evt_bias':>10}{'log10 sev':>11}{'sec':>7}")
    for k, v in scores.items():
        print(f"{k:<28}{v['n']:>4}{v['spearman_severity']:>+9.3f}{v['spearman_event_fraction']:>+9.3f}"
              f"{v['event_fraction_mae']:>9.3f}{v['event_fraction_bias']:>+10.3f}"
              f"{v['severity_log10_ratio_median']:>+11.3f}{v['seconds_median']:>7.2f}")
    path = save("e3_simulator" + ("_quick" if args.quick else "") + ".json",
                {"rows": rows, "scores": scores, "seconds": time.perf_counter() - t0})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
