"""
E2: calibration of the probe estimators against exact PLRF ground truth.

At fixed checkpoints (alpha in {0.5, 0.9}, optimizer in {sgd, adamw}, lr = 0.5 x
edge, step 750, two seeds), repeat the probe ``reps`` times for each K and
compare with exact values:

  * R = tr(H P Sigma P^T): relative bias, relative RMSE, empirical SD versus the
    mean jackknife SE, and coverage of +/-2 SE and +/- t_{K-1} SE intervals;
  * the log-scale jackknife interval ``risk_trace_ci`` (coverage);
  * D_eff: abstention rate, relative bias when reported, and whether abstention
    agrees with the exact D_eff exceeding max_rank_fraction * (K - 1); also with
    the degrees-of-freedom minimum relaxed from 16 to 4.

Each repetition is run with the true PLRF minibatch noise (products of Gaussians,
heavy-tailed) and with Gaussian microbatch gradients of the *same* per-example
covariance, isolating the effect of non-Gaussian noise.

Run: uv run python studies/stability_study/e2_estimator_calibration.py [--quick]
"""

import argparse
import math
import time

import torch
import torch.nn as nn

from common import bp, load_edges, save
from snr_grad import ExactQuadraticCurvature, StabilityProbe, exact_noise_geometry, frozen_update_map

T_975 = {7: 2.365, 15: 2.131, 31: 2.040, 63: 1.998, 127: 1.979}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    edges = load_edges()
    Ks = (8, 16, 32, 64, 128)
    reps = 200
    if args.quick:
        Ks, reps = (8, 32), 20
    cfg = bp.Config()
    B = cfg.batch_size
    t0 = time.perf_counter()
    results = []
    for a in (0.5, 0.9):
        prob = bp.PLRF(cfg, a)
        for name in ("sgd", "adamw"):
            lr = 0.5 * edges[(name, a)]
            _, snaps = bp.train(prob, name, lr, 2, 751, checkpoints=(750,))
            theta_stack, opt = snaps[750]
            for s in range(2):
                p, o = bp.slice_optimizer(name, opt, theta_stack, s, lr)
                theta = p.detach()
                mu = prob.pop_grad(theta[None])[0]
                sigma = prob.sigma(theta)
                pmap = frozen_update_map(o, at_gradient={p: mu})
                pdiag = pmap({p: torch.ones_like(mu)})[p]
                exact = exact_noise_geometry(prob.H, sigma, pdiag)
                curv = ExactQuadraticCurvature([p], prob.H)
                sig_sqrt = torch.linalg.cholesky(sigma + 1e-12 * torch.eye(prob.D, dtype=sigma.dtype))
                for K in Ks:
                    for noise in ("plrf", "gaussian"):
                        gen = torch.Generator().manual_seed(1 + 1000 * K + s)
                        rs, ses, ds, statuses, cis, ds_relaxed, st_relaxed = [], [], [], [], [], [], []
                        for _ in range(reps):
                            if noise == "plrf":
                                g = prob.minibatch_grad(theta[None].expand(K, -1), gen)
                            else:
                                z = torch.randn(K, B, prob.D, generator=gen, dtype=mu.dtype)
                                g = mu + (z @ sig_sqrt.t()).mean(1)
                            grads = [{p: g[k]} for k in range(K)]
                            probe = StabilityProbe(grads, B, update_map=pmap, curvature=curv)
                            m = probe.measure()
                            probe.min_concentration_dof = 4
                            m_relaxed = probe.measure()
                            ds_relaxed.append(m_relaxed.concentration)
                            st_relaxed.append(m_relaxed.concentration_status)
                            cis.append(m.risk_trace_ci)
                            rs.append(m.risk_trace_per_example)
                            ses.append(m.risk_trace_se)
                            ds.append(m.concentration)
                            statuses.append(m.concentration_status)
                        r_true = exact["risk_trace"]
                        rt = torch.tensor(rs, dtype=torch.float64)
                        se = torch.tensor(ses, dtype=torch.float64)
                        err = rt - r_true
                        tq = T_975.get(K - 1, 2.0)
                        ok = [d for d, st in zip(ds, statuses) if st == "ok"]
                        should_abstain = exact["concentration"] > 0.25 * (K - 1)
                        cover_log = sum(ci is not None and ci[0] <= r_true <= ci[1] for ci in cis) / reps
                        ok_rel = [d for d, st in zip(ds_relaxed, st_relaxed) if st == "ok"]
                        row = {
                            "alpha": a, "optimizer": name, "seed": s, "K": K, "noise": noise,
                            "R_true": r_true, "D_true": exact["concentration"],
                            "R_rel_bias": float(err.mean() / r_true),
                            "R_rel_rmse": float(err.pow(2).mean().sqrt() / r_true),
                            "R_sd_over_mean_se": float(rt.std() / se.mean()),
                            "cover_2se": float((err.abs() <= 2 * se).double().mean()),
                            "cover_t": float((err.abs() <= tq * se).double().mean()),
                            "R_skew": float(((rt - rt.mean()) ** 3).mean() / rt.std() ** 3),
                            "cover_log_t": cover_log,
                            "abstain_rate": 1 - len(ok) / reps,
                            "abstain_rate_dof4": 1 - len(ok_rel) / reps,
                            "D_rel_bias_dof4": (sum(ok_rel) / len(ok_rel) / exact["concentration"] - 1)
                            if ok_rel else None,
                            "D_rel_rmse_dof4": (math.sqrt(sum((d - exact["concentration"]) ** 2 for d in ok_rel)
                                                          / len(ok_rel)) / exact["concentration"]) if ok_rel else None,
                            "abstain_should": bool(should_abstain),
                            "D_rel_bias": (sum(ok) / len(ok) / exact["concentration"] - 1) if ok else None,
                            "D_rel_rmse": (math.sqrt(sum((d - exact["concentration"]) ** 2 for d in ok) / len(ok))
                                           / exact["concentration"]) if ok else None,
                        }
                        results.append(row)
                        print(f"a={a} {name:>5} s={s} K={K:>3} {noise:>8}: Rbias={row['R_rel_bias']:+.3f} "
                              f"Rrmse={row['R_rel_rmse']:.3f} sd/se={row['R_sd_over_mean_se']:.2f} "
                              f"cov2={row['cover_2se']:.2f} covt={row['cover_t']:.2f} covlog={row['cover_log_t']:.2f} "
                              f"D_true={row['D_true']:.1f} abst={row['abstain_rate']:.2f} "
                              f"Dbias={row['D_rel_bias'] if row['D_rel_bias'] is None else round(row['D_rel_bias'], 3)} "
                              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    path = save("e2_estimator_calibration" + ("_quick" if args.quick else "") + ".json",
                {"rows": results, "reps": reps, "seconds": time.perf_counter() - t0})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
