"""
E8a: validate the closed-form stability indices on PLRF (snr_grad.stability_theory).

SGD part. For spectrum alpha in {0.3, 0.5, 0.7, 0.9}, batch B in {2, 8, 32} and
lr = m x (deterministic edge 2 / (lambda_max (1 + 1/B))), m in {0.2, ..., 0.95}:
  * the exact expected-loss curve from the mean recursion vs the seed average;
  * kappa, D_K vs intermittency of individual runs: late-loss tail ratio
    (95th percentile / median of L - L*), excursion events per run, severity.

Adam part. alpha in {0.5, 0.9}, B in {2, 8, 32, 128}, lr in {0.1, 0.3, 1.0} x the
AdamW edge measured at B=8. At steps 1000 / 2000 / 3000 the edge ratio
lr * lambda_max(P^1/2 H P^1/2) / 38 is computed with the actual Adam second moment,
with the stationary one (mu^2 + diag(Sigma)/B) and with the noise floor
(diag(Sigma)/B), and compared with excursions in the next 1000 steps of that run.

Run: uv run python studies/stability_study/e8_plrf_theory.py [--quick]
"""

import argparse
import math
import time
from dataclasses import replace

import torch

from common import bp, load_edges, save
from snr_grad import adam_edge_threshold, detect_excursions, sgd_stochastic_edge

EXC = {"window": 50, "k_mad": 6.0, "rel_floor": 0.25}


def expected_curve(prob, lam, vecs, lr, B, steps):
    """Exact E[L(t)] for SGD from theta=0 via the mean recursion."""
    theta_star = torch.linalg.lstsq(prob.H, prob.c[:, None]).solution[:, 0]
    c = (vecs.t() @ (-theta_star)) ** 2
    a = 1 - 2 * lr * lam + lr * lr * lam * lam * (1 + 1 / B)
    l_star = prob.loss_star
    out = [l_star + 0.5 * float(lam @ c)]
    for _ in range(steps):
        c = a * c + (lr * lr / B) * lam * (float(lam @ c) + 2 * l_star)
        out.append(l_star + 0.5 * float(lam @ c))
        if not math.isfinite(out[-1]) or out[-1] > 1e12:
            out += [float("inf")] * (steps + 1 - len(out))
            break
    return torch.tensor(out, dtype=torch.float64)


def sgd_part(cfg, alphas, Bs, mults, seeds, steps):
    rows = []
    for a in alphas:
        for B in Bs:
            prob = bp.PLRF(replace(cfg, batch_size=B), a)
            lam, vecs = torch.linalg.eigh(prob.H)
            lam = lam.clamp_min(0)
            det_edge = 2.0 / (float(lam.max()) * (1 + 1 / B))
            for m in mults:
                lr = m * det_edge
                th = sgd_stochastic_edge(lam, lr, B)
                losses, _ = bp.train(prob, "sgd", lr, seeds, steps)
                exp_curve = expected_curve(prob, lam, vecs, lr, B, steps)
                late = slice(steps // 2, steps + 1)
                fin = torch.isfinite(losses[:, late]).all(1)
                emp_mean = float(losses[fin][:, late].mean()) if bool(fin.any()) else float("inf")
                th_mean = float(exp_curve[late].mean())
                ex = (losses[:, late] - prob.loss_star).clamp_min(1e-30)
                tail = [float(torch.quantile(ex[s], 0.95) / ex[s].median()) if bool(fin[s]) else float("inf")
                        for s in range(seeds)]
                reps = [detect_excursions(losses[s], **EXC) for s in range(seeds)]
                row = {"alpha": a, "B": B, "m": m, "lr": lr, **th,
                       "emp_late_mean": emp_mean, "theory_late_mean": th_mean,
                       "loss_star": prob.loss_star,
                       "tail_ratio_median": sorted(tail)[seeds // 2],
                       "events_per_run": sum(r.num_events for r in reps) / seeds,
                       "severity_median": sorted(min(r.max_excursion, 1e4) for r in reps)[seeds // 2],
                       "divergence_fraction": sum(r.diverged for r in reps) / seeds}
                rows.append(row)
                print(f"SGD a={a} B={B:>2} m={m:.2f} kappa={th['kappa']:.3f} D_K={th['D_K']:.2f} "
                      f"late mean emp={emp_mean:.4g} theory={th_mean:.4g} tail={row['tail_ratio_median']:.2f} "
                      f"events={row['events_per_run']:.1f} sev={row['severity_median']:.2f} "
                      f"div={row['divergence_fraction']:.2f}", flush=True)
    return rows


def adam_part(cfg, alphas, Bs, mults, seeds, steps, ckpts, horizon):
    edges = load_edges()
    thr = adam_edge_threshold(0.9)
    rows = []
    for a in alphas:
        for B in Bs:
            prob = bp.PLRF(replace(cfg, batch_size=B), a)
            for m in mults:
                lr = m * edges[("adamw", a)]
                losses, snaps = bp.train(prob, "adamw", lr, seeds, steps, checkpoints=ckpts)
                for t, (theta_stack, opt) in snaps.items():
                    st = next(iter(opt.state.values()))
                    step = float(st["step"])
                    v_hat = st["exp_avg_sq"] / (1 - 0.999 ** step)
                    for s in range(seeds):
                        theta = theta_stack[s]
                        if not bool(torch.isfinite(theta).all()):
                            continue
                        mu = prob.pop_grad(theta[None])[0]
                        sig2 = prob.sigma(theta).diagonal()
                        precs = {
                            "actual": 1 / (v_hat[s].sqrt() + 1e-8),
                            "stationary": 1 / ((mu * mu + sig2 / B).sqrt() + 1e-8),
                            "floor": 1 / ((sig2 / B).sqrt() + 1e-8),
                        }
                        row = {"alpha": a, "B": B, "m": m, "lr": lr, "step": t, "seed": s,
                               "loss": float(prob.loss(theta[None])[0]), "loss_star": prob.loss_star}
                        for k, pdiag in precs.items():
                            r = pdiag.sqrt()
                            lam = float(torch.linalg.eigvalsh(r[:, None] * prob.H * r[None, :]).max())
                            row[f"ratio_{k}"] = lr * lam / thr
                        window = losses[s, t: t + horizon + 1]
                        rep = detect_excursions(window, window=20, warmup=20, k_mad=6.0, rel_floor=0.25)
                        row["events_next"] = rep.num_events
                        row["severity_next"] = min(rep.max_excursion, 1e4)
                        rows.append(row)
                print(f"Adam a={a} B={B:>3} m={m}: {len(rows)} checkpoint rows", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg = bp.Config()
    t0 = time.perf_counter()
    if args.quick:
        sgd = sgd_part(cfg, (0.9,), (8,), (0.5, 0.9), 4, 600)
        adam = adam_part(cfg, (0.9,), (8,), (0.3,), 2, 1200, (400,), 300)
    else:
        sgd = sgd_part(cfg, (0.3, 0.5, 0.7, 0.9), (2, 8, 32), (0.2, 0.4, 0.6, 0.8, 0.95), 8, 3000)
        adam = adam_part(cfg, (0.5, 0.9), (2, 8, 32, 128), (0.1, 0.3, 1.0), 4, 4000,
                         (1000, 2000, 3000), 1000)
    print("wrote", save("e8_plrf_theory" + ("_quick" if args.quick else "") + ".json",
                        {"sgd": sgd, "adam": adam, "seconds": time.perf_counter() - t0}))


if __name__ == "__main__":
    main()
