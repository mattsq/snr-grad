"""
E8a (Adam): do the noise-fluctuation index Phi and the dominant-mode tail explain PLRF
Adam excursions better than the adaptive edge ratio alone?

Rebuilds the E8a Adam checkpoints (same deterministic runs) and, with exact H and Sigma,
computes in Adam-preconditioned coordinates (P = 1/sqrt(v_hat); h_i, u_i eigenpairs of
P^1/2 H P^1/2; sigma_i = u_i^T P^1/2 Sigma P^1/2 u_i; edge = 38):
  ratio  = lr h_max / 38                   (adaptive edge fraction)
  Phi    = kappa_hat sqrt(2 / D_hat)       (noise_fluctuation_index, horizon 1000)
  xi     = dominant-mode loss tail exponent at lr*lambda = 2 * ratio
for P from the actual and from the stationary second moment, then scores them against
excursions in the next 1000 steps.

Run: uv run python studies/stability_study/e8_adam_indices.py
"""

import math
import time
from dataclasses import replace

import torch

from common import bp, load_edges, save
from snr_grad import adam_edge_threshold, detect_excursions, dominant_mode_tail, noise_fluctuation_index


def main():
    torch.set_num_threads(4)
    edges = load_edges()
    cfg = bp.Config()
    thr = adam_edge_threshold(0.9)
    rows = []
    t0 = time.perf_counter()
    for a in (0.5, 0.9):
        for B in (2, 8, 32, 128):
            prob = bp.PLRF(replace(cfg, batch_size=B), a)
            for m in (0.1, 0.3, 1.0):
                lr = m * edges[("adamw", a)]
                losses, snaps = bp.train(prob, "adamw", lr, 4, 4000, checkpoints=(1000, 2000, 3000))
                for t, (theta_stack, opt) in snaps.items():
                    st = next(iter(opt.state.values()))
                    v_hat = st["exp_avg_sq"] / (1 - 0.999 ** float(st["step"]))
                    for s in range(4):
                        theta = theta_stack[s]
                        if not bool(torch.isfinite(theta).all()):
                            continue
                        L = float(prob.loss(theta[None])[0])
                        mu = prob.pop_grad(theta[None])[0]
                        sigma = prob.sigma(theta)
                        rep = detect_excursions(losses[s, t: t + 1001], window=20, warmup=20)
                        row = {"alpha": a, "B": B, "m": m, "step": t, "seed": s, "loss": L,
                               "events_next": rep.num_events, "severity_next": min(rep.max_excursion, 1e4)}
                        for name, pdiag in (("act", 1 / (v_hat[s].sqrt() + 1e-8)),
                                            ("stat", 1 / ((mu * mu + sigma.diagonal() / B).sqrt() + 1e-8))):
                            r = pdiag.sqrt()
                            h, u = torch.linalg.eigh(r[:, None] * prob.H * r[None, :])
                            sig = ((u * ((r[:, None] * sigma * r[None, :]) @ u)).sum(0))
                            ratio = lr * float(h.max()) / thr
                            idx = noise_fluctuation_index(h, sig, lr, B, L, edge=thr, horizon=1000)
                            xi = dominant_mode_tail(min(2 * ratio, 4.0) * B / (B + 1), B, samples=20_000)
                            row.update({f"ratio_{name}": ratio, f"Phi_{name}": idx["Phi"],
                                        f"kappa_{name}": idx["kappa_hat"], f"D_{name}": idx["D_hat"],
                                        f"xi_{name}": xi["loss_tail_exponent"]})
                        rows.append(row)
                print(f"a={a} B={B:>3} m={m}: {len(rows)} rows ({time.perf_counter() - t0:.0f}s)", flush=True)

    def fin(x):
        return x if math.isfinite(x) else 1e9

    def auroc(score, label):
        pos = [s for s, l in zip(score, label) if l]
        neg = [s for s, l in zip(score, label) if not l]
        return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

    sev = [r["severity_next"] for r in rows]
    big = [r["severity_next"] > 1.0 for r in rows]
    scores = {}
    for key in ("ratio_act", "ratio_stat", "Phi_act", "Phi_stat", "kappa_act", "D_act", "xi_act", "xi_stat"):
        sign = -1 if key.startswith(("xi", "D")) else 1
        xs = [sign * fin(r[key]) for r in rows]
        scores[key] = {"spearman_severity": bp.spearman(xs, sev), "auroc_severe": auroc(xs, big)}
        print(f"{key:<11} spearman(sev)={scores[key]['spearman_severity']:+.3f} "
              f"AUROC(sev>1)={scores[key]['auroc_severe']:.3f}")
    X = torch.tensor([[1.0, math.log(max(fin(r["Phi_act"]), 1e-6)), math.log(max(r["ratio_stat"], 1e-6))]
                      for r in rows], dtype=torch.float64)
    y = torch.tensor([math.log(r["severity_next"] + 1e-2) for r in rows], dtype=torch.float64)
    for cols, name in (([0, 1], "log Phi_act"), ([0, 2], "log ratio_stat"), ([0, 1, 2], "both")):
        b = torch.linalg.lstsq(X[:, cols], y[:, None]).solution[:, 0]
        p = X[:, cols] @ b
        r2 = 1 - float(((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        scores[f"r2_{name}"] = r2
        print(f"log(severity) ~ {name}: R^2 = {r2:.2f}  coef={[round(float(v), 2) for v in b]}")
    print("wrote", save("e8_adam_indices.json", {"rows": rows, "scores": scores}))


if __name__ == "__main__":
    main()
