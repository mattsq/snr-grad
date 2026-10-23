"""
E1: decouple sharpness from gradient noise.

At a fixed learning rate and spectrum, the batch size B and the target exponent
beta change the gradient noise but leave H (and, for SGD, lr * lambda_max(H))
unchanged. A sensor that only tracks sharpness cannot explain severity changes
inside such a stratum; a noise-aware one should.

Grid: alpha_spectrum in {0.5, 0.9}, optimizer in {sgd, adamw}, lr = 0.5 x edge
(edge measured at B=8, beta=0.7), B in {2, 4, 8, 16, 64}, beta in {0.5, 0.7, 1.0}.

Run: uv run python studies/stability_study/e1_decoupling.py [--quick]
"""

import argparse
import time
from dataclasses import replace

import torch

from common import bp, checkpoint_rows, finite, load_edges, save

SENSORS = ["lr_sharpness", "gns_optimizer", "noise_penalty", "noise_to_loss", "rel_sd",
           "upper_tail_ratio", "probe_noise_to_loss", "sim_probe_fixed_severity",
           "sim_probe_lossprop_severity", "sim_exact_lossprop_severity"]


def strata_scores(rows, group_keys):
    out = {}
    for s in SENSORS:
        vals = []
        for key in sorted({tuple(r[k] for k in group_keys) for r in rows}):
            rs = [r for r in rows if tuple(r[k] for k in group_keys) == key
                  and finite(r.get(s)) and finite(r["severity"])]
            if len(rs) < 6 or len({round(r[s], 12) for r in rs}) < 2:
                continue
            rho = bp.spearman([r[s] for r in rs], [r["severity"] for r in rs])
            if rho == rho:
                vals.append(rho)
        out[s] = {"strata": len(vals), "mean": sum(vals) / len(vals) if vals else float("nan")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    edges = load_edges()
    alphas = (0.5, 0.9)
    opts = ("sgd", "adamw")
    Bs = (2, 4, 8, 16, 64)
    betas = (0.5, 0.7, 1.0)
    base = bp.Config(continuations=16, horizon=200, probe_k=32, sim_seeds=16)
    steps, seeds, ckpt_seeds = 3000, 6, 4
    if args.quick:
        Bs, betas, steps, seeds, ckpt_seeds = (4, 32), (0.7,), 800, 2, 2
        base = replace(base, continuations=6, horizon=80, sim_seeds=6, probe_k=16)
    t0 = time.perf_counter()
    rows, summaries = [], []
    for a in alphas:
        for beta in betas:
            for B in Bs:
                cfg = replace(base, batch_size=B, beta=beta)
                prob = bp.PLRF(cfg, a)
                for name in opts:
                    lr = 0.5 * edges[(name, a)]
                    tag = {"alpha": a, "beta": beta, "B": B}
                    r, summ, _ = checkpoint_rows(prob, cfg, name, lr, tag=tag, seeds=seeds,
                                                 steps=steps, ckpt_seeds=ckpt_seeds)
                    rows += r
                    summaries.append(summ)
                    print(f"alpha={a} beta={beta} B={B:>3} {name:>6}: events/run="
                          f"{summ['mean_events']:6.2f} div={summ['divergence_fraction']:.2f} "
                          f"final_excess={summ['final_excess_loss_median']:.3g} "
                          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # Varying B at fixed (alpha, beta, optimizer, checkpoint step): sharpness is fixed for SGD.
    by_B = strata_scores(rows, ("alpha", "beta", "optimizer", "step"))
    # Varying beta at fixed (alpha, B, optimizer, step).
    by_beta = strata_scores(rows, ("alpha", "B", "optimizer", "step"))
    for title, sc in (("vary B", by_B), ("vary beta", by_beta)):
        print(f"\n=== within-stratum mean Spearman with severity ({title}) ===")
        for s, v in sorted(sc.items(), key=lambda kv: -(kv[1]["mean"] if kv[1]["mean"] == kv[1]["mean"] else -9)):
            print(f"   {s:<32} strata={v['strata']:>3} mean={v['mean']:+.3f}")
    path = save("e1_decoupling" + ("_quick" if args.quick else "") + ".json",
                {"rows": rows, "summaries": summaries, "scores_vary_B": by_B,
                 "scores_vary_beta": by_beta, "seconds": time.perf_counter() - t0})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
