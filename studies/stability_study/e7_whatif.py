"""
E7c: can one simulator build answer "which learning rates are safe from here?"
for a small Transformer LM, and is it cheaper than asking reality?

Checkpoints: AdamW at a moderate base lr, --ckpts steps, --seeds seeds. From each
checkpoint, for every candidate lr in --lrs (a "query"):

  real  S continuations of T steps (fresh data, the real network and optimizer state,
        held-out loss every step): excursion severity / event fraction (shared
        definition), median loss change over the horizon, divergence fraction;
  sim   the same quantities from ONE QuadraticStabilitySimulator build per checkpoint
        (probe K=32 x B/4 sequences, rank-16 GGN on 16 held-out sequences, resampled
        loss-proportional noise), re-run per candidate lr;
  base  lr * lambda_max(P^1/2 H P^1/2) with P the frozen AdamW map and H the low-rank
        GGN (edge-of-stability sharpness; > 2 predicts instability for a quadratic).

Scores: Spearman over (checkpoint, lr) of simulated vs real severity and loss change;
the "unsafe" classification (real loss change > +0.05 or divergence) predicted by the
simulator vs by sharpness > 2; wall-clock seconds of both.

Run: uv run python studies/stability_study/e7_whatif.py --d 32 --layers 2 --L 32 --B 8
"""

import argparse
import copy
import math
import time

import torch

from common import save
from benchmark_stability_plrf import auroc, spearman
from e7_fidelity import EXC, K, RANK, baselines, build_simulator
from tinylm import MarkovLanguage, TinyLM, lm_loss
from snr_grad import detect_excursions


def real_queries(lang, model, opt, lr, a, val, seed):
    t0 = time.perf_counter()
    reps, changes, div = [], [], 0
    for i in range(a.S):
        m2 = copy.deepcopy(model)
        o2 = torch.optim.AdamW(m2.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
        o2.load_state_dict(copy.deepcopy(opt.state_dict()))
        for g in o2.param_groups:
            g["lr"] = lr
        gen = torch.Generator().manual_seed(seed + i)
        with torch.no_grad():
            losses = [float(lm_loss(m2, val))]
        for _ in range(a.T):
            o2.zero_grad(set_to_none=True)
            lm_loss(m2, lang.sample(a.B, a.L, gen)).backward()
            o2.step()
            with torch.no_grad():
                losses.append(float(lm_loss(m2, val)))
            if not math.isfinite(losses[-1]):
                losses += [float("inf")] * (a.T + 1 - len(losses))
                break
        r = detect_excursions(losses, **EXC)
        reps.append(r)
        div += r.diverged
        changes.append(losses[-1] - losses[0] if math.isfinite(losses[-1]) else float("inf"))
    return {"severity": sum(min(r.max_excursion, 100.0) for r in reps) / a.S,
            "event_fraction": sum(r.spiked for r in reps) / a.S,
            "loss_change": sorted(changes)[a.S // 2], "diverged": div / a.S,
            "seconds": time.perf_counter() - t0}


def sim_query(sim, lr, a, seed):
    t0 = time.perf_counter()
    dist = sim.run(a.S, a.T, seed=seed, lr=lr, excursion_kwargs=EXC)
    sm = dist.summary()
    final = dist.losses[:, -1]
    change = float(final.median()) - sim.loss0 if bool(torch.isfinite(final).all()) else float("inf")
    return {"severity": sm["excursion_severity_mean"], "event_fraction": sm["spike_fraction"],
            "loss_change": change, "diverged": sm["divergence_fraction"],
            "seconds": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--base-lr", type=float, default=0.01)
    ap.add_argument("--lrs", type=str, default="0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.2")
    ap.add_argument("--ckpts", type=str, default="300,800")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--unsafe", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    lang = MarkovLanguage()
    val = lang.sample(16, a.L, torch.Generator().manual_seed(12345))
    ckpts = tuple(int(x) for x in a.ckpts.split(","))
    lrs = tuple(float(x) for x in a.lrs.split(","))
    rows, costs = [], []
    t_start = time.perf_counter()
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        model = TinyLM(lang.vocab, a.d, a.layers, heads=max(1, a.d // 32), max_len=a.L)
        opt = torch.optim.AdamW(model.parameters(), lr=a.base_lr, betas=(0.9, 0.999), weight_decay=0.0)
        gen = torch.Generator().manual_seed(100 + seed)
        for t in range(1, max(ckpts) + 1):
            opt.zero_grad(set_to_none=True)
            lm_loss(model, lang.sample(a.B, a.L, gen)).backward()
            opt.step()
            if t not in ckpts:
                continue
            cs = 9000 + 31 * seed + t
            sim, probe, low, t_build = build_simulator(lang, model, opt, a.base_lr, a, val,
                                                       torch.Generator().manual_seed(cs))
            real_s = sim_s = 0.0
            for lr in lrs:
                real = real_queries(lang, model, opt, lr, a, val, cs + 100)
                simr = sim_query(sim, lr, a, cs + 200)
                base = baselines(opt, probe, low, lr, a)
                real_s += real["seconds"]
                sim_s += simr["seconds"]
                rows.append({"seed": seed, "step": t, "lr": lr, "loss0": sim.loss0,
                             **{f"real_{k}": v for k, v in real.items()},
                             **{f"sim_{k}": v for k, v in simr.items()},
                             "lr_sharpness": base.get("lr_sharpness"),
                             "noise_penalty": base.get("noise_penalty")})
                r = rows[-1]
                print(f"seed={seed} step={t:>4} lr={lr:<6g} real: sev={real['severity']:7.3f} "
                      f"dL={real['loss_change']:+7.3f} div={real['diverged']:.2f} | sim: "
                      f"sev={simr['severity']:7.3f} dL={simr['loss_change']:+7.3f} "
                      f"div={simr['diverged']:.2f} | sharp={r['lr_sharpness']:.2f} "
                      f"({real['seconds']:.1f}s vs {simr['seconds']:.1f}s) "
                      f"[{time.perf_counter() - t_start:.0f}s]", flush=True)
            costs.append({"seed": seed, "step": t, "build_seconds": t_build,
                          "real_seconds_all_queries": real_s, "sim_seconds_all_queries": sim_s})

    def unsafe(row, pre):
        ch = row[f"{pre}_loss_change"]
        return (not math.isfinite(ch)) or ch > a.unsafe or row[f"{pre}_diverged"] > 0.5

    def clip(v):
        return v if math.isfinite(v) else 1e6

    labels = [unsafe(r, "real") for r in rows]
    sim_unsafe = [unsafe(r, "sim") for r in rows]
    sharp_unsafe = [r["lr_sharpness"] > 2.0 for r in rows]

    def acc(pred):
        return sum(p == l for p, l in zip(pred, labels)) / len(labels)

    scores = {
        "spearman_severity_sim": spearman([r["sim_severity"] for r in rows], [r["real_severity"] for r in rows]),
        "spearman_severity_sharpness": spearman([r["lr_sharpness"] for r in rows], [r["real_severity"] for r in rows]),
        "spearman_loss_change_sim": spearman([clip(r["sim_loss_change"]) for r in rows],
                                             [clip(r["real_loss_change"]) for r in rows]),
        "spearman_loss_change_sharpness": spearman([r["lr_sharpness"] for r in rows],
                                                   [clip(r["real_loss_change"]) for r in rows]),
        "unsafe_fraction_real": sum(labels) / len(labels),
        "unsafe_accuracy_sim": acc(sim_unsafe),
        "unsafe_accuracy_sharpness_gt2": acc(sharp_unsafe),
        "unsafe_auroc_sim_loss_change": auroc([clip(r["sim_loss_change"]) for r in rows], labels),
        "unsafe_auroc_sharpness": auroc([r["lr_sharpness"] for r in rows], labels),
    }
    tot_build = sum(c["build_seconds"] for c in costs)
    tot_real = sum(c["real_seconds_all_queries"] for c in costs)
    tot_sim = sum(c["sim_seconds_all_queries"] for c in costs)
    cost = {"queries_per_checkpoint": len(lrs), "build_seconds_total": tot_build,
            "real_seconds_total": tot_real, "sim_seconds_total": tot_sim,
            "speedup_all_queries": tot_real / (tot_build + tot_sim),
            "speedup_single_query": (tot_real / len(lrs)) / (tot_build + tot_sim / len(lrs))}
    print("\nScores:", {k: round(v, 3) for k, v in scores.items()})
    print("Cost:", {k: round(v, 2) for k, v in cost.items()})
    tag = f"d{a.d}_l{a.layers}_L{a.L}_B{a.B}"
    print("wrote", save(f"e7_whatif_{tag}.json", {"config": vars(a), "rows": rows, "costs": costs,
                                                 "scores": scores, "cost": cost,
                                                 "seconds": time.perf_counter() - t_start}))


if __name__ == "__main__":
    main()
