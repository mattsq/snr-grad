"""
E7d: does a higher curvature rank fix the simulator's late-training optimism?

Rebuilds the E7c checkpoints (same deterministic training) and re-runs only the
simulator with GGN rank 4 / 16 / 64, scoring against the real query results already
stored in results/e7_whatif_<config>.json.

Run: uv run python studies/stability_study/e7_rank_ablation.py --d 64 --L 64 --B 8
"""

import argparse
import json
import math
import os
import time

import torch

from common import RESULTS, save
from benchmark_stability_plrf import spearman
from e7_fidelity import build_simulator
from e7_whatif import sim_query
from tinylm import MarkovLanguage, TinyLM, lm_loss


def onset(pairs, thr):
    for lr, change, div in sorted(pairs):
        if (not math.isfinite(change)) or change > thr or div > 0.5:
            return lr
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--ranks", type=str, default="4,16,64")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tag = f"d{a.d}_l{a.layers}_L{a.L}_B{a.B}"
    ref = json.load(open(os.path.join(RESULTS, f"e7_whatif_{tag}.json")))
    cfg = ref["config"]
    for k in ("S", "T", "base_lr", "seeds", "unsafe"):
        setattr(a, k, cfg[k])
    ckpts = tuple(int(x) for x in cfg["ckpts"].split(","))
    lrs = sorted({r["lr"] for r in ref["rows"]})
    lang = MarkovLanguage()
    val = lang.sample(16, a.L, torch.Generator().manual_seed(12345))
    out = []
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
            real = {r["lr"]: r for r in ref["rows"] if r["seed"] == seed and r["step"] == t}
            with torch.no_grad():
                l_now = float(lm_loss(model, val))
            assert abs(l_now - real[lrs[0]]["loss0"]) < 1e-4, "checkpoint mismatch"
            for rank in (int(x) for x in a.ranks.split(",")):
                t0 = time.perf_counter()
                sim, _, low, t_build = build_simulator(lang, model, opt, a.base_lr, a, val,
                                                       torch.Generator().manual_seed(cs), rank=rank)
                res = {lr: sim_query(sim, lr, a, cs + 200) for lr in lrs}
                row = {"seed": seed, "step": t, "rank": rank, "build_seconds": t_build,
                       "seconds": time.perf_counter() - t0,
                       "residual": low.residual, "top_eig": float(low.eigenvalues[0]),
                       "spearman_loss_change": spearman(
                           [min(res[lr]["loss_change"], 1e6) for lr in lrs],
                           [min(real[lr]["real_loss_change"], 1e6) for lr in lrs]),
                       "onset_sim": onset([(lr, res[lr]["loss_change"], res[lr]["diverged"]) for lr in lrs], a.unsafe),
                       "onset_real": onset([(lr, real[lr]["real_loss_change"], real[lr]["real_diverged"]) for lr in lrs], a.unsafe),
                       "sim_loss_change": {lr: res[lr]["loss_change"] for lr in lrs}}
                out.append(row)
                print(f"seed={seed} step={t} rank={rank:>3}: onset sim={row['onset_sim']} real={row['onset_real']} "
                      f"rho(dL)={row['spearman_loss_change']:+.3f} residual={low.residual:.3g} "
                      f"top={row['top_eig']:.3g} ({row['seconds']:.0f}s)", flush=True)
    print("wrote", save(f"e7_rank_ablation_{tag}.json", {"rows": out}))


if __name__ == "__main__":
    main()
