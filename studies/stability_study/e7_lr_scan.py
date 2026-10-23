"""
E7 helper: where does training of the tiny LM become spiky / diverge?

For each configuration and learning rate, train AdamW for --steps steps (3 seeds)
and evaluate the held-out loss every 5 steps; report excursion events on the
held-out curve, divergence, and the final loss.

Run: uv run python studies/stability_study/e7_lr_scan.py --d 32 --layers 2 --L 32 --B 32
"""

import argparse
import math

import torch

from common import save
from tinylm import MarkovLanguage, TinyLM, lm_loss
from snr_grad import detect_excursions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lrs", type=str, default="0.03,0.1,0.15,0.2,0.3,0.4,0.6")
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    lang = MarkovLanguage()
    val = lang.sample(16, a.L, torch.Generator().manual_seed(12345))
    out = []
    for lr in (float(x) for x in a.lrs.split(",")):
        reps, finals = [], []
        for seed in range(3):
            torch.manual_seed(seed)
            model = TinyLM(lang.vocab, a.d, a.layers, heads=max(1, a.d // 32), max_len=a.L)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, a.beta2), weight_decay=0.0)
            gen = torch.Generator().manual_seed(100 + seed)
            curve = []
            for t in range(a.steps):
                opt.zero_grad(set_to_none=True)
                lm_loss(model, lang.sample(a.B, a.L, gen)).backward()
                opt.step()
                if t % 5 == 0:
                    with torch.no_grad():
                        curve.append(float(lm_loss(model, val)))
                    if not math.isfinite(curve[-1]):
                        break
            reps.append(detect_excursions(curve, window=10, warmup=10))
            finals.append(curve[-1])
        row = {"lr": lr, "events": sum(r.num_events for r in reps) / 3,
               "spiked": sum(r.spiked for r in reps) / 3, "diverged": sum(r.diverged for r in reps) / 3,
               "max_excursion_median": sorted(r.max_excursion for r in reps)[1],
               "final_median": sorted(finals)[1]}
        out.append(row)
        print(f"d={a.d} L={a.L} B={a.B} beta2={a.beta2} lr={lr:<5g} events={row['events']:.2f} "
              f"spiked={row['spiked']:.2f} diverged={row['diverged']:.2f} "
              f"maxexc={row['max_excursion_median']:.3f} final={row['final_median']:.3f}", flush=True)
    save(f"e7_lr_scan_d{a.d}_l{a.layers}_L{a.L}_B{a.B}_b2{a.beta2}.json", {"config": vars(a), "rows": out})


if __name__ == "__main__":
    main()
