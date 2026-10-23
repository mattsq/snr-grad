"""
E9c-2: seed repeats of the one Adam spike seen in E9c (clean language, T = 0.03, no copy
rule, B = 32, beta2 = 0.999, lr = 0.01) and of its controls (beta2 = 0.95; B = 128).

Run: uv run python studies/stability_study/e9_lm_repeat.py --part {b32,b128}
"""

import argparse
import time

import torch

from common import save
from e9_lm_confident import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("b32", "b128"), default="b32")
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.part == "b32":
        cells = [(32, 0.999, s) for s in range(5)] + [(32, 0.95, s) for s in range(5)]
    else:
        cells = [(128, 0.999, s) for s in range(1, 4)]
    rows = []
    t0 = time.perf_counter()
    for B, beta2, seed in cells:
        r = run(0.03, B, beta2, 0.01, copy_prob=0.0, seed=seed)
        r["seed"] = seed
        rows.append(r)
        print(f"B={B:>3} beta2={beta2:<5} seed={seed}: events={r['events']} sev={r['severity']:.2f} "
              f"peak/med={r['peak_over_median']:.2f} at step {r['peak_step']} | R_act@1000={r['R_act@1000']:.3f} "
              f"R_act@2000={r['R_act@2000']:.3f} R_floor@2000={r['R_floor@2000']:.3f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print("wrote", save(f"e9_lm_repeat_{args.part}.json", {"rows": rows}))


if __name__ == "__main__":
    main()
