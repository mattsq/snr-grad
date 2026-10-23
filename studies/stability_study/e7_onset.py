"""
E7 analysis: predicted vs real instability onset per checkpoint.

Onset = smallest candidate lr whose real (or simulated) median loss change over the
horizon exceeds +0.05 nats, or that diverges. The sharpness rule predicts the smallest
lr with lr * lambda_max(P^1/2 H P^1/2) > 2. Reports the ratio predicted / real onset
(1 = exact; > 1 = optimistic, i.e. would allow an unsafe lr).

Run: uv run python studies/stability_study/e7_onset.py results/e7_whatif_*.json
"""

import json
import math
import sys


def onset(rows, key):
    for r in sorted(rows, key=lambda r: r["lr"]):
        if key(r):
            return r["lr"]
    return None


def main(paths):
    out = {}
    for path in paths:
        d = json.load(open(path))
        thr = d["config"]["unsafe"]
        ck = sorted({(r["seed"], r["step"]) for r in d["rows"]})
        res = []
        for seed, step in ck:
            rs = [r for r in d["rows"] if r["seed"] == seed and r["step"] == step]
            real = onset(rs, lambda r: (not math.isfinite(r["real_loss_change"])) or r["real_loss_change"] > thr
                         or r["real_diverged"] > 0.5)
            sim = onset(rs, lambda r: (not math.isfinite(r["sim_loss_change"])) or r["sim_loss_change"] > thr
                        or r["sim_diverged"] > 0.5)
            sharp = onset(rs, lambda r: r["lr_sharpness"] > 2.0)
            res.append({"seed": seed, "step": step, "real": real, "sim": sim, "sharpness": sharp,
                        "sim_ratio": (sim / real) if sim and real else None,
                        "sharp_ratio": (sharp / real) if sharp and real else None})
            print(f"{path.split('/')[-1]:<34} seed={seed} step={step:>4}: onset real={real} sim={sim} "
                  f"sharpness>2={sharp}")
        out[path] = res
    json.dump(out, open("results/e7_onset.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
