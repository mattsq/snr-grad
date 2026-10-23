"""
E0: does any curvature-aware sensor carry information beyond lr * lambda_max(P H)?

Reads benchmarks/stability_plrf_results.json (from benchmark_stability_plrf.py) and
reports, on held-out checkpoints:
  * partial Spearman of each sensor with severity, controlling for sharpness
    (residuals of rank-on-rank linear fits);
  * out-of-sample R^2 of log-severity regressions fitted on dev checkpoints:
    sharpness only vs sharpness + sensor.

Run: uv run python studies/stability_study/e0_incremental.py
"""

import json
import math
import os
import sys

import torch

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "..", "benchmarks", "stability_plrf_results.json")
OUT = os.path.join(HERE, "results", "e0_incremental.json")

SENSORS = [
    "noise_to_loss", "upper_tail_ratio", "rel_sd", "noise_penalty", "gns_optimizer",
    "probe_noise_to_loss", "sim_probe_lossprop_severity", "sim_exact_lossprop_severity",
    "activation_participation_ratio",
]


def ranks(x):
    t = torch.tensor(x, dtype=torch.float64)
    r = torch.empty_like(t)
    r[torch.argsort(t)] = torch.arange(len(x), dtype=torch.float64)
    return r


def resid(y, xs):
    X = torch.stack([torch.ones_like(y)] + xs, 1)
    beta = torch.linalg.lstsq(X, y[:, None]).solution
    return y - (X @ beta)[:, 0]


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def ok(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def r2_oos(train, test, feats, target="severity"):
    def design(rows):
        return torch.tensor([[1.0] + [math.log(r[f]) for f in feats] for r in rows],
                            dtype=torch.float64)
    y_tr = torch.tensor([math.log(r[target] + 1e-3) for r in train], dtype=torch.float64)
    y_te = torch.tensor([math.log(r[target] + 1e-3) for r in test], dtype=torch.float64)
    beta = torch.linalg.lstsq(design(train), y_tr[:, None]).solution
    pred = (design(test) @ beta)[:, 0]
    return 1.0 - float(((y_te - pred) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum())


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    rows = json.load(open(src))["checkpoints"]
    out = {}
    for group in ["pooled", "sgd", "adamw", "snr_adamw"]:
        rs = [r for r in rows if group == "pooled" or r["optimizer"] == group]
        res = {}
        for s in SENSORS:
            sub = [r for r in rs if ok(r.get(s)) and ok(r.get("lr_sharpness"))]
            held = [r for r in sub if r["split"] == "heldout"]
            dev = [r for r in sub if r["split"] == "dev"]
            if len(held) < 10:
                continue
            y = ranks([r["severity"] for r in held])
            sh = ranks([r["lr_sharpness"] for r in held])
            x = ranks([r[s] for r in held])
            partial = corr(resid(y, [sh]), resid(x, [sh]))
            res[s] = {
                "n": len(held),
                "partial_spearman_given_sharpness": partial,
                "r2_sharpness_only": r2_oos(dev, held, ["lr_sharpness"]),
                "r2_sharpness_plus_sensor": r2_oos(dev, held, ["lr_sharpness", s]),
                "r2_sensor_only": r2_oos(dev, held, [s]),
            }
        out[group] = res
        print(f"-- {group}")
        print(f"   {'sensor':<32}{'partial|sharp':>14}{'R2 sharp':>10}{'R2 +sensor':>12}{'R2 sensor':>11}")
        for s, v in sorted(res.items(), key=lambda kv: -kv[1]["partial_spearman_given_sharpness"]):
            print(f"   {s:<32}{v['partial_spearman_given_sharpness']:>+14.3f}"
                  f"{v['r2_sharpness_only']:>10.3f}{v['r2_sharpness_plus_sensor']:>12.3f}"
                  f"{v['r2_sensor_only']:>11.3f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
