"""Summarise E7a timings: speedup range and a simple fitted cost model."""

import json
import math
import os

import torch

from common import RESULTS

d = json.load(open(os.path.join(RESULTS, "e7_crossover_timing.json")))
rows = d["rows"]
sp = torch.tensor([r["speedup_1_query"] for r in rows])
spn = torch.tensor([r["speedup_1_query_noeval"] for r in rows])
print(f"configs={len(rows)}  1-query speedup min={sp.min():.2f} median={sp.median():.2f} max={sp.max():.2f}")
print(f"without held-out eval: min={spn.min():.2f} median={spn.median():.2f} max={spn.max():.2f}")
print("configs with speedup < 1 (no eval):",
      [(r['d'], r['layers'], r['L'], r['B']) for r in rows if r["speedup_1_query_noeval"] < 1])

# Cost model: t_step = a + b * params * tokens ; t_sim = c + e * params ; build = f + g * params * L
def fit(xs, ys):
    X = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    beta = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
    pred = X @ beta
    r2 = 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return beta.tolist(), r2

step_beta, step_r2 = fit([[1.0, r["params"] * r["tokens_per_step"]] for r in rows], [r["t_step"] for r in rows])
sim_beta, sim_r2 = fit([[1.0, r["params"]] for r in rows], [r["t_sim_step"] for r in rows])
build = [r["t_probe"] + r["t_build_lowrank"] for r in rows]
build_beta, build_r2 = fit([[1.0, r["params"] * r["L"] * 16, r["params"] * r["L"] * max(1, r["B"] // 4) * 32]
                            for r in rows], build)
print(f"t_step  = {step_beta[0] * 1e3:.2f} ms + {step_beta[1] * 1e12:.3f} ps * params * tokens   (R2 {step_r2:.2f})")
print(f"t_sim   = {sim_beta[0] * 1e3:.2f} ms + {sim_beta[1] * 1e9:.3f} ns * params            (R2 {sim_r2:.2f})")
print(f"build   = {build_beta[0]:.2f} s + ggn {build_beta[1] * 1e12:.3f} ps*params*L*16 + probe {build_beta[2] * 1e12:.3f} ps*params*L*b*K  (R2 {build_r2:.2f})")
print(f"per-step speedup grows ~ linearly in tokens/step: at params=1M, tokens=4096 the model predicts "
      f"{(step_beta[0] + step_beta[1] * 1e6 * 4096) / (sim_beta[0] + sim_beta[1] * 1e6):.0f}x")
out = {"step_model": step_beta, "step_r2": step_r2, "sim_model": sim_beta, "sim_r2": sim_r2,
       "build_model": build_beta, "build_r2": build_r2,
       "speedup_1q": [float(sp.min()), float(sp.median()), float(sp.max())],
       "speedup_1q_noeval": [float(spn.min()), float(spn.median()), float(spn.max())]}
json.dump(out, open(os.path.join(RESULTS, "e7_timing_summary.json"), "w"), indent=1)
print(f"{'d':>4}{'lay':>4}{'L':>5}{'B':>4}{'params':>9}{'tok':>6}{'step ms':>9}{'sim ms':>8}{'build s':>9}{'1q':>6}{'5q':>6}{'step x':>8}")
for r in rows:
    if r["layers"] == 2 and r["B"] in (2, 32):
        print(f"{r['d']:>4}{r['layers']:>4}{r['L']:>5}{r['B']:>4}{r['params'] / 1e6:>8.2f}M{r['tokens_per_step']:>6}"
              f"{r['t_step'] * 1e3:>9.1f}{r['t_sim_step'] * 1e3:>8.2f}{r['t_probe'] + r['t_build_lowrank']:>9.1f}"
              f"{r['speedup_1_query']:>6.1f}{r['speedup_5_queries']:>6.1f}{r['speedup_per_step']:>8.1f}")
