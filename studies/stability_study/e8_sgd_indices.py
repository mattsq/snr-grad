"""
E8a analysis (SGD): which closed-form index predicts spikiness?

Per PLRF SGD configuration (results/e8_plrf_theory.json):
  Phi    = kappa * sqrt(2 / D_K)   (stationary noise-driven fraction x relative SD of a
                                     D_K-mode chi-square: Gaussian fluctuation level)
  xi     = loss tail exponent of the dominant-mode Kesten recursion (lr*lambda_max, B)
  kappa  = mean-loss feedback gain
Targets: late-loss tail ratio (95th pct / median of L - L*), excursion events per run,
severity. Rows that diverged are reported separately.
"""

import json
import math
import os

import torch

from common import RESULTS, bp
from snr_grad import dominant_mode_tail

d = json.load(open(os.path.join(RESULTS, "e8_plrf_theory.json")))
rows = d["sgd"]
for r in rows:
    lr_lam = r["m"] * 2.0 / (1 + 1 / r["B"])
    k = dominant_mode_tail(lr_lam, r["B"], samples=100_000)
    r["xi"] = k["loss_tail_exponent"]
    r["lyapunov"] = k["lyapunov"]
    r["Phi"] = r["kappa"] * math.sqrt(2 / r["D_K"]) if math.isfinite(r["kappa"]) else float("inf")

ok = [r for r in rows if r["divergence_fraction"] == 0]
div = [r for r in rows if r["divergence_fraction"] > 0]
print(f"non-divergent rows: {len(ok)}, divergent: {len(div)}")
for target in ("tail_ratio_median", "events_per_run", "severity_median"):
    out = {}
    for idx in ("Phi", "kappa", "xi", "D_K"):
        xs = [(-r[idx] if idx in ("xi", "D_K") else r[idx]) for r in ok]
        out[idx] = round(bp.spearman([min(x, 1e9) for x in xs], [r[target] for r in ok]), 3)
    print(f"Spearman with {target:<18} (xi, D_K sign-flipped): {out}")

# Joint model: log severity ~ log Phi + 1/xi (non-divergent rows)
X = torch.tensor([[1.0, math.log(max(r["Phi"], 1e-6)), 1.0 / max(r["xi"], 1e-3)] for r in ok], dtype=torch.float64)
y = torch.tensor([math.log(r["severity_median"] + 1e-2) for r in ok], dtype=torch.float64)
beta = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
pred = X @ beta
r2 = 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
print(f"log(severity) = {beta[0]:.2f} + {beta[1]:.2f} log(Phi) + {beta[2]:.2f}/xi   R^2 = {r2:.2f}")
for cols, name in (([0, 1], "Phi only"), ([0, 2], "1/xi only")):
    b = torch.linalg.lstsq(X[:, cols], y[:, None]).solution[:, 0]
    p = X[:, cols] @ b
    print(f"   {name}: R^2 = {1 - float(((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()):.2f}")

print("\nDivergent configurations (kappa, D_K, lyapunov):")
for r in div:
    print(f"   a={r['alpha']} B={r['B']} m={r['m']}: kappa={r['kappa']:.2f} D_K={r['D_K']:.1f} "
          f"lyap={r['lyapunov']:+.3f} div={r['divergence_fraction']:.2f}")
print("\nkappa > 1 but NOT divergent:")
for r in ok:
    if r["kappa"] > 1:
        print(f"   a={r['alpha']} B={r['B']} m={r['m']}: kappa={r['kappa']:.2f} D_K={r['D_K']:.2f} xi={r['xi']:.2f} "
              f"sev={r['severity_median']:.0f}")
json.dump(rows, open(os.path.join(RESULTS, "e8_sgd_indices.json"), "w"), indent=1)
