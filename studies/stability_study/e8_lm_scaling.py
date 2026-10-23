"""
E8b analysis: power-law fits of the LM spike indices and extrapolation.

Fits log kappa_hat, log D_hat and log ratio (rows below the adaptive edge) on
log d, log B, log lr, log L and log step, then asks where Phi = kappa sqrt(2/D)
would reach the PLRF "visible spike" band (0.15) and where the adaptive edge
(ratio = 1, stalling) is reached first.
"""

import json
import math
import os

import torch

from common import RESULTS

rows = json.load(open(os.path.join(RESULTS, "e8_lm_indices.json")))["rows"]
trend_path = os.path.join(RESULTS, "e8_lm_trend.json")
if os.path.exists(trend_path):
    rows = rows + json.load(open(trend_path))["rows"]
ok = [r for r in rows if math.isfinite(r["kappa_hat"]) and r["ratio"] < 1]
stalled = [r for r in rows if r["ratio"] >= 1]
print(f"rows below adaptive edge: {len(ok)}; at/above edge (stall): {len(stalled)}")
feats = ["d", "B", "lr", "L", "step"]
X = torch.tensor([[1.0] + [math.log(r[f]) for f in feats] for r in ok], dtype=torch.float64)
fits = {}
for target in ("kappa_hat", "D_hat", "ratio", "Phi"):
    y = torch.tensor([math.log(r[target]) for r in ok], dtype=torch.float64)
    beta = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
    pred = X @ beta
    r2 = 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    fits[target] = {"intercept": float(beta[0]), **{f: float(b) for f, b in zip(feats, beta[1:])}, "r2": r2}
    print(f"log {target:<9} = {beta[0]:+.2f} " + " ".join(f"{float(b):+.2f}*log({f})" for f, b in zip(feats, beta[1:]))
          + f"   R^2={r2:.2f}")


def predict(target, **kw):
    f = fits[target]
    return math.exp(f["intercept"] + sum(f[k] * math.log(v) for k, v in kw.items()))


print("\nExtrapolation (L=64, lr=0.01, step=4000): width needed for Phi = 0.15 at each B")
out = {"fits": fits, "extrapolation": []}
for B in (8, 32, 128):
    for d in (256, 1024, 4096, 16384, 65536):
        phi = predict("Phi", d=d, B=B, lr=0.01, L=64, step=4000)
        ratio = predict("ratio", d=d, B=B, lr=0.01, L=64, step=4000)
        out["extrapolation"].append({"B": B, "d": d, "Phi": phi, "ratio": ratio})
        print(f"   B={B:>3} d={d:>6}: Phi={phi:.4f} ratio={ratio:.3f}")
# Solve for d with Phi = 0.15 analytically from the Phi fit (if the width exponent is positive).
fp = fits["Phi"]
for B in (8, 32, 128):
    base = fp["intercept"] + fp["B"] * math.log(B) + fp["lr"] * math.log(0.01) + fp["L"] * math.log(64) + fp["step"] * math.log(4000)
    if fp["d"] > 0:
        d_star = math.exp((math.log(0.15) - base) / fp["d"])
        print(f"   Phi=0.15 at d ~ {d_star:.3g} for B={B} (lr=0.01, L=64, 4000 steps)")
json.dump(out, open(os.path.join(RESULTS, "e8_lm_scaling.json"), "w"), indent=1)
