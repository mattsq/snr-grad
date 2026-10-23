"""E1 breakdown: within-stratum Spearman (varying B) per optimizer and spectrum."""

import json
import os

from common import RESULTS, bp, finite

SENSORS = ["lr_sharpness", "gns_optimizer", "noise_penalty", "noise_to_loss",
           "sim_probe_lossprop_severity", "sim_exact_lossprop_severity"]

d = json.load(open(os.path.join(RESULTS, "e1_decoupling.json")))
rows = d["rows"]
out = {}
print(f"{'opt':<6}{'alpha':>6}  " + "".join(f"{s[:18]:>20}" for s in SENSORS))
for o in ("sgd", "adamw"):
    for a in (0.5, 0.9):
        line = {}
        for s in SENSORS:
            vals = []
            for beta in sorted({r["beta"] for r in rows}):
                for step in sorted({r["step"] for r in rows}):
                    rs = [r for r in rows if r["optimizer"] == o and r["alpha"] == a and r["beta"] == beta
                          and r["step"] == step and finite(r.get(s))]
                    if len(rs) >= 6 and len({round(r[s], 12) for r in rs}) > 1:
                        vals.append(bp.spearman([r[s] for r in rs], [r["severity"] for r in rs]))
            line[s] = sum(vals) / len(vals) if vals else float("nan")
        out[f"{o}_{a}"] = line
        print(f"{o:<6}{a:>6}  " + "".join(f"{line[s]:>+20.3f}" for s in SENSORS))
json.dump(out, open(os.path.join(RESULTS, "e1_breakdown.json"), "w"), indent=1)
