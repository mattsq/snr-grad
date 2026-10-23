"""
E8b-4: test the extrapolation on the LM's most spike-favourable corner.

The E8b fits say Phi grows with width and lr and falls with batch and sequence length;
heavy global Zipf at a small vocabulary lowers D. Here: vocab 64, Zipf 1.5, L=16,
B in {2, 4}, width {128, 256}, lr {0.01, 0.02}. Checks whether measured Phi lands near
the fit's prediction and whether observed severity rises with Phi.

Run: uv run python studies/stability_study/e8_lm_favorable.py
"""

import json
import math
import os
import time

import torch

import e8_lm_indices as e8
from common import RESULTS, save
from tinylm import MarkovLanguage


def main():
    torch.set_num_threads(4)
    fits = json.load(open(os.path.join(RESULTS, "e8_lm_scaling.json")))["fits"]["Phi"]
    lang = MarkovLanguage(vocab=64, global_zipf=1.5)
    rows = []
    t0 = time.perf_counter()
    for d in (128, 256):
        for B in (2, 4):
            for lr in (0.01, 0.02):
                for r in e8.run(lang, d, 16, B, lr, (300, 600)):
                    pred = math.exp(fits["intercept"] + fits["d"] * math.log(d) + fits["B"] * math.log(B)
                                    + fits["lr"] * math.log(lr) + fits["L"] * math.log(16)
                                    + fits["step"] * math.log(r["step"]))
                    r.update({"vocab": 64, "zipf": 1.5, "Phi_fit_pred_uniform_data": pred})
                    rows.append(r)
                    print(f"d={d} B={B} lr={lr} step={r['step']}: loss={r['loss']:.3f} ratio={r['ratio']:.3f} "
                          f"kappa={r['kappa_hat']:.3f} D={r['D_hat']:.0f} Phi={r['Phi']:.4f} "
                          f"(fit, uniform-data: {pred:.4f}) sev={r['severity_next']:.3f} "
                          f"events={r['events_next']} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print("wrote", save("e8_lm_favorable.json", {"rows": rows}))


if __name__ == "__main__":
    main()
