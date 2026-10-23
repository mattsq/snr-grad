"""
E8b-2: do the LM spike indices drift toward the spiky band with longer training?

Width 64 and 128 (L=64, AdamW lr=0.01, B in {8, 32}) trained for 4000 steps, measuring
kappa_hat, D_hat and Phi (e8_lm_indices.indices) every 500 steps plus the observed
excursion severity over the following 200 steps.

Run: uv run python studies/stability_study/e8_lm_trend.py
"""

import time

import torch

from common import save
from e8_lm_indices import run
from tinylm import MarkovLanguage


def main():
    torch.set_num_threads(4)
    lang = MarkovLanguage()
    rows = []
    t0 = time.perf_counter()
    for d in (64, 128):
        for B in (8, 32):
            for r in run(lang, d, 64, B, 0.01, tuple(range(500, 4001, 500))):
                rows.append(r)
                print(f"d={d} B={B} step={r['step']:>4}: loss={r['loss']:.3f} ratio={r['ratio']:.3f} "
                      f"kappa={r['kappa_hat']:.3f} D={r['D_hat']:.0f} Phi={r['Phi']:.4f} "
                      f"top-noise={r['noise_top_fraction']:.2f} sev={r['severity_next']:.3f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print("wrote", save("e8_lm_trend.json", {"rows": rows}))


if __name__ == "__main__":
    main()
