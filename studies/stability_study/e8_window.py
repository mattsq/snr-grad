"""
E8c: the spike-window width w = (noise along the sharpest direction) / (total noise).

With a frozen preconditioner, raising lr by s gives mode i a stationary excess
E_i ~ s sigma_i / (1 - s r_i). Only the sharpest mode diverges at the edge, and it
dominates the noise-driven loss (D -> 1, large Phi) only within the last fraction
w ~ sigma_1 / sum sigma of the way to the edge. So w is the relative width of the
learning-rate band in which large noise-driven spikes are possible.

Part 1 (PLRF, exact): w = lambda_1 / tr(H) for SGD vs the observed spike onset in E8a
(smallest edge fraction m with severity >= 1 among non-divergent rows).
Part 2 (tiny LM): measured w at representative and spike-favourable configurations.

Run: uv run python studies/stability_study/e8_window.py
"""

import json
import os

import torch

import e8_lm_indices as e8
from common import RESULTS, bp, save
from tinylm import MarkovLanguage


def main():
    torch.set_num_threads(4)
    sgd = json.load(open(os.path.join(RESULTS, "e8_plrf_theory.json")))["sgd"]
    part1 = []
    for a in (0.3, 0.5, 0.7, 0.9):
        prob = bp.PLRF(bp.Config(), a)
        lam = torch.linalg.eigvalsh(prob.H).clamp_min(0)
        w = float(lam.max() / lam.sum())
        for B in (2, 8, 32):
            rs = sorted([r for r in sgd if r["alpha"] == a and r["B"] == B], key=lambda r: r["m"])
            onset = next((r["m"] for r in rs if r["divergence_fraction"] == 0 and r["severity_median"] >= 1.0), None)
            first_div = next((r["m"] for r in rs if r["divergence_fraction"] > 0), None)
            part1.append({"alpha": a, "B": B, "w": w, "predicted_window_start": 1 - w,
                          "observed_spike_onset_m": onset, "first_divergent_m": first_div})
            print(f"PLRF a={a} B={B:>2}: w=lambda1/trH={w:.3f} -> window m in ({1 - w:.2f}, 1); "
                  f"observed: severity>=1 from m={onset}, divergence from m={first_div}", flush=True)
    part2 = []
    for (vocab, zipf, d, L, B, lr) in ((64, 0.0, 64, 64, 8, 0.01), (64, 0.0, 128, 64, 32, 0.01),
                                        (64, 1.5, 128, 16, 4, 0.01), (64, 1.5, 256, 16, 2, 0.01),
                                        (4096, 1.5, 64, 64, 8, 0.01)):
        lang = MarkovLanguage(vocab=vocab, global_zipf=zipf)
        for r in e8.run(lang, d, L, B, lr, (300, 600)):
            r.update({"vocab": vocab, "zipf": zipf})
            part2.append(r)
            print(f"LM vocab={vocab} zipf={zipf} d={d} L={L} B={B} step={r['step']}: w={r['window_w']:.4f} "
                  f"ratio={r['ratio']:.3f} Phi={r['Phi']:.4f}", flush=True)
    print("wrote", save("e8_window.json", {"plrf": part1, "lm": part2}))


if __name__ == "__main__":
    main()
