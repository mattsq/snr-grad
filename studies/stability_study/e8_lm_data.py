"""
E8b-3: does a heavy-tailed token distribution concentrate the gradient noise (lower
D_hat) and raise Phi?

Tiny LM (d=64, L=64, B=8, AdamW lr=0.01), synthetic language with vocab size in
{64, 512, 4096} and global Zipf exponent in {0, 1, 1.5}; indices at steps 300/600
(e8_lm_indices.indices) plus the observed excursion severity over the next 200 steps.

Run: uv run python studies/stability_study/e8_lm_data.py
"""

import time

import torch

import e8_lm_indices as e8
from common import save
from tinylm import MarkovLanguage


def main():
    torch.set_num_threads(4)
    rows = []
    t0 = time.perf_counter()
    for vocab in (64, 512, 4096):
        for zipf in (0.0, 1.0, 1.5):
            lang = MarkovLanguage(vocab=vocab, global_zipf=zipf)
            for r in e8.run(lang, 64, 64, 8, 0.01, (300, 600)):
                r.update({"vocab": vocab, "zipf": zipf})
                rows.append(r)
                print(f"vocab={vocab:>4} zipf={zipf}: step={r['step']} loss={r['loss']:.3f} ratio={r['ratio']:.3f} "
                      f"kappa={r['kappa_hat']:.3f} D={r['D_hat']:.0f} Phi={r['Phi']:.4f} "
                      f"top-noise={r['noise_top_fraction']:.2f} sev={r['severity_next']:.3f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print("wrote", save("e8_lm_data.json", {"rows": rows}))


if __name__ == "__main__":
    main()
