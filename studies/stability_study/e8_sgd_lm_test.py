"""
E8d: falsification test of the window theory on the tiny LM.

Prediction (E8c): in SGD geometry the configuration d=128, L=16, B=4 with a Zipf-1.5
64-token language has noise share along the sharpest direction w ~ 0.1-0.16 (PLRF
spike-capable range), while d=64, L=64, B=8 on the uniform language has w ~ 0.01.
So plain SGD near its stability edge should produce intermittent excursions in the
first configuration and go straight from smooth to divergent in the second. (With
Adam, w collapses to ~1e-3 and neither spikes; E8b.)

Procedure per configuration: bisect the SGD learning rate at which training diverges
or stalls (held-out loss above its starting value at the end of 1500 steps, majority
of 2 seeds); then train 3 seeds for 2000 steps at m in {0.5, 0.7, 0.85, 0.95} x edge,
evaluating the held-out loss every 5 steps, and report excursion events and severity.

Run: uv run python studies/stability_study/e8_sgd_lm_test.py
"""

import math
import time

import torch

from common import save
from tinylm import MarkovLanguage, TinyLM, lm_loss
from snr_grad import detect_excursions

CONFIGS = {
    "aligned (w_sgd~0.1-0.16)": dict(vocab=64, zipf=1.5, d=128, L=16, B=4),
    "control (w_sgd~0.01)": dict(vocab=64, zipf=0.0, d=64, L=64, B=8),
}


def train(lang, c, lr, seed, steps, every=5):
    torch.manual_seed(seed)
    model = TinyLM(lang.vocab, c["d"], 2, heads=max(1, c["d"] // 32), max_len=c["L"])
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(100 + seed)
    val = lang.sample(16, c["L"], torch.Generator().manual_seed(12345))
    curve = []
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        lm_loss(model, lang.sample(c["B"], c["L"], gen)).backward()
        opt.step()
        if t % every == 0:
            with torch.no_grad():
                curve.append(float(lm_loss(model, val)))
            if not math.isfinite(curve[-1]) or curve[-1] > 50:
                curve += [float("inf")] * (steps // every - len(curve))
                break
    return curve


def failed(curve):
    return not math.isfinite(curve[-1]) or curve[-1] > curve[0]


def main():
    torch.set_num_threads(4)
    out = {}
    t0 = time.perf_counter()
    for name, c in CONFIGS.items():
        lang = MarkovLanguage(vocab=c["vocab"], global_zipf=c["zipf"])
        lo, hi = 0.01, 20.0
        for _ in range(9):
            mid = math.sqrt(lo * hi)
            fails = sum(failed(train(lang, c, mid, s, 1500)) for s in range(2))
            if fails >= 1:
                hi = mid
            else:
                lo = mid
        edge = lo
        print(f"[{name}] SGD edge lr ~ {edge:.3g} ({time.perf_counter() - t0:.0f}s)", flush=True)
        rows = []
        for m in (0.5, 0.7, 0.85, 0.95):
            reps, finals = [], []
            for s in range(3):
                curve = train(lang, c, m * edge, 10 + s, 2000)
                reps.append(detect_excursions(curve, window=10, warmup=10))
                finals.append(curve[-1])
            row = {"m": m, "lr": m * edge, "events_per_run": sum(r.num_events for r in reps) / 3,
                   "severity_median": sorted(min(r.max_excursion, 100.0) for r in reps)[1],
                   "severity_max": max(min(r.max_excursion, 100.0) for r in reps),
                   "diverged": sum(r.diverged for r in reps) / 3, "final_median": sorted(finals)[1]}
            rows.append(row)
            print(f"   m={m:.2f}: events/run={row['events_per_run']:.2f} severity median={row['severity_median']:.3f} "
                  f"max={row['severity_max']:.3f} diverged={row['diverged']:.2f} final={row['final_median']:.3f} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
        out[name] = {"config": c, "edge": edge, "rows": rows}
    print("wrote", save("e8_sgd_lm_test.json", out))


if __name__ == "__main__":
    main()
