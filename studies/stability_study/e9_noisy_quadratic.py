"""
E9a: noise-floor protection and spike amplitude for Adam on a noisy quadratic.

Adam (beta1 = 0.9, eps = 1e-8) on L = h x^2 / 2 with minibatch noise g = h x + xi,
xi ~ N(0, s^2), s = sigma / sqrt(B). Theory (docs/adam-spikes-literature.md):

  1. Protection. The noise floor bounds sqrt(v) >= s, so lr h / sqrt(v) <= lr h / s =
     38 R_floor. If R_floor < 1, Adam stays inside its edge (lr h / sqrt(v) < 38); if
     R_floor > 1, v decays until lr h / sqrt(v) pins near the adaptive edge, about 38.
  2. Amplitude. At the edge the gradient must reach about sqrt(v) = lr h / 38 to refill
     the memory, a displacement of about lr / 38, so excursions cost about
     Delta L ~ h (lr / 38)^2 / 2, independent of the noise and of the gradient scale.

We sweep R_floor in {0.25, 0.5, 1, 2, 4, 16, inf (noise-free)} and lr in {0.003, 0.01,
0.03} (h = 1, beta2 = 0.99; plus beta2 = 0.999 at lr = 0.01). Late-phase statistics:
median of lr h / sqrt(v_hat), late peak loss, and peak / (lr^2 h / 2888).

Run: uv run python studies/stability_study/e9_noisy_quadratic.py
"""

import math
import time

import torch

from common import save
from snr_grad import adam_edge_threshold


def run(lr, s, beta2, steps, seeds=16, h=1.0, seed=0):
    x = torch.nn.Parameter(torch.ones(seeds, 1, dtype=torch.float64))
    opt = torch.optim.Adam([x], lr=lr, betas=(0.9, beta2), eps=1e-8)
    gen = torch.Generator().manual_seed(seed)
    late_ratio, late_loss = [], []
    for t in range(steps):
        with torch.no_grad():
            g = h * x + torch.randn(seeds, 1, generator=gen, dtype=torch.float64) * s
        x.grad = g
        opt.step()
        if t >= steps // 2:
            st = opt.state[x]
            v_hat = st["exp_avg_sq"] / (1 - beta2 ** float(st["step"]))
            late_ratio.append(lr * h / v_hat.sqrt()[:, 0])
            late_loss.append(0.5 * h * x.detach()[:, 0] ** 2)
    return torch.stack(late_ratio), torch.stack(late_loss)


def main():
    torch.set_num_threads(1)
    thr = adam_edge_threshold(0.9)
    rows = []
    t0 = time.perf_counter()
    configs = [(lr, 0.99) for lr in (0.003, 0.01, 0.03)] + [(0.01, 0.999)]
    for lr, beta2 in configs:
        steps = 8000 if beta2 == 0.99 else 30000
        for r_floor in (0.25, 0.5, 1.0, 2.0, 4.0, 16.0, float("inf")):
            s = 0.0 if math.isinf(r_floor) else lr / (thr * r_floor)   # h = 1
            ratio, loss = run(lr, s, beta2, steps)
            scale = lr ** 2 / (2 * thr ** 2)
            row = {"lr": lr, "beta2": beta2, "R_floor": r_floor, "noise_std": s,
                   "ratio_median": float(ratio.median()), "ratio_q95": float(torch.quantile(ratio, 0.95)),
                   "late_loss_median": float(loss.median()),
                   "late_loss_peak_median_over_seeds": float(loss.max(0).values.median()),
                   "amplitude_scale": scale}
            row["peak_over_scale"] = row["late_loss_peak_median_over_seeds"] / scale
            rows.append(row)
            print(f"lr={lr:<6} beta2={beta2} R_floor={r_floor:>5}: lr*h/sqrt(v) median={row['ratio_median']:6.1f} "
                  f"(floor value {min(38 * r_floor, 1e9):8.1f}) q95={row['ratio_q95']:6.1f} | late loss median="
                  f"{row['late_loss_median']:.2e} peak={row['late_loss_peak_median_over_seeds']:.2e} "
                  f"peak/(lr^2 h/2888)={row['peak_over_scale']:7.2f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print("wrote", save("e9_noisy_quadratic.json", {"rows": rows}))


if __name__ == "__main__":
    main()
