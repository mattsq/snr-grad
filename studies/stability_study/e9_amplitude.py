"""
E9a-2: validate the Adam edge-oscillation amplitude formula on rotated quadratics.

Noise-free and weakly noisy Adam (beta1 = 0.9, beta2 = 0.99) on L = x^T H x / 2 with
H = Q diag(lambda) Q^T in d = 20 (random rotation Q, lambda log-spaced over 2 decades),
for several learning rates. In the late phase (pinned at the adaptive edge) we compare
the observed peak loss with

    Delta L_edge = lr^2 h_w / (2 * 38^2 * max_i P_i u_i^2)

computed from the late Adam state (P = 1/sqrt(v_hat); h_w, u the top eigenpair of
P^1/2 H P^1/2). Also a "delocalized" case: H = lambda_1 * 1 1^T / d + small diagonal, whose
top direction spreads equally over all d coordinates (predicts d times the 1-D scale).

Run: uv run python studies/stability_study/e9_amplitude.py
"""

import math

import torch

from common import save
from snr_grad import adam_edge_amplitude


def run(H, lr, noise, steps=12000, seeds=8, seed=0):
    d = H.shape[0]
    gen = torch.Generator().manual_seed(seed)
    x = torch.nn.Parameter(torch.randn(seeds, d, generator=gen, dtype=torch.float64))
    opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99), eps=1e-12)
    peaks = torch.zeros(seeds, dtype=torch.float64)
    preds = []
    for t in range(steps):
        with torch.no_grad():
            g = x @ H + noise * torch.randn(seeds, d, generator=gen, dtype=torch.float64)
        x.grad = g
        opt.step()
        if t >= steps // 2:
            with torch.no_grad():
                peaks = torch.maximum(peaks, 0.5 * ((x @ H) * x).sum(1))
            if t % 500 == 0:
                st = opt.state[x]
                v_hat = st["exp_avg_sq"] / (1 - 0.99 ** float(st["step"]))
                for k in range(seeds):
                    P = 1.0 / (v_hat[k].sqrt() + 1e-12)
                    r = P.sqrt()
                    ev, vec = torch.linalg.eigh(r[:, None] * H * r[None, :])
                    preds.append(adam_edge_amplitude(float(ev[-1]), vec[:, -1], P, lr))
    return float(peaks.median()), float(torch.tensor(preds).median())


def main():
    torch.set_num_threads(1)
    d = 20
    gen = torch.Generator().manual_seed(1)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    H_rot = (q * torch.logspace(0, -2, d, dtype=torch.float64)) @ q.t()
    ones = torch.ones(d, dtype=torch.float64) / math.sqrt(d)
    H_deloc = torch.outer(ones, ones) * 1.0 + torch.diag(torch.full((d,), 0.01, dtype=torch.float64))
    rows = []
    for name, H in (("rotated", H_rot), ("delocalized", H_deloc)):
        for lr in (0.003, 0.01, 0.03):
            for noise in (0.0, 1e-6):
                peak, pred = run(H, lr, noise)
                rows.append({"case": name, "lr": lr, "noise": noise, "peak": peak, "pred": pred,
                             "ratio": peak / pred, "one_d_scale": lr ** 2 * float(torch.linalg.eigvalsh(H).max()) / (2 * 38 ** 2)})
                print(f"{name:<11} lr={lr:<6} noise={noise:g}: late peak={peak:.3e} predicted dL_edge={pred:.3e} "
                      f"peak/pred={peak / pred:6.1f}  (1-D scale lr^2 lambda_max/2888={rows[-1]['one_d_scale']:.2e})", flush=True)
    print("wrote", save("e9_amplitude.json", {"rows": rows}))


if __name__ == "__main__":
    main()
