"""
E8d-2: predicted vs observed severity for SGD on the tiny LM.

For the two E8d configurations, train with SGD at 0.5 / 0.95 x the measured edge to
step 1000 and compute kappa_hat, D_hat, Phi in SGD geometry (raw GGN, edge 2) plus
the dominant-mode tail exponent xi(lr * h_max, B). The severity predicted by the PLRF
SGD calibration (refit here from results/e8_sgd_indices.json) is compared with the
severity observed in E8d.

Run: uv run python studies/stability_study/e8_sgd_lm_phi.py
"""

import json
import math
import os

import torch

from common import RESULTS, save
from e8_sgd_lm_test import CONFIGS
from tinylm import MarkovLanguage, TinyLM, lm_loss, output_loss_fn
from snr_grad import (
    GGNCurvature,
    LowRankCurvature,
    collect_microbatch_gradients,
    dominant_mode_tail,
    noise_fluctuation_index,
)


def calibration():
    rows = [r for r in json.load(open(os.path.join(RESULTS, "e8_sgd_indices.json")))
            if r["divergence_fraction"] == 0]
    X = torch.tensor([[1.0, math.log(max(r["Phi"], 1e-6)), 1.0 / max(r["xi"], 1e-3)] for r in rows],
                     dtype=torch.float64)
    y = torch.tensor([math.log(r["severity_median"] + 1e-2) for r in rows], dtype=torch.float64)
    full = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
    phi_only = torch.linalg.lstsq(X[:, :2], y[:, None]).solution[:, 0]
    return full, phi_only


def main():
    torch.set_num_threads(4)
    full, phi_only = calibration()
    tests = json.load(open(os.path.join(RESULTS, "e8_sgd_lm_test.json")))
    out = []
    for name, c in CONFIGS.items():
        lang = MarkovLanguage(vocab=c["vocab"], global_zipf=c["zipf"])
        edge = tests[name]["edge"]
        for m in (0.5, 0.95):
            lr = m * edge
            torch.manual_seed(10)
            model = TinyLM(lang.vocab, c["d"], 2, heads=max(1, c["d"] // 32), max_len=c["L"])
            opt = torch.optim.SGD(model.parameters(), lr=lr)
            gen = torch.Generator().manual_seed(110)
            val = lang.sample(16, c["L"], torch.Generator().manual_seed(12345))
            for _ in range(1000):
                opt.zero_grad(set_to_none=True)
                lm_loss(model, lang.sample(c["B"], c["L"], gen)).backward()
                opt.step()
            params = [p for p in model.parameters() if p.requires_grad]
            b = max(1, c["B"] // 4)
            grads = collect_microbatch_gradients(model, lm_loss, lang.sample(32 * b, c["L"], gen), num_splits=32)
            model.eval()
            low = LowRankCurvature.from_proxy(GGNCurvature(model, val[:, :-1], output_loss_fn(val), params=params),
                                              params, rank=32, oversample=8, power_iters=1, trace_probes=8)
            model.train()
            dev = torch.stack([torch.cat([g[p].reshape(-1) for p in params]) for g in grads])
            dev = dev - dev.mean(0)
            sig = b * (dev @ low.eigenvectors).pow(2).sum(0) / 31
            tot = b * float(dev.pow(2).sum()) / 31
            with torch.no_grad():
                loss = float(lm_loss(model, val))
            idx = noise_fluctuation_index(low.eigenvalues, sig, lr, c["B"], loss, edge=2.0, horizon=400,
                                          rest=(low.residual, max(0.0, tot - float(sig.sum())),
                                                dev.shape[1] - 32))
            xi = dominant_mode_tail(min(lr * float(low.eigenvalues[0]), 4.0), c["B"])["loss_tail_exponent"]
            pred_full = math.exp(float(full[0] + full[1] * math.log(max(idx["Phi"], 1e-6)) + full[2] / max(xi, 1e-3)))
            pred_phi = math.exp(float(phi_only[0] + phi_only[1] * math.log(max(idx["Phi"], 1e-6))))
            obs = next(r for r in tests[name]["rows"] if r["m"] == m)
            row = {"config": name, "m": m, "lr_h_max": lr * float(low.eigenvalues[0]), **idx, "xi": xi,
                   "w": float(sig[int(torch.argmax(low.eigenvalues))]) / tot,
                   "pred_severity_phi_xi": pred_full, "pred_severity_phi": pred_phi,
                   "obs_severity_median": obs["severity_median"]}
            out.append(row)
            print(f"[{name}] m={m}: lr*h_max={row['lr_h_max']:.2f} kappa={idx['kappa_hat']:.3f} D={idx['D_hat']:.1f} "
                  f"Phi={idx['Phi']:.4f} xi={xi:.2f} w={row['w']:.3f} | predicted severity "
                  f"{pred_phi:.3f} (Phi) / {pred_full:.3f} (Phi+xi) vs observed {obs['severity_median']:.3f}",
                  flush=True)
    print("wrote", save("e8_sgd_lm_phi.json", {"rows": out}))


if __name__ == "__main__":
    main()
