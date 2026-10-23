"""
E8b: measure the spike indices on the tiny LM and see how they scale.

For each configuration (width d, sequence length L, batch B sequences, AdamW lr), at
checkpoints (steps 300 and 600) we compute, in Adam-preconditioned coordinates
(P = 1/sqrt(v_hat), actual optimizer state):

  ratio      lr * lambda_max(P^1/2 G P^1/2) / 38          (adaptive edge fraction)
  kappa_hat  noise-driven fraction of the held-out loss     (noise_fluctuation_index)
  D_hat      number of modes carrying it
  Phi        kappa_hat * sqrt(2 / D_hat)

using a rank-32 GGN of P^1/2 G P^1/2 on 16 held-out sequences (rest lumped at the
residual curvature) and per-sequence noise from K=32 microbatches. The run then
continues for 200 steps with the held-out loss evaluated every step, giving the
observed excursion severity right after the checkpoint.

Run: uv run python studies/stability_study/e8_lm_indices.py [--quick]
"""

import argparse
import math
import time

import torch

from common import save
from tinylm import MarkovLanguage, TinyLM, lm_loss, output_loss_fn
from snr_grad import (
    GGNCurvature,
    LowRankCurvature,
    PreconditionedCurvature,
    adam_edge_threshold,
    collect_microbatch_gradients,
    detect_excursions,
    noise_fluctuation_index,
)

K, RANK, WINDOW = 32, 32, 200


def indices(lang, model, opt, lr, B, L, val, gen):
    params = [p for p in model.parameters() if p.requires_grad]
    prec = {}
    for group in opt.param_groups:
        beta2, eps = group["betas"][1], group["eps"]
        for p in group["params"]:
            st = opt.state[p]
            v_hat = st["exp_avg_sq"] / (1 - beta2 ** float(st["step"]))
            prec[p] = 1.0 / (v_hat.sqrt() + eps)
    b = max(1, B // 4)
    grads = collect_microbatch_gradients(model, lm_loss, lang.sample(K * b, L, gen), num_splits=K)
    model.eval()
    ggn = GGNCurvature(model, val[:, :-1], output_loss_fn(val), params=params)
    low = LowRankCurvature.from_proxy(PreconditionedCurvature(ggn, prec), params, rank=RANK,
                                      oversample=8, power_iters=1, trace_probes=8)
    model.train()
    root = torch.cat([prec[p].sqrt().reshape(-1) for p in params])
    dev = torch.stack([torch.cat([g[p].reshape(-1) for p in params]) for g in grads])
    dev = (dev - dev.mean(0)) * root                       # whitened microbatch deviations
    proj = dev @ low.eigenvectors                          # [K, r]
    sig_top = b * proj.pow(2).sum(0) / (K - 1)             # per-sequence variance along u_i
    sig_tot = b * float(dev.pow(2).sum()) / (K - 1)
    n = root.numel()
    with torch.no_grad():
        loss = float(lm_loss(model, val))
    thr = adam_edge_threshold(opt.param_groups[0]["betas"][0])
    idx = noise_fluctuation_index(low.eigenvalues, sig_top, lr, B, loss, edge=thr, horizon=WINDOW,
                                  rest=(low.residual, max(0.0, sig_tot - float(sig_top.sum())), n - RANK))
    top = int(torch.argmax(low.eigenvalues))
    return {"loss": loss, "ratio": lr * float(low.eigenvalues[0]) / thr, "h_max": float(low.eigenvalues[0]),
            "noise_top_fraction": float(sig_top.sum()) / sig_tot if sig_tot > 0 else float("nan"),
            # Spike-window width: share of the gradient noise along the sharpest direction.
            "window_w": float(sig_top[top]) / sig_tot if sig_tot > 0 else float("nan"), **idx}


def run(lang, d, L, B, lr, ckpts, gen_seed=0):
    torch.manual_seed(gen_seed)
    model = TinyLM(lang.vocab, d, 2, heads=max(1, d // 32), max_len=L)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    gen = torch.Generator().manual_seed(100 + gen_seed)
    val = lang.sample(16, L, torch.Generator().manual_seed(12345))
    out = []
    pending = {}
    for t in range(1, max(ckpts) + WINDOW + 1):
        opt.zero_grad(set_to_none=True)
        lm_loss(model, lang.sample(B, L, gen)).backward()
        opt.step()
        for c in list(pending):
            with torch.no_grad():
                pending[c]["curve"].append(float(lm_loss(model, val)))
            if len(pending[c]["curve"]) > WINDOW:
                rep = detect_excursions(pending[c]["curve"], window=20, warmup=10)
                row = pending.pop(c)
                row.pop("curve")
                row.update({"severity_next": min(rep.max_excursion, 100.0), "events_next": rep.num_events,
                            "diverged_next": rep.diverged})
                out.append(row)
        if t in ckpts:
            idx = indices(lang, model, opt, lr, B, L, val, torch.Generator().manual_seed(7 * t))
            with torch.no_grad():
                start = float(lm_loss(model, val))
            pending[t] = {"d": d, "L": L, "B": B, "lr": lr, "step": t,
                          "params": sum(p.numel() for p in model.parameters()), **idx, "curve": [start]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    lang = MarkovLanguage()
    grid = [(d, 64, B, lr) for d in (32, 64, 128, 256) for B in (8, 32) for lr in (0.003, 0.01, 0.03)]
    grid += [(64, L, 8, 0.01) for L in (16, 256)]
    grid += [(d, 64, 128, lr) for d in (32, 64) for lr in (0.01, 0.03)]
    ckpts = (300, 600)
    if args.quick:
        grid, ckpts = [(32, 32, 8, 0.01)], (100,)
    rows = []
    t0 = time.perf_counter()
    for d, L, B, lr in grid:
        for r in run(lang, d, L, B, lr, ckpts):
            rows.append(r)
            print(f"d={d:>3} L={L:>3} B={B:>3} lr={lr:<6g} step={r['step']:>3}: loss={r['loss']:.3f} "
                  f"ratio={r['ratio']:.3f} kappa={r['kappa_hat']:.3f} D={r['D_hat']:.1f} Phi={r['Phi']:.3f} "
                  f"top-noise={r['noise_top_fraction']:.2f} | next200: sev={r['severity_next']:.3f} "
                  f"events={r['events_next']} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print("wrote", save("e8_lm_indices" + ("_quick" if args.quick else "") + ".json", {"rows": rows}))


if __name__ == "__main__":
    main()
