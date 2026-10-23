"""
E9b: do confident (low-entropy) targets turn on Adam memory spikes in the tiny LM, as
the noise-floor / edge-amplitude theory predicts?

Language temperature in {0.5 (E7/E8 default), 0.1, 0.03} (lower = more deterministic
transitions, copy rule unchanged), batch B in {4, 32} sequences, beta2 in {0.95, 0.999},
lr in {0.003, 0.01}; d=64, L=32, 3000 AdamW steps, held-out loss every 5 steps.

At steps 1000 and 2000, measured indices (whitened by the actual Adam P = 1/sqrt(v_hat)):
  R_act    lr * lambda_max(P^1/2 G P^1/2) / 38          (current distance to the edge)
  R_floor  same with the noise-floor P_f = 1/(sigma/sqrt(B))  (how close noise lets v get)
  dL_edge  lr^2 h_w / (2 * 38^2 * max_i P_i u_i^2)          (edge-oscillation loss scale)
  S        30 * dL_edge / L                                  (predicted relative spike size)

Prediction: spikes iff R_floor is large (noise does not hold v up) and S is >~ 1, more
so with beta2 = 0.999. The high-entropy default should stay quiet.

Grids: "base" uses the E7/E8 language (copy_prob 0.3, whose copy-versus-Markov ambiguity
keeps the loss near 0.3 nats even at low temperature); "clean" removes the copy rule
(copy_prob 0, T = 0.03, entropy floor about 0.02 nats) and adds B = 128, the
large-batch, confident regime where the literature reports Adam spikes.

The per-sequence variance for the noise floor is estimated from 8-sequence microbatches
for every B: single-sequence microbatches (used at B = 4 in a first run) give zero
variance to rarely touched embedding rows and inflate R_floor.

Run: uv run python studies/stability_study/e9_lm_confident.py [--grid base|clean]
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
    adam_edge_amplitude,
    adam_edge_threshold,
    collect_microbatch_gradients,
    detect_excursions,
)

K = 32


def indices(lang, model, opt, lr, B, L, val, gen):
    params = [p for p in model.parameters() if p.requires_grad]
    beta2 = opt.param_groups[0]["betas"][1]
    P = {}
    for p in params:
        st = opt.state[p]
        P[p] = 1.0 / ((st["exp_avg_sq"] / (1 - beta2 ** float(st["step"]))).sqrt() + 1e-8)
    b = 8
    grads = collect_microbatch_gradients(model, lm_loss, lang.sample(K * b, L, gen), num_splits=K)
    dev = torch.stack([torch.cat([g[p].reshape(-1) for p in params]) for g in grads])
    sig2 = b * dev.var(0)                                          # per-sequence variance per coord
    flat_floor = 1.0 / ((sig2 / B).sqrt() + 1e-8)
    off, Pf = 0, {}
    for p in params:
        Pf[p] = flat_floor[off: off + p.numel()].reshape(p.shape)
        off += p.numel()
    model.eval()
    ggn = GGNCurvature(model, val[:, :-1], output_loss_fn(val), params=params)
    thr = adam_edge_threshold(0.9)
    out = {}
    for name, prec in (("act", P), ("floor", Pf)):
        low = LowRankCurvature.from_proxy(PreconditionedCurvature(ggn, prec), params, rank=4,
                                          oversample=4, power_iters=3)
        out[f"R_{name}"] = lr * float(low.eigenvalues[0]) / thr
        if name == "act":
            Pflat = torch.cat([prec[p].reshape(-1) for p in params])
            out["dL_edge"] = adam_edge_amplitude(float(low.eigenvalues[0]), low.eigenvectors[:, 0], Pflat, lr)
    model.train()
    with torch.no_grad():
        loss = float(lm_loss(model, val))
    out["loss"] = loss
    out["S_pred"] = 30.0 * out["dL_edge"] / loss
    return out


def run(temp, B, beta2, lr, copy_prob=0.3, d=64, L=32, steps=3000, seed=0, measure=(1000, 2000)):
    lang = MarkovLanguage(temperature=temp, copy_prob=copy_prob)
    torch.manual_seed(seed)
    model = TinyLM(lang.vocab, d, 2, heads=max(1, d // 32), max_len=L)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, beta2), weight_decay=0.0)
    gen = torch.Generator().manual_seed(100 + seed)
    val = lang.sample(32, L, torch.Generator().manual_seed(12345))
    curve, idx = [], {}
    for t in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        lm_loss(model, lang.sample(B, L, gen)).backward()
        opt.step()
        if t % 5 == 0:
            with torch.no_grad():
                curve.append(float(lm_loss(model, val)))
        if t in measure:
            idx[t] = indices(lang, model, opt, lr, B, L, val, torch.Generator().manual_seed(t))
    rep = detect_excursions(curve[100:], window=20, warmup=10, rel_floor=0.25)   # after step 500
    tail = torch.tensor(curve[100:])
    return {"temp": temp, "copy_prob": copy_prob, "entropy_floor": lang.entropy_floor(), "B": B,
            "beta2": beta2, "lr": lr, "events": rep.num_events,
            "severity": min(rep.max_excursion, 1e4), "diverged": rep.diverged,
            "final_loss": curve[-1], "min_loss": float(tail.min()),
            "peak_over_median": float(tail.max() / tail.median()),
            "peak_step": 5 * (100 + int(tail.argmax()) + 1), "curve": curve,
            **{f"{k}@{t}": v for t, dct in idx.items() for k, v in dct.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=("base", "clean"), default="base")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.grid == "base":
        cells = [(t, 0.3, B) for t in (0.5, 0.1, 0.03) for B in (4, 32)]
    else:
        cells = [(0.03, 0.0, B) for B in (4, 32, 128)]
    rows = []
    t0 = time.perf_counter()
    for temp, copy_prob, B in cells:
        for beta2 in (0.95, 0.999):
            for lr in (0.003, 0.01):
                r = run(temp, B, beta2, lr, copy_prob=copy_prob)
                rows.append(r)
                print(f"T={temp:<4} copy={copy_prob} B={B:>3} beta2={beta2:<5} lr={lr:<5}: events={r['events']:>3} "
                      f"sev={r['severity']:8.2f} peak/med={r['peak_over_median']:7.2f} loss final={r['final_loss']:.4f} "
                      f"| @2000 L={r['loss@2000']:.4f} R_act={r['R_act@2000']:.3f} R_floor={r['R_floor@2000']:.3f} "
                      f"dL_edge={r['dL_edge@2000']:.2e} S={r['S_pred@2000']:.3f} ({time.perf_counter() - t0:.0f}s)",
                      flush=True)
    print("wrote", save("e9_lm_confident.json" if args.grid == "base" else "e9_lm_clean.json", {"rows": rows}))


if __name__ == "__main__":
    main()
