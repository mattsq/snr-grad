"""
E11: does Muon have the Adam memory spikes of E9/E10, and what should a guard do for it?

Setting: the E9c/E10 spiking configuration (T = 0.03 without the copy rule, d = 64, L = 32,
B = 32, 3000 steps, no weight decay), 6 seeds and the same data order as E10. As is usual
for Muon, the hidden weight matrices of the blocks (qkv, proj, fc1, fc2) use torch Muon
(momentum 0.95, Nesterov, ``adjust_lr_fn="match_rms_adamw"``, so its update RMS matches
AdamW's at the same lr); the embeddings, the output head, the biases and the layer norms
use AdamW (0.9, 0.999), which is where E9d located the spikes.

Variants (``--part spiking``, lr 0.01 for both halves):
  adamw           AdamW everywhere (the E10 baseline; recorded again for the weight-norm and
                  attention-logit traces; the "muon" columns then describe AdamW's step on
                  the same matrices)
  muon            Muon + AdamW
  muon_guard      both wrapped in snr_grad.GuardedOptimizer with default settings: the
                  AdamW half gets the per-coordinate refill guard, the Muon half the
                  spectral step cap
  muon_adam_b2    Muon + AdamW (0.9, 0.95): attribution check with no guard

``--part lr`` scans the Muon lr (0.01, 0.03, 0.1, 0.3) with the AdamW half guarded and at
lr 0.01, to look for the Muon-specific failure reported at scale (weight-norm growth and
attention-logit growth; Musec, arXiv:2609.11655; Kimi K2, arXiv:2507.20534). Variants
``muon_lr{X}`` (plain Muon half), ``muon_lr{X}_guard`` (Muon half guarded too) and
``muon_lr{X}_wd`` (plain Muon half with decoupled weight decay 0.1, the Moonlight remedy).

Per step we record the largest normalized update of each half:
  adam_ratio      max |delta theta| / lr over the AdamW parameters
  muon_spec       max ||delta W||_2 / lr_eff over the Muon matrices (lr_eff includes
                  Muon's shape-dependent lr adjustment); 1 is Muon's nominal step
  muon_coord      max |delta W_ij| / lr over the Muon matrices
and every 50 steps the largest weight spectral norm among the Muon matrices and the largest
attention logit on the held-out batch.

Run: uv run python studies/stability_study/e11_muon.py [--part spiking|lr] [--variants ...]
"""

import argparse
import math
import time

import torch

from common import save
from tinylm import MarkovLanguage, TinyLM, lm_loss
from snr_grad import GuardedOptimizer, detect_excursions

SPIKING = ("adamw", "muon", "muon_guard", "muon_adam_b2")
LR_SCAN = tuple(f"muon_lr{lr}{g}" for lr in ("0.01", "0.03", "0.1", "0.3") for g in ("", "_guard"))


def split_params(model):
    hidden = [p for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2]
    ids = {id(p) for p in hidden}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return hidden, rest


def make_optimizers(variant, model, lr):
    hidden, rest = split_params(model)
    muon_lr = lr
    if variant == "adamw":
        return (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0),), hidden, rest, lr
    if variant.startswith("muon_lr"):
        muon_lr = float(variant[len("muon_lr"):].removesuffix("_guard").removesuffix("_wd"))
    muon = torch.optim.Muon(hidden, lr=muon_lr, weight_decay=0.1 if variant.endswith("_wd") else 0.0,
                            adjust_lr_fn="match_rms_adamw")
    betas = (0.9, 0.95) if variant == "muon_adam_b2" else (0.9, 0.999)
    adam = torch.optim.AdamW(rest, lr=lr, betas=betas, weight_decay=0.0)
    guard_muon = variant.endswith("guard")
    guard_adam = variant == "muon_guard" or variant.startswith("muon_lr")
    return (GuardedOptimizer(muon) if guard_muon else muon,
            GuardedOptimizer(adam) if guard_adam else adam), hidden, rest, muon_lr


def lr_eff(p, lr):
    a, b = p.shape
    return lr * 0.2 * math.sqrt(max(a, b))      # torch Muon, adjust_lr_fn="match_rms_adamw"


@torch.no_grad()
def max_attention_logit(model, batch):
    logits = []

    def hook(mod, inp, out):
        b, l, d3 = out.shape
        d, h = d3 // 3, model.blocks[0].heads
        q, k, _ = out.split(d, dim=-1)
        q = q.view(b, l, h, d // h).transpose(1, 2)
        k = k.view(b, l, h, d // h).transpose(1, 2)
        logits.append(float(((q @ k.transpose(-2, -1)) / math.sqrt(d // h)).abs().max()))

    handles = [blk.qkv.register_forward_hook(hook) for blk in model.blocks]
    model(batch[:, :-1])
    for h in handles:
        h.remove()
    return max(logits)


def run(variant, seed, temp=0.03, copy_prob=0.0, B=32, L=32, d=64, lr=0.01, steps=3000):
    lang = MarkovLanguage(temperature=temp, copy_prob=copy_prob)
    torch.manual_seed(seed)
    model = TinyLM(lang.vocab, d, 2, heads=max(1, d // 32), max_len=L)
    opts, hidden, rest, muon_lr = make_optimizers(variant, model, lr)
    gen = torch.Generator().manual_seed(100 + seed)
    val = lang.sample(32, L, torch.Generator().manual_seed(12345))
    curve, adam_ratio, muon_spec, muon_coord, wspec, logit = [], [], [], [], [], []
    for t in range(1, steps + 1):
        old_h = [p.detach().clone() for p in hidden]
        old_r = [p.detach().clone() for p in rest]
        for o in opts:
            o.zero_grad(set_to_none=True)
        lm_loss(model, lang.sample(B, L, gen)).backward()
        for o in opts:
            o.step()
        adam_ratio.append(max(float((p.detach() - o).abs().max()) for p, o in zip(rest, old_r)) / lr)
        deltas = [(p.detach() - o) for p, o in zip(hidden, old_h)]
        muon_spec.append(max(float(torch.linalg.matrix_norm(dw.double(), ord=2)) / lr_eff(p, muon_lr)
                             for dw, p in zip(deltas, hidden)))
        muon_coord.append(max(float(dw.abs().max()) for dw in deltas) / muon_lr)
        if t % 5 == 0:
            with torch.no_grad():
                curve.append(float(lm_loss(model, val)))
        if t % 50 == 0:
            wspec.append(max(float(torch.linalg.matrix_norm(p.detach().double(), ord=2)) for p in hidden))
            logit.append(max_attention_logit(model, val))
        if not math.isfinite(curve[-1] if curve else 0.0):
            break
    finite = [c if math.isfinite(c) else 1e4 for c in curve]
    rep = detect_excursions(finite[100:], window=20, warmup=10, rel_floor=0.25)
    tail = torch.tensor(finite[100:])
    return {"variant": variant, "seed": seed, "muon_lr": muon_lr, "events": rep.num_events,
            "severity": min(rep.max_excursion, 1e4),
            "peak_over_median": float(tail.max() / tail.median()),
            "loss@500": finite[99], "loss@1000": finite[199], "final_loss": finite[-1],
            "late_median": float(torch.tensor(finite[400:]).median()),
            "adam_ratio_max": max(adam_ratio[500:]), "muon_spec_max": max(muon_spec[500:]),
            "muon_spec_median": float(torch.tensor(muon_spec[500:]).median()),
            "muon_coord_max": max(muon_coord[500:]),
            "muon_coord_median": float(torch.tensor(muon_coord[500:]).median()),
            "weight_spec_final": wspec[-1], "weight_spec_max": max(wspec),
            "logit_final": logit[-1], "logit_max": max(logit),
            "curve": [float(f"{c:.4g}") for c in finite],
            "adam_ratio": [float(f"{x:.3g}") for x in adam_ratio],
            "muon_spec": [float(f"{x:.3g}") for x in muon_spec],
            "weight_spec": [float(f"{x:.4g}") for x in wspec],
            "logit": [float(f"{x:.4g}") for x in logit]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("spiking", "lr"), default="spiking")
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    variants = args.variants or list(SPIKING if args.part == "spiking" else LR_SCAN)
    seeds = args.seeds if args.seeds is not None else list(range(6 if args.part == "spiking" else 2))
    rows = []
    t0 = time.perf_counter()
    for variant in variants:
        for seed in seeds:
            r = run(variant, seed)
            rows.append(r)
            print(f"{args.part} {variant:<16} seed={seed}: events={r['events']} sev={r['severity']:6.2f} "
                  f"late={r['late_median']:.4f} adam_ratio={r['adam_ratio_max']:.2f} "
                  f"muon_spec={r['muon_spec_max']:.2f} (med {r['muon_spec_median']:.2f}) "
                  f"muon_coord={r['muon_coord_max']:.2f} (med {r['muon_coord_median']:.3f}) "
                  f"|W|_2={r['weight_spec_final']:.1f} (max {r['weight_spec_max']:.1f}) "
                  f"logit={r['logit_final']:.1f} (max {r['logit_max']:.1f}) ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    tag = args.tag or "_".join(variants)
    print("wrote", save(f"e11_muon_{args.part}_{tag}.json", {"rows": rows}))


if __name__ == "__main__":
    main()
