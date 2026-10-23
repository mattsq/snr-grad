"""
E10: mitigation trial for the Adam memory spikes of E9.

Setting (spiking): the E9c configuration, T = 0.03 without the copy rule, d = 64, L = 32,
B = 32, lr = 0.01, (beta1, beta2) = (0.9, 0.999), 3000 steps; 6 seeds per variant.
Setting (progress check): the E7/E8 language (T = 0.5, copy rule 0.3), same model and
optimizer settings, 2 seeds per variant, where no variant should spike and only the
loss matters.

Variants (all without weight decay):
  adamw        torch AdamW (baseline)
  cap          per-coordinate update ratio |m_hat| / (sqrt(v_hat) + eps) clipped to 1
  refill       when that ratio exceeds 1, raise v_hat so that it equals 1 (the memory
               is refilled, so the next steps are damped too)
  amsgrad      torch AdamW(amsgrad=True): v_hat never decreases
  eps1d_1e-4   eps = 1e-4 on one-dimensional parameters (biases, layer norms), about
               their typical sqrt(v_hat) after 1500 steps; 1e-8 elsewhere
  eps1d_1e-5   the same with eps = 1e-5
  beta1_0.98   (beta1, beta2) = (0.98, 0.999)
  beta2_0.95   (beta1, beta2) = (0.9, 0.95)

Run: uv run python studies/stability_study/e10_mitigation.py [--part spiking|progress]
"""

import argparse
import math
import time

import torch

from common import save
from tinylm import MarkovLanguage, TinyLM, lm_loss
from snr_grad import detect_excursions

VARIANTS = ("adamw", "cap", "refill", "amsgrad", "eps1d_1e-4", "eps1d_1e-5", "beta1_0.98", "beta2_0.95")


class GuardedAdam(torch.optim.Optimizer):
    """Adam (no weight decay) whose per-coordinate update ratio is bounded by ``limit``.

    ``mode="cap"`` clips ``m_hat / (sqrt(v_hat) + eps)`` to ``[-limit, limit]``.
    ``mode="refill"`` instead raises ``v_hat`` to ``(|m_hat| / limit - eps)^2`` where the
    ratio would exceed ``limit``, and stores it, so the memory is no longer stale.
    With ``limit = inf`` both reduce to torch Adam.
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, limit=1.0, mode="cap"):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, limit=limit, mode=mode))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                m_hat = st["m"] / bc1
                v_hat = st["v"] / bc2
                if group["mode"] == "refill" and math.isfinite(group["limit"]):
                    need = (m_hat.abs() / group["limit"] - group["eps"]).clamp_min(0) ** 2
                    v_hat = torch.maximum(v_hat, need)
                    st["v"].copy_(v_hat * bc2)
                ratio = m_hat / (v_hat.sqrt() + group["eps"])
                if group["mode"] == "cap":
                    ratio.clamp_(-group["limit"], group["limit"])
                p.add_(ratio, alpha=-group["lr"])


def make_optimizer(variant, model, lr):
    if variant in ("cap", "refill"):
        return GuardedAdam(model.parameters(), lr, limit=1.0, mode=variant)
    if variant.startswith("eps1d_"):
        eps = float(variant.split("_")[1])
        one_d = [p for p in model.parameters() if p.dim() == 1]
        rest = [p for p in model.parameters() if p.dim() != 1]
        return torch.optim.AdamW([{"params": one_d, "eps": eps}, {"params": rest}], lr=lr,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    betas = {"beta1_0.98": (0.98, 0.999), "beta2_0.95": (0.9, 0.95)}.get(variant, (0.9, 0.999))
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=0.0,
                             amsgrad=variant == "amsgrad")


def run(variant, seed, temp, copy_prob, B=32, L=32, d=64, lr=0.01, steps=3000):
    lang = MarkovLanguage(temperature=temp, copy_prob=copy_prob)
    torch.manual_seed(seed)
    model = TinyLM(lang.vocab, d, 2, heads=max(1, d // 32), max_len=L)
    opt = make_optimizer(variant, model, lr)
    gen = torch.Generator().manual_seed(100 + seed)
    val = lang.sample(32, L, torch.Generator().manual_seed(12345))
    curve, max_ratio = [], []
    for t in range(1, steps + 1):
        old = [p.detach().clone() for p in model.parameters()]
        opt.zero_grad(set_to_none=True)
        lm_loss(model, lang.sample(B, L, gen)).backward()
        opt.step()
        max_ratio.append(max(float((p.detach() - o).abs().max()) for p, o in zip(model.parameters(), old)) / lr)
        if t % 5 == 0:
            with torch.no_grad():
                curve.append(float(lm_loss(model, val)))
    rep = detect_excursions(curve[100:], window=20, warmup=10, rel_floor=0.25)
    tail = torch.tensor(curve[100:])
    return {"variant": variant, "seed": seed, "temp": temp, "copy_prob": copy_prob,
            "events": rep.num_events, "severity": min(rep.max_excursion, 1e4),
            "peak_over_median": float(tail.max() / tail.median()),
            "loss@500": curve[99], "loss@1000": curve[199], "final_loss": curve[-1],
            "late_median": float(torch.tensor(curve[400:]).median()),
            "max_update_ratio": max(max_ratio[500:]),
            "curve": [float(f"{c:.4g}") for c in curve]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("spiking", "progress"), default="spiking")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    temp, copy_prob, seeds = (0.03, 0.0, 6) if args.part == "spiking" else (0.5, 0.3, 2)
    rows = []
    t0 = time.perf_counter()
    for variant in args.variants:
        for seed in range(seeds):
            r = run(variant, seed, temp, copy_prob)
            rows.append(r)
            print(f"{args.part} {variant:<11} seed={seed}: events={r['events']} sev={r['severity']:6.2f} "
                  f"loss@500={r['loss@500']:.4f} @1000={r['loss@1000']:.4f} late median={r['late_median']:.4f} "
                  f"final={r['final_loss']:.4f} max ratio={r['max_update_ratio']:.2f} ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    tag = "_".join(args.variants) if args.variants != list(VARIANTS) else "all"
    print("wrote", save(f"e10_mitigation_{args.part}_{tag}.json", {"rows": rows}))


if __name__ == "__main__":
    main()
