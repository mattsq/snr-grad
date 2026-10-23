"""
E8c-2: is the noise / sharpest-direction misalignment caused by Adam's whitening?

At LM checkpoints, compute the window width w (noise share along the sharpest
direction) and D_hat for the raw GGN (P = I, SGD geometry) and for Adam's actual
preconditioner, using the same probe.

Run: uv run python studies/stability_study/e8_alignment.py
"""

import torch

from common import save
from tinylm import MarkovLanguage, TinyLM, lm_loss, output_loss_fn
from snr_grad import (
    GGNCurvature,
    LowRankCurvature,
    PreconditionedCurvature,
    collect_microbatch_gradients,
)


def geometry(model, grads, prec, val, b, K=32, rank=32):
    params = [p for p in model.parameters() if p.requires_grad]
    model.eval()
    ggn = GGNCurvature(model, val[:, :-1], output_loss_fn(val), params=params)
    proxy = ggn if prec is None else PreconditionedCurvature(ggn, prec)
    low = LowRankCurvature.from_proxy(proxy, params, rank=rank, oversample=8, power_iters=1)
    model.train()
    root = torch.cat([(prec[p].sqrt() if prec is not None else torch.ones_like(p)).reshape(-1) for p in params])
    dev = torch.stack([torch.cat([g[p].reshape(-1) for p in params]) for g in grads])
    dev = (dev - dev.mean(0)) * root
    sig = b * (dev @ low.eigenvectors).pow(2).sum(0) / (K - 1)
    tot = b * float(dev.pow(2).sum()) / (K - 1)
    top = int(torch.argmax(low.eigenvalues))
    share_sharp = float(sig[top]) / tot
    share_top32 = float(sig.sum()) / tot
    # How sharp are the noise-heaviest directions, relative to the sharpest one?
    noisiest = int(torch.argmax(sig))
    return {"w": share_sharp, "noise_in_top32": share_top32,
            "h_of_noisiest_over_hmax": float(low.eigenvalues[noisiest] / low.eigenvalues[top])}


def main():
    torch.set_num_threads(4)
    rows = []
    for (vocab, zipf, d, L, B) in ((64, 0.0, 64, 64, 8), (64, 1.5, 128, 16, 4), (64, 1.5, 256, 16, 2)):
        lang = MarkovLanguage(vocab=vocab, global_zipf=zipf)
        torch.manual_seed(0)
        model = TinyLM(lang.vocab, d, 2, heads=max(1, d // 32), max_len=L)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0.0)
        gen = torch.Generator().manual_seed(100)
        val = lang.sample(16, L, torch.Generator().manual_seed(12345))
        for t in range(1, 601):
            opt.zero_grad(set_to_none=True)
            lm_loss(model, lang.sample(B, L, gen)).backward()
            opt.step()
            if t not in (300, 600):
                continue
            b = max(1, B // 4)
            grads = collect_microbatch_gradients(model, lm_loss, lang.sample(32 * b, L, gen), num_splits=32)
            prec = {}
            for p in model.parameters():
                st = opt.state[p]
                prec[p] = 1.0 / ((st["exp_avg_sq"] / (1 - 0.999 ** float(st["step"]))).sqrt() + 1e-8)
            for name, P in (("sgd_geometry", None), ("adam_geometry", prec)):
                g = geometry(model, grads, P, val, b)
                row = {"vocab": vocab, "zipf": zipf, "d": d, "L": L, "B": B, "step": t, "geometry": name, **g}
                rows.append(row)
                print(f"vocab={vocab} zipf={zipf} d={d} L={L} B={B} step={t} {name:<13}: w={g['w']:.4f} "
                      f"noise-in-top32={g['noise_in_top32']:.3f} h(noisiest)/h_max={g['h_of_noisiest_over_hmax']:.3f}",
                      flush=True)
    print("wrote", save("e8_alignment.json", {"rows": rows}))


if __name__ == "__main__":
    main()
