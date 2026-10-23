"""
E9d: what drives the Adam spikes of E9c, and does removing the suspected cause remove them?

Hypothesis (reactivation of dormant coordinates). At T = 0.03 without a copy rule the
Markov chain collapses onto a few tokens; the other 60 tokens appear only as the
uniformly random first token of a sequence. Their embedding rows get gradients only
intermittently, so their second moments decay between visits. When such a row's gradient
returns, Adam's update m/sqrt(v) restarts from a decayed memory and can reach

    A(beta1, beta2) = max_t (1 - beta1^t) / sqrt(1 - beta2^t)

times lr per step (6.6 for beta2 = 0.999, 1.1 for 0.95; snr_grad.adam_reactivation_gain),
far above a typical noise-dominated update.

Result (README, E9d): the hypothesis is rejected. The spikes persist without the rare
tokens and with beta2 = 0.95 on the embedding; they are carried by converged bias
vectors of the first block. A still bounds the overshoot (E9e).

Per step we record, for every parameter tensor, max |delta theta| / lr, and for the token
embedding the steps since the updated row last had a nonzero gradient. Variants:
  baseline    : E9c setting (T = 0.03, copy 0, B = 32, beta2 = 0.999, lr = 0.01)
  stationary  : same, but sequences start inside the chain (8-token burn-in), so the rare
                tokens never occur
  embed_b2    : same as baseline, but the token embedding uses beta2 = 0.95
  betas_B1_B2 : baseline with (beta1, beta2) = (B1, B2) everywhere, e.g. betas_0.9_0.95
                (dose-response of the spike rate in A)

Run: uv run python studies/stability_study/e9_lm_diagnose.py
"""

import argparse
import time

import torch

from common import save
from tinylm import MarkovLanguage, TinyLM, lm_loss
from snr_grad import adam_reactivation_gain, detect_excursions

VARIANTS = ("baseline", "stationary", "embed_b2", "betas_0.9_0.95")


def run(variant, seed, B=32, L=32, d=64, lr=0.01, steps=3000):
    lang = MarkovLanguage(temperature=0.03, copy_prob=0.0)
    burn = 8 if variant == "stationary" else 0

    def sample(n, g):
        return lang.sample(n, L + burn, g)[:, burn:]

    torch.manual_seed(seed)
    model = TinyLM(lang.vocab, d, 2, heads=max(1, d // 32), max_len=L)
    beta1, beta2 = (float(b) for b in variant.split("_")[1:]) if variant.startswith("betas_") else (0.9, 0.999)
    tok = model.tok.weight
    rest = [p for p in model.parameters() if p is not tok]
    opt = torch.optim.AdamW([{"params": [tok], "betas": (beta1, 0.95 if variant == "embed_b2" else beta2)},
                             {"params": rest, "betas": (beta1, beta2)}], lr=lr, weight_decay=0.0)
    names = {p: n for n, p in model.named_parameters()}
    gen = torch.Generator().manual_seed(100 + seed)
    val = sample(32, torch.Generator().manual_seed(12345))
    curve, max_u = [], {n: [] for n in names.values()}
    tok_row, tok_silence, global_u = [], [], []
    last_seen = torch.zeros(lang.vocab, dtype=torch.long)
    for t in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        lm_loss(model, sample(B, gen)).backward()
        seen = tok.grad.abs().sum(1) > 0
        old = {p: p.detach().clone() for p in names}
        opt.step()
        for p, n in names.items():
            u = ((p.detach() - old[p]).abs() / lr)
            max_u[n].append(float(u.max()))
            if p is tok:
                row = int(u.max(1).values.argmax())
                tok_row.append(row)
                tok_silence.append(int(t - 1 - last_seen[row]) if seen[row] else -1)
        global_u.append(max(v[-1] for v in max_u.values()))
        last_seen[seen] = t
        if t % 5 == 0:
            with torch.no_grad():
                curve.append(float(lm_loss(model, val)))
    rep = detect_excursions(curve[100:], window=20, warmup=10, rel_floor=0.25)
    tail = torch.tensor(curve[100:])
    peak_step = 5 * (100 + int(tail.argmax()) + 1)
    lo, hi = max(0, peak_step - 60), peak_step       # steps (1-based) just before the peak
    window = {n: max(v[lo:hi]) for n, v in max_u.items()}
    late = {n: float(torch.tensor(v[500:]).median()) for n, v in max_u.items()}
    top = max(window, key=lambda n: window[n] / late[n])
    tok_name = names[tok]
    k = lo + max(range(hi - lo), key=lambda i: max_u[tok_name][lo + i])
    return {"variant": variant, "seed": seed, "betas": [beta1, beta2],
            "gain_A": adam_reactivation_gain(beta1, beta2), "events": rep.num_events,
            "severity": min(rep.max_excursion, 1e4), "peak_over_median": float(tail.max() / tail.median()),
            "peak_step": peak_step, "final_loss": curve[-1],
            "window_max_u": window, "median_max_u": late, "most_amplified_tensor": top,
            "tok_max_u_in_window": max_u[tok_name][k], "tok_max_u_step": k + 1,
            "tok_row": tok_row[k], "tok_row_silence": tok_silence[k],
            "tok_max_u_run": max(max_u[tok_name][500:]), "curve": curve,
            "tok_max_u": max_u[tok_name], "global_max_u": global_u,
            "global_max_u_run": max(global_u[500:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="e9_lm_diagnose.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    print("reactivation gain A:", {b2: round(adam_reactivation_gain(0.9, b2), 2) for b2 in (0.95, 0.99, 0.999)})
    rows = []
    t0 = time.perf_counter()
    for variant in args.variants:
        for seed in range(args.seeds):
            r = run(variant, seed)
            rows.append(r)
            print(f"{variant:<10} seed={seed}: events={r['events']} sev={r['severity']:6.2f} peak at {r['peak_step']} | "
                  f"most amplified before peak: {r['most_amplified_tensor']} "
                  f"({r['window_max_u'][r['most_amplified_tensor']]:.2f} vs median {r['median_max_u'][r['most_amplified_tensor']]:.2f}) | "
                  f"tok max|u| {r['tok_max_u_in_window']:.2f} at {r['tok_max_u_step']} row {r['tok_row']} "
                  f"silent {r['tok_row_silence']} | run max tok {r['tok_max_u_run']:.2f} all {r['global_max_u_run']:.2f} ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    print("wrote", save(args.out, {"rows": rows}))


if __name__ == "__main__":
    main()
