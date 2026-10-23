"""
E7a: where does the simulator become cheaper than real continuations?

For a small decoder LM (tinylm.py) across width, depth, sequence length and batch
size, time:

  real  per continuation step: training step (fwd + bwd + AdamW) and a held-out
        loss evaluation (needed to detect excursions in the real run);
  sim   build: probe (K=32 microbatches of B/4 sequences) + LowRankCurvature
        (rank 16, 1 power iteration, 8 trace probes) on the GGN of 16 held-out
        sequences; per simulated step (fused low-rank path, resampled noise).

A "query" is the question "how spiky are the next T=100 steps under this
configuration?", answered with S=8 continuations. Real cost: S*T*(step + eval).
Simulator cost: build + (S+1)*T*sim_step (one extra noiseless path). The build is
paid once per checkpoint, so Q what-if queries (e.g. candidate learning rates) cost
build + Q*(S+1)*T*sim_step versus Q*S*T*(step + eval).

Run: uv run python studies/stability_study/e7_crossover_timing.py [--quick] [--threads 4]
"""

import argparse
import itertools
import time

import torch

from common import save
from tinylm import MarkovLanguage, TinyLM, lm_loss, output_loss_fn
from snr_grad import (
    GGNCurvature,
    LowRankCurvature,
    QuadraticStabilitySimulator,
    StabilityProbe,
    collect_microbatch_gradients,
)

S, T, K, RANK = 8, 100, 32, 16


def timed(fn, reps):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def measure(lang, d, layers, L, B, gen):
    model = TinyLM(lang.vocab, d, layers, heads=max(1, d // 32), max_len=L)
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = [lang.sample(B, L, gen) for _ in range(4)]
    val = lang.sample(16, L, gen)
    it = itertools.cycle(batches)

    def step():
        opt.zero_grad(set_to_none=True)
        lm_loss(model, next(it)).backward()
        opt.step()

    def evaluate():
        with torch.no_grad():
            lm_loss(model, val)

    t_step = timed(step, 4)
    t_eval = timed(evaluate, 4)

    b = max(1, B // 4)
    probe_batch = lang.sample(K * b, L, gen)
    t0 = time.perf_counter()
    grads = collect_microbatch_gradients(model, lm_loss, probe_batch, num_splits=K)
    t_probe = time.perf_counter() - t0
    params = [p for p in model.parameters() if p.requires_grad]
    model.eval()
    ggn = GGNCurvature(model, val[:, :-1], output_loss_fn(val), params=params)
    t0 = time.perf_counter()
    low = LowRankCurvature.from_proxy(ggn, params, rank=RANK, oversample=8, power_iters=1,
                                      trace_probes=8)
    t_build = time.perf_counter() - t0
    model.train()
    probe = StabilityProbe.from_microbatches(grads, b, opt, None)
    sim = QuadraticStabilitySimulator.from_checkpoint(
        opt, probe, low, model=model, batch_size=B, loss0=float(lm_loss(model, val).detach()),
        noise_model="resample", noise_scaling="loss_proportional")
    assert sim._fused_low_rank(torch.Generator())
    t0 = time.perf_counter()
    sim.run(2, 20)
    t_sim_step = (time.perf_counter() - t0) / (3 * 20)

    real_query = S * T * (t_step + t_eval)
    sim_query = (S + 1) * T * t_sim_step
    build = t_probe + t_build
    row = {
        "d": d, "layers": layers, "L": L, "B": B, "params": n, "tokens_per_step": B * L,
        "t_step": t_step, "t_eval": t_eval, "t_probe": t_probe, "t_build_lowrank": t_build,
        "build_matvecs": low.metadata.cost["build_matvecs"], "t_sim_step": t_sim_step,
        "real_query_s": real_query, "real_query_noeval_s": S * T * t_step,
        "sim_query_s": build + sim_query, "sim_query_marginal_s": sim_query,
        "speedup_1_query": real_query / (build + sim_query),
        "speedup_1_query_noeval": S * T * t_step / (build + sim_query),
        "speedup_per_step": (t_step + t_eval) / t_sim_step,
        "queries_to_break_even": (build / (real_query - sim_query)) if real_query > sim_query else None,
    }
    for q in (5, 20):
        row[f"speedup_{q}_queries"] = q * real_query / (build + q * sim_query)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(0)
    lang = MarkovLanguage()
    grid = list(itertools.product((32, 64, 128, 256), (2, 4), (8, 32, 128, 256), (2, 8, 32)))
    if args.quick:
        grid = [(64, 2, 32, 8), (128, 2, 128, 32)]
    rows = []
    t_start = time.perf_counter()
    for d, layers, L, B in grid:
        row = measure(lang, d, layers, L, B, gen)
        rows.append(row)
        print(f"d={d:>3} layers={layers} L={L:>3} B={B:>2} params={row['params'] / 1e6:5.2f}M "
              f"tokens={row['tokens_per_step']:>5}  step={row['t_step'] * 1e3:7.1f}ms "
              f"eval={row['t_eval'] * 1e3:6.1f}ms sim_step={row['t_sim_step'] * 1e3:6.2f}ms "
              f"build={row['t_probe'] + row['t_build_lowrank']:6.1f}s  "
              f"speedup: 1q={row['speedup_1_query']:5.2f} (no eval {row['speedup_1_query_noeval']:5.2f}) "
              f"5q={row['speedup_5_queries']:5.2f} per-step={row['speedup_per_step']:6.1f} "
              f"({time.perf_counter() - t_start:.0f}s)", flush=True)
    path = save("e7_crossover_timing" + ("_quick" if args.quick else "") + ".json",
                {"rows": rows, "S": S, "T": T, "K": K, "rank": RANK, "threads": args.threads})
    print("wrote", path)


if __name__ == "__main__":
    main()
