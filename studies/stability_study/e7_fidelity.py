"""
E7b: simulator fidelity on a small Transformer LM at / beyond the cost crossover.

For one LM configuration (``--d --layers --L --B``), AdamW runs over a learning-rate
grid are checkpointed. At each checkpoint:

  real   S=8 continuations of T=100 steps with fresh data; held-out loss every step;
         severity = mean capped max excursion (shared definition), event fraction;
  sim    probe (K=32 microbatches of B/4 sequences) + rank-16 GGN (held-out batch),
         resampled loss-proportional noise, S=8 x T=100 simulated steps;
  base   lr * lambda_max(P^1/2 H P^1/2) on the low-rank GGN (sharpness), the one-step
         noise penalty with the same curvature, and optimizer-aware GNS.

What-if queries: at the last checkpoint of seed 0, the *same* simulator build is
asked about lr x {0.5, 2, 4}, and real continuations at those learning rates are run
for comparison (the question a batch / lr controller would ask).

Wall-clock seconds of every component are recorded, so fidelity and cost come from
the same runs.

Run: uv run python studies/stability_study/e7_fidelity.py --d 64 --layers 2 --L 32 --B 8
"""

import argparse
import copy
import math
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
    detect_excursions,
    frozen_update_map,
)

S, T, K, RANK = 8, 100, 32, 16
EXC = {"window": 20, "warmup": 10}
WHATIF = (0.5, 2.0, 4.0)


def make_model(lang, a, seed):
    torch.manual_seed(seed)
    return TinyLM(lang.vocab, a.d, a.layers, heads=max(1, a.d // 32), max_len=a.L)


def real_continuations(lang, model, opt, lr, a, val, seed):
    t0 = time.perf_counter()
    reps = []
    for i in range(S):
        m2 = copy.deepcopy(model)
        o2 = torch.optim.AdamW(m2.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
        o2.load_state_dict(copy.deepcopy(opt.state_dict()))
        for g in o2.param_groups:
            g["lr"] = lr
        gen = torch.Generator().manual_seed(seed + i)
        losses = []
        with torch.no_grad():
            losses.append(float(lm_loss(m2, val)))
        for _ in range(T):
            o2.zero_grad(set_to_none=True)
            lm_loss(m2, lang.sample(a.B, a.L, gen)).backward()
            o2.step()
            with torch.no_grad():
                losses.append(float(lm_loss(m2, val)))
            if not math.isfinite(losses[-1]):
                losses += [float("inf")] * (T + 1 - len(losses))
                break
        reps.append(detect_excursions(losses, **EXC))
    return {
        "severity": sum(min(r.max_excursion, 100.0) for r in reps) / S,
        "event_fraction": sum(r.spiked for r in reps) / S,
        "seconds": time.perf_counter() - t0,
    }


def build_simulator(lang, model, opt, lr, a, val, gen, rank=RANK):
    t0 = time.perf_counter()
    b = max(1, a.B // 4)
    grads = collect_microbatch_gradients(model, lm_loss, lang.sample(K * b, a.L, gen), num_splits=K)
    t_probe = time.perf_counter() - t0
    params = [p for p in model.parameters() if p.requires_grad]
    model.eval()
    ggn = GGNCurvature(model, val[:, :-1], output_loss_fn(val), params=params)
    t0 = time.perf_counter()
    low = LowRankCurvature.from_proxy(ggn, params, rank=rank, oversample=8, power_iters=1,
                                      trace_probes=8)
    t_build = time.perf_counter() - t0
    model.train()
    probe = StabilityProbe.from_microbatches(grads, b, opt, low)
    with torch.no_grad():
        loss0 = float(lm_loss(model, val))
    sim = QuadraticStabilitySimulator.from_checkpoint(
        opt, probe, low, model=model, batch_size=a.B, loss0=loss0,
        noise_model="resample", noise_scaling="loss_proportional")
    return sim, probe, low, t_probe + t_build


def baselines(opt, probe, low, lr, a):
    m = probe.measure(lr=lr, batch_size=a.B)
    pmap = frozen_update_map(opt)
    out = {"noise_penalty": m.noise_penalty, "gns_optimizer": m.gns_optimizer,
           "risk_trace": m.risk_trace_per_example}
    if pmap.supported:
        ones = pmap({p: torch.ones_like(p) for p in probe.params})
        pdiag = torch.cat([ones[p].reshape(-1) for p in probe.params]).clamp_min(0).sqrt()
        w = pdiag[:, None] * low.eigenvectors * low.eigenvalues.clamp_min(0).sqrt()[None, :]
        out["lr_sharpness"] = lr * float(torch.linalg.eigvalsh(w.t() @ w).max())
    return out


def simulate(sim, lr, seed):
    t0 = time.perf_counter()
    sm = sim.run(S, T, seed=seed, lr=lr, excursion_kwargs=EXC).summary()
    return {"severity": sm["excursion_severity_mean"], "event_fraction": sm["spike_fraction"],
            "seconds": time.perf_counter() - t0, "below_zero": sm["surrogate_below_zero"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--lrs", type=str, default="0.001,0.003,0.01,0.03,0.1")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--ckpts", type=str, default="150,400")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-whatif", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    lang = MarkovLanguage()
    vgen = torch.Generator().manual_seed(12345)
    val = lang.sample(16, a.L, vgen)
    ckpts = tuple(int(x) for x in a.ckpts.split(","))
    lrs = tuple(float(x) for x in a.lrs.split(","))
    rows, whatif, train_curves = [], [], {}
    t_start = time.perf_counter()
    for lr in lrs:
        for seed in range(a.seeds):
            model = make_model(lang, a, seed)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
            gen = torch.Generator().manual_seed(100 + seed)
            curve = []
            for t in range(1, a.steps + 1):
                opt.zero_grad(set_to_none=True)
                loss = lm_loss(model, lang.sample(a.B, a.L, gen))
                loss.backward()
                opt.step()
                curve.append(float(loss.detach()))
                if not math.isfinite(curve[-1]):
                    break
                if t in ckpts:
                    cs = 7000 + 31 * seed + t
                    real = real_continuations(lang, model, opt, lr, a, val, cs)
                    sim, probe, low, t_build = build_simulator(lang, model, opt, lr, a, val,
                                                               torch.Generator().manual_seed(cs + 1))
                    simr = simulate(sim, None, cs + 2)
                    row = {"lr": lr, "seed": seed, "step": t, "real_severity": real["severity"],
                           "real_event_fraction": real["event_fraction"], "real_seconds": real["seconds"],
                           "sim_severity": simr["severity"], "sim_event_fraction": simr["event_fraction"],
                           "sim_seconds": simr["seconds"], "build_seconds": t_build,
                           "sim_below_zero": simr["below_zero"],
                           "ggn_top_eig": float(low.eigenvalues[0]), **baselines(opt, probe, low, lr, a)}
                    rows.append(row)
                    print(f"lr={lr:<6g} seed={seed} step={t:>4}: real sev={real['severity']:7.3f} "
                          f"evt={real['event_fraction']:.2f} | sim sev={simr['severity']:7.3f} "
                          f"evt={simr['event_fraction']:.2f} | real {real['seconds']:6.1f}s vs "
                          f"sim {t_build + simr['seconds']:6.1f}s (build {t_build:.1f}s) "
                          f"[{time.perf_counter() - t_start:.0f}s]", flush=True)
                    if not a.no_whatif and seed == 0 and t == ckpts[-1]:
                        for f in WHATIF:
                            rq = real_continuations(lang, model, opt, lr * f, a, val, cs + 50)
                            sq = simulate(sim, lr * f, cs + 51)
                            whatif.append({"base_lr": lr, "factor": f, "lr": lr * f,
                                           "real_severity": rq["severity"],
                                           "real_event_fraction": rq["event_fraction"],
                                           "real_seconds": rq["seconds"],
                                           "sim_severity": sq["severity"],
                                           "sim_event_fraction": sq["event_fraction"],
                                           "sim_seconds": sq["seconds"]})
                            print(f"   what-if lr x{f}: real sev={rq['severity']:7.3f} | "
                                  f"sim sev={sq['severity']:7.3f}  ({rq['seconds']:.1f}s vs "
                                  f"{sq['seconds']:.1f}s)", flush=True)
            train_curves[f"{lr}_{seed}"] = curve

    from benchmark_stability_plrf import spearman  # noqa: E402

    def fin(v):
        return isinstance(v, (int, float)) and math.isfinite(v)

    scores = {}
    for key in ("sim_severity", "lr_sharpness", "noise_penalty", "gns_optimizer", "risk_trace"):
        rs = [r for r in rows if fin(r.get(key))]
        scores[key] = spearman([r[key] for r in rs], [r["real_severity"] for r in rs]) if len(rs) > 2 else None
    calib = {
        "event_fraction_mae": sum(abs(r["sim_event_fraction"] - r["real_event_fraction"]) for r in rows) / len(rows),
        "event_fraction_bias": sum(r["sim_event_fraction"] - r["real_event_fraction"] for r in rows) / len(rows),
    }
    wi = {}
    if whatif:
        wi["spearman_severity"] = spearman([w["sim_severity"] for w in whatif],
                                           [w["real_severity"] for w in whatif])
        wi["event_fraction_mae"] = sum(abs(w["sim_event_fraction"] - w["real_event_fraction"])
                                       for w in whatif) / len(whatif)
    cost = {
        "real_seconds_per_query_median": float(torch.tensor([r["real_seconds"] for r in rows]).median()),
        "sim_seconds_per_query_median": float(torch.tensor([r["sim_seconds"] for r in rows]).median()),
        "build_seconds_median": float(torch.tensor([r["build_seconds"] for r in rows]).median()),
    }
    cost["speedup_1_query"] = cost["real_seconds_per_query_median"] / (
        cost["build_seconds_median"] + cost["sim_seconds_per_query_median"])
    cost["speedup_4_queries"] = 4 * cost["real_seconds_per_query_median"] / (
        cost["build_seconds_median"] + 4 * cost["sim_seconds_per_query_median"])
    print("\nSpearman with real severity:", {k: (round(v, 3) if v is not None else None) for k, v in scores.items()})
    print("Event-fraction calibration:", {k: round(v, 3) for k, v in calib.items()})
    print("What-if queries:", {k: round(v, 3) for k, v in wi.items()})
    print("Cost:", {k: round(v, 2) for k, v in cost.items()})
    tag = f"d{a.d}_l{a.layers}_L{a.L}_B{a.B}"
    path = save(f"e7_fidelity_{tag}.json", {"config": vars(a), "rows": rows, "whatif": whatif,
                                           "scores": scores, "calibration": calib,
                                           "whatif_scores": wi, "cost": cost,
                                           "train_curves": train_curves,
                                           "seconds": time.perf_counter() - t_start})
    print("wrote", path)


if __name__ == "__main__":
    main()
