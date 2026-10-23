"""
E6: simulator fidelity and cost vs network width, with low-rank curvature.

Teacher-student MLP regression (benchmark_stability_layers task) at hidden widths
{32, 128, 512}. At checkpoints of runs with several optimizers / learning rates:

  * real continuations (fresh data, real network and optimizer) -> severity;
  * simulator with curvature = full GGN, low-rank GGN (rank 4 / 16 / 64, with an
    isotropic remainder), activation proxy; noise = resampled probe deviations
    (loss-proportional). Gaussian noise with the full GGN is kept as a reference.

Reports per width: Spearman(sim severity, real severity), and median seconds of
the simulator (including the low-rank build) vs the real continuations, and of
one real training step.

Run: uv run python studies/stability_study/e6_network_scale.py [--quick]
"""

import argparse
import copy
import math
import time
from dataclasses import replace

import torch

from common import ROOT, save  # noqa: F401  (sets sys.path)
import benchmark_stability_layers as bl
from snr_grad import (
    ActivationCurvature,
    GGNCurvature,
    LowRankCurvature,
    QuadraticStabilitySimulator,
    StabilityProbe,
    collect_microbatch_gradients,
    detect_excursions,
)

EXC = {"window": 20, "warmup": 10}


def analyse(task, model, opt, name, lr, cfg, seed, ranks, n_real, n_sim, horizon):
    out = {}
    gen = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    batch = task.sample(cfg.probe_k * cfg.batch_size, gen)
    grads = collect_microbatch_gradients(model, bl.mse, batch, num_splits=cfg.probe_k)
    x_c, y_c = task.x_val[:256], task.y_val[:256]
    out_loss = lambda o: 0.5 * ((o - y_c) ** 2).sum(1).mean()  # noqa: E731
    base = bl.val_loss(task, model)

    t0 = time.perf_counter()
    reals = []
    for i in range(n_real):
        m2 = copy.deepcopy(model)
        o2 = bl.make_opt(name, m2.parameters(), lr)
        o2.load_state_dict(copy.deepcopy(opt.state_dict()))
        reals.append(bl.run_real(task, m2, o2, horizon, torch.Generator().manual_seed(seed + 100 + i)))
    out["real_seconds"] = time.perf_counter() - t0
    reps = [detect_excursions(r, **EXC) for r in reals]
    out["real_severity"] = sum(min(r.max_excursion, 100.0) for r in reps) / len(reps)
    out["real_event_fraction"] = sum(r.spiked for r in reps) / len(reps)

    ggn = GGNCurvature(model, x_c, out_loss, params=params)
    curvs = {"ggn": (ggn, 0.0)}
    for r in ranks:
        t0 = time.perf_counter()
        low = LowRankCurvature.from_proxy(ggn, params, rank=r, trace_probes=16, seed=seed)
        curvs[f"ggn_r{r}"] = (low, time.perf_counter() - t0)
        out[f"ggn_r{r}_top_eig"] = float(low.eigenvalues[0])
        out[f"ggn_r{r}_residual"] = low.residual
    t0 = time.perf_counter()
    act = ActivationCurvature.from_model(model, lambda m: m(x_c))
    curvs["act"] = (act, time.perf_counter() - t0)

    probe = StabilityProbe.from_microbatches(grads, cfg.batch_size, opt, None)
    jobs = [(k, c, b, "resample") for k, (c, b) in curvs.items()] + [("ggn_gauss", ggn, 0.0, "gaussian")]
    for key, curv, build, model_kind in jobs:
        t0 = time.perf_counter()
        sim = QuadraticStabilitySimulator.from_checkpoint(
            opt, probe, curv, model=model, batch_size=cfg.batch_size, loss0=base,
            noise_scaling="loss_proportional", noise_model=model_kind,
        )
        dist = sim.run(n_sim, horizon, seed=seed + 7, excursion_kwargs=EXC)
        sm = dist.summary()
        out[f"sim_{key}_severity"] = sm["excursion_severity_mean"]
        out[f"sim_{key}_event_fraction"] = sm["spike_fraction"]
        out[f"sim_{key}_seconds"] = time.perf_counter() - t0 + build
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    widths, ranks = (32, 128, 512), (4, 16, 64)
    arms = (("sgd", 0.1), ("sgd", 1.0), ("adamw", 3e-3), ("adamw", 3e-2), ("adamw", 0.1),
            ("snr_adamw", 0.1))
    seeds, steps, ckpts, n_real, n_sim, horizon = 2, 1000, (300, 700), 8, 8, 100
    if args.quick:
        widths, ranks, arms = (32,), (4,), (("sgd", 1.0), ("adamw", 0.1))
        seeds, steps, ckpts, n_real, n_sim, horizon = 1, 200, (100,), 3, 3, 30
    t_start = time.perf_counter()
    rows = []
    for width in widths:
        cfg = replace(bl.Config(), hidden=width, probe_k=32)
        task = bl.Task(cfg)
        for name, lr in arms:
            for seed in range(seeds):
                model = task.student(seed)
                opt = bl.make_opt(name, model.parameters(), lr)
                gen = torch.Generator().manual_seed(1000 + seed)
                t_step = []
                for t in range(1, steps + 1):
                    ts = time.perf_counter()
                    opt.zero_grad(set_to_none=True)
                    bl.mse(model, task.sample(cfg.batch_size, gen)).backward()
                    opt.step()
                    t_step.append(time.perf_counter() - ts)
                    if not math.isfinite(bl.val_loss(task, model)):
                        break
                    if t in ckpts:
                        res = analyse(task, model, opt, name, lr, cfg, 50_000 + 31 * seed + t,
                                      ranks, n_real, n_sim, horizon)
                        rows.append({"width": width, "optimizer": name, "lr": lr, "seed": seed,
                                     "step": t, "train_step_seconds": sorted(t_step)[len(t_step) // 2],
                                     **res})
                print(f"width={width} {name} lr={lr} seed={seed}: {len(rows)} checkpoints "
                      f"({time.perf_counter() - t_start:.0f}s)", flush=True)

    keys = ["ggn", "ggn_gauss", *[f"ggn_r{r}" for r in ranks], "act"]
    summary = {}
    for width in widths:
        rs = [r for r in rows if r["width"] == width]
        if len(rs) < 3:
            continue
        summ = {"n": len(rs),
                "real_seconds": float(torch.tensor([r["real_seconds"] for r in rs]).median()),
                "train_step_seconds": float(torch.tensor([r["train_step_seconds"] for r in rs]).median()),
                "n_params": sum(p.numel() for p in bl.Task(replace(bl.Config(), hidden=width)).student(0).parameters())}
        for k in keys:
            summ[f"{k}_spearman"] = bl.spearman([r[f"sim_{k}_severity"] for r in rs],
                                                [r["real_severity"] for r in rs])
            summ[f"{k}_seconds"] = float(torch.tensor([r[f"sim_{k}_seconds"] for r in rs]).median())
        summary[width] = summ
        print(f"\n== width {width} ({summ['n_params']} params, n={summ['n']}): real continuations "
              f"{summ['real_seconds']:.2f}s, train step {summ['train_step_seconds'] * 1e3:.2f}ms")
        for k in keys:
            print(f"   sim[{k:<9}] spearman={summ[f'{k}_spearman']:+.3f}  seconds={summ[f'{k}_seconds']:.2f}")
    path = save("e6_network_scale" + ("_quick" if args.quick else "") + ".json",
                {"rows": rows, "summary": summary, "seconds": time.perf_counter() - t_start})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
