"""
E6b: is the full-GGN simulator's weakness at large width the same unbounded-surrogate
problem as unprojected low-rank curvature?

A GGN on N held-out samples with d_out outputs has rank <= N * d_out (2048 here),
below the parameter count at widths 128 / 512. At width-512 checkpoints we compare
the full GGN simulator with and without projection onto its top-64 eigenvectors
(deterministic surrogate minimum and Spearman with real severity).

Run: uv run python studies/stability_study/e6b_full_ggn_projection.py
"""

import copy
import math
from dataclasses import replace

import torch

from common import save
import benchmark_stability_layers as bl
from e6_network_scale import EXC
from snr_grad import (
    GGNCurvature,
    LowRankCurvature,
    QuadraticStabilitySimulator,
    StabilityProbe,
    collect_microbatch_gradients,
    detect_excursions,
)


def main():
    torch.set_num_threads(1)
    width = 512
    cfg = replace(bl.Config(), hidden=width, probe_k=32)
    task = bl.Task(cfg)
    arms = (("sgd", 0.1), ("sgd", 1.0), ("adamw", 3e-3), ("adamw", 3e-2), ("adamw", 0.1),
            ("snr_adamw", 0.1))
    rows = []
    for name, lr in arms:
        model = task.student(0)
        opt = bl.make_opt(name, model.parameters(), lr)
        gen = torch.Generator().manual_seed(1000)
        for _ in range(300):
            opt.zero_grad(set_to_none=True)
            bl.mse(model, task.sample(cfg.batch_size, gen)).backward()
            opt.step()
        seed = 99
        params = [p for p in model.parameters() if p.requires_grad]
        grads = collect_microbatch_gradients(model, bl.mse, task.sample(32 * 16, gen), num_splits=32)
        x_c, y_c = task.x_val[:256], task.y_val[:256]
        ggn = GGNCurvature(model, x_c, lambda o: 0.5 * ((o - y_c) ** 2).sum(1).mean(), params=params)
        base = bl.val_loss(task, model)
        reals = []
        for i in range(8):
            m2 = copy.deepcopy(model)
            o2 = bl.make_opt(name, m2.parameters(), lr)
            o2.load_state_dict(copy.deepcopy(opt.state_dict()))
            reals.append(bl.run_real(task, m2, o2, 100, torch.Generator().manual_seed(seed + i)))
        reps = [detect_excursions(r, **EXC) for r in reals]
        row = {"optimizer": name, "lr": lr,
               "real_severity": sum(min(r.max_excursion, 100.0) for r in reps) / len(reps)}
        probe = StabilityProbe.from_microbatches(grads, cfg.batch_size, opt, None)
        top = LowRankCurvature.from_proxy(ggn, params, rank=64, seed=seed)
        for key, sub in (("full_unprojected", None), ("full_projected64", top.eigenvectors)):
            sim = QuadraticStabilitySimulator.from_checkpoint(
                opt, probe, ggn, model=model, batch_size=cfg.batch_size, loss0=base,
                noise_scaling="loss_proportional", noise_model="resample", subspace=sub)
            sm = sim.run(8, 100, seed=seed + 7, excursion_kwargs=EXC).summary()
            row[f"{key}_severity"] = sm["excursion_severity_mean"]
            row[f"{key}_det_min_over_L0"] = sm["deterministic_min_loss"] / base
            row[f"{key}_below_zero"] = sm["surrogate_below_zero"]
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    for key in ("full_unprojected", "full_projected64"):
        rho = bl.spearman([r[f"{key}_severity"] for r in rows], [r["real_severity"] for r in rows])
        print(f"{key}: spearman={rho:+.3f}")
    print("wrote", save("e6b_full_ggn_projection.json", {"rows": rows}))


if __name__ == "__main__":
    main()
