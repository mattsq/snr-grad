"""Shared helpers for the stability study (reuses benchmark_stability_plrf)."""

import json
import math
import os
import sys
from dataclasses import replace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import benchmark_stability_plrf as bp  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")

# Measured stability edges at the benchmark defaults (D=256, V=4096, B=8, beta=0.7),
# from benchmarks/stability_plrf_results.json, so experiments share one lr scale.
EDGES = {}


def load_edges():
    if not EDGES:
        path = os.path.join(ROOT, "benchmarks", "stability_plrf_results.json")
        for r in json.load(open(path))["stage0"]:
            EDGES[(r["optimizer"], r["alpha"])] = r["edge_lr"]
    return EDGES


def checkpoint_rows(prob, cfg, name, lr, *, tag, seeds=8, steps=3000,
                    ckpt_fracs=(0.25, 0.5), ckpt_seeds=4, data_seed=1000,
                    sensors=True):
    """Train, then compute real continuations (+ sensors) at checkpoints."""
    ckpts = tuple(int(f * steps) for f in ckpt_fracs)
    losses, snaps = bp.train(prob, name, lr, seeds, steps, checkpoints=ckpts,
                             data_seed=data_seed)
    rows = []
    for t, (theta_stack, opt) in snaps.items():
        for s in range(min(ckpt_seeds, seeds)):
            if not bool(torch.isfinite(theta_stack[s]).all()):
                continue
            seed = 10_000 + 97 * s + t
            real = bp.real_continuations(prob, name, opt, theta_stack, s, lr, cfg, seed)
            row = {**tag, "optimizer": name, "lr": lr, "step": t, "seed": s, **real}
            if sensors:
                row.update(bp.checkpoint_sensors(prob, name, opt, theta_stack, s, lr, cfg,
                                                 seed + 5))
            rows.append(row)
    reps = [bp.detect_excursions(losses[s], window=cfg.window, k_mad=cfg.k_mad,
                                 rel_floor=cfg.rel_floor,
                                 divergence_factor=cfg.divergence_factor)
            for s in range(seeds)]
    summary = {**tag, "optimizer": name, "lr": lr,
               "spike_fraction": sum(r.spiked and not r.diverged for r in reps) / seeds,
               "divergence_fraction": sum(r.diverged for r in reps) / seeds,
               "mean_events": sum(r.num_events for r in reps) / seeds,
               "final_excess_loss_median": float((losses[:, -1] - prob.loss_star).median())}
    return rows, summary, losses


def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    return path
