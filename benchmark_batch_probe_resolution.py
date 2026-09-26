"""At fixed parameters and examples, show the effect of split count on GNS sensors.

Run: python benchmark_batch_probe_resolution.py
The highest split count is a finite-sample reference, not population truth.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_batch_control import build_model, data, digit_loss, optimizer_for, update
from snr_grad import probe_batch


def measure(seed, *, batch_size=128, train_steps=150, lr=.1):
    train, _ = data(seed, "digits")
    model = build_model(seed, "digits")
    optimizer = optimizer_for(model, "snr_muon", lr, len(train[0]))
    for step in range(train_steps):
        rng = torch.Generator().manual_seed(seed + 60000 + step)
        idx = torch.randint(len(train[0]), (16,), generator=rng)
        optimizer.zero_grad(set_to_none=True)
        digit_loss(model, (train[0][idx], train[1][idx])).backward()
        update(optimizer, 16)
    idx = torch.randperm(len(train[0]), generator=torch.Generator().manual_seed(seed + 70000))[:batch_size]
    batch = train[0][idx], train[1][idx]
    results = []
    for splits in (4, 8, 16, 32, 64, 128):
        if batch_size % splits:
            continue
        probe = probe_batch(model, digit_loss, batch, splits=splits, optimizer=optimizer)
        def finite(value):
            return value if math.isfinite(value) else None

        results.append(dict(seed=seed, splits=splits, examples=batch_size,
                            seconds=probe.elapsed_seconds, euclidean=finite(probe.euclidean.scale),
                            adamw=finite(probe.adamw.scale),
                            muon=finite(probe.muon.scale)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--output", type=Path, default=Path("benchmarks/benchmark_batch_probe_resolution.json"))
    parser.add_argument("--figure", type=Path, default=Path("benchmarks/benchmark_batch_probe_resolution.png"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [item for seed in args.seeds for item in measure(seed)]
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf8")
    splits = sorted({r["splits"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, color in (("euclidean", "#9467bd"), ("adamw", "#5470a6"),
                        ("muon", "#c44153")):
        matrix = np.array([[r[name] if r[name] is not None else np.nan
                            for r in rows if r["seed"] == seed and r["splits"] in splits]
                           for seed in args.seeds])
        axes[0].plot(splits, np.nanmedian(matrix, axis=0), "o-", label=name, color=color)
        axes[0].fill_between(splits, np.nanquantile(matrix, .25, axis=0),
                             np.nanquantile(matrix, .75, axis=0), color=color, alpha=.15)
    costs = np.array([[r["seconds"] for r in rows if r["seed"] == seed] for seed in args.seeds])
    axes[1].plot(splits, np.median(costs, axis=0) * 1000, "o-", color="#666666")
    axes[1].fill_between(splits, np.quantile(costs, .25, axis=0) * 1000,
                         np.quantile(costs, .75, axis=0) * 1000, color="#666666", alpha=.15)
    axes[0].set_ylabel("Raw noise scale (examples)")
    axes[1].set_ylabel("Probe time (ms)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(splits, labels=[str(x) for x in splits])
        ax.set_xlabel("Disjoint microbatches from the same 128 examples")
        ax.grid(alpha=.2)
    axes[0].legend()
    unresolved = sum(r["euclidean"] is None for r in rows)
    axes[0].text(.02, .02, f"{unresolved}/{len(rows)} Euclidean probes unresolved (held)",
                 transform=axes[0].transAxes, fontsize=8)
    fig.suptitle("Sensor resolution at a fixed digits checkpoint (median and interquartile range)")
    fig.tight_layout()
    fig.savefig(args.figure, dpi=150)
    plt.close(fig)
    print(args.figure)


if __name__ == "__main__":
    main()
