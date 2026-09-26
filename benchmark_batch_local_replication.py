"""Replicate paired local batch continuations around a digit-label shift.

This checks whether a 12-step batch-efficiency knee is stable across training
draws and whether the best batch depends on per-step versus per-example gain.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_batch_control import build_model, data, digit_loss, evaluate, optimizer_for, update
from snr_grad import probe_batch


def run_seed(seed, *, repetitions, horizon, sizes):
    train, validation = data(seed, "digits_shift")
    model = build_model(seed, "digits_shift")
    optimizer = optimizer_for(model, "snr_muon", .1, len(train[0]))
    rows = []
    for step in range(1, 563):
        rng = torch.Generator().manual_seed(seed + 20000 + step - 1)
        indices = torch.randint(len(train[0]), (4,), generator=rng)
        target = train[1] if step <= 375 else train[2]
        optimizer.zero_grad(set_to_none=True)
        digit_loss(model, (train[0][indices], target[indices])).backward()
        update(optimizer, 4)
        if step not in (187, 562):
            continue
        phase = "pre-shift" if step == 187 else "post-shift"
        target = train[1] if step == 187 else train[2]
        baseline = evaluate(model, validation, step == 562, digit_loss)
        probe_ids = torch.randperm(len(train[0]), generator=torch.Generator().manual_seed(seed + 50000))[:64]
        probe = probe_batch(model, digit_loss, (train[0][probe_ids], target[probe_ids]),
                            splits=8, optimizer=optimizer)
        state = copy.deepcopy(model.state_dict())
        opt_state = copy.deepcopy(optimizer.state_dict())
        for repeat in range(repetitions):
            # Same draw prefix across candidate batches within each repeat.
            draws = [torch.randint(len(train[0]), (max(sizes),), generator=
                                  torch.Generator().manual_seed(seed * 100000 + step * 100 + repeat * horizon + t))
                     for t in range(horizon)]
            for B in sizes:
                branch = build_model(seed, "digits_shift")
                branch.load_state_dict(state)
                branch_opt = optimizer_for(branch, "snr_muon", .1, len(train[0]))
                branch_opt.load_state_dict(copy.deepcopy(opt_state))
                for indices in draws:
                    branch_opt.zero_grad(set_to_none=True)
                    digit_loss(branch, (train[0][indices[:B]], target[indices[:B]])).backward()
                    update(branch_opt, B)
                gain = baseline - evaluate(branch, validation, step == 562, digit_loss)
                rows.append(dict(seed=seed, phase=phase, checkpoint_step=step,
                                 repeat=repeat, batch=B, horizon=horizon,
                                 gain_per_step=gain / horizon,
                                 gain_per_example=gain / (horizon * B),
                                 euclidean_scale=(probe.euclidean.scale if math.isfinite(probe.euclidean.scale)
                                                  else None),
                                 muon_scale=probe.muon.scale))
    return rows


def plot(rows, output):
    sizes = sorted({r["batch"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, phase in enumerate(("pre-shift", "post-shift")):
        relevant = [r for r in rows if r["phase"] == phase]
        for j, (metric, scaling, title) in enumerate((("gain_per_step", 1, "per optimizer step"),
                                                      ("gain_per_example", sizes[0], "per training example (×4)"))):
            ax = axes[i, j]
            for seed in sorted({r["seed"] for r in relevant}):
                seed_rows = [r for r in relevant if r["seed"] == seed]
                mean = [np.mean([r[metric] * scaling for r in seed_rows if r["batch"] == B]) for B in sizes]
                ax.plot(sizes, mean, color="#888888", lw=.8, alpha=.45)
            # Aggregate at seed level; uncertainty across five independent splits.
            matrix = np.array([[np.mean([r[metric] * scaling for r in relevant
                                         if r["batch"] == B and r["seed"] == seed])
                                for B in sizes] for seed in sorted({r["seed"] for r in relevant})])
            ax.plot(sizes, matrix.mean(0), "o-", color="#c44153", lw=2, label="mean across seeds")
            ax.fill_between(sizes, matrix.mean(0) - matrix.std(0), matrix.mean(0) + matrix.std(0),
                            color="#c44153", alpha=.15, label="seed SD")
            ax.axhline(0, color="black", lw=.7)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sizes, labels=[str(B) for B in sizes])
            ax.set_xlabel("Candidate batch size")
            ax.set_ylabel("Validation CE reduction " + title)
            ax.set_title(phase + " | " + title)
            ax.grid(alpha=.2)
            if i == j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Replicated 12-step local batch efficiency: same checkpoint, paired draws")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--output", type=Path, default=Path("benchmarks/benchmark_batch_local_replication.json"))
    parser.add_argument("--figure", type=Path, default=Path("benchmarks/benchmark_batch_local_replication.png"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for seed in args.seeds for row in run_seed(seed, repetitions=args.repetitions,
                                                           horizon=args.horizon, sizes=args.sizes)]
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf8")
    plot(rows, args.figure)
    print(args.figure)


if __name__ == "__main__":
    main()
