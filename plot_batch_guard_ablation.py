"""Compare AdamW batch decisions with the CPU probe-time guard on and off.

Uses the paired outputs of benchmark_batch_control.py. Each input contains
AdamW Euclidean and aware policies on digits and digits_shift, seeds 2–6.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_batch_control import read_rows


def runs(path):
    result = {}
    for row in read_rows(path):
        if row.get("kind") is None and row["optimizer"] == "adamw":
            result.setdefault((row["task"], row["policy"], row["seed"]), []).append(row)
    return result


def plot(guarded, permissive, output):
    conditions = (("Euclidean, guarded", guarded, "euclidean", "#7466a7"),
                  ("Aware, guarded", guarded, "aware", "#c44153"),
                  ("Euclidean, permissive", permissive, "euclidean", "#7466a7"),
                  ("Aware, permissive", permissive, "aware", "#c44153"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, task in enumerate(("digits", "digits_shift")):
        loss_ax, batch_ax = axes[i]
        for j, (label, source, policy, color) in enumerate(conditions):
            trials = [source[(task, policy, seed)] for seed in range(2, 7)]
            final = np.array([trial[-1]["validation"] for trial in trials])
            loss_ax.scatter(np.full(5, j) + np.linspace(-.09, .09, 5), final,
                            color=color, alpha=.6, s=20)
            loss_ax.errorbar(j, final.mean(), yerr=final.std(ddof=1), fmt="o",
                             color=color, capsize=4, ms=6)
            grid = np.arange(4, 3001, 4)
            paths = []
            for trial in trials:
                samples = np.array([r["samples"] for r in trial])
                batches = np.array([r["actual_batch"] for r in trial])
                paths.append(batches[np.clip(np.searchsorted(samples, grid, side="left"),
                                             0, len(batches) - 1)])
            paths = np.array(paths)
            batch_ax.plot(grid, paths.mean(axis=0), color=color,
                          ls="--" if "permissive" in label else "-", label=label)
        loss_ax.set_xticks(range(4), [x[0].replace(", ", "\n") for x in conditions], fontsize=8)
        loss_ax.set_ylabel("Final validation cross-entropy")
        loss_ax.set_title(task + ": five paired seeds")
        batch_ax.set_xlabel("Training examples")
        batch_ax.set_ylabel("Mean actual batch size")
        batch_ax.set_title(task + ": action path")
        if task == "digits_shift":
            batch_ax.axvline(1500, color="black", ls=":", lw=1, label="label switch")
        for ax in (loss_ax, batch_ax):
            ax.grid(alpha=.2)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("AdamW: probe-time guard changes the policy and the outcome")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guarded", type=Path)
    parser.add_argument("permissive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/benchmark_batch_guard_ablation.png"))
    args = parser.parse_args()
    plot(runs(args.guarded), runs(args.permissive), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
