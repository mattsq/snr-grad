"""Plot the exploratory SNRMuon scale-multiplier sensitivity on shifted digits."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_batch_control import read_rows


def group(path, policy):
    rows = read_rows(path)
    return {seed: [r for r in rows if r.get("kind") is None and
                   r["optimizer"] == "snr_muon" and r["task"] == "digits_shift"
                   and r["policy"] == policy and r["seed"] == seed]
            for seed in range(2, 7)}


def plot(baseline, trials, output):
    fig, (loss_ax, batch_ax) = plt.subplots(1, 2, figsize=(11, 4.5))
    fixed = np.array([v[-1]["validation"] for v in baseline.values()])
    multipliers = sorted(trials)
    for i, value in enumerate(multipliers):
        paths = trials[value]
        final = np.array([v[-1]["validation"] for v in paths.values()])
        mean_batch = np.array([np.mean([r["actual_batch"] for r in v]) for v in paths.values()])
        loss_ax.plot(np.full(5, i), final, "o", color="#c44153", alpha=.45)
        loss_ax.errorbar(i, final.mean(), yerr=final.std(ddof=1), fmt="o",
                         color="#c44153", capsize=4)
        batch_ax.plot(np.full(5, i), mean_batch, "o", color="#5470a6", alpha=.45)
        batch_ax.errorbar(i, mean_batch.mean(), yerr=mean_batch.std(ddof=1),
                          fmt="o", color="#5470a6", capsize=4)
    loss_ax.axhline(fixed.mean(), color="black", ls="--", label="fixed B=4 mean")
    batch_ax.axhline(4, color="black", ls="--", label="fixed B=4")
    loss_ax.set_ylabel("Final validation cross-entropy")
    batch_ax.set_ylabel("Mean actual batch size")
    for ax in (loss_ax, batch_ax):
        ax.set_xticks(range(len(multipliers)), [str(x) for x in multipliers])
        ax.set_xlabel("Scale multiplier")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Corrected Muon sensor: multiplier sensitivity after label shift")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corrected", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/benchmark_batch_muon_multiplier.png"))
    args = parser.parse_args()
    trials = {0.2: group(args.corrected, "aware")}
    for multiplier in (0.05, 0.1, 0.4):
        trials[multiplier] = group(Path(f"benchmarks/benchmark_batch_muon_multiplier_{multiplier}.jsonl.gz"), "aware")
    plot(group(args.corrected, "fixed_small"), trials, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
