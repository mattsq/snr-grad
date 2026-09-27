"""Plot held-out batch-choice regret and predicted sizes from paired continuations."""

import argparse
import gzip
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(records, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = {"constant": "#777777", "euclidean": "#6385ba", "aware": "#c44153"}
    for row, optimizer in enumerate(("snr_muon", "adamw")):
        for col, objective in enumerate(("example", "time")):
            ax = axes[row, col]
            for k, sensor in enumerate(colors):
                for phase, shifted in enumerate((False, True)):
                    subset = [p for p in records if p["optimizer"] == optimizer and
                              p["objective"] == objective and p["sensor"] == sensor and
                              p["shifted"] == shifted]
                    # Average source checkpoints within a seed before showing
                    # dispersion across independent held-out digit splits.
                    seeds = sorted({p["seed"] for p in subset})
                    by_seed = [np.mean([p["regret"] for p in subset if p["seed"] == seed])
                               for seed in seeds]
                    x = phase * 4 + k
                    ax.bar(x, np.mean(by_seed), color=colors[sensor], width=.8,
                           label=sensor if phase == 0 else None)
                    ax.errorbar(x, np.mean(by_seed), yerr=np.std(by_seed), fmt="none",
                                ecolor="black", capsize=2, lw=.8)
            ax.set_xticks([1, 5], ["before shift", "after shift"])
            ax.set_ylabel("Regret to local no-probe oracle")
            ax.set_title(f"{optimizer} | gain per {'processed example' if col == 0 else 'CPU second'}")
            ax.grid(axis="y", alpha=.2)
            if row == col == 0:
                ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Held-out checkpoint decisions, development seeds 0–1 (mean ± seed SD)")
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    opener = gzip.open if args.input.endswith(".gz") else open
    with opener(args.input, "rt", encoding="utf8") as handle:
        plot(json.load(handle)["predictions"], args.output)
