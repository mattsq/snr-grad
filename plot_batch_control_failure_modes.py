"""Compare fixed, adaptive, and change-point controls on matched digit budgets."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("benchmarks")
FILES = ("benchmark_batch_control_digits.jsonl.gz",
         "benchmark_batch_control_ablation.jsonl.gz",
         "benchmark_batch_control_alarm.jsonl.gz")
POLICIES = ("fixed_small", "fixed_mid", "fixed_reference", "aware",
            "shift_reset", "aware_shift_reset", "aware_alarm")
STYLE = {
    "fixed_small": ("B=4", "#47689e", "-"),
    "fixed_mid": ("B=8", "#2788aa", "-"),
    "fixed_reference": ("B=16", "#7a8a9a", "-"),
    "aware": ("aware", "#c44153", "-"),
    "shift_reset": ("oracle B=16→4", "#6a9b69", "--"),
    "aware_shift_reset": ("oracle aware reset", "#e09842", "--"),
    "aware_alarm": ("loss-alarm reset", "#71499d", "-"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true", help="plot the two-class shift instead")
    args = parser.parse_args()
    task = "digits_partial_shift" if args.partial else "digits_shift"
    files = ("benchmark_batch_control_partial_shift.jsonl.gz",) if args.partial else FILES
    policies = tuple(p for p in POLICIES if not args.partial or p != "aware_shift_reset")
    rows = []
    for filename in files:
        with gzip.open(ROOT / filename, "rt", encoding="utf8") as source:
            rows.extend(json.loads(line) for line in source)
    rows = [r for r in rows if r["task"] == task and r.get("kind") is None]
    grid = np.arange(0, 3001, 20)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for i, optimizer in enumerate(("snr_muon", "adamw")):
        loss_ax, batch_ax = axes[i]
        for policy in policies:
            label, color, style = STYLE[policy]
            trials = [[r for r in rows if r["optimizer"] == optimizer and r["policy"] == policy
                       and r["seed"] == seed] for seed in range(2, 7)]
            # Validation labels jump at the switch; interpolation must not
            # blend the pre-shift and post-shift losses across that boundary.
            for left, right in ((0, 1500), (1500, 3000)):
                segment_grid = grid[(grid >= left) & (grid <= right)]
                losses = np.array([
                    np.interp(segment_grid,
                              [r["samples"] for r in trial if (r["samples"] < 1500 if left == 0 else r["samples"] >= 1500)],
                              [r["validation"] for r in trial if (r["samples"] < 1500 if left == 0 else r["samples"] >= 1500)])
                    for trial in trials])
                loss_ax.plot(segment_grid, losses.mean(0), color=color, ls=style, lw=1.8,
                             label=label if left == 0 else None)
            if policy in ("aware", "shift_reset", "aware_shift_reset", "aware_alarm"):
                batches = np.array([np.interp(grid, [r["samples"] for r in trial],
                                             [r["actual_batch"] for r in trial]) for trial in trials])
                batch_ax.plot(grid, np.median(batches, axis=0), color=color, ls=style, lw=1.8,
                              label=label)
        for ax in (loss_ax, batch_ax):
            ax.axvline(1500, color="black", ls=":", lw=1)
            ax.set_xlim(0, 3000)
            ax.set_xlabel("Training examples")
            ax.grid(alpha=.2)
        loss_ax.set_ylim(bottom=0)
        loss_ax.set_ylabel("Mean held-out validation CE")
        batch_ax.set_ylabel("Median actual batch across five seeds")
        batch_ax.set_ylim(2, 18)
        loss_ax.set_title(optimizer + " | sample-aligned validation loss")
        batch_ax.set_title(optimizer + " | batch trajectory")
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(("Two-class label switch" if args.partial else "All-class label switch") +
                 ": adaptation lag versus sample efficiency")
    fig.tight_layout()
    output = ROOT / ("benchmark_batch_control_partial_shift.png" if args.partial else
                     "benchmark_batch_control_failure_modes.png")
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
