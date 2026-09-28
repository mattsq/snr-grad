"""Render the batch-control benchmark's comparisons and mechanism diagnostics.

python plot_batch_control.py benchmarks/benchmark_batch_control.jsonl
"""

import argparse
import gzip
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASKS = ("stationary", "shift", "matrix")
PRIMARY = {"stationary": "snr_adamw", "shift": "snr_adamw", "matrix": "snr_muon",
           "digits": "snr_muon", "digits_shift": "snr_muon"}
COLORS = {
    "fixed_small": "#5470a6", "fixed_reference": "#2788aa", "fixed_large": "#c7794a", "ramp": "#6a9b69",
    "euclidean": "#9467bd", "aware": "#c44153",
}
LABELS = {
    "fixed_small": "fixed small", "fixed_reference": "fixed reference", "fixed_large": "fixed large", "ramp": "preset ramp",
    "euclidean": "Euclidean control", "aware": "optimizer-aware control",
}


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8") as source:
        return [json.loads(line) for line in source]


def pick(rows, task, optimizer, policy=None, kind=None):
    return [r for r in rows if r["task"] == task and r["optimizer"] == optimizer
            and (policy is None or r["policy"] == policy)
            and (r.get("kind") == kind)]


def by_seed(rows):
    return [[r for r in rows if r["seed"] == seed]
            for seed in sorted({r["seed"] for r in rows})]


def plot_frontiers(rows, output):
    fig, axes = plt.subplots(len(TASKS), 3, figsize=(16, 3.5 * len(TASKS) + 1), squeeze=False)
    fig.suptitle("Batch control: validation loss under three cost measures", fontsize=15)
    for i, task in enumerate(TASKS):
        optimizer = PRIMARY[task]
        for j, (key, xlabel) in enumerate((("step", "Optimizer steps"), ("samples", "Training examples"),
                                          ("seconds", "Training time (s, including probes)"))):
            ax = axes[i, j]
            for policy, kind, style in [(p, optimizer, "-") for p in COLORS] + [
                ("fixed_small", "adamw", "--"), ("aware", "adamw", "--")]:
                trials = by_seed(pick(rows, task, kind, policy))
                if not trials:
                    continue
                curves = [(np.array([r[key] for r in trial]), np.array([r["validation"] for r in trial]))
                          for trial in trials if trial]
                if not curves:
                    continue
                xmax = min(x[-1] for x, _ in curves)
                xmin = max(x[0] for x, _ in curves)
                grid = np.linspace(xmin, xmax, 90)
                values = np.stack([np.interp(grid, x, y) for x, y in curves])
                label = LABELS[policy] + (" (AdamW)" if kind == "adamw" else "")
                ax.plot(grid, values.mean(0), color=COLORS[policy], ls=style,
                        lw=1.6, label=label, alpha=.85)
                if len(curves) > 1:
                    ax.fill_between(grid, values.mean(0) - values.std(0),
                                    values.mean(0) + values.std(0), color=COLORS[policy], alpha=.08)
            if task in ("shift", "digits_shift") and key == "samples":
                shift = pick(rows, task, optimizer, "fixed_small")[0]["shift_at"]
                ax.axvline(shift, ls=":", color="black", alpha=.65, label="target switch")
            ax.set_title((task + " | " + optimizer) if j == 0 else xlabel)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Validation loss")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .97))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_diagnostics(rows, output):
    fig, axes = plt.subplots(len(TASKS), 2, figsize=(14, 3.5 * len(TASKS) + 1), squeeze=False)
    fig.suptitle("Controller decisions: measured scales, chosen batches, and probe cost", fontsize=15)
    for i, task in enumerate(TASKS):
        optimizer = PRIMARY[task]
        ax, cost_ax = axes[i]
        for policy in ("euclidean", "aware"):
            trials = by_seed(pick(rows, task, optimizer, policy))
            if not trials:
                continue
            trial = trials[0]  # A single seed keeps each decision path legible.
            ax.step([r["step"] for r in trial], [r["actual_batch"] for r in trial],
                    where="post", color=COLORS[policy], lw=2, label=LABELS[policy] + " actual B")
            observed = [r for r in trial if r["recommendation"] is not None]
            sensor = "muon" if PRIMARY[task] == "snr_muon" and policy == "aware" else (
                "adamw" if policy == "aware" else "euclidean")
            ax.scatter([r["step"] for r in observed if r[sensor] is not None],
                       [r[sensor] for r in observed if r[sensor] is not None],
                       s=14, color=COLORS[policy], marker="x", alpha=.65,
                       label=LABELS[policy] + " raw scale")
            for seed_trial in trials:
                spent = np.cumsum([r.get("probe_seconds") or 0 for r in seed_trial])
                elapsed = np.array([r["seconds"] for r in seed_trial])
                cost_ax.plot([r["step"] for r in seed_trial], spent / np.maximum(elapsed, 1e-9),
                             color=COLORS[policy], alpha=.24, lw=1)
            common_steps = min(len(t) for t in trials)
            fractions = np.stack([
                np.cumsum([r.get("probe_seconds") or 0 for r in t[:common_steps]]) /
                np.maximum([r["seconds"] for r in t[:common_steps]], 1e-9)
                for t in trials
            ])
            cost_ax.plot([r["step"] for r in trials[0][:common_steps]],
                         fractions.mean(0), color=COLORS[policy], lw=2, label=LABELS[policy])
        ax.set_yscale("log", base=2)
        ax.set_ylim(bottom=2)
        first_seed = min(r["seed"] for r in pick(rows, task, optimizer))
        ax.set_title(task + f" | seed {first_seed} decision path")
        ax.set_ylabel("Batch size / raw GNS (log2)")
        ax.set_xlabel("Optimizer step")
        ax.legend(fontsize=7, loc="upper right")
        cost_ax.set_title(task + " | cumulative probe time / training time")
        cost_ax.set_ylabel("Probe time fraction")
        cost_ax.set_xlabel("Optimizer step")
        cost_ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        cost_ax.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_calibration(rows, output):
    fig, axes = plt.subplots(len(TASKS), 2, figsize=(14, 3.5 * len(TASKS) + 1), squeeze=False)
    fig.suptitle("Local continuation: do noise scales identify a useful batch range?", fontsize=15)
    for i, task in enumerate(TASKS):
        curves = pick(rows, task, PRIMARY[task], "fixed_small", "local_curve")
        if task in ("shift", "digits_shift") and any("phase" in r for r in curves):
            curves = [r for r in curves if r.get("phase") == "after"]
        if not curves:
            continue
        sizes = sorted({r["candidate"] for r in curves})
        ax, scatter = axes[i]
        for key, color, label in (("improvement_per_step", "#5470a6", "per optimizer step"),
                                  ("improvement_per_sample", "#c7794a", "per training example")):
            means = [np.mean([r[key] for r in curves if r["candidate"] == B]) for B in sizes]
            stds = [np.std([r[key] for r in curves if r["candidate"] == B]) for B in sizes]
            values = np.array(means)
            if key == "improvement_per_sample":
                # Rescale for display only; metric retains its meaning in JSONL.
                values *= sizes[0]
                stds = np.array(stds) * sizes[0]
                label += f" (x{sizes[0]} for display)"
            ax.plot(sizes, values, "o-", color=color, label=label)
            ax.fill_between(sizes, values - stds, values + stds, color=color, alpha=.15)
        ax.axhline(0, color="black", alpha=.35, lw=.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, labels=[str(B) for B in sizes])
        ax.set_xlabel("Candidate batch size")
        ax.set_ylabel("Validation improvement per step")
        ax.set_title(task + (" post-shift" if task in ("shift", "digits_shift") else "") +
                     " | paired continuations")
        ax.legend(fontsize=8)
        sensor = "muon" if PRIMARY[task] == "snr_muon" else "adamw"
        checkpoints = sorted({(r["seed"], r["checkpoint_step"]) for r in curves})
        for index, (seed, checkpoint) in enumerate(checkpoints):
            trial = [r for r in curves if r["seed"] == seed and r["checkpoint_step"] == checkpoint]
            gains = {r["candidate"]: r["improvement_per_step"] for r in trial}
            if max(gains.values()) <= 0:
                continue
            knee = next((B for B in sizes if gains[B] >= .8 * max(gains.values())), sizes[-1])
            for key, color, marker in (("euclidean", COLORS["euclidean"], "o"),
                                       (sensor, COLORS["aware"], "s")):
                estimate = trial[0][key]
                if estimate is not None and math.isfinite(estimate) and estimate > 0:
                    scatter.scatter(estimate, knee, color=color, marker=marker, s=45,
                                    label=key if index == 0 else None)
        bounds = [sizes[0], sizes[-1]]
        scatter.plot(bounds, bounds, color="gray", ls=":", label="identity (uncalibrated)")
        scatter.set_xscale("log", base=2)
        scatter.set_yscale("log", base=2)
        scatter.set_yticks(sizes, labels=[str(B) for B in sizes])
        scatter.set_xlabel("Predicted raw noise scale (examples)")
        scatter.set_ylabel("Smallest B within 80% of best per-step gain")
        scatter.set_title(task + (" post-shift" if task in ("shift", "digits_shift") else "") +
                          " | scale versus measured knee")
        scatter.legend(fontsize=8)
        for axis in (ax, scatter):
            axis.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_gates(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SNRAdamW: does changing the actual batch also change gate strength?", fontsize=14)
    for ax, task in zip(axes, ("stationary", "shift")):
        for policy in COLORS:
            trials = by_seed(pick(rows, task, "snr_adamw", policy))
            if not trials:
                continue
            grid = np.linspace(0, min(t[-1]["samples"] for t in trials), 100)[1:]
            values = np.stack([np.interp(grid, [r["samples"] for r in t],
                                         [r["gate_mean"] for r in t]) for t in trials])
            ax.plot(grid, values.mean(0), color=COLORS[policy], label=LABELS[policy], lw=1.7)
            ax.fill_between(grid, values.mean(0) - values.std(0),
                            values.mean(0) + values.std(0), color=COLORS[policy], alpha=.1)
        if task == "shift":
            shift = pick(rows, task, "snr_adamw", "fixed_small")[0]["shift_at"]
            ax.axvline(shift, ls=":", color="black", label="target switch")
        ax.set_title(task)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Mean SNR gate value")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    global TASKS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--tasks", nargs="+", choices=tuple(PRIMARY), default=TASKS)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    TASKS = tuple(args.tasks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.data)
    for name, function in (("frontiers", plot_frontiers), ("diagnostics", plot_diagnostics),
                           ("calibration", plot_calibration), ("gates", plot_gates)):
        if name == "gates" and not any(t in TASKS for t in ("stationary", "shift")):
            continue
        path = args.out_dir / ("benchmark_batch_control_" + args.tag + name + ".png")
        function(rows, path)
        print(path)


if __name__ == "__main__":
    main()
