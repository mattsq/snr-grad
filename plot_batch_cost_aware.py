"""Plot the cost-aware controller: its decision rule and the CPU mechanism check.

``benchmark_batch_cost_aware_decisions.png`` draws the batch that
``CostAwareBatchController`` targets as a function of the noise scale under
three prices, with the noise scales the controllers actually measured.
``benchmark_batch_cost_aware.png`` summarizes ``benchmark_batch_cost_aware.py``:
example-priced arms at the example cap (fixed and square-root learning rates),
time-priced arms at a CPU-time budget, and the controllers' batch paths.
"""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snr_grad import CostAwareBatchController, StepTimeModel

# Reference categorical slots 1-3 (validated all-pairs) and a neutral ramp.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAYS = {"fixed_4": "#3d3c39", "fixed_16": "#86857f", "fixed_64": "#bdbcb5"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
STYLE = {"fixed_4": dict(color=GRAYS["fixed_4"], ls="-"),
         "fixed_16": dict(color=GRAYS["fixed_16"], ls="--"),
         "fixed_64": dict(color=GRAYS["fixed_64"], ls=":"),
         "euclidean": dict(color=BLUE, ls="-"), "aware": dict(color=ORANGE, ls="-"),
         "old": dict(color=AQUA, ls="-")}
LABEL = {"fixed_4": "Fixed B=4", "fixed_16": "Fixed B=16", "fixed_64": "Fixed B=64",
         "euclidean": "Cost-aware, Euclidean", "aware": "Cost-aware, optimizer-aware",
         "old": "Earlier calibrated controller (aware)"}
TASK_LABEL = {"digits": "Digits", "digits_shift": "Label shift", "digits_rotate": "Rotation",
              "matrix_shift": "Matrix shift"}
OPT_LABEL = {"snr_muon": "SNRMuon", "adamw": "AdamW"}


def style_axes(ax):
    ax.grid(color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def load(path):
    runs = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf8") as handle:
        next(handle)
        for line in handle:
            r = json.loads(line)
            runs[r["budget"], r["task"], r["optimizer"], r["policy"], r["seed"]].append(r)
    return runs


def key_policy(policy):
    return policy.split("_", 1)[1] if policy.startswith(("example_", "time_")) else policy


def target_batch(noise, sizes, **prices):
    controller = CostAwareBatchController(sizes, initial=sizes[0], ema=0., **prices)
    scores = [controller.efficiency(b, noise) for b in sizes]
    return sizes[int(np.argmax(scores))]


def cpu_step_seconds(runs):
    """Median seconds per step of the fixed time-budget arms, all optimizers."""
    seconds = defaultdict(list)
    for (budget, _, _, policy, _), rows in runs.items():
        if budget == "time" and policy.startswith("fixed_"):
            b = int(policy.split("_")[1])
            seconds[b] += [n["seconds"] - p["seconds"] for p, n in zip(rows, rows[1:])]
    return {b: float(np.median(v)) for b, v in seconds.items()}


def decision_map(runs, output):
    sizes = tuple(2 ** k for k in range(2, 10))
    measured = cpu_step_seconds(runs)
    b = np.array(sorted(measured))
    t1, t0 = np.polyfit(b, [measured[x] for x in b], 1)
    cpu = StepTimeModel({s: t0 + t1 * s for s in sizes})
    accelerator = StepTimeModel({s: max(1., s / 128) for s in sizes})
    noise = np.geomspace(1, 1e5, 500)
    fig, (ax, bars) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True,
                                   gridspec_kw=dict(height_ratios=(3, 1.25)))
    style_axes(ax)
    style_axes(bars)
    curves = [("Price per example only", dict(example_price=1., time_price=0.), BLUE),
              (f"CPU step time, fitted to this run: {t0 * 1e3:.2f} ms + {t1 * 1e6:.1f} µs × B",
               dict(step_times=cpu, time_price=1.), ORANGE),
              ("Illustrative step time flat to B=128, then linear (target is always the knee)",
               dict(step_times=accelerator, time_price=1.), AQUA)]
    for label, prices, color in curves:
        chosen = [target_batch(n, sizes, **prices) for n in noise]
        ax.step(noise, chosen, where="mid", color=color, lw=2, label=label)
    ax.set_yscale("log", base=2)
    ax.set_yticks(sizes)
    ax.set_yticklabels([str(s) for s in sizes])
    ax.set_ylim(3.2, 700)
    ax.set_ylabel("Batch the controller moves toward")
    ax.set_title("Cost-aware target batch: argmax over B of [B / (B + B_noise)] / cost(B)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    scales = defaultdict(list)
    for (budget, _, optimizer, policy, _), rows in runs.items():
        if policy.startswith(("example_", "time_")):
            scales[optimizer, key_policy(policy)] += [
                r["noise_scale"] for r in rows if r.get("noise_scale") and np.isfinite(r["noise_scale"])]
    order = [(o, s) for o in ("snr_muon", "adamw") for s in ("euclidean", "aware")]
    for i, (optimizer, sensor) in enumerate(order):
        values = scales[optimizer, sensor]
        lo, mid, hi = np.percentile(values, [10, 50, 90])
        bars.plot([lo, hi], [i, i], color=STYLE[sensor]["color"], lw=5, solid_capstyle="round")
        bars.plot([mid], [i], "o", ms=8, color=STYLE[sensor]["color"], mec="white", mew=2)
    bars.set_yticks(range(len(order)))
    bars.set_yticklabels([f"{OPT_LABEL[o]}, {'Euclidean' if s == 'euclidean' else 'optimizer-aware'}"
                          for o, s in order], fontsize=8)
    bars.set_ylim(len(order) - .5, -.5)
    bars.set_xscale("log")
    bars.set_xlabel("Gradient noise scale B_noise (per-example units)")
    bars.set_title("Smoothed noise scales the controllers measured (10–90%, dot = median)", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def final(rows):
    return rows[-1]["validation_final"]


def example_panel(ax, runs, old_summary, include_old, title):
    style_axes(ax)
    pairs = [(t, o) for o in ("snr_muon", "adamw") for t in TASK_LABEL]
    policies = ["fixed_16", "euclidean", "aware"] + (["old"] if include_old else [])
    offsets = np.linspace(-.27, .27, len(policies))
    for row, (task, optimizer) in enumerate(pairs):
        seeds = sorted(s for (bu, t, o, p, s) in runs if (bu, t, o, p) == ("examples", task, optimizer, "fixed_4"))
        base = {s: final(runs["examples", task, optimizer, "fixed_4", s]) for s in seeds}
        for offset, policy in zip(offsets, policies):
            if policy == "old":
                diffs = [old_summary[task, optimizer, s, "aware_example"] - old_summary[task, optimizer, s, "fixed_small"]
                         for s in seeds]
            else:
                name = policy if policy.startswith("fixed") else "example_" + policy
                diffs = [final(runs["examples", task, optimizer, name, s]) - base[s] for s in seeds]
            y = row + offset
            ax.scatter(diffs, [y] * len(diffs), s=10, color=STYLE[policy]["color"], alpha=.45, lw=0)
            ax.plot([np.mean(diffs)] * 2, [y - .1, y + .1], color=STYLE[policy]["color"], lw=2.5)
    ax.axvline(0, color=MUTED, lw=1)
    # A few fixed-batch failures reach +2.5; symlog keeps small gaps readable.
    ax.set_xscale("symlog", linthresh=.05)
    ax.set_xlim(-.3, 3)
    ticks = [-.1, 0, .05, .1, .5, 1, 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{TASK_LABEL[t]} / {OPT_LABEL[o]}" for t, o in pairs], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Final loss minus fixed B=4 (symlog; dots: seeds, bar: mean)")
    ax.set_title(title, fontsize=9.5)
    return policies


def time_panel(ax, runs, optimizer):
    style_axes(ax)
    for policy in ("fixed_4", "fixed_16", "fixed_64", "time_euclidean", "time_aware"):
        curves = [rows for (bu, t, o, p, s), rows in runs.items()
                  if (bu, t, o, p) == ("time", "digits", optimizer, policy)]
        end = min(r[-1]["seconds"] for r in curves)
        grid = np.linspace(0, end, 120)
        mean = np.mean([np.interp(grid, [x["seconds"] for x in r], [x["validation_final"] for x in r])
                        for r in curves], axis=0)
        ax.plot(grid, mean, lw=2 if policy.startswith("time") else 1.5, **STYLE[key_policy(policy)])
    ax.set_yscale("log")
    ax.set_xlabel("Training CPU seconds (probe and timing pass included)")
    ax.set_ylabel("Validation cross-entropy")
    ax.set_title(f"Price per CPU second, stationary digits / {OPT_LABEL[optimizer]}", fontsize=9.5)


def path_panel(ax, runs):
    style_axes(ax)
    for budget, prefix, ls in (("examples", "example_", "-"), ("time", "time_", "--")):
        for sensor in ("euclidean", "aware"):
            paths = [rows for (bu, t, o, p, s), rows in runs.items() if bu == budget and p == prefix + sensor]
            grid = np.linspace(0, 3000, 100)
            mean = np.exp(np.mean([np.interp(grid, [x["samples"] for x in r],
                                             np.log([x["actual_batch"] for x in r])) for r in paths], axis=0))
            name = "Euclidean" if sensor == "euclidean" else "Optimizer-aware"
            ax.plot(grid, mean, color=STYLE[sensor]["color"], ls=ls, lw=2,
                    label=f"{name}, price per {'example' if budget == 'examples' else 'CPU second'}")
    ax.set_yscale("log", base=2)
    ax.set_yticks([4, 8, 16, 32, 64])
    ax.set_yticklabels(["4", "8", "16", "32", "64"])
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Batch size (geometric mean over runs)")
    ax.set_title("Controller batch paths, all runs from B=16", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)


def summary_figure(runs, runs_sqrt, old_summary, output):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    policies = example_panel(axes[0, 0], runs, old_summary, True,
                             "Price per example, fixed learning rate (3,000-example cap)")
    example_panel(axes[0, 1], runs_sqrt, old_summary, False,
                  "Price per example, square-root learning-rate coupling")
    path_panel(axes[0, 2], runs)
    time_panel(axes[1, 0], runs, "snr_muon")
    time_panel(axes[1, 1], runs, "adamw")
    axes[1, 2].axis("off")
    handles = [plt.Line2D([], [], lw=2, **STYLE[p]) for p in ("fixed_4", "fixed_16", "fixed_64", "euclidean", "aware", "old")]
    axes[1, 2].legend(handles, [LABEL[p] for p in ("fixed_4", "fixed_16", "fixed_64", "euclidean", "aware", "old")],
                      loc="center", frameon=False, fontsize=10, title="Policies", title_fontsize=10)
    fig.suptitle("Cost-aware batch control with the free accumulation probe | CPU mechanism check, seeds 11–15",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("benchmarks/benchmark_batch_cost_aware.jsonl.gz"))
    parser.add_argument("--input-sqrt", type=Path,
                        default=Path("benchmarks/benchmark_batch_cost_aware_sqrt.jsonl.gz"))
    parser.add_argument("--old-summary", type=Path,
                        default=Path("benchmarks/benchmark_batch_rollout_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/benchmark_batch_cost_aware.png"))
    args = parser.parse_args()
    runs, runs_sqrt = load(args.input), load(args.input_sqrt)
    old = {(r["task"], r["optimizer"], r["seed"], r["policy"]): r["final_loss"]
           for r in json.loads(args.old_summary.read_text())["results"]}
    decision_map(runs, args.output.with_name(args.output.stem + "_decisions.png"))
    summary_figure(runs, runs_sqrt, old, args.output)
    for name, data in (("fixed lr", runs), ("sqrt lr", runs_sqrt)):
        print(f"\n{name}: mean final loss")
        for (budget, task) in (("examples", t) for t in TASK_LABEL):
            for optimizer in OPT_LABEL:
                cells = []
                for policy in ("fixed_4", "fixed_16", "fixed_64", "example_euclidean", "example_aware"):
                    values = [final(r) for (bu, t, o, p, s), r in data.items() if (bu, t, o, p) == (budget, task, optimizer, policy)]
                    cells.append(f"{policy}={np.mean(values):.3f}")
                print(f"  {task:13s} {optimizer:8s} " + " ".join(cells))
        for optimizer in OPT_LABEL:
            cells = []
            for policy in ("fixed_4", "fixed_16", "fixed_64", "time_euclidean", "time_aware"):
                values = [final(r) for (bu, t, o, p, s), r in data.items() if (bu, t, o, p) == ("time", "digits", optimizer, policy)]
                cells.append(f"{policy}={np.mean(values):.3f}")
            print(f"  time-budget digits {optimizer:8s} " + " ".join(cells))


if __name__ == "__main__":
    main()
