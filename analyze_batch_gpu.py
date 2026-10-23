"""Analyze ``benchmark_batch_gpu.py`` output.

Rollout files: final validation loss at the shared time budget, with paired
per-seed differences on evaluation seeds only. The best fixed batch and the
ramp are the comparators; the best fixed batch is chosen on ``--tuning-seeds``
so that it is a deployable choice, and also reported as a post-hoc oracle.

Validation files: at each checkpoint, fit ``gain_per_step(B) = G B / (B + B_crit)``
to the paired continuations, then ask (1) whether each sensor's noise scale
ranks ``B_crit`` across checkpoints, and (2) how much gain per second is lost
when the cost-aware rule picks a batch from each sensor and the measured step
times, relative to the best measured batch.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path

import numpy as np


def rows(paths):
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf8") as handle:
            for line in handle:
                yield json.loads(line)


def paired(a, b):
    d = np.array(a) - np.array(b)
    sd = float(d.std(ddof=1)) if len(d) > 1 else math.nan
    return f"{d.mean():+.4f} (sd {sd:.4f}, better in {int((d < 0).sum())}/{len(d)})"


def analyze_rollouts(records, tuning, plot_path):
    final = defaultdict(dict)
    curves = defaultdict(list)
    for r in records:
        if r["kind"] != "eval":
            continue
        key = r["optimizer"], r["policy"], r["seed"]
        curves[key].append((r["seconds"], r["validation"]))
        if r.get("final"):
            final[r["optimizer"], r["seed"]][r["policy"]] = r
    summary = {}
    for optimizer in sorted({k[0] for k in final}):
        seeds = sorted(s for (o, s) in final if o == optimizer)
        evaluation = [s for s in seeds if s not in tuning]
        policies = sorted({p for (o, s), v in final.items() if o == optimizer for p in v})
        fixed = [p for p in policies if p.startswith("fixed:")]
        loss = {p: {s: final[optimizer, s][p]["validation"] for s in seeds if p in final[optimizer, s]}
                for p in policies}
        chosen = (min(fixed, key=lambda p: np.mean([loss[p][s] for s in tuning if s in loss[p]]))
                  if fixed and any(s in loss[fixed[0]] for s in tuning) else None)
        oracle = min(fixed, key=lambda p: np.mean([loss[p][s] for s in evaluation])) if fixed else None
        print(f"\n{optimizer}: evaluation seeds {evaluation}, tuning seeds {sorted(tuning)}")
        print(f"{'policy':24s} {'final loss':>10s} {'tokens':>12s} {'steps':>7s}")
        for p in policies:
            values = [loss[p][s] for s in evaluation if s in loss[p]]
            tokens = np.mean([final[optimizer, s][p]["tokens"] for s in evaluation if s in loss[p]])
            steps = np.mean([final[optimizer, s][p]["steps"] for s in evaluation if s in loss[p]])
            print(f"{p:24s} {np.mean(values):10.4f} {tokens:12.0f} {steps:7.0f}")
        comparisons = {}
        for controller in ("controller:aware", "controller:euclidean"):
            if controller not in loss:
                continue
            others = [chosen, oracle, "ramp",
                      "controller:euclidean" if controller == "controller:aware" else None]
            for other in dict.fromkeys(filter(None, others)):
                if other in loss and other != controller:
                    label = f"{controller} - {other}" + (" (tuned)" if other == chosen else
                                                         " (post-hoc oracle)" if other == oracle else "")
                    both = [s for s in evaluation if s in loss[controller] and s in loss[other]]
                    text = paired([loss[controller][s] for s in both], [loss[other][s] for s in both])
                    comparisons[label] = text
                    print(f"  {label}: {text}")
        summary[optimizer] = dict(tuned_fixed=chosen, oracle_fixed=oracle, comparisons=comparisons,
                                  final={p: loss[p] for p in policies})
    if plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        optimizers = sorted({k[0] for k in curves})
        fig, axes = plt.subplots(1, len(optimizers), figsize=(7 * len(optimizers), 5), squeeze=False)
        for ax, optimizer in zip(axes[0], optimizers):
            for policy in sorted({k[1] for k in curves if k[0] == optimizer}):
                runs = [sorted(v) for k, v in curves.items() if k[:2] == (optimizer, policy)]
                end = min(r[-1][0] for r in runs)
                grid = np.linspace(0, end, 100)
                mean = np.mean([np.interp(grid, *zip(*r)) for r in runs], axis=0)
                ax.plot(grid, mean, label=policy, lw=2 if policy.startswith("controller") else 1,
                        ls="-" if policy.startswith("controller") else "--")
            ax.set_xlabel("Training seconds (device-synchronized)")
            ax.set_ylabel("Validation loss (mean over seeds)")
            ax.set_title(optimizer)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
    return summary


def fit_critical_batch(sizes, gains):
    """Least-squares ``B_crit``; ``None`` when the best fit is at the grid edge."""
    grid = np.geomspace(min(sizes) / 16, max(sizes) * 16, 400)
    best = None
    for bc in grid:
        shape = np.array(sizes) / (np.array(sizes) + bc)
        g = float(shape @ gains / (shape @ shape))
        sse = float(((gains - g * shape) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(bc), g)
    # Gains that fall or stay flat with B (or are negative) pin the fit to an
    # edge; such a checkpoint has no identifiable critical batch.
    if best[1] in (float(grid[0]), float(grid[-1])) or best[2] <= 0:
        return None, best[2]
    return best[1], best[2]


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else math.nan


def analyze_validation(records):
    continuations = defaultdict(lambda: defaultdict(list))
    checkpoints = {}
    for r in records:
        key = r.get("optimizer"), r.get("seed"), r.get("fraction")
        if r["kind"] == "continuation":
            continuations[key][r["batch"]].append(r)
        elif r["kind"] == "checkpoint":
            checkpoints[key] = r
    results = defaultdict(list)
    for key, by_batch in continuations.items():
        if key not in checkpoints:
            continue
        sizes = sorted(by_batch)
        gain = np.array([np.mean([c["start_loss"] - c["end_loss"] for c in by_batch[b]]) / by_batch[b][0]["steps"]
                         for b in sizes])
        seconds = np.array([np.mean([c["seconds"] for c in by_batch[b]]) / by_batch[b][0]["steps"]
                            for b in sizes])
        per_second = gain / seconds
        bc, _ = fit_critical_batch(sizes, gain)
        best = sizes[int(np.argmax(per_second))]
        entry = dict(seed=key[1], fraction=key[2], critical_batch=bc, best_per_second=best,
                     best_rate=float(per_second.max()))
        for sensor in ("euclidean", "aware"):
            values = [s[sensor] for s in checkpoints[key]["sensors"] if s.get(sensor) is not None]
            if not values:
                continue
            scale = float(np.median(values))
            predicted = sizes[int(np.argmax([b / (b + scale) / t for b, t in zip(sizes, seconds)]))]
            entry[sensor] = scale
            entry[sensor + "_choice"] = predicted
            entry[sensor + "_regret"] = float(per_second.max() - per_second[sizes.index(predicted)])
        results[key[0]].append(entry)
    summary = {}
    for optimizer, entries in results.items():
        print(f"\n{optimizer}: {len(entries)} checkpoints")
        print(f"{'seed':>5s} {'frac':>5s} {'B_crit fit':>10s} {'best B/s':>8s} "
              f"{'eucl':>8s} {'aware':>8s} {'eucl->B':>7s} {'aware->B':>8s}")
        for e in sorted(entries, key=lambda e: (e["seed"], e["fraction"])):
            bc = e["critical_batch"] if e["critical_batch"] is not None else math.nan
            print(f"{e['seed']:5d} {e['fraction']:5.2f} {bc:10.1f} {e['best_per_second']:8d} "
                  f"{e.get('euclidean', math.nan):8.1f} {e.get('aware', math.nan):8.1f} "
                  f"{e.get('euclidean_choice', -1):7d} {e.get('aware_choice', -1):8d}")
        report = {}
        for sensor in ("euclidean", "aware"):
            usable = [e for e in entries if sensor in e]
            fitted = [e for e in usable if e["critical_batch"] is not None]
            if not usable:
                continue
            report[sensor] = dict(
                identified_checkpoints=len(fitted),
                spearman_with_critical_batch=spearman([e[sensor] for e in fitted],
                                                      [e["critical_batch"] for e in fitted]),
                mean_log_ratio_to_critical_batch=float(np.mean(
                    [math.log(e[sensor] / e["critical_batch"]) for e in fitted])) if fitted else math.nan,
                agreement_with_best=float(np.mean([e[sensor + "_choice"] == e["best_per_second"]
                                                   for e in usable])),
                mean_relative_regret=float(np.mean([e[sensor + "_regret"] / e["best_rate"]
                                                    for e in usable if e["best_rate"] > 0])))
            print(f"  {sensor}: " + ", ".join(f"{k} {v:.3f}" for k, v in report[sensor].items()))
        summary[optimizer] = dict(checkpoints=entries, sensors=report)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--tuning-seeds", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    records = list(rows(args.inputs))
    modes = {r["mode"] for r in records if r["kind"] == "config"}
    summary = {}
    if "rollout" in modes:
        summary["rollout"] = analyze_rollouts([r for r in records if r["kind"] == "eval"],
                                              set(args.tuning_seeds), args.plot)
    if "validate" in modes:
        summary["validate"] = analyze_validation(records)
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
