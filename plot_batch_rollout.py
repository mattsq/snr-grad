"""Summarize and plot the frozen on-policy study."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POLICIES = ("fixed_small", "fixed_8", "fixed_16", "fixed_32", "fixed_64", "fixed_time",
            "warmup_time",
            "euclidean_example", "aware_example",
            "euclidean_time", "aware_time")
COLORS = {"fixed_small": "#222222", "fixed_8": "#555555",
          "fixed_16": "#888888", "fixed_32": "#aaaaaa", "fixed_64": "#bbbbbb",
          "fixed_time": "#555555",
          "warmup_time": "#5c658c",
          "euclidean_example": "#6385ba", "aware_example": "#c44153",
          "euclidean_time": "#3b9278", "aware_time": "#c98725"}
ROLLOUT_SHA256 = "107a2fa192e466d341eacde3d3cb201ede7d433409517f881ab959f33dbb226e"


def source(path):
    if not path.is_dir():
        return gzip.open(path, "rt", encoding="utf8")
    parts = sorted(path.glob("part[0-9][0-9]"))
    if not parts:
        raise ValueError("Missing rollout archive parts")
    blob = b"".join(p.read_bytes() for p in parts)
    if hashlib.sha256(blob).hexdigest() != ROLLOUT_SHA256:
        raise ValueError("Rollout archive parts are incomplete or changed")
    return io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(blob)), encoding="utf8")


def load(paths):
    runs = defaultdict(list)
    configs = []
    for path in paths:
        with source(path) as handle:
            configs.append(json.loads(next(handle))["config"])
            for line in handle:
                row = json.loads(line)
                key = row["task"], row["optimizer"], row["seed"], row["policy"]
                runs[key].append(row)
    for (_, _, _, policy), rows in runs.items():
        if not rows or rows[-1]["samples"] != 3000:
            raise ValueError("Incomplete rollout trajectory")
        if policy.startswith(("euclidean_", "aware_")) and rows[0]["actual_batch"] != 4:
            raise ValueError("Adaptive arm did not begin at the prespecified B=4")
    return configs, runs


def summaries(runs):
    results = []
    for (task, kind, seed, policy), rows in sorted(runs.items()):
        same = [other for (t, k, s, _), other in runs.items()
                if (t, k, s) == (task, kind, seed)]
        common_seconds = min(other[-1]["seconds"] for other in same)
        timed_ce = float(np.interp(common_seconds, [r["seconds"] for r in rows],
                                   [r["validation_final"] for r in rows]))
        final = rows[-1]
        counts = {str(b): sum(r["actual_batch"] == b for r in rows)
                  for b in (4, 8, 16, 32, 64)}
        results.append(dict(task=task, optimizer=kind, seed=seed, policy=policy,
                            final_loss=final["validation_final"],
                            common_time_loss=timed_ce, common_seconds=common_seconds,
                            seconds=final["seconds"],
                            processed_examples=final["processed_examples"],
                            probes=(final["processed_examples"] - final["samples"]) // 64,
                            steps=final["steps"], batch_counts=counts))
    return results


def plot(runs, output):
    tasks = sorted({key[0] for key in runs})
    for task in tasks:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for oi, optimizer in enumerate(("snr_muon", "adamw")):
            relevant = {key: rows for key, rows in runs.items()
                        if key[0] == task and key[1] == optimizer}
            if not relevant:
                continue
            seeds = sorted({key[2] for key in relevant})
            x_examples = np.linspace(4, 3000, 100)
            for axis_index in range(2):
                ax = axes[oi, axis_index]
                for policy in POLICIES:
                    values = []
                    for seed in seeds:
                        rows = relevant[(task, optimizer, seed, policy)]
                        if axis_index == 0:
                            x = x_examples
                            abscissa = [r["samples"] for r in rows]
                        else:
                            end = min(relevant[(task, optimizer, seed, p)][-1]["seconds"]
                                      for p in POLICIES)
                            x = np.linspace(0, 1, 100)
                            abscissa = [r["seconds"] / end for r in rows]
                        values.append(np.interp(x, abscissa,
                                                [r["validation_final"] for r in rows]))
                    matrix = np.array(values)
                    ax.plot(x, matrix.mean(axis=0), color=COLORS[policy], lw=1.5,
                            ls="--" if policy == "fixed_time" else "-",
                            label=policy if oi == axis_index == 0 else None)
                if axis_index == 0 and task != "digits":
                    ax.axvline(1500, color="black", lw=.7, ls=":")
                ax.set_title(f"{task} / {optimizer} | {'training examples' if axis_index == 0 else 'fraction of common CPU-time support'}")
                ax.set_xlabel("Training examples" if axis_index == 0 else "Fraction of common time")
                ax.set_ylabel("Final-regime validation loss")
                ax.grid(alpha=.2)
        fig.legend(POLICIES, loc="lower center", bbox_to_anchor=(.5, .005),
                   ncol=6, fontsize=8)
        fig.suptitle(f"On-policy batch control | {task} | five new seeds", fontsize=14)
        fig.tight_layout(rect=(0, .11, 1, .96))
        fig.savefig(output.with_name(output.stem + "_" + task + output.suffix),
                    dpi=140, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--summary", type=Path,
                        default=Path("benchmarks/benchmark_batch_rollout_summary.json"))
    parser.add_argument("--figure", type=Path,
                        default=Path("benchmarks/benchmark_batch_rollout.png"))
    args = parser.parse_args()
    config, runs = load(args.input)
    results = summaries(runs)
    args.summary.write_text(json.dumps(dict(configs=config, results=results), indent=2) + "\n")
    plot(runs, args.figure)
    for task in sorted({r["task"] for r in results}):
        for kind in ("snr_muon", "adamw"):
            print(task, kind)
            for policy in POLICIES:
                subset = [r for r in results if (r["task"], r["optimizer"], r["policy"]) ==
                          (task, kind, policy)]
                print(" ", policy, "final", round(float(np.mean([r["final_loss"] for r in subset])), 4),
                      "common_time", round(float(np.mean([r["common_time_loss"] for r in subset])), 4),
                      "seconds", round(float(np.mean([r["seconds"] for r in subset])), 3))


if __name__ == "__main__":
    main()
