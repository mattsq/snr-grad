"""Run frozen budget-aware batch policies on fresh and shifted tasks.

The calibration is loaded from the prior paired-checkpoint experiment. Only
development seeds 0 and 1 are eligible; no evaluation outcomes refit it.
"""

import argparse
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_batch_budget import choose
from benchmark_batch_control import (build_model, data, digit_loss, evaluate,
                                     loss_fn, optimizer_for, update)
from snr_grad import probe_batch


TASKS = ("digits", "digits_shift", "digits_rotate", "matrix_shift")
POLICIES = ("fixed_small", "fixed_8", "fixed_16", "fixed_32", "fixed_64", "fixed_time",
            "warmup_time",
            "euclidean_example", "aware_example",
            "euclidean_time", "aware_time")
SIZES = (4, 8, 16, 32, 64)


def task_data(seed, task):
    if task == "matrix_shift":
        train, valid = data(seed, "shift")
        return train[0], train[0], train[1], train[2], valid[0], valid[0], valid[1], valid[2], loss_fn, "matrix"
    source = "digits_shift" if task == "digits_shift" else "digits"
    train, valid = data(seed, source)
    if task == "digits_rotate":
        rotate = lambda x: torch.rot90(x.reshape(-1, 8, 8), 1, (1, 2)).reshape(-1, 64)
        return (train[0], rotate(train[0]), train[1], train[1],
                valid[0], rotate(valid[0]), valid[1], valid[1], digit_loss, "digits")
    return (train[0], train[0], train[1], train[2] if task == "digits_shift" else train[1],
            valid[0], valid[0], valid[1], valid[2] if task == "digits_shift" else valid[1],
            digit_loss, "digits")


def development_calibration(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf8") as handle:
        data_set = json.load(handle)
    if set(data_set["config"]["development_seeds"]) != {0, 1}:
        raise ValueError("Calibration must use only the prespecified development seeds 0–1.")
    result = []
    for curve in data_set["curves"]:
        if curve["seed"] not in (0, 1):
            continue
        curve = dict(curve)
        curve["options"] = {int(batch): values for batch, values in curve["options"].items()}
        result.append(curve)
    return result


def fixed_from_development(curves, optimizer, objective):
    subset = [r for r in curves if r["optimizer"] == optimizer]
    return max(SIZES, key=lambda batch: (
        np.mean([r["options"][batch]["fixed_" + objective] for r in subset]), -batch))


def run(seed, task, kind, policy, calibration, budget, probe_every, probe_size, probe_splits):
    (x_pre, x_post, y_pre, y_post, xv_pre, xv_post, yv_pre, yv_post,
     criterion, model_task) = task_data(seed, task)
    model = build_model(seed, model_task)
    optimizer = optimizer_for(model, kind, .1 if kind == "snr_muon" else .003, len(x_pre))
    indices = torch.randint(len(x_pre), (budget,), generator=
                            torch.Generator().manual_seed(seed + 90000))
    fixed_time = fixed_from_development(calibration, kind, "time")
    fixed_example = fixed_from_development(calibration, kind, "example")
    fixed_batch = (4 if policy == "fixed_small" else
                   int(policy.split("_")[1]) if policy in ("fixed_8", "fixed_16", "fixed_32", "fixed_64")
                   else fixed_time if policy == "fixed_time" else None)
    next_batch = fixed_batch if fixed_batch is not None else 4
    consumed = steps = probe_examples = 0
    elapsed = 0.
    rows = []
    has_shift = task != "digits"
    while consumed < budget:
        # Cut the one crossing step at the change point. The resulting partial
        # minibatch is recorded and shared consistently across policies.
        remaining = budget - consumed
        if has_shift and consumed < budget // 2:
            remaining = min(remaining, budget // 2 - consumed)
        batch_size = min(next_batch, remaining)
        shifted = has_shift and consumed >= budget // 2
        x = x_post if shifted else x_pre
        y = y_post if shifted else y_pre
        draw = indices[consumed:consumed + batch_size]
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        criterion(model, (x[draw], y[draw])).backward()
        update(optimizer, batch_size)
        elapsed += time.perf_counter() - started
        consumed += batch_size
        steps += 1
        decision = None
        sensor_value = None
        if fixed_batch is None and policy != "warmup_time" and steps % probe_every == 0 and consumed < budget:
            shifted_now = has_shift and consumed >= budget // 2
            px = x_post if shifted_now else x_pre
            py = y_post if shifted_now else y_pre
            probe_ids = torch.randperm(len(x_pre), generator=
                                       torch.Generator().manual_seed(seed + 40000 + steps))[:probe_size]
            probe_examples += len(probe_ids)
            started = time.perf_counter()
            probe = probe_batch(model, criterion, (px[probe_ids], py[probe_ids]),
                                splits=probe_splits, optimizer=optimizer)
            elapsed += time.perf_counter() - started
            sensor = "euclidean" if policy.startswith("euclidean") else "aware"
            sensor_value = (probe.euclidean.scale if sensor == "euclidean" else
                            probe.muon.scale if kind == "snr_muon" else probe.adamw.scale)
            objective = "example" if policy.endswith("example") else "time"
            query = dict(optimizer=kind, options={b: None for b in SIZES},
                         **{sensor: sensor_value})
            decision = choose(calibration, query, sensor, objective)
            next_batch = decision if decision is not None else (
                fixed_example if objective == "example" else fixed_time)
            if not math.isfinite(sensor_value):
                sensor_value = None
        if fixed_batch is not None:
            next_batch = fixed_batch
        elif policy == "warmup_time":
            next_batch = 4 if steps < probe_every else fixed_time
        active_post = has_shift and consumed >= budget // 2
        current_x, current_y = ((xv_post, yv_post) if active_post else (xv_pre, yv_pre))
        rows.append(dict(seed=seed, task=task, optimizer=kind, policy=policy,
                         samples=consumed, processed_examples=consumed + probe_examples,
                         steps=steps, actual_batch=batch_size, next_batch=next_batch,
                         decision=decision, sensor_value=sensor_value, seconds=elapsed,
                         validation_current=evaluate(model, (current_x, current_y, current_y), False, criterion),
                         validation_final=evaluate(model, (xv_post, yv_post, yv_post), False, criterion)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path,
                        default=Path("benchmarks/benchmark_batch_budget.json.gz"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(11, 16)))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--optimizers", nargs="+", choices=("snr_muon", "adamw"),
                        default=("snr_muon", "adamw"))
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--probe-every", type=int, default=20)
    parser.add_argument("--probe-size", type=int, default=64)
    parser.add_argument("--probe-splits", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/benchmark_batch_rollout.jsonl.gz"))
    args = parser.parse_args()
    if (args.budget != 3000 or args.probe_every != 20 or
            args.probe_size != 64 or args.probe_splits != 8 or
            args.probe_size > 1437 or set(args.seeds) & {0, 1, 2, 3, 4, 5, 6}):
        parser.error("frozen protocol requires 3000 examples, 64/8 probe every 20 steps, and new seeds")
    torch.set_num_threads(1)
    calibration = development_calibration(args.calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf8") as handle:
        handle.write(json.dumps(dict(kind="config", config={
            **vars(args), "calibration": str(args.calibration), "output": str(args.output)}),
            default=str) + "\n")
        for kind in args.optimizers:
            for task in args.tasks:
                for seed in args.seeds:
                    order = torch.randperm(len(args.policies), generator=
                                           torch.Generator().manual_seed(seed + 80000)).tolist()
                    for position in order:
                        policy = args.policies[position]
                        rows = run(seed, task, kind, policy, calibration, args.budget,
                                   args.probe_every, args.probe_size, args.probe_splits)
                        for row in rows:
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
                        print(kind, task, seed, policy, "loss", round(rows[-1]["validation_final"], 4),
                              "time", round(rows[-1]["seconds"], 3), flush=True)


if __name__ == "__main__":
    main()
