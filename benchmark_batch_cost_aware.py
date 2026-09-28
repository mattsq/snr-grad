"""CPU check of the free probe and cost-aware controller on the on-policy tasks.

This reruns the tasks and seeds of ``benchmark_batch_rollout.py`` with the two
fixes that do not need an accelerator:

* noise is measured from the training step's own accumulation microbatches
  (``GradientNoiseAccumulator``), so no example is spent on probing; and
* ``CostAwareBatchController`` chooses the batch from an explicit price, with
  no fitted multiplier.

Each objective is scored at its own endpoint. Example-priced arms train to the
3,000-example cap. Time-priced arms train for a fixed CPU-time budget on
stationary digits, the one task where a time budget does not also change
when each arm reaches a change point. The seeds were already inspected in the
earlier study, and the CPU cannot reward large batches the way an
accelerator can, so this is a mechanism check, not evidence for the method.
"""

import argparse
import copy
import gzip
import json
import time
from pathlib import Path

import torch

from benchmark_batch_control import build_model, evaluate, optimizer_for, update
from benchmark_batch_rollout import TASKS, task_data
from snr_grad import CostAwareBatchController, GradientNoiseAccumulator, StepTimeModel, coupled_lr
from snr_grad.variance import tree_split


SIZES = (4, 8, 16, 32, 64)
EXAMPLE_POLICIES = ("fixed_4", "fixed_16", "fixed_64", "example_euclidean", "example_aware")
TIME_POLICIES = ("fixed_4", "fixed_16", "fixed_64", "time_euclidean", "time_aware")


def step(model, optimizer, criterion, x, y, batch_size, noise_splits):
    """One update; with ``noise_splits`` it accumulates and returns a probe."""
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    probe = None
    if noise_splits:
        noise = GradientNoiseAccumulator(model.parameters(), noise_splits, optimizer=optimizer)
        for chunk in tree_split((x, y), noise_splits):
            (criterion(model, chunk) / noise_splits).backward()
            noise.record()
        probe = noise.finish(batch_size)
    else:
        criterion(model, (x, y)).backward()
    update(optimizer, batch_size)
    return probe, time.perf_counter() - started


def timing_pass(model, optimizer, criterion, x, y, repeats=3):
    """Median seconds per unmeasured step for each size, on throwaway copies."""
    model, optimizer_copy = copy.deepcopy((model, optimizer))
    out = {}
    for batch in SIZES:
        seconds = sorted(step(model, optimizer_copy, criterion, x[:batch], y[:batch], batch, 0)[1]
                         for _ in range(repeats))
        out[batch] = seconds[len(seconds) // 2]
    return out


def run(seed, task, kind, policy, budget, time_budget, measure_every, splits,
        lr_rule="none", reference_batch=16):
    (x_pre, x_post, y_pre, y_post, xv_pre, xv_post, yv_pre, yv_post,
     criterion, model_task) = task_data(seed, task)
    model = build_model(seed, model_task)
    base_lr = .1 if kind == "snr_muon" else .003
    optimizer = optimizer_for(model, kind, base_lr, len(x_pre))
    timed = time_budget is not None
    # The same example-indexed draws as the earlier rollout study.
    count = budget if not timed else 200_000
    indices = torch.randint(len(x_pre), (count,), generator=torch.Generator().manual_seed(seed + 90000))
    elapsed = 0.
    controller = None
    if policy.startswith(("example_", "time_")):
        objective, sensor = policy.split("_")
        sensor = ("muon" if kind == "snr_muon" else "adamw") if sensor == "aware" else sensor
        times = None
        if objective == "time":
            started = time.perf_counter()
            times = StepTimeModel(timing_pass(model, optimizer, criterion, x_pre, y_pre), ema=0.8)
            elapsed += time.perf_counter() - started
        controller = CostAwareBatchController(
            SIZES, initial=16, sensor=sensor, step_times=times,
            time_price=1. if objective == "time" else 0., example_price=1. if objective == "example" else 0.,
            ema=0.8, warmup=2, dwell=2, deadband=0.05)
    next_batch = int(policy.split("_")[1]) if policy.startswith("fixed_") else controller.current
    consumed = steps = 0
    has_shift = task != "digits"
    rows = []
    while (elapsed < time_budget) if timed else (consumed < budget):
        batch_size = next_batch
        if not timed:
            remaining = budget - consumed
            if has_shift and consumed < budget // 2:
                remaining = min(remaining, budget // 2 - consumed)
            batch_size = min(batch_size, remaining)
        shifted = has_shift and consumed >= budget // 2
        x, y = (x_post, y_post) if shifted else (x_pre, y_pre)
        draw = indices[consumed:consumed + batch_size]
        for group in optimizer.param_groups:
            # The same coupling for every arm; "none" reproduces the earlier study.
            group["lr"] = coupled_lr(base_lr, batch_size, reference_batch, lr_rule)
        measure = controller is not None and steps % measure_every == 0 and batch_size % splits == 0
        probe, seconds = step(model, optimizer, criterion, x[draw], y[draw], batch_size,
                              splits if measure else 0)
        elapsed += seconds
        consumed += batch_size
        steps += 1
        reason = scale = None
        if controller is not None:
            if controller.step_times is not None and not measure:
                controller.step_times.record(batch_size, seconds)
            if measure:
                started = time.perf_counter()
                controller.observe(probe)
                decision = controller.recommend()
                elapsed += time.perf_counter() - started
                next_batch, reason = decision.batch_size, decision.reason
                scale = decision.estimated_scale
        active_post = has_shift and consumed >= budget // 2
        rows.append(dict(seed=seed, task=task, optimizer=kind, policy=policy, lr_rule=lr_rule,
                         budget="time" if timed else "examples", samples=consumed, steps=steps,
                         actual_batch=batch_size, next_batch=next_batch, reason=reason,
                         noise_scale=scale,
                         seconds=elapsed,
                         validation_final=evaluate(model, (xv_post, yv_post, yv_post), False, criterion),
                         validation_current=evaluate(
                             model, ((xv_post, yv_post, yv_post) if active_post else (xv_pre, yv_pre, yv_pre)),
                             False, criterion)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(11, 16)))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--optimizers", nargs="+", choices=("snr_muon", "adamw"),
                        default=("snr_muon", "adamw"))
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--time-budget", type=float, default=0.5,
                        help="CPU seconds for the time-priced comparison on stationary digits.")
    parser.add_argument("--measure-every", type=int, default=5)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--lr-rule", choices=("none", "sqrt", "linear"), default="none",
                        help="Learning-rate coupling for every arm, relative to --reference-batch.")
    parser.add_argument("--reference-batch", type=int, default=16)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/benchmark_batch_cost_aware.jsonl.gz"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf8") as handle:
        handle.write(json.dumps(dict(kind="config", config={k: str(v) for k, v in vars(args).items()})) + "\n")
        jobs = [(kind, task, seed, policy, None) for kind in args.optimizers for task in args.tasks
                for seed in args.seeds for policy in EXAMPLE_POLICIES]
        if "digits" in args.tasks:
            jobs += [(kind, "digits", seed, policy, args.time_budget) for kind in args.optimizers
                     for seed in args.seeds for policy in TIME_POLICIES]
        # A discarded run per optimizer absorbs one-off start-up cost (the
        # first step in a process can take over a second on CPU).
        for kind in args.optimizers:
            run(args.seeds[0], "digits", kind, "time_euclidean", args.budget, 0.2,
                args.measure_every, args.splits, args.lr_rule, args.reference_batch)
        # Interleave arms randomly to spread CPU-timing drift across policies.
        order = torch.randperm(len(jobs), generator=torch.Generator().manual_seed(80000)).tolist()
        for i in order:
            kind, task, seed, policy, time_budget = jobs[i]
            rows = run(seed, task, kind, policy, args.budget, time_budget, args.measure_every,
                       args.splits, args.lr_rule, args.reference_batch)
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(kind, task, seed, policy, "time" if time_budget else "examples",
                  "loss", round(rows[-1]["validation_final"], 4), "seconds", round(rows[-1]["seconds"], 3),
                  "mean batch", round(rows[-1]["samples"] / rows[-1]["steps"], 1), flush=True)


if __name__ == "__main__":
    main()
