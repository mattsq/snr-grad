"""Cross-seed test of noise sensors and explicit batch-resource objectives.

Run on the PR branch: python benchmark_batch_budget.py --output benchmarks/batch_budget.json
The fitted policy is evaluated offline at held-out checkpoints; its predicted
actions are not rolled out into new training trajectories.
"""

import argparse
import copy
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_batch_control import (build_model, data, digit_loss, evaluate,
                                     optimizer_for, update)
from snr_grad import probe_batch


def collect(seed, source, kind, sizes, checkpoints, horizon, repetitions):
    train, validation = data(seed, "digits_shift")
    model = build_model(seed, "digits_shift")
    lr = .1 if kind == "snr_muon" else .003
    optimizer = optimizer_for(model, kind, lr, len(train[0]))
    consumed, step = 0, 0
    rows = []
    for checkpoint in checkpoints:
        while consumed < checkpoint:
            idx = torch.randint(len(train[0]), (source,), generator=
                                torch.Generator().manual_seed(seed + 20000 + step))
            target = train[2] if consumed >= 1500 else train[1]
            optimizer.zero_grad(set_to_none=True)
            digit_loss(model, (train[0][idx], target[idx])).backward()
            update(optimizer, source)
            consumed += source
            step += 1
        shifted = consumed >= 1500
        target = train[2] if shifted else train[1]
        baseline = evaluate(model, validation, shifted, digit_loss)
        ids = torch.randperm(len(train[0]), generator=
                             torch.Generator().manual_seed(seed + 50000 + checkpoint))[:64]
        probe = probe_batch(model, digit_loss, (train[0][ids], target[ids]),
                            splits=8, optimizer=optimizer)
        state = copy.deepcopy(model.state_dict())
        opt_state = copy.deepcopy(optimizer.state_dict())
        for repeat in range(repetitions):
            # Prefix pairing makes each larger candidate include the small
            # candidate's draw at every step, while preserving independent reps.
            draws = [torch.randint(len(train[0]), (max(sizes),), generator=
                                   torch.Generator().manual_seed(
                                       900000 + seed * 100000 + checkpoint * 100 + repeat * horizon + t))
                     for t in range(horizon)]
            for batch in sizes:
                branch = build_model(seed, "digits_shift")
                branch.load_state_dict(state)
                branch_opt = optimizer_for(branch, kind, lr, len(train[0]))
                branch_opt.load_state_dict(copy.deepcopy(opt_state))
                start = time.perf_counter()
                for draw in draws:
                    branch_opt.zero_grad(set_to_none=True)
                    digit_loss(branch, (train[0][draw[:batch]], target[draw[:batch]])).backward()
                    update(branch_opt, batch)
                seconds = time.perf_counter() - start
                gain = baseline - evaluate(branch, validation, shifted, digit_loss)
                rows.append(dict(seed=seed, source=source, checkpoint=checkpoint,
                                 samples=consumed, shifted=shifted, optimizer=kind,
                                 repeat=repeat, batch=batch, horizon=horizon,
                                 gain=gain, train_seconds=seconds,
                                 probe_seconds=probe.elapsed_seconds,
                                 euclidean=probe.euclidean.scale,
                                 aware=probe.muon.scale if kind == "snr_muon" else probe.adamw.scale))
    return rows


def curves(rows, probe_interval):
    grouped = {}
    for row in rows:
        key = row["optimizer"], row["seed"], row["source"], row["checkpoint"]
        grouped.setdefault(key, []).append(row)
    result = []
    for (kind, seed, source, checkpoint), observations in grouped.items():
        first = observations[0]
        options = {}
        for batch in sorted({r["batch"] for r in observations}):
            subset = [r for r in observations if r["batch"] == batch]
            gain = float(np.mean([r["gain"] for r in subset]))
            seconds = float(np.median([r["train_seconds"] for r in subset]))
            # Spread a probe over its configured interval, not every local
            # continuation. No probe cost is charged to a fixed policy.
            charged = first["probe_seconds"] * first["horizon"] / probe_interval
            probe_examples = 64 * first["horizon"] / probe_interval
            options[batch] = dict(gain=gain, seconds=seconds,
                                  example=gain / (first["horizon"] * batch + probe_examples),
                                  fixed_example=gain / (first["horizon"] * batch),
                                  time=gain / (seconds + charged),
                                  fixed_time=gain / seconds)
        result.append(dict(optimizer=kind, seed=seed, source=source,
                           checkpoint=checkpoint, samples=first["samples"],
                           shifted=first["shifted"], euclidean=first["euclidean"],
                           aware=first["aware"], options=options,
                           probe_seconds=first["probe_seconds"]))
    return result


def choose(training, query, sensor, objective, neighbors=5):
    """Predict gain per declared resource from nearby development probes."""
    scale = query[sensor]
    eligible = [r for r in training if r["optimizer"] == query["optimizer"]
                and math.isfinite(r[sensor]) and r[sensor] > 0]
    if not math.isfinite(scale) or scale <= 0 or not eligible:
        return None
    nearest = sorted(eligible, key=lambda r: abs(math.log(r[sensor] / scale)))[:neighbors]
    sizes = sorted(query["options"])
    scores = {b: np.mean([r["options"][b][objective] for r in nearest]) for b in sizes}
    return max(sizes, key=lambda b: (scores[b], -b))


def evaluate_predictions(all_curves, development_seeds, objective, charge_probes=True):
    dev = [r for r in all_curves if r["seed"] in development_seeds]
    held = [r for r in all_curves if r["seed"] not in development_seeds]
    result = []
    for query in held:
        pool = [r for r in dev if r["optimizer"] == query["optimizer"]]
        sizes = sorted(query["options"])
        # Fixed batches incur no probe. Charge the adaptive predictions for
        # both probe examples and elapsed probe time in the matching objective.
        fixed_key = "fixed_" + objective
        sensor_key = objective if charge_probes else fixed_key
        constant = max(sizes, key=lambda b: (np.mean([r["options"][b][fixed_key] for r in pool]), -b))
        oracle = max(sizes, key=lambda b: (query["options"][b][fixed_key], -b))
        for sensor in ("constant", "euclidean", "aware"):
            predicted = constant if sensor == "constant" else choose(dev, query, sensor, sensor_key)
            if predicted is None:
                predicted = constant
            scores = query["options"]
            score = scores[predicted][fixed_key if sensor == "constant" else sensor_key]
            result.append(dict(optimizer=query["optimizer"], seed=query["seed"],
                               source=query["source"], checkpoint=query["checkpoint"],
                               shifted=query["shifted"],
                               objective=objective if charge_probes else objective + "_no_probe",
                               sensor=sensor, predicted=predicted, oracle=oracle,
                               score=score,
                               oracle_score=scores[oracle][fixed_key],
                               regret=scores[oracle][fixed_key] - score,
                               constant=constant))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(7)))
    parser.add_argument("--development-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--sources", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[750, 2250])
    parser.add_argument("--optimizers", nargs="+", choices=["snr_muon", "adamw"],
                        default=["snr_muon", "adamw"])
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--probe-interval", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/batch_budget.json.gz"))
    args = parser.parse_args()
    if (not set(args.development_seeds) < set(args.seeds) or args.horizon < 1
            or args.repetitions < 1 or args.probe_interval < 1 or
            any(b <= 0 for b in args.sources + args.sizes)):
        parser.error("need distinct development and evaluation seeds and positive sizes")
    torch.set_num_threads(1)
    observations = []
    for kind in args.optimizers:
        for seed in args.seeds:
            for source in args.sources:
                observations.extend(collect(seed, source, kind, args.sizes,
                                            args.checkpoints, args.horizon, args.repetitions))
                print(kind, seed, source, "complete", flush=True)
    local_curves = curves(observations, args.probe_interval)
    predictions = [p for objective in ("example", "time") for charge in (True, False)
                   for p in evaluate_predictions(local_curves, args.development_seeds,
                                                 objective, charge_probes=charge)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args))
    config["output"] = str(args.output)
    opener = gzip.open if str(args.output).endswith(".gz") else open
    with opener(args.output, "wt", encoding="utf8") as handle:
        json.dump(dict(config=config, observations=observations, curves=local_curves,
                       predictions=predictions), handle)
        handle.write("\n")
    for kind in args.optimizers:
        for objective in ("example", "time", "example_no_probe", "time_no_probe"):
            for sensor in ("constant", "euclidean", "aware"):
                subset = [p for p in predictions if p["optimizer"] == kind and
                          p["objective"] == objective and p["sensor"] == sensor]
                print(kind, objective, sensor, "mean_regret",
                      round(float(np.mean([p["regret"] for p in subset])), 6),
                      "oracle_matches", sum(p["predicted"] == p["oracle"] for p in subset),
                      "/", len(subset))


if __name__ == "__main__":
    main()
