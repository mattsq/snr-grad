"""Small paired batch-control experiment; writes JSONL observations.

Example: python benchmark_batch_control.py --steps 80 --seeds 3 --output batch-control.jsonl
The synthetic tasks are diagnostic, not evidence that a controller generalizes.
"""

import argparse
import copy
import gzip
import json
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from snr_grad import BatchController, SNRAdamW, SNRMuon, probe_batch


def loss_fn(model, batch):
    x, y = batch
    return (model(x) - y).square().mean()


def digit_loss(model, batch):
    x, y = batch
    return F.cross_entropy(model(x), y)


def data(seed, task, n=2048, d=12):
    if task.startswith("digits"):
        # A fixed, real classification dataset with an untouched validation split.
        # scikit-learn is needed only to run this optional benchmark.
        from sklearn.datasets import load_digits
        digits = load_digits()
        rng = torch.Generator().manual_seed(seed)
        x = torch.tensor(digits.data, dtype=torch.float32) / 16.
        y = torch.tensor(digits.target, dtype=torch.long)
        shifted_y = (torch.where(y == 0, 1, torch.where(y == 1, 0, y))
                     if task == "digits_partial_shift" else (y + 3) % 10)
        permutation = torch.randperm(len(x), generator=rng)
        train_ids, val_ids = permutation[:1437], permutation[1437:]
        return (x[train_ids], y[train_ids], shifted_y[train_ids]), (
            x[val_ids], y[val_ids], shifted_y[val_ids])
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=rng)
    w = torch.randn(d, 1, generator=rng)
    y = x @ w + .6 * torch.randn(n, 1, generator=rng)
    shifted = x @ torch.roll(w, 3, 0) + .6 * torch.randn(n, 1, generator=rng)
    xv = torch.randn(1024, d, generator=rng)
    yv = xv @ (torch.roll(w, 3, 0) if task == "shift" else w)
    return (x, y, shifted), (xv, xv @ w, xv @ torch.roll(w, 3, 0))


def build_model(seed, task):
    torch.manual_seed(seed + 10000)
    if task.startswith("digits"):
        return nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64),
                             nn.GELU(), nn.Linear(64, 10))
    if task == "matrix":
        return nn.Sequential(nn.Linear(12, 32), nn.Tanh(), nn.Linear(32, 1))
    return nn.Linear(12, 1)


def optimizer_for(model, kind, lr, n):
    if kind == "snr_adamw":
        return SNRAdamW(model.parameters(), lr=lr, alpha="finite", dataset_size=n,
                        track_stats=True)
    if kind == "snr_muon":
        return SNRMuon(model.parameters(), lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def update(optimizer, batch_size):
    if isinstance(optimizer, SNRAdamW):
        optimizer.step(batch_size=batch_size)
    else:
        optimizer.step()


def evaluate(model, validation, shifted=False, loss=loss_fn):
    with torch.no_grad():
        return float(loss(model, (validation[0], validation[2] if shifted else validation[1])))


def accuracy(model, validation, shifted=False):
    with torch.no_grad():
        labels = validation[2] if shifted else validation[1]
        return float((model(validation[0]).argmax(1) == labels).float().mean())


def gate_mean(optimizer):
    stats = getattr(optimizer, "last_stats", None)
    return None if stats is None else stats.mean_gate


def run(seed, task, kind, policy, *, sizes, steps, lr, probe_every, budget,
        coupled_lr=False, curve=False, probe_size=64, probe_splits=8,
        target_multiplier=1., continuation_steps=4, calibration_points="mid",
        reference_batch=None):
    train, validation = data(seed, task)
    model = build_model(seed, task)
    optimizer = optimizer_for(model, kind, lr, len(train[0]))
    criterion = digit_loss if task.startswith("digits") else loss_fn
    sensor = "euclidean" if policy == "euclidean" else ("muon" if kind == "snr_muon" else "adamw")
    controller = BatchController(sizes, initial=sizes[0], sensor=sensor,
                                 warmup=2, dwell=3, target_multiplier=target_multiplier,
                                 max_probe_fraction=15.) if policy in ("euclidean", "aware", "aware_shift_reset", "aware_alarm") else None
    probe_rng = torch.Generator().manual_seed(seed + 40000)
    consumed = 0
    shifting = task in ("shift", "digits_shift", "digits_partial_shift")
    shift_at = steps * sizes[0] // 2
    clock = 0.
    loss_ema = None
    last_alarm = -100000
    records = []
    checkpoints = ({steps // 4, steps // 2, 3 * steps // 4}
                   if calibration_points == "quarters" else {steps // 2})
    if shifting and calibration_points == "quarters":
        checkpoints.discard(steps // 2)
    for step in range(steps):
        if policy == "fixed_large":
            B = sizes[-1]
        elif policy == "fixed_mid":
            B = sizes[1]
        elif policy == "fixed_reference":
            B = sizes[min(2, len(sizes) - 1)] if reference_batch is None else reference_batch
        elif policy == "shift_reset":
            B = sizes[0] if shifting and consumed >= shift_at else (sizes[min(2, len(sizes) - 1)] if reference_batch is None else reference_batch)
        elif policy == "ramp":
            B = sizes[min(len(sizes) - 1, step * len(sizes) // steps)]
        elif controller:
            B = controller.current
        else:
            B = sizes[0]
        if budget is not None and consumed + B > budget:
            break
        optimizer.param_groups[0]["lr"] = lr * (math.sqrt(B / sizes[0]) if coupled_lr else 1.)
        # Same per-step draw prefix for every policy, even when batch sizes differ.
        sample_rng = torch.Generator().manual_seed(seed + 20000 + step)
        indices = torch.randint(len(train[0]), (B,), generator=sample_rng)
        target = train[2] if shifting and consumed >= shift_at else train[1]
        batch = train[0][indices], target[indices]
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        training_loss = criterion(model, batch)
        training_loss.backward()
        update(optimizer, B)
        train_seconds = time.perf_counter() - start
        consumed += B
        alarm = False
        if policy == "aware_alarm":
            observed_loss = float(training_loss.detach())
            alarm = (step >= 20 and step - last_alarm >= 80 and loss_ema is not None
                     and observed_loss > max(1., 4. * loss_ema))
            if alarm:
                last_alarm = step
                controller = BatchController(sizes, initial=sizes[0], sensor=sensor,
                                             warmup=2, dwell=3, target_multiplier=target_multiplier,
                                             max_probe_fraction=15.)
            loss_ema = observed_loss if loss_ema is None else .95 * loss_ema + .05 * observed_loss
        if policy == "aware_shift_reset" and shifting and consumed - B < shift_at <= consumed:
            # Diagnostic with oracle change-point knowledge: isolate slow
            # controller response from the sensor's intrinsic usefulness.
            controller = BatchController(sizes, initial=sizes[0], sensor=sensor,
                                         warmup=2, dwell=3, target_multiplier=target_multiplier,
                                         max_probe_fraction=15.)
        # The next probe and local continuations evaluate the post-step task.
        probe_target = train[2] if shifting and consumed >= shift_at else train[1]
        probe = None
        decision = None
        probe_examples = 0
        if controller:
            if (step + 1) % probe_every == 0:
                # Fresh disjoint examples at the same post-step parameters.
                pidx = torch.randperm(len(train[0]), generator=probe_rng)[:probe_size]
                probe_examples = len(pidx)
                pbatch = train[0][pidx], probe_target[pidx]
                try:
                    probe = probe_batch(model, criterion, pbatch, splits=probe_splits, optimizer=optimizer)
                except (ValueError, RuntimeError):
                    probe = None
                controller.observe(probe, step_seconds=train_seconds)
                decision = controller.recommend()
        clock += time.perf_counter() - start
        records.append(dict(seed=seed, task=task, optimizer=kind, policy=policy,
                            step=step + 1, samples=consumed, seconds=clock, actual_batch=B,
                            validation=evaluate(model, validation, shifting and consumed >= shift_at, criterion),
                            validation_accuracy=accuracy(model, validation, shifting and consumed >= shift_at)
                            if task.startswith("digits") else None,
                            shift_at=shift_at if shifting else None,
                            gate_mean=gate_mean(optimizer),
                            recommendation=None if decision is None else decision.batch_size,
                            reason=None if decision is None else decision.reason,
                            probe_examples=probe_examples,
                            probe_seconds=None if decision is None else decision.probe_seconds,
                            alarm=alarm,
                            euclidean=None if probe is None else probe.euclidean.scale,
                            adamw=None if probe is None else probe.adamw.scale,
                            muon=None if probe is None or probe.muon is None else probe.muon.scale,
                            muon_fallback=None if probe is None or probe.muon_fallback is None else probe.muon_fallback.scale))
        if curve and (step + 1) in checkpoints:
            # Paired continuations from the identical checkpoint, with fresh optimizer
            # state copies and the same random sample stream for each candidate.
            state = copy.deepcopy(model.state_dict())
            opt_state = copy.deepcopy(optimizer.state_dict())
            baseline = evaluate(model, validation, shifting and consumed >= shift_at, criterion)
            calibration_ids = torch.randperm(len(train[0]), generator=torch.Generator().manual_seed(seed + 50000))[:probe_size]
            calibration = probe_batch(model, criterion, (train[0][calibration_ids], probe_target[calibration_ids]),
                                      splits=probe_splits, optimizer=optimizer)
            for candidate in sizes:
                branch = build_model(seed, task)
                branch.load_state_dict(state)
                branch_optimizer = optimizer_for(branch, kind, lr, len(train[0]))
                branch_optimizer.load_state_dict(copy.deepcopy(opt_state))
                t0 = time.perf_counter()
                for local_step in range(continuation_steps):
                    local_rng = torch.Generator().manual_seed(seed + 30000 + step * 100 + local_step)
                    ids = torch.randint(len(train[0]), (candidate,), generator=local_rng)
                    branch_optimizer.zero_grad(set_to_none=True)
                    criterion(branch, (train[0][ids], probe_target[ids])).backward()
                    update(branch_optimizer, candidate)
                dt = time.perf_counter() - t0
                gain = baseline - evaluate(branch, validation, shifting and consumed >= shift_at, criterion)
                records.append(dict(seed=seed, task=task, optimizer=kind, policy=policy,
                                    kind="local_curve", checkpoint_step=step + 1, candidate=candidate,
                                    phase="after" if shifting and consumed >= shift_at else "before",
                                    euclidean=calibration.euclidean.scale, adamw=calibration.adamw.scale,
                                    muon=None if calibration.muon is None else calibration.muon.scale,
                                    improvement_per_step=gain / continuation_steps,
                                    improvement_per_sample=gain / (continuation_steps * candidate),
                                    improvement_per_second=gain / dt))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--reference-batch", type=int)
    parser.add_argument("--probe-every", type=int, default=5)
    parser.add_argument("--probe-size", type=int, default=64)
    parser.add_argument("--probe-splits", type=int, default=8)
    parser.add_argument("--target-multiplier", type=float, default=1.)
    parser.add_argument("--continuation-steps", type=int, default=4)
    parser.add_argument("--calibration-points", choices=("mid", "quarters"), default="mid")
    parser.add_argument("--tasks", nargs="+", choices=("stationary", "shift", "matrix", "digits", "digits_shift", "digits_partial_shift"),
                        default=("stationary", "shift", "matrix"))
    parser.add_argument("--policies", nargs="+", choices=("fixed_small", "fixed_mid", "fixed_reference", "fixed_large", "ramp", "euclidean", "aware", "shift_reset", "aware_shift_reset", "aware_alarm"),
                        default=("fixed_small", "fixed_reference", "fixed_large", "ramp", "euclidean", "aware"))
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--muon-lr", type=float)
    parser.add_argument("--adamw-lr", type=float)
    parser.add_argument("--snr-adamw-lr", type=float)
    parser.add_argument("--sample-budget", type=int)
    parser.add_argument("--coupled-lr", action="store_true")
    parser.add_argument("--output", default="batch-control.jsonl")
    args = parser.parse_args()
    if args.probe_every < 1 or args.steps < 2 or args.seeds < 1 or args.continuation_steps < 1 or args.probe_size < args.probe_splits * 2 or args.probe_size % args.probe_splits or (args.reference_batch is not None and args.reference_batch not in args.sizes):
        parser.error("steps, seeds and probe-every must be positive")
    opener = gzip.open if args.output.endswith(".gz") else open
    with opener(args.output, "wt", encoding="utf8") as output:
        for task in args.tasks:
            for kind in (("snr_muon", "adamw") if task in ("matrix", "digits", "digits_shift", "digits_partial_shift") else ("snr_adamw", "adamw")):
                kind_lr = {"snr_muon": args.muon_lr, "snr_adamw": args.snr_adamw_lr,
                           "adamw": args.adamw_lr}[kind]
                kind_lr = args.lr if kind_lr is None else kind_lr
                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    for policy in args.policies:
                        rows = run(seed, task, kind, policy, sizes=args.sizes, steps=args.steps,
                                   lr=kind_lr, probe_every=args.probe_every, budget=args.sample_budget,
                                   coupled_lr=args.coupled_lr, curve=policy == "fixed_small",
                                   probe_size=args.probe_size, probe_splits=args.probe_splits,
                                   target_multiplier=args.target_multiplier,
                                   continuation_steps=args.continuation_steps,
                                   calibration_points=args.calibration_points,
                                   reference_batch=args.reference_batch)
                        for row in rows:
                            output.write(json.dumps(row) + "\n")
                        print(task, kind, seed, policy, rows[-1].get("validation"))


if __name__ == "__main__":
    main()
