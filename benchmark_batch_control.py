"""Small paired batch-control experiment; writes JSONL observations.

Example: python benchmark_batch_control.py --steps 80 --seeds 3 --output batch-control.jsonl
The synthetic tasks are diagnostic, not evidence that a controller generalizes.
"""

import argparse
import copy
import json
import math
import time

import torch
from torch import nn

from snr_grad import BatchController, SNRAdamW, SNRMuon, probe_batch


def loss_fn(model, batch):
    x, y = batch
    return (model(x) - y).square().mean()


def data(seed, task, n=2048, d=12):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=rng)
    w = torch.randn(d, 1, generator=rng)
    y = x @ w + .6 * torch.randn(n, 1, generator=rng)
    shifted = x @ torch.roll(w, 3, 0) + .6 * torch.randn(n, 1, generator=rng)
    xv = torch.randn(1024, d, generator=rng)
    yv = xv @ (torch.roll(w, 3, 0) if task == "shift" else w)
    return (x, y, shifted), (xv, yv)


def build_model(seed, task):
    torch.manual_seed(seed + 10000)
    if task == "matrix":
        return nn.Sequential(nn.Linear(12, 32), nn.Tanh(), nn.Linear(32, 1))
    return nn.Linear(12, 1)


def optimizer_for(model, kind, lr, n):
    if kind == "snr_adamw":
        return SNRAdamW(model.parameters(), lr=lr, alpha="finite", dataset_size=n)
    if kind == "snr_muon":
        return SNRMuon(model.parameters(), lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def update(optimizer, batch_size):
    if isinstance(optimizer, SNRAdamW):
        optimizer.step(batch_size=batch_size)
    else:
        optimizer.step()


def evaluate(model, validation):
    with torch.no_grad():
        return float(loss_fn(model, validation))


def gate_mean(optimizer):
    stats = getattr(optimizer, "last_stats", None)
    return None if stats is None else stats.mean_gate


def run(seed, task, kind, policy, *, sizes, steps, lr, probe_every, budget,
        coupled_lr=False, curve=False):
    train, validation = data(seed, task)
    model = build_model(seed, task)
    optimizer = optimizer_for(model, kind, lr, len(train[0]))
    sensor = "euclidean" if policy == "euclidean" else ("muon" if kind == "snr_muon" else "adamw")
    controller = BatchController(sizes, initial=sizes[0], sensor=sensor,
                                 warmup=2, dwell=3, target_multiplier=1.,
                                 max_probe_fraction=5.) if policy in ("euclidean", "aware") else None
    probe_rng = torch.Generator().manual_seed(seed + 40000)
    consumed = 0
    clock = 0.
    records = []
    for step in range(steps):
        if policy == "fixed_large":
            B = sizes[-1]
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
        target = train[2] if task == "shift" and consumed >= steps * sizes[0] // 2 else train[1]
        batch = train[0][indices], target[indices]
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_fn(model, batch).backward()
        update(optimizer, B)
        train_seconds = time.perf_counter() - start
        consumed += B
        probe = None
        decision = None
        probe_examples = 0
        if controller:
            if (step + 1) % probe_every == 0:
                # Fresh disjoint examples at the same post-step parameters.
                pidx = torch.randperm(len(train[0]), generator=probe_rng)[:sizes[0]]
                probe_examples = len(pidx)
                pbatch = train[0][pidx], target[pidx]
                try:
                    probe = probe_batch(model, loss_fn, pbatch, splits=4, optimizer=optimizer)
                except (ValueError, RuntimeError):
                    probe = None
                controller.observe(probe, step_seconds=train_seconds)
                decision = controller.recommend()
        clock += time.perf_counter() - start
        records.append(dict(seed=seed, task=task, optimizer=kind, policy=policy,
                            step=step + 1, samples=consumed, seconds=clock, actual_batch=B,
                            validation=evaluate(model, validation), gate_mean=gate_mean(optimizer),
                            recommendation=None if decision is None else decision.batch_size,
                            reason=None if decision is None else decision.reason,
                            probe_examples=probe_examples,
                            probe_seconds=None if decision is None else decision.probe_seconds,
                            euclidean=None if probe is None else probe.euclidean.scale,
                            adamw=None if probe is None else probe.adamw.scale,
                            muon=None if probe is None or probe.muon is None else probe.muon.scale,
                            muon_fallback=None if probe is None or probe.muon_fallback is None else probe.muon_fallback.scale))
        if curve and step == steps // 2:
            # Paired continuations from the identical checkpoint, with fresh optimizer
            # state copies and the same random sample stream for each candidate.
            state = copy.deepcopy(model.state_dict())
            opt_state = copy.deepcopy(optimizer.state_dict())
            baseline = evaluate(model, validation)
            for candidate in sizes:
                model.load_state_dict(state)
                optimizer.load_state_dict(opt_state)
                t0 = time.perf_counter()
                for local_step in range(4):
                    local_rng = torch.Generator().manual_seed(seed + 30000 + step * 4 + local_step)
                    ids = torch.randint(len(train[0]), (candidate,), generator=local_rng)
                    optimizer.zero_grad(set_to_none=True)
                    loss_fn(model, (train[0][ids], target[ids])).backward()
                    update(optimizer, candidate)
                dt = time.perf_counter() - t0
                gain = baseline - evaluate(model, validation)
                records.append(dict(seed=seed, task=task, optimizer=kind, policy=policy,
                                    kind="local_curve", checkpoint_step=step + 1, candidate=candidate,
                                    improvement_per_step=gain / 4, improvement_per_sample=gain / (4 * candidate),
                                    improvement_per_second=gain / dt))
            model.load_state_dict(state)
            optimizer.load_state_dict(opt_state)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--probe-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--sample-budget", type=int)
    parser.add_argument("--coupled-lr", action="store_true")
    parser.add_argument("--output", default="batch-control.jsonl")
    args = parser.parse_args()
    if args.probe_every < 1 or args.steps < 2 or args.seeds < 1:
        parser.error("steps, seeds and probe-every must be positive")
    with open(args.output, "w", encoding="utf8") as output:
        for task in ("stationary", "shift", "matrix"):
            for kind in (("snr_muon", "adamw") if task == "matrix" else ("snr_adamw", "adamw")):
                for seed in range(args.seeds):
                    for policy in ("fixed_small", "fixed_large", "ramp", "euclidean", "aware"):
                        rows = run(seed, task, kind, policy, sizes=args.sizes, steps=args.steps,
                                   lr=args.lr, probe_every=args.probe_every, budget=args.sample_budget,
                                   coupled_lr=args.coupled_lr, curve=policy == "fixed_small")
                        for row in rows:
                            output.write(json.dumps(row) + "\n")
                        print(task, kind, seed, policy, rows[-1].get("validation"))


if __name__ == "__main__":
    main()
