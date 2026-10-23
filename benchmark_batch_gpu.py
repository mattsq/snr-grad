"""Accelerator study of optimizer-aware, cost-aware batch control.

The CPU study in README.md could not show a benefit because an example cap
makes the smallest batch optimal, the probe used extra examples, and step time
on a tiny CPU model is mostly fixed overhead. This harness removes those
confounds: it trains a small byte-level transformer on an accelerator, measures
noise from the training step's own accumulation microbatches, prices steps by
measured device time, couples the learning rate to the batch, and scores every
policy at the same wall-clock budget. See
``docs/proposals/batch-control-gpu-study.md`` for the protocol.

Three modes share one model, data stream and optimizer setup:

``throughput``
    Seconds per optimizer step for each candidate batch. Its JSON output seeds
    the controllers' step-time model in the other modes.
``validate``
    Offline sensor validation. Train a fixed-batch reference, then at several
    checkpoints measure each sensor and run paired short continuations at
    every candidate batch. The analysis fits a critical batch to the measured
    per-step gains and asks which sensor predicts it and the time-optimal batch.
``rollout``
    Train each policy for the same device-time budget and record validation
    loss against time and tokens.

Run ``python benchmark_batch_gpu.py --help`` for options and
``analyze_batch_gpu.py`` on the JSONL output.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from snr_grad import (CostAwareBatchController, GradientNoiseAccumulator, SNRMuon, StepTimeModel,
                      coupled_lr)


# ---------------------------------------------------------------------------
# Data: byte-level text, one deterministic example stream per seed.
# ---------------------------------------------------------------------------

def load_bytes(path: str | None, synthetic: int, seed: int) -> torch.Tensor:
    if path:
        return torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.uint8).long()
    # A structured synthetic stream for smoke tests: a sparse random Markov
    # chain over 64 symbols, so the loss can fall well below log(64).
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(64, 64, generator=g) * 3
    probs = torch.softmax(logits, -1)
    out = torch.empty(synthetic, dtype=torch.long)
    state = 0
    draws = torch.rand(synthetic, generator=g)
    cdf = probs.cumsum(-1)
    for i in range(synthetic):
        state = int(torch.searchsorted(cdf[state], draws[i].clamp_max(cdf[state, -1] - 1e-6)))
        out[i] = state
    return out


class Stream:
    """Example ``i`` is the same window for every policy of a seed."""

    def __init__(self, tokens: torch.Tensor, seq_len: int, seed: int, chunk: int = 1 << 16):
        self.tokens, self.seq_len, self.seed, self.chunk = tokens, seq_len, seed, chunk
        self.cache: dict[int, torch.Tensor] = {}

    def starts(self, first: int, count: int) -> torch.Tensor:
        out = []
        while count:
            c, offset = divmod(first, self.chunk)
            if c not in self.cache:
                g = torch.Generator().manual_seed(self.seed * 1_000_003 + c)
                self.cache = {c: torch.randint(len(self.tokens) - self.seq_len - 1, (self.chunk,),
                                               generator=g)}
            take = min(count, self.chunk - offset)
            out.append(self.cache[c][offset:offset + take])
            first, count = first + take, count - take
        return torch.cat(out)

    def batch(self, first: int, count: int, device: torch.device):
        idx = self.starts(first, count)[:, None] + torch.arange(self.seq_len + 1)
        window = self.tokens[idx].to(device, non_blocking=True)
        return window[:, :-1], window[:, 1:]


def split_data(tokens: torch.Tensor, seq_len: int, valid_windows: int, seed: int, device):
    cut = int(len(tokens) * 0.9)
    train, valid = tokens[:cut], tokens[cut:]
    g = torch.Generator().manual_seed(seed + 777)
    starts = torch.randint(len(valid) - seq_len - 1, (valid_windows,), generator=g)
    window = valid[starts[:, None] + torch.arange(seq_len + 1)].to(device)
    return train, (window[:, :-1], window[:, 1:])


# ---------------------------------------------------------------------------
# Model: a small pre-norm GPT without dropout (dropout would add split noise).
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.up = nn.Linear(width, 4 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, c // self.heads).unbind(2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class GPT(nn.Module):
    def __init__(self, vocab: int, seq_len: int, width: int, depth: int, heads: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.position = nn.Parameter(torch.zeros(seq_len, width))
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, tokens):
        x = self.embed(tokens) + self.position[:tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def lm_loss(model, batch, dtype):
    x, y = batch
    with torch.autocast(x.device.type, dtype=dtype, enabled=dtype != torch.float32):
        logits = model(x)
    return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.reshape(-1))


@torch.no_grad()
def evaluate(model, valid, micro, dtype):
    x, y = valid
    losses = [lm_loss(model, (x[i:i + micro], y[i:i + micro]), dtype) * len(x[i:i + micro])
              for i in range(0, len(x), micro)]
    return float(torch.stack(losses).sum() / len(x))


def build(args, seed, device):
    torch.manual_seed(seed + 10_000)
    model = GPT(256, args.seq_len, args.width, args.depth, args.heads).to(device)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                      weight_decay=args.weight_decay,
                                      fused=device.type == "cuda")
    else:
        optimizer = SNRMuon(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    return model, optimizer


# ---------------------------------------------------------------------------
# One optimizer step with free noise measurement.
# ---------------------------------------------------------------------------

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def lr_at(args, batch, tokens_seen, time_fraction):
    lr = coupled_lr(args.lr, batch, args.reference_batch, args.lr_rule)
    if args.warmup_tokens:
        lr *= min(1., (tokens_seen + 1) / args.warmup_tokens)
    if args.schedule == "cosine":
        lr *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1., time_fraction)))
    return lr


def train_step(model, optimizer, stream, first, batch, args, device, dtype, lr, measure):
    """Return (probe or None, seconds). Every example trains the model."""
    micro = min(args.micro_batch, batch)
    if batch % micro:
        raise ValueError(f"Batch {batch} is not a multiple of micro-batch {micro}.")
    k = batch // micro
    for group in optimizer.param_groups:
        group["lr"] = lr
    sync(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    noise = (GradientNoiseAccumulator(model.parameters(), k, optimizer=optimizer,
                                      matrix_sensor=args.optimizer == "snr_muon")
             if measure and k >= 2 else None)
    for i in range(k):
        (lm_loss(model, stream.batch(first + i * micro, micro, device), dtype) / k).backward()
        if noise is not None:
            noise.record()
    probe = noise.finish(batch) if noise is not None else None
    optimizer.step()
    sync(device)
    return probe, time.perf_counter() - start


def sensor_values(probe, optimizer_name):
    if probe is None:
        return {}
    aware = probe.muon if optimizer_name == "snr_muon" else probe.adamw
    out = {"euclidean": probe.euclidean.scale, "aware": aware.scale if aware else math.inf,
           "microbatches": probe.microbatches, "probe_seconds": probe.elapsed_seconds}
    return {k: (v if isinstance(v, int) or math.isfinite(v) else None) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Modes.
# ---------------------------------------------------------------------------

def measure_throughput(args, device, dtype, tokens, seed):
    train, _ = split_data(tokens, args.seq_len, args.valid_windows, seed, device)
    stream = Stream(train, args.seq_len, seed + 5)
    out = {}
    for batch in args.sizes:
        model, optimizer = build(args, seed, device)
        seconds = []
        for step in range(args.throughput_warmup + args.throughput_steps):
            _, s = train_step(model, optimizer, stream, step * batch, batch, args, device, dtype,
                              args.lr, measure=False)
            if step >= args.throughput_warmup:
                seconds.append(s)
        out[batch] = sorted(seconds)[len(seconds) // 2]
        print(f"B={batch}: {out[batch] * 1e3:.2f} ms/step", flush=True)
    return out


def policy_batch(policy, controller, fraction, sizes):
    if policy.startswith("fixed:"):
        return int(policy.split(":")[1])
    if policy == "ramp":
        # Predetermined geometric ramp over the time budget, no sensor.
        index = min(len(sizes) - 1, int(fraction * len(sizes)))
        return sizes[index]
    return controller.current


def rollout(args, device, dtype, tokens, seed, policy, step_times, handle):
    train, valid = split_data(tokens, args.seq_len, args.valid_windows, seed, device)
    stream = Stream(train, args.seq_len, seed)
    model, optimizer = build(args, seed, device)
    controller = None
    if policy.startswith("controller:"):
        sensor = policy.split(":")[1]
        sensor = ("muon" if args.optimizer == "snr_muon" else "adamw") if sensor == "aware" else sensor
        prices = dict(time_price=1., example_price=0.) if args.objective == "time" else \
            dict(time_price=0., example_price=1.)
        controller = CostAwareBatchController(
            args.sizes, initial=args.initial_batch, sensor=sensor,
            step_times=StepTimeModel(step_times, ema=0.95) if args.objective == "time" else None,
            ema=args.controller_ema, warmup=args.controller_warmup, dwell=args.controller_dwell,
            deadband=args.deadband, max_probe_fraction=math.inf, **prices)
    measuring = controller is not None or args.measure_all
    elapsed = 0.
    examples = steps = 0
    next_eval = 0.
    while elapsed < args.time_budget and (not args.token_budget or
                                          examples * args.seq_len < args.token_budget):
        if elapsed >= next_eval:
            row = dict(kind="eval", seed=seed, optimizer=args.optimizer, policy=policy,
                       seconds=elapsed, steps=steps, examples=examples,
                       tokens=examples * args.seq_len,
                       validation=evaluate(model, valid, args.micro_batch, dtype))
            handle.write(json.dumps(row) + "\n")
            next_eval += args.eval_every
        batch = policy_batch(policy, controller, elapsed / args.time_budget, args.sizes)
        lr = lr_at(args, batch, examples * args.seq_len, elapsed / args.time_budget)
        measure = measuring and steps % args.measure_every == 0
        probe, seconds = train_step(model, optimizer, stream, examples, batch, args, device, dtype,
                                    lr, measure)
        elapsed += seconds
        examples += batch
        steps += 1
        decision = None
        if controller is not None:
            if controller.step_times is not None and not measure:
                # Measured steps carry the probe overhead, which does not
                # depend on the chosen batch; keep it out of the price.
                controller.step_times.record(batch, seconds)
            if measure:
                # Unmeasured steps neither observe nor decide, so warmup and
                # dwell count measurements, not optimizer steps.
                start = time.perf_counter()
                controller.observe(probe)
                decision = controller.recommend()
                elapsed += time.perf_counter() - start
        if steps % args.log_every == 0 or (decision and decision.reason in ("increase", "decrease")):
            handle.write(json.dumps(dict(
                kind="step", seed=seed, optimizer=args.optimizer, policy=policy, steps=steps,
                seconds=elapsed, examples=examples, batch=batch, lr=lr, step_seconds=seconds,
                reason=decision.reason if decision else None,
                next_batch=decision.batch_size if decision else None,
                **sensor_values(probe, args.optimizer))) + "\n")
    handle.write(json.dumps(dict(kind="eval", final=True, seed=seed, optimizer=args.optimizer,
                                 policy=policy, seconds=elapsed, steps=steps, examples=examples,
                                 tokens=examples * args.seq_len,
                                 validation=evaluate(model, valid, args.micro_batch, dtype))) + "\n")
    handle.flush()


def validate(args, device, dtype, tokens, seed, handle):
    train, valid = split_data(tokens, args.seq_len, args.valid_windows, seed, device)
    stream = Stream(train, args.seq_len, seed)
    model, optimizer = build(args, seed, device)
    checkpoints = sorted(args.checkpoints)
    elapsed = 0.
    examples = 0
    # Fresh continuation data lives far beyond the reference run's prefix.
    fresh = 1 << 40
    for fraction in checkpoints:
        while elapsed < fraction * args.time_budget:
            lr = lr_at(args, args.reference_batch, examples * args.seq_len, elapsed / args.time_budget)
            _, s = train_step(model, optimizer, stream, examples, args.reference_batch, args,
                              device, dtype, lr, False)
            elapsed += s
            examples += args.reference_batch
        state = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
        base = evaluate(model, valid, args.micro_batch, dtype)
        # Sensors from a large, fixed probe batch at the checkpoint (no step).
        sensors = []
        for rep in range(args.repetitions):
            model.load_state_dict(state[0])
            optimizer.load_state_dict(state[1])
            k = args.probe_batch // args.micro_batch
            optimizer.zero_grad(set_to_none=True)
            noise = GradientNoiseAccumulator(model.parameters(), k, optimizer=optimizer,
                                             matrix_sensor=args.optimizer == "snr_muon")
            first = fresh + rep * args.probe_batch
            for i in range(k):
                (lm_loss(model, stream.batch(first + i * args.micro_batch, args.micro_batch, device),
                         dtype) / k).backward()
                noise.record()
            sensors.append(sensor_values(noise.finish(args.probe_batch), args.optimizer))
        # Paired continuations: identical start, same draw prefix within a repetition.
        for rep in range(args.repetitions):
            first = fresh * 2 + rep * (1 << 30)
            for batch in args.sizes:
                model.load_state_dict(state[0])
                optimizer.load_state_dict(state[1])
                seconds = 0.
                for step in range(args.continuation_steps):
                    lr = lr_at(args, batch, examples * args.seq_len, elapsed / args.time_budget)
                    _, s = train_step(model, optimizer, stream, first + step * batch, batch, args,
                                      device, dtype, lr, False)
                    seconds += s
                handle.write(json.dumps(dict(
                    kind="continuation", seed=seed, optimizer=args.optimizer, fraction=fraction,
                    repetition=rep, batch=batch, steps=args.continuation_steps, seconds=seconds,
                    start_loss=base, end_loss=evaluate(model, valid, args.micro_batch, dtype))) + "\n")
        handle.write(json.dumps(dict(kind="checkpoint", seed=seed, optimizer=args.optimizer,
                                     fraction=fraction, loss=base, sensors=sensors,
                                     probe_batch=args.probe_batch,
                                     microbatches=args.probe_batch // args.micro_batch)) + "\n")
        handle.flush()
        model.load_state_dict(state[0])
        optimizer.load_state_dict(state[1])
        print(f"seed {seed} checkpoint {fraction:.2f} loss {base:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("throughput", "validate", "rollout"))
    parser.add_argument("--data", help="Text file read as bytes. Omit for a synthetic stream.")
    parser.add_argument("--synthetic", type=int, default=200_000,
                        help="Synthetic tokens when --data is omitted (smoke tests only).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--optimizer", choices=("adamw", "snr_muon"), default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate at --reference-batch.")
    parser.add_argument("--lr-rule", choices=("none", "sqrt", "linear"), default="sqrt")
    parser.add_argument("--schedule", choices=("constant", "cosine"), default="cosine",
                        help="Cosine decays by fraction of the time budget, identically for all policies.")
    parser.add_argument("--warmup-tokens", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--micro-batch", type=int, default=16,
                        help="Sequences per accumulation pass; the noise probe needs batch >= 2x this.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--reference-batch", type=int, default=64)
    parser.add_argument("--initial-batch", type=int, default=32)
    parser.add_argument("--policies", nargs="+", default=None,
                        help="fixed:B, ramp, controller:euclidean, controller:aware. "
                             "Default: every fixed size, ramp, both controllers.")
    parser.add_argument("--objective", choices=("time", "tokens"), default="time")
    parser.add_argument("--time-budget", type=float, default=600., help="Training seconds per run.")
    parser.add_argument("--token-budget", type=int, default=0, help="Optional additional token cap.")
    parser.add_argument("--eval-every", type=float, default=15., help="Training seconds between evals.")
    parser.add_argument("--valid-windows", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--controller-ema", type=float, default=0.95)
    parser.add_argument("--controller-warmup", type=int, default=5,
                        help="Measurements before the first decision.")
    parser.add_argument("--controller-dwell", type=int, default=5,
                        help="Measurements between batch changes.")
    parser.add_argument("--measure-every", type=int, default=10,
                        help="Record noise on every Nth step; the spectral sensor's "
                             "eigendecompositions are the main overhead.")
    parser.add_argument("--deadband", type=float, default=0.05)
    parser.add_argument("--measure-all", action="store_true",
                        help="Also record sensors on fixed policies (adds their overhead).")
    parser.add_argument("--step-times", type=Path, help="JSON from throughput mode.")
    parser.add_argument("--throughput-steps", type=int, default=20)
    parser.add_argument("--throughput-warmup", type=int, default=5)
    parser.add_argument("--checkpoints", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--continuation-steps", type=int, default=20)
    parser.add_argument("--probe-batch", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.sizes = sorted(set(args.sizes))
    if min(args.sizes) < 2 * args.micro_batch or any(b % args.micro_batch for b in args.sizes):
        parser.error("Every size must be a multiple of --micro-batch and at least twice it.")
    if args.initial_batch not in args.sizes:
        parser.error("--initial-batch must be one of --sizes.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tokens = load_bytes(args.data, args.synthetic, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "throughput":
        result = measure_throughput(args, device, dtype, tokens, args.seeds[0])
        args.output.write_text(json.dumps(dict(
            config={k: str(v) for k, v in vars(args).items()},
            device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            seconds={str(b): s for b, s in result.items()}), indent=2) + "\n")
        return

    step_times = None
    if args.step_times:
        measured = json.loads(args.step_times.read_text())
        step_times = {int(b): s for b, s in measured["seconds"].items()}
        for key in ("optimizer", "micro_batch", "seq_len", "width", "depth", "heads", "dtype"):
            if measured["config"][key] != str(getattr(args, key)):
                parser.error(f"--step-times was measured with a different {key}.")
        if set(step_times) != set(args.sizes):
            parser.error("--step-times must cover exactly the candidate --sizes.")
    policies = args.policies or ([f"fixed:{b}" for b in args.sizes] +
                                 ["ramp", "controller:euclidean", "controller:aware"])
    if args.mode == "rollout" and args.objective == "time" and step_times is None and any(
            p.startswith("controller:") for p in policies):
        # A one-off per hardware and model; it is logged, not charged.
        print("No --step-times given; measuring throughput first.", flush=True)
        started = time.perf_counter()
        step_times = measure_throughput(args, device, dtype, tokens, args.seeds[0])
        print(f"throughput pass took {time.perf_counter() - started:.1f}s", flush=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "wt", encoding="utf8") as handle:
        handle.write(json.dumps(dict(kind="config", mode=args.mode, step_times=step_times,
                                     config={k: str(v) for k, v in vars(args).items()})) + "\n")
        for seed in args.seeds:
            if args.mode == "validate":
                validate(args, device, dtype, tokens, seed, handle)
                continue
            # Randomize arm order within a seed to spread device-timing drift.
            order = torch.randperm(len(policies), generator=torch.Generator().manual_seed(seed + 80_000))
            for i in order.tolist():
                rollout(args, device, dtype, tokens, seed, policies[i], step_times, handle)
                print(f"seed {seed} {policies[i]} done", flush=True)


if __name__ == "__main__":
    main()
