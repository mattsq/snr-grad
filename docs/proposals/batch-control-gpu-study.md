# Accelerator study of cost-aware batch control

This protocol should be fixed before any accelerator result is seen. It tests
optimizer-aware batch control in a setting where a batch-size decision can
matter. The CPU study ([`../batch-control-cpu-results.md`](../batch-control-cpu-results.md))
could not answer the question, because of how it was set up.

## Question

On an accelerator, with wall-clock time as the budget, does a controller
that reads the **optimizer-aware** gradient noise scale reach lower validation
loss than:

- the best fixed batch,
- a predetermined batch ramp, and
- the same controller reading the **Euclidean** noise scale?

It is worth asking only if a prior condition holds: the aware sensor must
predict the measured critical batch better than the Euclidean sensor. The
study therefore has two stages, and stage 2 does not start unless stage 1
passes.

## What changed relative to the CPU study

| CPU study problem | Fix in this study |
| --- | --- |
| An example cap makes the smallest batch optimal | The primary budget is device-synchronized training time. There is also a token-budget sanity run |
| Probes spent 80% extra examples | `GradientNoiseAccumulator` reuses the training step's own accumulation microbatches. The measured overhead (about one eigendecomposition per matrix, every `--measure-every` steps) is charged to the controllers' clock |
| Step time was fixed overhead on a tiny CPU model | A 10.6M-parameter transformer on a GPU, where step time is flat up to some batch size and then grows |
| Fixed learning rate across batches | `coupled_lr` scales the rate with the batch. The same rule applies to every arm, fixed ones included |
| A heuristic multiplier fitted to one objective | `CostAwareBatchController` maximizes `[B / (B + B_noise)] / seconds(B)` from measured step times, with no fitted multiplier |
| Time policies scored at an example cap | Every arm is scored at the same time budget |

## Setup

- **Hardware:** one GPU. Record its name; `benchmark_batch_gpu.py throughput` does this. Do not share the device with other jobs while timing.
- **Data:** enwik8 (the first 100 MB of English Wikipedia), read as bytes. The vocabulary is 256, so no tokenizer is needed. The first 90% is used for training. Validation uses 512 fixed windows from the last 10%. Any other text file of at least about 50 MB can replace it with `--data`.
- **Model:** a pre-norm GPT with width 384, depth 6, 6 heads, context 256, and no dropout. These are the script defaults.
- **Precision:** bf16 autocast; float32 loss and statistics.
- **Batches:** `--sizes 32 64 128 256 512` sequences, with gradient accumulation over micro-batches of 16. Every arm uses the same accumulation structure. If `B=512` makes 32 accumulation passes unreasonably slow on the device, raise `--micro-batch` for all arms together.
- **Optimizers:** `adamw` (torch fused AdamW, betas 0.9/0.95, weight decay 0.1) and `snr_muon` (`SNRMuon`). The aware sensor is the frozen-AdamW geometry for AdamW and the nuclear-norm row-covariance geometry for SNRMuon matrices.
- **Schedule:** cosine decay by fraction of the time budget, with the same schedule for every arm. The warmup is 2M tokens.
- **Seeds:** tuning seeds 0–1 and evaluation seeds 10–14. Evaluation seeds are used only once the protocol is frozen.

## Stage 0: throughput and tuning (tuning seeds only)

1. Measure step time for each optimizer:
   `python benchmark_batch_gpu.py throughput --data enwik8 --optimizer adamw --output runs/tp_adamw.json`, and likewise for `snr_muon`.
   If step time is proportional to `B` across the whole grid, the device is already saturated at the smallest batch. In that case the study cannot show a benefit: stop and choose a smaller model or a larger grid.
2. On seeds 0–1 and at `--reference-batch 64`, tune the base learning rate over `{3e-4, 1e-3, 3e-3}` for AdamW and `{3e-3, 1e-2, 3e-2}` for SNRMuon, and the coupling rule over `{sqrt, linear}`. Use fixed-batch rollouts and pick the combination with the best mean final loss across all fixed sizes.
3. Choose `--time-budget` so that fixed `B=64` passes its loss plateau onset within the budget. A 600 s budget is the planned default; a different value must be chosen at this stage, not later.
4. Freeze everything in a config note committed before stage 1: learning rate, rule, budget, `--measure-every`, controller EMA, warmup, dwell and deadband.

## Stage 1: offline sensor validation (go / no-go)

```bash
python benchmark_batch_gpu.py validate --data enwik8 --optimizer adamw \
    --seeds 10 11 12 13 14 --lr <tuned> --lr-rule <tuned> --warmup-tokens 2000000 --time-budget <frozen> \
    --output runs/validate_adamw.jsonl.gz
python analyze_batch_gpu.py runs/validate_adamw.jsonl.gz runs/validate_snr_muon.jsonl.gz \
    --summary runs/validate_summary.json
```

The script trains the reference batch and stops at 10, 30, 50, 70 and 90% of the budget. At each checkpoint it:

1. measures both sensors on a 512-sequence probe, three times; and
2. runs three paired 20-step continuations at every candidate batch from identical model and optimizer states.

The analysis then:

- fits `gain_per_step(B) = G · B / (B + B_crit)` to the continuations, marking unidentifiable fits;
- computes the Spearman correlation between each sensor and `B_crit` across checkpoints; and
- computes the relative loss in gain per second when the cost-aware rule picks a batch from each sensor and the measured step times, compared with the best measured batch.

**Pass condition** (per optimizer, over 25 checkpoints):

- the aware sensor has a higher Spearman correlation with `B_crit` than the Euclidean sensor, and
- the aware sensor has a lower mean relative regret than the Euclidean sensor.

If the Euclidean sensor is at least as good on both, the optimizer-aware hypothesis is rejected for that optimizer, and stage 2 for it is reported only as a Euclidean-controller study.

Known limitation: the nuclear-norm sensor depends on the number of accumulation passes `K`. It is biased low at small `K`. The rollout controller sees `K = B / 16`, which varies from 2 to 32. To check this, repeat the validation probe with `--probe-batch` set to each candidate size and report the drift of the aware scale with `K`.

## Stage 2: time-budget rollouts

```bash
python benchmark_batch_gpu.py rollout --data enwik8 --optimizer adamw \
    --seeds 0 1 10 11 12 13 14 --step-times runs/tp_adamw.json \
    --lr <tuned> --lr-rule <tuned> --warmup-tokens 2000000 --time-budget <frozen> \
    --output runs/rollout_adamw.jsonl.gz
python analyze_batch_gpu.py runs/rollout_adamw.jsonl.gz runs/rollout_snr_muon.jsonl.gz \
    --tuning-seeds 0 1 --plot runs/rollout.png --summary runs/rollout_summary.json
```

Arms run in randomized order within each seed. The arms are:

- every fixed size;
- `ramp`, a geometric ramp from 32 to 512 over the time budget;
- `controller:euclidean`; and
- `controller:aware`.

The controllers start at `B=32`. Rows from seeds 0–1 are used only to choose the "tuned" fixed batch. The analysis also reports the post-hoc best fixed batch on the evaluation seeds, as an unattainable oracle.

**Primary endpoint:** validation loss at the time budget.
**Secondary endpoints:** validation loss against time (the curve), tokens processed, and the controller's batch path and decision reasons.

**Supported** (per optimizer) if, on the five evaluation seeds, `controller:aware` meets all three of these:

- it has lower final loss than the tuned fixed batch in at least 4 of 5 seeds, with a negative mean paired difference;
- it has a negative mean paired difference against `ramp`; and
- it has a mean paired difference against `controller:euclidean` that is no worse than zero.

Anything else is reported as not supported, whatever the direction of the post-hoc oracle comparison.

**Sanity check:** a rollout with `--objective tokens --initial-batch 128 --policies controller:euclidean controller:aware fixed:32` and a `--token-budget` must drive both controllers down to `B=32`, the smallest size, and match `fixed:32` up to measurement overhead. If it does not, the controller implementation is wrong and the other results are void.

## Cost estimate

Stage 2 is 8 arms × 7 seeds × 2 optimizers × 10 minutes, about 19 GPU-hours. Stage 1 is about 1 GPU-hour per optimizer. Stage 0 tuning is about 6 × 5 × 2 × 2 × 10 minutes, about 20 GPU-hours on the default grid. That is roughly 40 GPU-hours on one A100-class device. Halving the time budget halves it, if the plateau criterion in stage 0 still holds.

## Out of scope

- Multi-GPU data parallelism. `GradientNoiseAccumulator` measures per-process microbatches; under DDP it must record inside `no_sync`, or its statistics must be aggregated across ranks.
- Models large enough that the eigendecomposition of an `m × m` row covariance is costly. For those, use `--measure-every` or the Euclidean and AdamW sensors, which need only elementwise updates.
- Throughput from `torch.compile`, because variable batch shapes cause recompilation.
