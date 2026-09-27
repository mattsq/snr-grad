# Proposal: optimizer-aware batch control

## Question

Can the gradient statistics already used by `snr-grad` tell us when another sample is worth its cost? The existing gates decide which **update directions** to trust. This experiment adds a separate decision about **how many samples** to use for the next update. It should not equate the mean coordinate gate value with a critical batch size.

## Measurement

Start with two or more disjoint microbatches evaluated at the same parameters, before the optimizer step. Record the aggregate mean gradient and between-microbatch variability. Normalize every noise estimate to a **per-example** variance scale before comparing candidate batch sizes: the variance of a batch mean then falls approximately as `1/B`. The existing `backward_with_microbatch_variance` and exact probe provide useful plumbing, but their returned `s` is already on the *current batch-mean* scale. Reuse their raw subbatch statistics or recover the per-example scale explicitly; do not feed `s` unchanged into a batch-size formula.

Compare three sensors, computed from the same probe:

1. **Euclidean:** a conventional aggregate gradient-noise-scale estimate.
2. **AdamW-aware:** apply a frozen, detached Adam preconditioner to both the mean and microbatch deviations before measuring signal and noise. This is a local proxy, since Adam's moments themselves change with batch size.
3. **Muon-aware:** measure matrix-gradient noise in the dual-norm geometry of spectral descent, with a separate treatment for the AdamW fallback tensors. Follow the non-Euclidean GNS derivation rather than substituting the SVD-coordinate gates in `SpectralSNRMuon`: directional gating and a scalar batch-control sensor answer different questions.

Log each sensor against a small, directly measured local batch-efficiency curve: from a checkpoint, run short paired continuations at several fixed `B` values and compare loss improvement per optimizer step, per training example, and per wall-clock second. This checks whether a sensor predicts the useful batch range before it controls anything.

## Controller and integration

Implement a training-loop-side `BatchController` with an `observe(probe)` / `recommend()` interface. The data loader and gradient-accumulation loop apply its recommendation; the optimizer does not silently change the batch. Start with a finite set of feasible effective batch sizes, EMA smoothing, a warmup, a deadband, a minimum dwell time, and a maximum one-rung change per decision. Hold the batch when the probe is missing, unstable, or too expensive. Log the recommendation, actual batch, probe cost, and reason for each change.

First test **batch-only** adaptation with a fixed learning-rate schedule. Then add `(B, lr)` as paired actions, compared against a predetermined batch/LR ramp. Learning rate should not be inferred mechanically from the noise scale: evaluate candidate coupling rules at equal sample and compute budgets. If `SNRAdamW(alpha="finite")` is used, pass the *actual* batch size at each step so `alpha = B/(N-B)` remains valid. Check whether changing `B` also changes gate density enough to confound an apparent controller gain; log gate statistics and include an ungated AdamW/Muon comparison.

## Small first experiment

Use one stationary task and one task with a distribution or signal-support shift, then a small matrix-heavy model for Muon. Compare fixed small/large batch, a predetermined batch ramp, Euclidean-sensor control, and optimizer-aware-sensor control, with matched starting checkpoints and seeds. Report validation loss against **samples, optimizer steps, and wall time**, plus probe overhead and controller trajectories. The proposal is supported if the optimizer-aware sensor predicts the measured useful batch range better and its controller improves the chosen cost/loss frontier across seeds; otherwise retain the diagnostics and drop the adaptive policy.

Related work: [Naganuma et al., *Adaptive Batch Sizes Using Non-Euclidean Gradient Noise Scales*](https://arxiv.org/abs/2602.03001) motivates optimizer-specific dual norms; [Merrill et al., *Critical Batch Size Revisited*](https://arxiv.org/abs/2505.23971) motivates checking a noise-scale proxy against measured batch efficiency, especially under Adam.

## Initial implementation

`snr_grad.batch_control.probe_batch` takes equal, disjoint microbatches and returns
per-example Euclidean, frozen AdamW, and nuclear-norm matrix noise scales, with
a separate L1 fallback for nonmatrix Muon parameters. The nuclear sensor uses
the singular values of centered matrix residuals concatenated along their
columns to estimate the square root of the row covariance and compute
`(tr(C_row**0.5) / ||mean_gradient||_*)**2`; it never reads the SVD gates.
The probe does not change parameter gradients or optimizer state. Training
BatchNorm and Dropout are rejected because they confound the variance estimate.
For Euclidean and frozen AdamW quadratic norms, the measured mean-gradient
square includes `noise / probe_batch_size`. Subtract this finite-probe bias
before forming the scale, clipping an unresolved signal to zero and holding
the current batch. The nuclear-norm matrix sensor remains a finite-split
heuristic; its scale depends substantially on the number of disjoint splits.

`BatchController.observe(probe)` and `recommend()` operate in the training loop.
The caller must use the recommendation on the *next* step, record the actual
size, and pass it to `SNRAdamW.step(batch_size=actual_size)` when using finite
alpha. The target multiplier is an empirical calibration parameter. With no
local batch-efficiency calibration, the initial value of 1 is illustrative.

Run `python benchmark_batch_control.py --steps 80 --seeds 3 --output batch-control.jsonl.gz`
for paired synthetic stationary, shifted, and matrix-heavy comparisons. The
compressed JSONL records validation loss against steps, examples, and elapsed training
time, probe overhead, gate mean where available, and local batch-efficiency
continuations from a shared mid-run checkpoint. `--sample-budget` caps consumed
training examples; `--coupled-lr` enables a separate square-root LR coupling
run to compare against the predetermined ramp under the same budget. Probe
examples and their cost are recorded separately from training examples. The
small synthetic study is a starting diagnostic; assess calibration and
cross-seed frontiers before interpreting policy gains.

`python plot_batch_control.py benchmarks/benchmark_batch_control.jsonl.gz` renders
the same three views as the other benchmarks: validation loss against steps,
samples, and measured training time; controller batch trajectories and probe
cost; and paired local-continuation efficiency alongside the raw sensor scales.
The plots expose both the chosen policy and the limits of its calibration.

## Held-out validation and outcome

`README.md` gives the exact digit benchmark and probe-resolution commands,
figures, and cross-seed results. This follow-up uses real handwritten digit
images with a held-out validation partition, a 1,500-example synthetic label
permutation, five held-out seeds, a 3,000-training-example cap, and fixed
`B=4`, `B=16`, `B=128`, and preset-ramp controls. The learning rates and the
heuristic `0.2` multiplier were chosen on separate development seeds. Short
paired continuations at checkpoints on either side of the shift measure local
per-step, per-example, and per-second improvement. Probe examples are separate
from the training-example cap; probe wall time is included in policy time.

The result does not support deploying this adaptive policy as a default.
Stationary SNRMuon has nearly equal final validation cross-entropy at fixed
`B=16` (0.166) and optimizer-aware control (0.161), while fixed `B=16` takes
about 0.25 CPU seconds versus 0.55. Following the label permutation,
optimizer-aware SNRMuon ends at 0.250 versus 0.199 for fixed `B=4`; its
cross-entropy is worse in all five seeds. AdamW-aware control helps relative
to fixed `B=16` after the shift, but has no clear advantage over other adaptive
or small-batch controls once probe cost is considered. These results use fixed
learning rates and a small CPU model; they do not establish GPU throughput or
batch/LR coupling behavior.

Keeping the sensor diagnostics is useful. On the same 128 digit examples and
the same trained checkpoint, increasing the number of disjoint probe splits
from eight to 128 raises the median raw Muon scale from roughly 143 to 350
and median probe time from 10 ms to 127 ms across five seeds. No split count
is a population ground truth. Calibration must specify probe resolution,
compute cost, and a measured loss horizon before a noise scale can justify a
controller action. A subsequent controller study should tune its policy on
development tasks and require held-out gains against fixed and Euclidean
controls at matched example and wall-time budgets.

## Failure-mode follow-up

The follow-up in `README.md` and `benchmark_batch_local_replication.py`
separates three possible explanations for the first result. Eight paired
12-step continuations at each of two checkpoints and five seeds show that
larger batches reliably improve **gain per step**, whereas `B=4` or `B=8`
maximizes mean **gain per training example** at all five post-shift
checkpoints. The original controller was calibrated to an 80%-of-best
per-step knee, then judged under an example cap. Its sensor can be useful
for identifying diminishing returns per update and still choose the wrong
action for the evaluated resource. A cost-aware decision must directly
specify the relative prices of examples, steps, and device time.

`benchmark_batch_control.py` now supports `fixed_mid`, `shift_reset`,
`aware_shift_reset`, and `aware_alarm` as explicit diagnostic policies.
With an abrupt permutation of all digit labels, resetting from `B=16` to
`B=4` at the known change point reduces SNRMuon's final mean CE from
0.336 to 0.203. Resetting the aware controller reduces it from 0.250 to
0.219. A training-loss alarm reduces it to 0.204, near the fixed `B=4`
result of 0.199. Thus lag across a large change point explains part of
the gap, while small-batch sample efficiency explains the remaining
advantage of `B=4`. The oracle policies know the change time and do not
represent deployable performance.

The two-class label swap (0↔1) is a less extreme follow-up. Aware SNRMuon
and fixed `B=4` tie at roughly 0.173 and 0.172; the known-change reset
is worse at 0.209. The simple loss alarm has false positives even on the
stationary digit task, especially with AdamW. Its apparent gains on a
large shift are not evidence for reliable unsupervised shift detection.
The alarm and second task were designed after seeing the initial held-out
outcomes and require fresh, independently selected problems for
confirmatory inference. The figures `benchmark_batch_control_failure_modes.png`,
`benchmark_batch_control_partial_shift.png`, and
`benchmark_batch_local_replication.png` show sample-aligned losses,
controller paths, and the competing local efficiency objectives.

The corrected row-covariance implementation and a separate probe-cost-guard
ablation are reported in the README. Earlier raw Muon scales and policies
derived from them are superseded; their figures remain historical records of
the erroneous flattening. The permissive AdamW-aware controller chooses
larger batches but fares badly after the label change. On this CPU, the
probe-time guard often holds AdamW at its initial size. An exploratory sweep
of the corrected Muon multiplier improves shifted sample efficiency as it
shrinks, reaching the fixed-small result only when the controller never
leaves that initial batch.

The viable next question is whether optimizer-specific geometry predicts
**throughput-adjusted** local improvement on a larger model and device,
with probe time amortized and a controller calibrated for that exact cost.
These CPU experiments do not answer it; they rule out treating a raw
dual-norm noise scale as a generally useful batch action under an example
budget.
