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
the singular values of the centered subbatch gradient matrix to compute
`(tr(C_row**0.5) / ||mean_gradient||_*)**2`; it never reads the SVD gates.
The probe does not change parameter gradients or optimizer state. Training
BatchNorm and Dropout are rejected because they confound the variance estimate.

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
