# Stability study: deeper tests of the curvature-aware stability sensor

A follow-up to [the proposal](../../docs/proposals/curvature-aware-stability-control.md)
and the first-round benchmarks (`benchmark_stability_plrf.py`,
`benchmark_stability_layers.py`). The first round left these questions open:

1. Does any noise-aware sensor add information beyond the sharpness baseline
   `lr * lambda_max(P H)`, which matched or beat the one-step sensors?
2. Is the gap a real limitation, or a quirk of a design where sharpness and noise vary
   together?
3. Why did the `R` standard errors cover the truth only 84% of the time?
4. Which simulator ingredients matter (noise model, horizon, seeds, probe size), and
   are its event *frequencies* calibrated, not just ranked?
5. Does SNR gating suppress spikes, delay them, or make them worse (proposal Stage 2)?
6. What happens with correlated data and with larger networks?

All scripts run from this directory with `uv run python <script> [--quick]` and write
JSON (and figures, via `plots.py`) to `results/`. Experiments E1-E5 use the PLRF problem
of `benchmark_stability_plrf.py` (D=256, V=4096, B=8 unless varied), where the Hessian
and gradient covariance are exact. Stability edges come from the first-round run.

## Summary

- **Noise-aware sensing matters, and it is decisive when noise varies at fixed
  sharpness (E1).** Varying only the batch size, the SGD sharpness is constant and
  cannot rank, while the noise penalty and the simulator reach Spearman 0.95-0.96.
  The overall picture is in E0 and E1 below.
- **Only the multi-step simulator survives Adam's coupling between noise and the
  preconditioner (E1).** For AdamW at the concentrated spectrum, spike severity *rises*
  with batch size (from 5.0 at B=2 to 8.4 at B=64). Every one-step sensor gets this
  wrong in at least one regime; the simulator, which replays Adam's second-moment
  dynamics, ranks correctly in all four (optimizer, spectrum) cells (0.66-0.96 with
  exact noise).
- **The simulator is well calibrated on PLRF (E3).** With true noise it reaches Spearman
  0.99 and an event-fraction error of 0.045. With noise resampled from 128 microbatches
  and loss-proportional scaling, it reaches 0.95-0.96, an error of 0.07-0.09 and no
  measurable bias. Loss-proportional scaling, enough microbatches (K >= 32), and 16 or
  more simulated seeds matter. Gaussian versus resampled noise matters less than the
  quick check suggested.
- **The probe's `R` is unbiased, but its jackknife SE is optimistic (E2).** The cause is
  right skew at small K, made worse by heavy-tailed noise. The new log-scale interval
  (`risk_trace_ci`) lifts coverage from 76-92% to 89-93%. The concentration estimate is
  usable from K=16, so the default minimum was relaxed.
- **SNR gating cuts both ways (E4).** At a moderately concentrated spectrum it removes
  most excursions *and* reaches a lower loss than AdamW at the same learning rate. At the most
  concentrated spectrum the default gate does not suppress spikes, and a slow variance
  EMA (`rho=0.999`) makes them rarer but enormous (maximum excursion about 5e4x). The
  soft gate nearly eliminates spikes there, at some cost in progress.
- **Correlated rows spread across microbatches make the probe blind to noise (E5).** It
  underestimates the variance by up to 100%. The library previously recommended exactly
  this (shuffling); it now provides `group_ids` to prevent it.
- **On networks, low-rank projected curvature is what works (E6).** The first run
  exposed a real pitfall: truncated curvature, including a GGN on fewer samples x
  outputs than parameters, makes the surrogate unbounded below, and rankings collapse
  or invert. With projection onto the top curvature subspace (now automatic for
  `LowRankCurvature`, with a warning otherwise), a rank 4-16 GGN ranks real severity at
  0.85-0.98 up to 284k parameters. The activation proxy fails at every width. A cost
  advantage over real continuations is still not demonstrated at these model sizes.

- **Scaling up (E7): cost stops being the issue at a few hundred tokens per step.** On a
  small Transformer LM, after optimising the simulator loop, the per-step speedup is
  about tokens-per-step / 200 at any width. A single what-if query pays off from about
  1k tokens per step (3.8-6.4x at 31k-430k parameters; 4.6-9.6x for 8 queries).
  - The simulator places the learning-rate instability onset within about 2x of reality
    (rank 64: geometric mean 0.85x, slightly conservative). Adam's edge-of-stability rule
    (threshold 3.8) never flags the unsafe rates in half the checkpoints and is otherwise
    about 3-4x too permissive.
  - It cannot predict how large a blow-up will be.
  - The noise-driven intermittent spikes the method targets barely occur at these sizes,
    so that part of the question needs larger models than this CPU budget allows.

- **When to expect a large spike effect (E8).** Closed-form indices predict PLRF spike
  severity almost perfectly: `Phi = kappa sqrt(2/D)` gives R^2 0.95 for SGD and AUROC
  0.97 for Adam. A spike window needs gradient noise aligned with the sharpest direction
  (`w >~ 0.1`).
  - Tiny Adam-trained LMs sit 30-300x below the spike band, and their noise is
    misaligned: Adam's whitening drops `w` by up to 270x (1.3-270x) below its SGD-geometry
    value.
  - Scaling trends do not close the gap, so look in SGD-trained, aligned settings rather
    than larger Adam LMs, and screen with `w` and `Phi` first.
  - Along the way: the E7 sharpness rule used the SGD threshold (2) instead of Adam's
    (3.8); corrected.

## E0 - incremental information beyond sharpness (`e0_incremental.py`)

Uses the first-round held-out checkpoints. It reports the partial Spearman correlation
with severity after controlling for `lr * lambda_max(P H)`, and the out-of-sample R^2 of
a log-severity regression fitted on dev seeds.

| group | sharpness R^2 | + exact-noise simulator | partial rho | + probe simulator | partial rho |
| --- | --- | --- | --- | --- | --- |
| pooled | 0.54 | 0.95 | +0.95 | 0.86 | +0.88 |
| SGD | 0.87 | 0.96 | +0.20 | 0.95 | +0.20 |
| AdamW | 0.97 | 0.98 | -0.05 | 0.98 | +0.12 |
| SNRAdamW | 0.64 | 0.92 | +0.92 | 0.86 | +0.79 |

Within plain SGD and AdamW, sharpness already explains almost everything in that design.
Across optimizers, and for the gated optimizer whose effective step depends on the gate,
the simulator adds a lot. The one-step ratios (`noise_to_loss`, `rel_sd`) add little or
nothing anywhere.

## E1 - decoupling sharpness from noise (`e1_decoupling.py`, `e1_breakdown.py`)

The grid is spectrum {0.5, 0.9} x {SGD, AdamW} x lr = 0.5 x edge x B {2, 4, 8, 16, 64} x
target exponent {0.5, 0.7, 1.0}, with 4 checkpoint seeds at 2 times each (576
checkpoints, 16 real continuations each). Each cell below is the mean within-stratum
Spearman with severity when **only B varies**:

| | sharpness | GNS (opt.) | noise penalty | noise/loss | probe simulator | exact simulator |
| --- | --- | --- | --- | --- | --- | --- |
| SGD, alpha 0.5 | n/a (constant) | -0.89 | **+0.96** | +0.52 | **+0.96** | **+0.96** |
| SGD, alpha 0.9 | n/a (constant) | -0.64 | **+0.95** | **+0.96** | **+0.94** | **+0.96** |
| AdamW, alpha 0.5 | -0.41 | -0.40 | +0.44 | -0.41 | +0.45 | +0.66 |
| AdamW, alpha 0.9 | +0.51 | -0.09 | -0.29 | +0.60 | +0.53 | +0.72 |

Mean severity by batch size (target exponent 0.7):

| | B=2 | 4 | 8 | 16 | 64 |
| --- | --- | --- | --- | --- | --- |
| SGD, alpha 0.5 | 11.9 | 2.0 | 0.65 | 0.28 | 0.07 |
| SGD, alpha 0.9 | 37.2 | 7.4 | 1.6 | 0.56 | 0.11 |
| AdamW, alpha 0.5 | 0.63 | 0.69 | 0.73 | 0.64 | 0.46 |
| AdamW, alpha 0.9 | 5.0 | 5.9 | 6.9 | 7.6 | 8.4 |

For SGD, larger batches cure spikes, as the `Sigma / B` picture predicts. For AdamW on the
concentrated spectrum, larger batches make spikes *worse*. A plausible mechanism is that
less noise shrinks the second moment and enlarges the effective step along the sharp
mode. So **"increase the batch size" is not a safe universal response to spike risk**,
and a batch controller built on a one-step sensor would get Adam wrong.

When only the target exponent varies, every noise sensor is weak (at most +0.29), and
sharpness (Adam strata only) scores +0.71. That axis changes severity little, so this is
mostly noise.

## E2 - estimator calibration (`e2_estimator_calibration.py`)

Setup: 200 repeated probes at each of 8 exact checkpoints (spectrum {0.5, 0.9} x
{SGD, AdamW} x 2 seeds; true `D_eff` between 1.2 and 3.7), using the true heavy-tailed
PLRF noise and a Gaussian control with the same covariance.

| K | R bias | R rel. RMSE (PLRF / Gauss) | +/-2 SE coverage | log-t CI coverage | D_eff bias (min dof 4) |
| --- | --- | --- | --- | --- | --- |
| 8 | +0.5% | 0.51 / 0.42 | 0.76 / 0.84 | 0.89 / 0.92 | -24% (abstains 62%) |
| 16 | -0.8% | 0.33 / 0.27 | 0.83 / 0.90 | 0.89 / 0.93 | -0.3% |
| 32 | +0.2% | 0.24 / 0.18 | 0.88 / 0.90 | 0.92 / 0.93 | +6% |
| 64 | -0.8% | 0.16 / 0.13 | 0.90 / 0.92 | 0.92 / 0.94 | +3% |
| 128 | -0.1% | 0.12 / 0.09 | 0.92 / 0.95 | 0.93 / 0.95 | +1% |

`R` is unbiased at every K. Undercoverage comes from the right skew of `R_hat`
(skewness 1.6 at K=8) and from heavy tails; the log-scale interval removes most of it.
At small K, PLRF noise costs about 20-25% more RMSE than Gaussian noise. Changes made:
added `risk_trace_ci`, and changed `min_concentration_dof` from 16 to 15 (K=16 is
unbiased; K=8 is not).

![E2](results/e2_estimator_calibration.png)

## E3 - simulator ablations and calibration (`e3_simulator.py`)

Setup: 72 checkpoints (4 spectra x {SGD, AdamW, SNRAdamW} x lr {0.3, 0.7, 0.95} x edge
x 2 seeds), each with 32 real continuations of 800 steps scored at prefixes 50 / 200 /
800. Columns: Spearman of severity, mean absolute error and bias of the simulated
event fraction, and the median log10 ratio of simulated to real severity.

| variant (horizon 200) | rho | event-fraction MAE | bias | log10 severity ratio |
| --- | --- | --- | --- | --- |
| oracle (true PLRF noise) | 0.991 | 0.048 | +0.003 | -0.03 |
| exact Sigma, loss-proportional | 0.972 | 0.083 | -0.046 | -0.04 |
| exact Sigma, fixed | 0.918 | 0.169 | -0.158 | -0.12 |
| resampled, K=128, loss-prop. | 0.951 | 0.094 | -0.011 | +0.01 |
| resampled, K=32, loss-prop. | 0.943 | 0.129 | -0.048 | +0.02 |
| Gaussian, K=32, loss-prop. | 0.950 | 0.102 | -0.062 | -0.03 |
| Gaussian, K=32, fixed | 0.907 | 0.155 | -0.142 | -0.11 |
| Student-t(5), K=32, loss-prop. | 0.954 | 0.128 | +0.097 | +0.20 |
| Gaussian, K=8, loss-prop. | 0.804 | 0.184 | -0.072 | -0.03 |

With 4 / 16 / 64 simulated seeds (exact, loss-proportional), rho is 0.955 / 0.972 /
0.982. Horizon has little effect on ranking. At 800 steps, resampled K=128 is the best
practical variant (rho 0.961, MAE 0.073, bias +0.016). The oracle result confirms that the
optimizer-state replay is correct. The remaining error is the noise model: it must
follow the loss (state dependence), and it needs enough microbatches to represent the
covariance.

## E4 - optimizer timescales and SNR gating (`e4_timescales.py`)

Setup: 11 variants on a *shared* learning-rate grid (0.3 / 0.6 / 1.0 x the AdamW edge),
8 seeds, 4000 steps.

- **alpha=0.5:** SNR gating with the default `snr` gate cuts excursions from 1.6-4.9 per
  run (AdamW) to 0-1.4 (0-3.5 for `rho=0.999`; 3.1 for `lambda_pop=0.25`), and reaches a 4-40x lower final loss. `lambda_pop=4` is best on
  both counts (0-0.1 events per run, lowest loss). The soft gate has zero events but
  learns more slowly (it never reaches the common loss target).
- **alpha=0.9:** AdamW has about 30 events per run whatever `beta2` is. The `snr` gate
  still has 9-24, so it does **not** suppress spikes there. A slow variance EMA
  (`rho=0.999`) makes events rarer (about 9 per run) but much larger: the median maximum
  excursion is 1.8e3 to 5.5e4x the trailing baseline, versus 20-50x for AdamW. That is
  the "misidentified signal under stale moment estimates" failure the proposal
  anticipated. The **soft gate nearly eliminates spikes** (0-1 per run) at a final
  excess loss about 3-6x that of `snr` with `lambda_pop=4`.
- **AdamW `beta2`** (0.95 / 0.99 / 0.999) has no consistent effect on spike counts here.
  The paper's `beta2` recipe does not carry over to this setting as a general rule.

![E4](results/e4_timescales.png)

## E5 - correlated examples (`e5_correlated_data.py`)

Setup: each distinct example appears r times in a batch of 64. The table shows the
estimated / true variance of the batch-mean gradient for a K=8 probe.

| r | grouped split | interleaved split |
| --- | --- | --- |
| 1 | 1.00 | 1.00 |
| 2 | 0.98 | 0.43 |
| 4 | 0.95 | 0.14 |
| 8 | 1.03 | 0.00 |

Keeping correlated rows in one microbatch gives the right batch noise (counting groups
as the units gives the right per-example covariance). Spreading them across
microbatches hides the noise. `collect_microbatch_gradients` now takes `group_ids` and
refuses splits that would cut a group.

## E6 - network scale and low-rank curvature (`e6_network_scale.py`)

Setup: teacher-student MLP regression at hidden widths 32 / 128 / 512 (2.4k / 22k / 284k
parameters), with 6 optimizer and learning-rate arms x 2 seeds x 2 checkpoints. Each
checkpoint has 8 real continuations and 8 simulated ones, 100 steps each. Noise is
resampled from K=32 microbatches with loss-proportional scaling. The GGN is computed on 256
held-out samples (8 outputs).

| simulator curvature | rho, width 32 | rho, width 128 | rho, width 512 | cost vs real, width 32 / 128 / 512 |
| --- | --- | --- | --- | --- |
| full GGN | 0.98 | 0.62 | 0.60 | 4.2x / 3.7x / 3.6x |
| low-rank GGN, r=4 | 0.96 | 0.90 | **0.90** | 0.9x / 1.2x / 2.4x |
| low-rank GGN, r=16 | **0.98** | **0.97** | 0.85 | 1.0x / 1.4x / 2.8x |
| low-rank GGN, r=64 | 0.99 | 0.96 | 0.87 | 1.8x / 2.9x / 6.5x |
| activation proxy | -0.25 | -0.58 | -0.41 | 0.6x / 0.7x / 0.1x |

**A pitfall the first run exposed.** Before the subspace fix, the low-rank variants had
rank correlations of -0.58 to +0.22
(`results/e6_network_scale_before_subspace_fix.json`). With truncated curvature, `g0` keeps components that have
almost no restoring force, so the surrogate loss falls without bound. On one checkpoint
it went from 0.06 to -1.08. Projecting `g0` and the noise onto the curvature's
eigenvector span fixes this; it is now automatic for `LowRankCurvature`.

**The full GGN suffers from the same problem (`e6b_full_ggn_projection.py`).** A GGN on
256 samples x 8 outputs has rank at most 2048, which is below the parameter count at
widths 128 and 512. At width 512, the unprojected full-GGN surrogate falls to -2.2x and
-16x the starting loss in the two spiky AdamW arms, and its simulated severity collapses
(0.23 against a real 2.30). Projecting onto its top-64 eigenvectors gives 2.17 vs 2.30
and 1.48 vs 1.53. The Spearman across the 6 arms goes from 0.66 to 1.00, though n is
small. The simulator now warns whenever the noiseless surrogate path goes below zero
for a non-negative loss.

Takeaways:

- With projection, a **rank 4-16 GGN keeps the ranking at 0.85-0.98** up to 284k
  parameters. The top few curvature directions carry the excursions.
- The **activation proxy is not usable for simulation**: it is anti-correlated at every
  width.
- **Cost has not been shown to be a win.** On these tiny models a real training step
  takes 1-2.5 ms, so real continuations are cheap, and the simulator (which still runs
  the optimizer over all parameters, plus the GGN build) costs 1-3x as much. The case
  for the simulator rests on models where a real step costs far more than an optimizer
  update plus a rank-r product. These experiments do not test that.

![E6](results/e6_network_scale.png)

## E7 - scaling up: where does the simulator pay off? (`e7_*.py`, `tinylm.py`)

The question: what is the smallest scale at which the checkpoint simulator is
cheaper than simply running real continuations, while still telling which
configurations are unsafe? The test model is a small decoder-only Transformer
(`tinylm.py`: explicit causal attention, so `torch.func` JVPs work) trained with AdamW
on a synthetic language (an order-1 Zipf-like Markov chain plus a copy-8-back rule).
Each row is one sequence, so microbatches never split correlated tokens (E5).

**Simulator speed (library change).** The per-step loop was rewritten: flat parameter
and gradient buffers, noise pre-projected into subspace coordinates, and a fused
rank-r path for `LowRankCurvature` that makes two passes over `V` per step. A simulated
step went from 15x a bare AdamW step to 2.6x (n = 1M, r = 16). A test checks that the
fused path reproduces the generic one to 1e-9.

**Cost crossover (`e7_crossover_timing.py`, 96 configurations, 4 CPU threads).** Width
is 32-256, depth 2-4, sequence length 8-256, batch 2-32. A query is 8 continuations x
100 steps, including a held-out loss evaluation per real step. The simulator build is
the probe (32 microbatches) plus a rank-16 GGN on 16 held-out sequences.

- Fitted cost model (R^2 0.95): real step = 11 ms + 36 ps x params x tokens; simulated
  step = 1.8 ms + 6.9 ns x params. **The per-step speedup is about tokens-per-step / 200**:
  the simulator wins per step once a training step processes more than about 200
  tokens, at any width.
- Single-query speedup including the build: median 2.0x, range 0.5-10x. It is below 1
  only for tiny batches with comparatively large models (B=2, or L=8 at width >= 128),
  where the GGN build dominates. With 5 queries from one build, nearly all
  configurations win.
- So **cost is not the binding constraint**. The crossover is at a few hundred tokens
  per step. Real LM training (1e5-1e6 tokens per step) is about three orders of
  magnitude past it, with a predicted per-step speedup in the thousands.

![E7 crossover](results/e7_crossover.png)

**Is the targeted effect present at this scale? (`e7_lr_scan.py`).** Mostly not.
- Across learning rates 0.005-0.6, `beta2` 0.95 and 0.999, batches 2-32 and widths
  32-64, raising the learning rate makes training *stall* smoothly at a higher loss.
  Intermittent excursions appear only weakly: 0.3-0.7 events per 1500-step run, of
  25-37% size, at lr about 0.05 with B = 4-8.
- An earlier attempt with an order-2 Markov language never learned (loss stayed at
  ln 64), and its results were discarded.

The question that remains testable, and is the one a controller would ask, is **"which
learning rates are unsafe from this checkpoint?"**

**What-if queries (`e7_whatif.py`).** Setup: AdamW at lr = 0.01, checkpoints at steps
300 and 800. From each checkpoint, 8 candidate learning rates (0.005-0.2) are each run
with 8 real continuations of 300 steps, and also answered by **one** simulator build per
checkpoint (rank-16 GGN, resampled loss-proportional noise). "Unsafe" means the held-out
loss rises by more than 0.05 nats over the horizon. Sharpness is `lr * lambda_max(P H)`
on the same low-rank GGN, including the `(1 - beta1)` one-step factor. In those units
Adam's edge-of-stability threshold is `2 (1 + beta1) = 3.8` (Cohen et al. 2022; see
`docs/stability-theory.md`). **Correction:** the first version of this section used
threshold 2, which is the SGD value. Both are shown below.

| model | params | severity rho (sim / sharpness) | unsafe accuracy (sim / sharpness > 3.8 / > 2) | speedup (8 queries / 1 query) |
| --- | --- | --- | --- | --- |
| d=32, L=32, B=8 | 31k | 0.93 / 0.97 | **0.81** / 0.59 / 0.72 | 4.6x / 3.8x |
| d=64, L=64, B=8 | 112k | 0.98 / 0.97 | **0.81** / 0.53 / 0.63 | 6.7x / 5.2x |
| d=128, L=128, B=8 | 430k | 0.97 / 0.97 | **0.88** / 0.69 / 0.88 | 9.6x / 6.4x |

Onset ratio (predicted / real smallest unsafe learning rate) per checkpoint. "never"
means no tested learning rate (up to 0.2) was flagged:

| model | simulator, rank 16 | sharpness > 3.8 (correct) | sharpness > 2 |
| --- | --- | --- | --- |
| d=32 | 0.7, 2.3, 0.5, 1.4 | never (x4) | 2.0, 6.7, 2.0, 4.0 |
| d=64 | 1.0, 3.5, 0.7, 2.3 | 2.9, never, 2.9, never | 2.9, 10, 1.4, 6.7 |
| d=128 | 0.7, 1.4 | 2.9, 4.0 | 1.0, 2.0 |

![E7 what-if](results/e7_whatif.png)

- Within one checkpoint, sharpness is just proportional to the learning rate, so any
  monotone score ranks the queries. The ranking columns therefore do not discriminate;
  the absolute decision does.
- **The simulator places the instability onset within about 2x of reality; the
  edge-of-stability rule does not.** With the correct threshold the rule never flags
  any tested learning rate in 5 of the 10 checkpoints, and is otherwise 2.9-4x too
  permissive. The real instabilities here happen far below the deterministic adaptive
  edge, which is the gap that noise-aware simulation fills.
- **The simulator cannot predict magnitudes.** Real blow-ups raise the loss by 2 to 80
  nats; the quadratic surrogate tops out at about 0.3. It also misses continued learning
  at small learning rates (real loss falls by up to 0.2; simulated stays about flat),
  because the rank-16 projection discards most descent directions. It is an onset
  detector, not a forecaster.

**Rank ablation (`e7_rank_ablation.py`, d=64, same checkpoints).** The simulator was
optimistic late in training. Higher curvature rank fixes most of that:

| GGN rank | onset ratios (4 checkpoints) | geometric mean | build + 8 queries |
| --- | --- | --- | --- |
| 4 | 1.4, 10, 2.9, 6.7 | 3.6 (optimistic) | about 67 s |
| 16 | 1.0, 3.5, 0.7, 2.3 | 1.6 | about 52 s |
| 64 | 0.43, 1.5, 0.43, 1.7 | 0.85 (slightly conservative) | about 72 s |

For comparison, the 8 real queries take about 340 s per checkpoint. **Rank 64 gives the
best safety profile** (worst case 2.3x either way) and stays 4-5x cheaper here.

**Where the critical point is.**
- **Cost:** about 200 tokens per step for per-step parity. A single query pays off from
  roughly 1k tokens per step (or 5 queries per build), at every width tested.
- **Usefulness:** in this model family the simulator beats the (correctly thresholded)
  edge-of-stability rule at every size tested (31k-430k parameters).
- **Not reached:** the regime the proposal targets, with noise-driven intermittent spikes
  at a fixed learning rate, does not appear clearly in models this small. Testing it
  needs larger models or longer training than this CPU budget allows. The prediction from
  E1 (Adam spike severity *rising* with batch size on concentrated spectra) has not been
  checked on a Transformer.

## E8 - when should a large spike effect appear? Theory plus small runs (`e8_*.py`)

Derivations are in [`docs/stability-theory.md`](../../docs/stability-theory.md); the code
is in `snr_grad.stability_theory`. Four closed-form quantities, each checked here:

| quantity | meaning |
| --- | --- |
| `kappa` | SGD feedback gain: the noise-driven fraction of the stationary loss (exact for Gaussian design); the mean loss diverges at 1 |
| `D_K` / `D_hat` | number of curvature modes carrying that noise-driven loss |
| `Phi = kappa sqrt(2/D)` | predicted relative fluctuation of the loss |
| `w` | share of gradient noise along the sharpest (preconditioned) direction: the relative width of the learning-rate window in which large noise-driven spikes are possible |

Adam enters through its preconditioner `P = 1/sqrt(v)` and its momentum edge
`2(1+beta1)/(1-beta1) = 38`.

**E8a - PLRF validation (exact H and Sigma; `e8_plrf_theory.py`, `e8_sgd_indices.py`,
`e8_adam_indices.py`).**
- **SGD (60 configurations).** The closed-form expected loss matches the seed-averaged
  runs to about three digits wherever `kappa < 1` (for example 0.3187 vs 0.3185).
  - Log severity is explained by `Phi` (R^2 0.95) and by the Kesten tail exponent `xi`
    (R^2 0.97); jointly R^2 = 0.99.
  - Phase diagram: `kappa >= 1` diverges unless the feedback is single-mode
    (`D_K <= 1.4`). In that case it gives infinite-mean bursts instead: severity
    3000-4000 with no divergence.
- **Adam (288 checkpoints).** In Adam-preconditioned coordinates, `Phi` predicts whether
  the next 1000 steps contain a severe excursion with AUROC 0.97 (log-severity R^2 0.86),
  against 0.81 (R^2 0.43) for the adaptive edge ratio.
  - Calibration: `severity ~ 15 * Phi^1.38`, so a 100% excursion corresponds to
    `Phi ~ 0.14` and a 10x one to `Phi ~ 0.7`.
  - The edge ratio rises with batch size (`v` shrinks), which explains E1's
    "bigger batch, worse Adam spikes".

**E8b - the tiny LM is far from that band (`e8_lm_indices.py`, `e8_lm_trend.py`,
`e8_lm_data.py`, `e8_lm_favorable.py`, `e8_lm_scaling.py`).** The same indices were
measured at about 130 LM checkpoints: width 32-256, batch 8-128, sequence 16-256, lr 0.003-0.03,
up to 4000 steps, and vocabularies of 64-4096 tokens with or without Zipf frequencies.

- **Standard settings:** `Phi = 0.0005-0.005`, 30-300x below the band. Two factors keep
  it small:
  - only 1-10% of the loss is noise-driven (`kappa_hat`);
  - that part is spread over `D_hat` = 100-6000 modes.
- **Fit** (R^2 0.91): `Phi ~ d^0.36 B^-0.39 L^-0.44 lr^0.52`. At lr 0.01, `Phi` reaches 0.15
  only at widths of order 1e7-1e8, so width alone will not get there.
- **Raising the learning rate does not help.** Above Adam's edge (ratio 2-950 here), runs
  *stall* rather than spike.
- **Longer training does not help.** `D_hat` falls 3-7x over 4000 steps, but
  `kappa_hat` falls in step, so `Phi` stays flat.
- **Data:** heavy Zipf frequencies at a small vocabulary halve `D_hat` (`Phi` x2), but
  larger vocabularies spread the noise over more embedding rows (`Phi` falls).
- **Most favourable corner** (width 256, B = 2-4, L = 16, Zipf): `D_hat` = 8-43 and
  `Phi` = 0.004-0.02, within about 2x of the fit. Observed severity was 0.03-0.07,
  matching the PLRF calibration (0.026-0.07). That corner also stalls easily.

**E8c - why: the noise is not aligned with the sharpest direction (`e8_window.py`,
`e8_alignment.py`).**
- **PLRF:** `w = lambda_1 / tr(H)` correctly separates the spike-capable spectra (w >= 0.12)
  from the diffuse one (w = 0.02, which diverges without spiking). It does not locate
  the onset precisely; batch size shifts it.
- **LM, Adam geometry:** `w = 1e-4 to 7e-3`, below even the non-spiking PLRF case.
- **LM, SGD geometry, same Adam-trained checkpoints:** `w` is up to 270x higher, reaching
  0.10-0.16 for the favourable configuration. So **Adam's per-coordinate whitening
  removes the alignment** that noise-driven spikes need.

**E8d - targeted test with SGD (`e8_sgd_lm_test.py`, `e8_sgd_lm_phi.py`).** The theory
predicted that the favourable configuration should show spikes under plain SGD near its
edge, while the control (w ~ 0.01) should not.
- **Result:** the aligned configuration's excursions are 2-4x larger than the control's
  (median severity 0.07-0.09, maximum 0.23, against 0.02-0.04). No event crosses the
  25% threshold, and there is no growth toward the edge.
- **Explanation:** when *trained* with SGD, the configuration's `w` is only 0.036. The
  geometry differs from the Adam-trained checkpoint, and 0.036 is below the
  spike-capable range, so no spikes is consistent with the theory.
- `Phi` predicted the observed severity within about 1-8x. The Kesten heavy-tail term
  over-predicted by orders of magnitude: it needs noise proportional to the error along
  the sharp mode, which does not hold here.

**Estimate: when is a big spike effect worth testing?**

A configuration qualifies only if, in the optimizer's own geometry and below its edge,
`w >~ 0.1`, `Phi >~ 0.15`, and either `kappa < 1` or `D ~ 1`. The tiny LMs miss on every
count:
- **Adam:** `w` is 15-1000x too small and `Phi` 7-300x too small.
- **Trends with scale:** `w` falls with width and vocabulary; `Phi` grows only as
  `d^0.36`.
- **Conclusion:** **scaling up a standard Adam-trained LM is not a path to the noise-driven
  spikes the proposal targets.** Spikes seen in large Adam LMs are more likely other
  mechanisms (in this study, crossing the adaptive edge produced stalls); the current
  theory does not cover those.

Where to look instead:
- **SGD or momentum training** of models whose sharpest direction is also their noisiest:
  PLRF-like random-feature or linear-head models, or tasks with a dominant output
  direction.
- **Screen first:** measure `w`, `Phi` and `kappa` at a few checkpoints before
  committing to an expensive run. That costs seconds to minutes per checkpoint on these
  models.

## Library changes made during the study

- `snr_grad.stability_theory`: `sgd_stochastic_edge`, `dominant_mode_tail`,
  `noise_fluctuation_index`, `adam_edge_threshold`, `adam_edge_ratio`,
  `adam_stationary_preconditioner`, `preconditioned_top_eigenvalue`; plus
  `PreconditionedCurvature` in `snr_grad.curvature` (tests in
  `tests/test_stability_theory.py`).

- Faster simulator: flat parameter and gradient buffers, noise pre-projected into
  subspace coordinates, and a fused low-rank path (`LowRankCurvature.matvec_flat`). A step
  went from 15x a bare AdamW step to 2.6x.

- `ResampledProbeNoise` and `noise_model="resample"`: non-Gaussian probe noise.
- `LowRankCurvature`: randomized subspace iteration with a deflated Hutchinson remainder.
- Simulator `subspace` projection (automatic for low-rank curvature), plus
  `deterministic_min_loss` and `surrogate_below_zero` in the summary, and a
  `RuntimeWarning` when the noiseless surrogate goes below zero.
- `StabilityMetrics.risk_trace_ci`: log-scale jackknife-t interval.
- `min_concentration_dof` default changed from 16 to 15.
- `collect_microbatch_gradients(group_ids=...)`, with the docstring's shuffling advice
  corrected.
- `tests/test_stability_deep.py`: Monte Carlo unbiasedness of the progress and
  mean-curvature U-statistics, interval coverage, SGD momentum and Nesterov maps against
  finite differences, the stationary momentum factor, group aggregation, edge cases,
  low-rank equivalence, and subspace projection.

## Caveats

PLRF is a convex quadratic, so the simulator's surrogate is exact there apart from the
noise model; the network results (E6) are the realistic test. The first-round MLP
benchmark used an unprojected GGN whose rank (at most 4096) was below the parameter
count (6.7k). Its simulator result (0.96) may therefore be affected by the unbounded
surrogate, although it was not invalidated there. Checkpoints within a
configuration are correlated, and the reported correlations are descriptive. The
thresholds of the excursion definition were fixed before these experiments and not
tuned.
