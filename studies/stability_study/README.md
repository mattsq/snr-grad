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

## Library changes made during the study

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
