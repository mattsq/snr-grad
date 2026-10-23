# Proposal: curvature-aware stochastic stability sensing and control

**Status:** Diagnostics implemented (M0, M1, M3 done; M2 partial; M4 not started; see section 11)  
**Target repository:** `mattsq/snr-grad`  
**Companion proposal:** [Optimizer-aware batch control](optimizer-aware-batch-control.md)  
**Primary reference:** Ran Tian and Courtney Paquette, [*Simulating Training Spikes of Deep Neural Networks by Layer-wise Random Feature Models*](https://arxiv.org/abs/2610.13524) (2026)

## 1. Question and motivation

Can the gradient-noise, optimizer-state, and activation-covariance measurements already available in `snr-grad` predict *stochastic loss excursions* better than a conventional (even optimizer-aware) scalar gradient-noise scale (GNS)? If so, can they improve batch-size or optimizer decisions at matched training compute, without making ordinary optimisation unnecessarily conservative?

The existing repository separates two decisions:

- **SNR gates** decide which *parameter-update coordinates* to trust from the relationship between estimated mean gradient and coordinate-wise minibatch-mean gradient variance.
- The [optimizer-aware batch-control proposal](optimizer-aware-batch-control.md) asks how many *additional training examples* are worth consuming for the next update, comparing Euclidean, frozen-Adam-aware, and Muon-aware noise sensors to directly measured batch-efficiency curves.

Tian and Paquette supply a third concern. Convex quadratic random-feature models can exhibit intermittent large validation-loss spikes under stochastic optimisation, especially with a concentrated power-law input spectrum and learning rates near instability. A layer-wise quadratic simulator distinguishes deliberately stable versus spike-prone Adam configurations of a Transformer from checkpoints, although it **does not** predict exact future spike times or reproduce the entire network's loss dynamics. Thus good *mean* training progress does not ensure acceptable *tail* behaviour.

This proposal adds a **read-only, curvature-aware stability sensor** and an optional **checkpoint-local stochastic continuation simulator**. Only if they prove predictive should a later batch controller or optimizer policy use them. Do not interpret the paper's power-law threshold as a universal necessary/sufficient criterion for spikes in arbitrary neural networks.

## 2. Falsifiable hypotheses

- **H1: Geometry adds predictive information.** At matched conventional GNS, learning-rate schedule, optimizer, and budget, concentrated curvature-weighted stochastic updates predict more frequent/larger loss excursions than diffuse updates. The effect should survive comparisons with plain activation spectral slope and maximum Hessian eigenvalue.
- **H2: Optimizer state matters.** Conditioning on actual preconditioning, SNR gates, and Adam second-moment state should improve prediction relative to raw-gradient spectra alone, especially when `beta2` is close to one.
- **H3: Multi-step state matters.** A simulator that evolves optimizer states and noise-driven displacement should predict the *distribution* of excursion severity better than a single-step expected-loss penalty.
- **H4 (only after H1-H3): Policy usefulness.** A stability-aware batch/learning-rate policy can reduce spike risk at comparable validation-loss-versus-samples/compute trade-offs, relative to predefined ramps and scalar-GNS control.

**Null outcomes matter:** If spectral sensors add no out-of-sample predictive value, or if the improvement costs more than it saves, preserve useful diagnostics and **do not** ship a controller.

## 3. Mathematical bridge

Let the per-example gradient have mean `mu` and covariance `Sigma`; under independent examples a batch-mean gradient is `g_B = mu + epsilon_B`, with `Cov(epsilon_B) = Sigma / B`. At a fixed checkpoint, locally approximate the *validation* loss by a quadratic with gradient `g` and Hessian `H`. For a **frozen linearized** optimizer response `P`, define the update

```text
Delta_theta = -eta P g_B.
```

Then

```text
E[Delta L] ≈ -eta g^T P mu
             + (eta^2 / 2) mu^T P^T H P mu
             + (eta^2 / (2 B)) tr(H P Sigma P^T).
```

The final term is the **curvature-weighted stochastic update penalty**, not a substitute for expected training progress. A large gradient variance may be harmless in flat directions and costly in sharp directions.

For a positive-semidefinite curvature proxy `H_plus`, define

```text
A = H_plus^(1/2) P Sigma P^T H_plus^(1/2)
R = tr(A)
D_eff = R^2 / tr(A^2)
```

For zero-mean Gaussian noise and a *noise-only* quadratic loss contribution,

```text
E[noise penalty] = eta^2 R / (2 B)
relative SD(noise penalty) = sqrt(2 / D_eff).
```

Thus two complementary measurements emerge:

1. **Noise magnitude/risk scale** `R/B`: decreases with `B` under the independent-sample model.
2. **Concentration** `D_eff`: small when relatively few curvature-sensitive modes dominate. It need not increase with nominal parameter count and, under pure `1/B` scaling at fixed `P`, is unchanged by changing `B`.

Important qualifications:

- This is a **local surrogate derivation**, not a formula claimed by Tian and Paquette. The expected-loss expression also has the deterministic linear term and finite-step mean-curvature term. The noise-only relative SD is **not** a calibrated probability of a total training-loss spike.
- The true Hessian can be indefinite; `H_plus` may be a PSD generalized Gauss-Newton (GGN), empirical Fisher (with its own caveats), or carefully validated layer quadratic proxy. Never silently apply the PSD effective-rank derivation to an indefinite Hessian.
- Adam, adaptive SNR gating, momentum, Muon orthogonalization, and DoPr with freshly estimated activation covariances are **stateful and/or nonlinear**. A detached/frozen `P` is a *one-step local approximation*. A multi-step simulator must replay the real optimizer state transitions.
- Adjacent language tokens, duplicated data, and correlated microbatches invalidate the naive `Sigma/B` scaling. The physical token count is not automatically the effective number of independent observations.
- Do **not** confuse the PLRF spectrum exponent `alpha_spectrum` with `SNRAdamW(alpha=...)`, the finite-sample/online coefficient in the population-risk gate.

### Layer-specific tractable proxy

For a linear layer `y = W h`, a simplified squared-loss/isotropic-output-curvature model yields `H_W ≈ I_out ⊗ C_h`, where `C_h = E[h h^T]` is the *uncentered* input activation correlation/covariance already collected by DoPr. With a noise-only matrix update `U`,

```text
quadratic_penalty(U) ≈ 0.5 * tr(U C_h U^T).
```

This expression is exact only under its stated model assumptions. For general networks, contrast this proxy with GGN/Hessian-vector-product estimates and actual measured finite-difference validation-loss changes. An optional separable `C_out ⊗ C_h` approximation may be explored later; `C_out = I` must not be treated as a fact about Transformers.

## 4. Measurement design and estimator interfaces

Keep **all new instrumentation outside the optimizer's step path**, as with `snr_grad.variance` and `snr_grad.activation`. There must be no new overhead or changed parameter updates unless the user explicitly enables the probe.

### 4.1 Gradient probes

Use `ExactVarianceEstimator` and `backward_with_microbatch_variance` plumbing. At *identical parameters and optimizer state*, take `K` disjoint, appropriately randomized/equal-weighted microbatches of size `b`. Obtain mean gradients `g_k` before any update.

```text
g_bar = mean_k(g_k)
C_micro = sum_k (g_k - g_bar) (g_k - g_bar)^T / (K - 1)
Sigma_hat_per_example = b * C_micro     # only for independent examples
```

The existing `grad_variances` hook returns **diagonal variance of the current batch mean**, not per-example variance. A batch controller must recover the correct per-example scale from raw microbatch statistics or by an explicit conversion. Preserve reduction conventions and record the number of independent *units* (examples/sequences rather than blindly tokens).

Support streaming *matrix-free* operations on centered microbatch gradient deviations; never materialize a full `num_parameters × num_parameters` covariance. Allow stratification by layer/parameter group.

Probe metadata: batch size and units, data-sampling scheme, RNG seed, parameter and optimizer checkpoint IDs, effective grouping, normalization, probe latency/memory, and estimation confidence.

### 4.2 Optimizer-aware update deviations

Provide a read-only function that maps a probe gradient to an **approximate update** at frozen optimizer state:

- **AdamW:** detached per-coordinate `1/(sqrt(v_hat)+eps)`; the mean first-moment contribution and bias correction must be modeled correctly for the chosen linearization.
- **SNRAdamW:** include frozen `q` derived from `m_hat`, `s_hat`, `alpha`, `lambda_pop`; report separate ungated and gated results. A gate that itself changes with noise requires a replay in the simulator.
- **DoPr:** include damped activation right-preconditioning when enabled; never assume covariance inversion reduces tail risk. Rank-deficient per-batch covariance can amplify noisy directions.
- **Muon/SpectralSNRMuon:** initially use empirical finite-difference/Jacobian-vector approximations where feasible or mark the frozen-linear sensor **unsupported**. Muon's polar/orthogonalized update is nonlinear, so a diagonal Adam approximation is incorrect. Continue using optimizer-appropriate dual-norm GNS as an independent baseline.

Only evaluate preconditioned *deviations* after subtracting their common mean at the same checkpoint. Report gates, preconditioner magnitudes, `beta2`, `rho`, `lambda_pop`, and any active adaptive-threshold policy; these are important explanatory variables.

### 4.3 Curvature backends

Offer a common minimal `CurvatureProxy` interface:

- `quad(update) -> scalar` for `u^T H_plus u`
- optional `matvec(update) -> tensor`
- `metadata` with proxy assumptions, rank, damping, checkpoint age, and cost.

Backends in increasing cost/fidelity:

1. **Exact diagonal/quadratic toy Hessian:** gold standard in PLRF and small regression.
2. **Activation correlation:** reuse DoPr's measurements via a new *read-only snapshot* API with explicit layer mappings, uncentered-vs-centered convention, damping, and sample count. Avoid duplicate hooks where possible.
3. **GGN or HVP:** on a small held-out set for independent validation of the activation proxy; permit randomized trace/low-rank sketches.

For a quadratic proxy `H_plus` and frozen update deviations `d_k = P(g_k-g_bar)`, an unbiased-in-sampling (not necessarily unbiased-in-model) trace estimator under independent microbatches is

```text
R_hat = b/(K-1) * sum_k quad(d_k).
```

This estimates `tr(H_plus P Sigma P^T)` on the per-example variance scale, before applying `eta^2/(2B)`. Include uncertainty across **independent groups of microbatch probes**, not just standard error over correlated gradient coordinates.

### 4.4 Concentration and rank diagnostics

**Do not estimate `D_eff` naively from a small `K`.** An empirical covariance of `K` microbatch gradients has rank at most `K-1`; with `K=4`, apparent low effective rank is essentially guaranteed and provides no evidence of true spectral concentration.

Plan two tiers:

- **Cheap proxy:** log eigenvalue slope, participation ratio, top-mode energy share, and effective rank of adequately sampled/shrunk activation correlation `C_h`. Label it *activation spectrum*, not curvature-weighted update-noise `D_eff`.
- **Costly validation:** estimate `tr(A)` and `tr(A^2)` using enough independent gradient replicates, a justified shrinkage/structured covariance model, or stochastic matrix-free estimators. Quantify finite-`K` bias and calibration on exact PLRF ground truth. Abstain rather than emit a spurious effective-rank value when diagnostics are unreliable.

A high-curvature mode with very little stochastic update variance may be safe; steep activation spectra alone do not identify this.

## 5. Checkpoint-local stochastic continuation simulator

After the sensor has been independently validated, build a *separate research simulator* using layerwise quadratic loss and stochastic synthetic gradients. It should answer **distributional** questions ("How often/how large an excursion under this configuration?") and never imply exact future spike-time prediction.

Initial model at checkpoint `theta0`:

```text
L_hat(theta) = L0 + g0^T (theta-theta0)
                 + 0.5 (theta-theta0)^T H_plus (theta-theta0)

g_sim(theta) = g0 + H_plus (theta-theta0) + noise(theta, t, seed)
```

The first baseline uses additive Gaussian noise with fixed covariance fitted from microbatches. A second model may make covariance state dependent, reflecting the paper's random-feature construction; test it **against** the fixed-covariance baseline. Gaussianity is an explicit assumption, not an empirical given. Examine heavy-tailed and correlated noise as failure modes.

For each simulated seed:

1. Clone parameters and **all** optimizer state: Adam `m,v`, SNR `s`/`rho`, step/bias correction, adaptive-gate thresholds, and relevant schedules. Do not mutate the live model.
2. Sample gradients under the chosen noise model and **execute the optimizer's actual update/gate code**, including `beta2`, `rho`, and optional weight decay. No full-model forward/backward per simulated step.
3. Recalculate the quadratic surrogate gradient at the new point and simulate for `T` steps; log quadratic loss, expected deterministic change, noise contribution, gates, moment lag, and projection along leading modes.
4. Aggregate over simulated seeds: maximum excursion, frequency above preregistered thresholds, excursion duration, tail quantiles, expected progress, and simulated divergence.
5. Refit or abort when checkpoint age, activation drift, nonlinear finite-difference error, or observed gradient-statistic drift exceeds configured validity tolerances.

Begin with exact quadratic and PLRF simulations to validate state replay; **only then** use approximate layer simulators. Multi-layer models may have cross-layer coupling and higher-order loss terms missed by a single-layer surrogate. The paper itself reports this discrepancy. Do not represent an independently simulated collection of layers as the entire network loss.

Provide a mechanism to compare the distribution of simulations **with held-out real continuations** from identical checkpoints. Distinguish an observed-risk *ranking* from calibrated exceedance probabilities.

## 6. Integration with the existing batch-control proposal

Do not create an optimizer-owned magic batch-size parameter. The external `BatchController.observe(probe) / recommend()` API proposed in [optimizer-aware batch control](optimizer-aware-batch-control.md) remains responsible for selecting feasible effective batch sizes, with the data loader/accumulation loop applying them.

Sequence:

1. **Diagnostics only:** collect Euclidean GNS, optimizer-aware GNS, curvature-weighted trace penalty, concentration confidence, and optional simulator risk without affecting training.
2. **Compare at matched budgets:** assess what each diagnostic predicts about useful batch range, validation-loss excursions, and measured batch efficiency.
3. **Batch-only safety guard (conditional):** from feasible `B`, choose the smallest batch on a suitable efficiency frontier whose **empirically calibrated** upper risk bound is at most `delta`. Abstain or fall back to predefined safe settings if calibration is missing.
4. **Joint action research (conditional):** evaluate `(B, lr)`; only then consider `beta2`, `rho`, or gate threshold changes as additional controlled actions. A smaller learning rate may be more efficient than extra batching. Do not mechanically infer LR from nominal noise scale.

The controller must retain warmup, EMA/deadband, dwell time, max one-rung moves, and explicit "hold" reasons from the existing proposal. As in that proposal, correct `alpha="finite"` to the **actual** training batch size, and measure whether batch changes inadvertently change SNR gate density.

**No uncalibrated risk score should trigger autonomous changes to `beta2` or `rho`.** Their coupled adaptation timescales are an ablation target, not a justified scheduling rule. The paper's `beta2` recipe is an experimental proof of concept, not a universal policy.

## 7. Evaluation plan: preregister before implementing a controller

### Stage 0: reproduce the failure mode (must pass first)

Reproduce Tian–Paquette's PLRF setting as closely as practical (e.g. `D=1024`, `V=20_000`, `B=8`; `alpha_spectrum=0.3` versus `0.9`, fixed random features, squared loss). Sweep LR around the **empirically measured** stability edge for SGD and AdamW, with multiple independent data and optimizer seeds.

Acceptance: reproduce a regime with intermittent *validation* loss excursions before sustained divergence for the concentrated spectrum, distinct from smooth convergence and monotone instability. Record all settings required for reproducibility. If this does not replicate, stop and diagnose before using the new metric for decisions.

### Stage 1: compare sensors on exact curvature

For every PLRF checkpoint, compute:

- Euclidean scalar GNS.
- Frozen-Adam/optimizer-aware GNS.
- Largest eigenvalue / plain activation spectrum and slope.
- Exact `R = tr(H_plus P Sigma P^T)`, exact `D_eff` where tractable.
- Probe-based estimates of `R` and concentration, with uncertainty/cost.
- Distribution of spikes over independent **real** short continuations, including matched-gradient-SNR, differing-spectrum pairs.

Measure out-of-sample rank correlation, threshold ranking, precision/recall or AUROC where suitable, and calibration curves when probabilities are claimed. Use held-out seeds and checkpoint configurations; avoid choosing spike thresholds after seeing predictions.

The core falsification test: do two situations with similar aggregate GNS but different spectral concentration show different spike rates, and does the curvature-aware diagnostic predict the difference?

### Stage 2: optimizer and timescale ablations

Compare AdamW, `SNRAdamW`, `RotatedSNRAdamW`, `SpectralSNRMuon` where supported, and optionally DoPr; explicitly log unsupported assumptions. Use a factorial subset of:

- `beta2` (including a deliberately slow 0.999 second moment).
- SNR variance EMA `rho`.
- Gate type/`lambda_pop` and static versus adaptive threshold.
- Batch size `B`, LR near stability edge, and spectral exponent.

Test whether SNR gating suppresses noise-driven excursions, merely delays them, or worsens them by misidentifying signal under stale moment estimates. Guard against interpreting a lack of spikes under slower or stalled learning as success: report learning progress and compute alongside risk.

### Stage 3: fidelity and cost beyond PLRF

Start with a small matrix-heavy MLP/Transformer, then optionally a modest decoder LM if earlier results warrant it.

- Compare the activation-correlation quadratic proxy with GGN/HVP directions and actual local validation-loss differences.
- Check true simulator risk rankings against real short continuations from the **same checkpoint**, varying optimizer hyperparameters/seeds.
- Evaluate proxy failure under correlated tokens, non-Gaussian noise, nonlinear drift, cross-layer effects, stale activation covariance, and gradient accumulation.
- Report measurement memory/time and amortized cost: a five-second *simulation* does **not** imply free checkpoint statistics (the paper's initial model statistics cost substantial real-model evaluation).

### Stage 4: controller trial (only if 0-3 justify it)

Compare fixed small/large batch, predetermined ramp, ordinary GNS control, optimizer-aware GNS control, and calibrated curvature-aware risk control. Use matched initial checkpoints and paired data/initialization seeds.

Report **loss vs tokens/samples, optimizer steps, wall time, and total probe overhead**; spike count/severity, 95th/99th percentile maximum excursion, recovery time, and sustained divergence; controller trajectories, violations of target risk, and probability calibration.

A worthwhile controller must improve the efficiency/risk Pareto frontier **across held-out seeds**. Fewer spikes obtained by doing less useful learning or spending prohibitive probe compute is a failure.

### Spike definition and statistical rigor

Use a fixed held-out validation set and preregister a scale-aware threshold (e.g. deviation above a locally smoothed baseline exceeding a robust-noise multiple plus an absolute/relative floor). Track transient excursions separately from sustained divergence. Use paired seeds and confidence intervals, and state the number of *independent trajectories*, not simply the number of evaluated steps. Preserve data-shuffle seeds independently of model-initialization and simulator seeds.

Prefer tens of seeds for cheap PLRF studies. Small-network runs should include enough paired repetitions to distinguish a single lucky/unlucky spike from a policy effect. Do not claim reliable 99th-percentile risk from a handful of continuations.

## 8. Proposed code changes (incremental, no implementation mandated by this document)

| Proposed path | Responsibility |
| --- | --- |
| `src/snr_grad/stability.py` | Read-only `StabilityProbe`, `StabilityMetrics`, gradient noise scale normalization, frozen-update adapters, uncertainty/abstention |
| `src/snr_grad/curvature.py` | `CurvatureProxy` and exact quadratic, activation, and validation GGN/HVP backends |
| `src/snr_grad/stability_sim.py` | Isolated checkpoint-local quadratic surrogate and seeded optimizer-state replay |
| `src/snr_grad/activation.py` | Optional read-only covariance snapshot extraction for DoPr, without changing default preconditioning |
| `benchmark_stability_plrf.py` | Reproduce PLRF spike/no-spike regimes; sensor ablations and ground-truth curvature |
| `benchmark_stability_layers.py` | Small network/Transformer approximation and held-out-continuation checks |
| `tests/test_stability.py` | Exact quadratic formulas, scaling conventions, reproducibility, estimator confidence and failure handling |
| `tests/test_stability_sim.py` | Optimizer-state replay parity and proof live model/state are unchanged |

Illustrative **external** API, subject to benchmarking before it becomes public:

```python
probe = StabilityProbe.from_microbatches(
    gradients=microbatch_gradients,
    microbatch_size=b,
    optimizer=optimizer,
    curvature=curvature_proxy,
)
metrics = probe.measure()  # diagnostics only, no optimizer.step()
# metrics: risk_trace_per_example, estimated_noise_penalty_at_B,
#          concentration_proxy + uncertainty, validity flags, probe_cost

sim = QuadraticStabilitySimulator.from_checkpoint(
    model=model,
    optimizer=optimizer,
    probe=probe,
    curvature=curvature_proxy,
)
distribution = sim.run(n_seeds=..., horizon=...)  # cloned state
```

The precise interface for non-linear optimizers must distinguish supported *exact replay* from an unsupported or approximate *frozen linear sensor*.

## 9. Correctness, safety and performance invariants

- **No silently changed training:** instrumentation is opt-in and does not modify `p.grad`, `requires_grad`, parameters, optimizer state, or training RNG. Unit tests snapshot all of them before/after probes. Simulator RNG is independent.
- **State parity:** a synthetic deterministic gradient sequence applied to a simulator must give the same optimizer state and parameter updates as the actual optimizer (including `beta2`, `rho`, gate adaptivity, bias correction and weight decay) within numerical tolerance.
- **Correct scale:** batch-mean gradient variance versus per-example variance is explicit everywhere. For correlated data, annotate violated independence assumptions and disable unsupported `1/B` extrapolations.
- **Positive semidefinite geometry:** indicate precisely which curvature proxy and damping were used; report indefiniteness/approximation mismatch rather than taking square roots of negative eigenvalues.
- **Honest confidence:** rank-deficient low-`K` probe covariances cannot produce meaningful high-dimensional effective-rank claims. Return `insufficient_samples`/uncertainty rather than deceptively precise numbers.
- **No risk guarantees without calibration:** use `risk_score` for an uncalibrated ranking and reserve `spike_probability` for externally validated probabilities.
- **Resource boundaries:** cap probing frequency, Hessian-vector products, layer count, memory, surrogate horizon and simulation seeds; log costs. On expensive/invalid/stale probes, retain current controller action.
- **Optimizer specificity:** DoPr covariance reuse, distributed gradient aggregation, dropout, shared weights, and fused projections must respect the existing documented implementation limitations.
- **No cherry-picking:** predefined splits, ablation matrices and spike thresholds; export full seeds/configurations/results.

## 10. Milestones and go/no-go decisions

1. **M0: exact toy replication.** Reproduce power-law-dependent spikes. If not reproducible, stop.
2. **M1: sensor validity.** Demonstrate correct `R` and variance scaling on quadratic ground truth and establish uncertainty limits for concentration. If low-cost concentration cannot be estimated reliably, retain only the well-estimated magnitude diagnostic.
3. **M2: predictive increment.** On held-out PLRF checkpoints, demonstrate improvement over optimizer-aware GNS, max eigenvalue and activation-only spectral metrics for predicting real excursion severity. If no incremental value, do not build controls.
4. **M3: simulator validation.** Establish stable-vs-spiky ranking and rough distributional agreement on held-out continuations. If simulation only reproduces spikes qualitatively, report that and avoid calibrated probability claims.
5. **M4: policy benefit.** Demonstrate a reproducible risk/compute/progress Pareto improvement over batch/LR baselines, including probe overhead. Only then integrate with `BatchController`.

The likely worthwhile initial deliverable is **a stability benchmark and a trustworthy diagnostic**, not a new optimizer algorithm. It should remain possible to conclude that SNR gating, DoPr, moment schedules, and better batches already solve enough of the problem that a curvature-aware controller is unjustified.

## References and local context

- Tian & Paquette (2026), [*Simulating Training Spikes of Deep Neural Networks by Layer-wise Random Feature Models*](https://arxiv.org/abs/2610.13524): PLRF quadratic spikes, failure of concentration for concentrated power-law spectra, checkpoint-local layerwise stochastic simulation, Adam `beta2` case study.
- [Existing optimizer-aware batch control proposal](optimizer-aware-batch-control.md): Euclidean, Adam-aware and Muon-aware sensors, direct batch-efficiency measurement, controlled batch/LR actions.
- [Current optimizer implementation](../../src/snr_grad/_core.py): `SNRAdamW`, spectral variants and moment/gate update semantics.
- [Variance estimators](../../src/snr_grad/variance.py): exact and microbatch gradient variance, correctly normalized to batch-mean variance.
- [Activation preconditioner](../../src/snr_grad/activation.py): damped activation covariance and DoPr.
- [Naganuma et al., *Adaptive Batch Sizes Using Non-Euclidean Gradient Noise Scales*](https://arxiv.org/abs/2602.03001): optimizer-specific gradient noise geometries.
- [Merrill et al., *Critical Batch Size Revisited*](https://arxiv.org/abs/2505.23971): empirical batch-efficiency validation beyond simple noise-scale proxies.

## 11. Implementation status and results

Implemented: `src/snr_grad/stability.py`, `curvature.py`, `stability_sim.py`, the read-only
`ActivationPreconditioner.covariance_snapshot()`, `benchmark_stability_plrf.py`,
`benchmark_stability_layers.py`, `tests/test_stability.py` and `tests/test_stability_sim.py`.
There are two deviations from the sketch above. First, `frozen_update_map(..., at_gradient=g)` gives Adam's *exact* diagonal
one-step Jacobian: holding `v` fixed was measurably wrong early in training, when
`(1-beta2)/(1-beta2**t)` is large. Second, PLRF minibatches are sampled exactly in the
D-dimensional feature space, because `(W^T x, <x, b>)` is jointly Gaussian.

Default-run outcomes (D=256, V=4096, B=8; `alpha_spectrum` in {0.3, 0.5, 0.7, 0.9}; SGD,
AdamW, SNRAdamW; 240 held-out checkpoints):

- **M0 passed.** The diffuse spectrum (0.3) stays smooth up to 0.95x the measured edge,
  and concentrated spectra show intermittent excursions without sustained divergence.
- **M1 passed with caveats.** `R` has a median relative error of 13% at K=32, but the
  jackknife 2-SE coverage is only 84%, so reported SEs are optimistic. Concentration is
  bias-corrected and abstains at 25% of checkpoints.
- **M2 is partial.** Curvature-weighted noise-to-loss beats GNS: GNS is anti-correlated
  with severity, and pairs at matched GNS are ordered correctly 75% of the time. But it
  does not beat `lr * lambda_max(P H)`, which reaches Spearman 0.99 within SGD and AdamW.
  H2 holds for AdamW (optimizer-aware penalty 0.70 vs raw 0.30).
- **M3: the simulator ranks well.** Spearman is 0.975 pooled, matched-GNS concordance is
  0.92 (sharpness: 0.85), and SNRAdamW scores 0.97 (sharpness: 0.82). The probe-fitted
  noise variant reaches 0.935. On a small MLP the GGN simulator ranks real continuations
  well (0.96), but it costs more than running them; the activation-proxy simulator fails.
- **M4 was not attempted**, per the go/no-go rule. The useful deliverable so far is the
  diagnostic and the benchmark, as section 10 anticipated.

## 12. Follow-up study

[`studies/stability_study`](../../studies/stability_study/README.md) tests the first-round
conclusions more deeply. The findings that change the picture:

- **H1 holds once noise varies independently of sharpness.** At a fixed learning rate,
  varying only the batch size leaves SGD's sharpness constant (it cannot rank), while
  the noise penalty and the simulator rank severity at about 0.95.
- **One-step sensors are not enough for Adam.** On a concentrated spectrum, AdamW spike
  severity *increases* with batch size (B=2: 5.0, B=64: 8.4). Only the multi-step
  simulator ranks every (optimizer, spectrum) cell correctly. This supports H3, and it
  argues against any controller that responds to spike risk by simply raising the batch
  size.
- **The simulator is calibrated on PLRF.** Resampled noise from 128 microbatches with
  loss-proportional scaling gives an event-fraction error of 0.07-0.09 and no bias; the
  oracle noise gives 0.045 (M3 is strengthened).
- **SNR gating (Stage 2):** it suppresses spikes and improves progress at a moderately
  concentrated spectrum. At the most concentrated spectrum the default gate does not
  suppress them, and a slow variance EMA (`rho=0.999`) produces rare but very large
  excursions. The soft gate suppresses spikes at a modest cost in progress.
- **Networks:** truncated curvature (low-rank, or a GGN on fewer samples x outputs than
  parameters) makes the surrogate unbounded below unless the dynamics are projected
  onto the curvature subspace. With projection, a rank 4-16 GGN ranks MLP severity at
  0.85-0.98 up to 284k parameters. A cost advantage over real continuations is not yet
  shown.

M4 (a controller) remains unstarted. The evidence now favours a simulator-based risk
check over one-step sensors, but the cost case needs a model where real steps are
expensive.
