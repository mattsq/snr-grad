# When do stochastic loss spikes get large? Closed-form indices

These derivations back `src/snr_grad/stability_theory.py`. They are validated in
`studies/stability_study` (E8). The setting is a loss that is locally quadratic, with
Hessian `H` and per-example gradient covariance `Sigma`, trained with batch `B` and
learning rate `eta`.

## 1. SGD with Gaussian-design noise: exact mean dynamics

For least squares with Gaussian features `z ~ N(0, H)`, write `e = theta - theta*`. The
least-squares residual `eps = z.theta* - y` is independent of `z`, with
`E[eps^2] = 2 L*`. One SGD step is

    e' = (I - eta S) e + eta (1/B) sum_b z_b eps_b,        S = (1/B) sum_b z_b z_b^T.

The Wishart identity `E[S A S] = (1 + 1/B) H A H + tr(H A) H / B` gives, for
`c_i = E[e_i^2]` in the eigenbasis of `H`,

    c_i' = a_i c_i + (eta^2 / B) lambda_i (sum_j lambda_j c_j + 2 L*),
    a_i  = 1 - 2 eta lambda_i + eta^2 lambda_i^2 (1 + 1/B).

This is a diagonal map plus a rank-one feedback. Its stationary point gives
`sum_j lambda_j c_j = 2 L* kappa / (1 - kappa)`, with

    kappa = (eta / B) sum_i w_i,     w_i = lambda_i / (2 - eta lambda_i (1 + 1/B)).

So:

- **Stationary expected loss.** `E[L] = L* / (1 - kappa)`, so the excess is
  `E[L] - L* = L* kappa / (1 - kappa)`, and **`kappa` is exactly the noise-driven
  fraction of the loss**.
- **Stochastic edge.** The mean loss diverges at `kappa = 1`, which lies below the
  deterministic edge `eta lambda_max (1 + 1/B) = 2`. For small `eta lambda`,
  `kappa ~ eta tr(H) / (2B)`.
- **Concentration.** The stationary excess is spread over modes in proportion to
  `w_i`, so `D_K = (sum w)^2 / sum w^2` is the number of modes carrying the
  noise-driven loss.

## 2. Fluctuations: Gaussian part, `Phi = kappa sqrt(2 / D_K)`

If each mode's excess is approximately an independent scaled chi-square variable, the
excess loss has relative standard deviation `sqrt(2 / D_K)`. Since the excess is a
fraction `kappa` of the loss, the loss itself fluctuates by about

    Phi = kappa * sqrt(2 / D_K).

Large relative excursions therefore need *both* a large noise-driven share of the
loss *and* few modes carrying it. With one dominant mode (`D_K ~ 1`) the loss behaves
like `chi^2_1`, whose 95th percentile is about 8x its median. Such a loss looks spiky
even without any instability.

## 3. Heavy tails near the edge: Kesten recursion for the dominant mode

Along a dominant mode, `x' = (1 - eta lambda chi) x + noise` with `chi ~ chi^2_B / B`.
Random linear recursions of this kind have stationary power-law tails,
`P(|x| > u) ~ u^-a`, where `a` solves `E|1 - eta lambda chi|^a = 1` (Kesten). The loss
(`~ x^2`) therefore has tail exponent `xi = a / 2`:

- `xi <= 1`: the mean loss is infinite, giving large intermittent bursts. For a single
  mode this coincides exactly with `kappa >= 1`.
- `1 < xi <= 2`: the variance is infinite, giving visible spikes.
- The Lyapunov exponent `E log|1 - eta lambda chi|` stays negative well beyond
  `kappa = 1`, so typical trajectories keep contracting.

Combining sections 1-3 gives a phase diagram, confirmed on PLRF:

| regime | condition | behaviour |
| --- | --- | --- |
| A | `kappa < 1` | finite mean; excursion size set by `Phi` and `xi` |
| B | `kappa >= 1`, `D_K ~ 1` | infinite mean but typical runs contract: rare huge spikes (severity about 1e3) |
| C | `kappa >= 1`, `D_K >~ 2` | mean-field feedback is self-averaging, so runs diverge |

A diffuse spectrum has a large `D_K`, so its `kappa` reaches 1, and it diverges, before
its tails ever get heavy. Large spikes need a concentrated spectrum.

## 4. General preconditioned optimizers (Adam) and networks

Linearize at a frozen diagonal preconditioner `P` (Adam: `P = 1/sqrt(v_hat)`). With
`h_i, u_i` the eigenpairs of `P^1/2 H P^1/2`, the per-example noise along `u_i` is
`sigma_i = u_i^T P^1/2 Sigma P^1/2 u_i`. Mode `i` then contributes a stationary excess
of

    E_i = eta sigma_i / (2 B * 2 (1 - eta h_i / edge)),

where `edge = 2` for SGD and `edge = 2(1 + beta1)/(1 - beta1) = 38` for Adam's EMA
momentum (Cohen et al. 2022). For Gaussian design this reduces to section 1. Then

    kappa_hat = sum E_i / L,    D_hat = (sum E)^2 / sum E^2,    Phi = kappa_hat sqrt(2 / D_hat).

A horizon factor `1 - exp(-2 eta h_i T)` removes slow modes that cannot equilibrate
within a window of `T` steps. For networks, `h_i, u_i` come from a low-rank
preconditioned GGN (`PreconditionedCurvature` with `LowRankCurvature`), and `sigma_i`
from probe deviations. Modes beyond the rank are lumped at the residual curvature.

**Adam's adaptive edge** is `ratio = eta h_max / 38`. Once `v` equilibrates to
`mu^2 + sigma^2 / B`, `P` grows as the loss and per-example variance fall, and grows
roughly as `sqrt(B)`. That predicts Adam instabilities at low loss and *large* batch.
On PLRF the ratio rises with B as predicted, but it explains severity much less well
than `Phi` (R^2 0.43 vs 0.86).

## 5. Width of the spike window: alignment of noise with the sharpest direction

With a frozen preconditioner, scaling the learning rate by `s` gives mode `i` an excess
of `E_i ~ s sigma_i / (1 - s r_i)`, where `r_i = eta h_i / edge`. Only the sharpest mode
diverges at the edge. It dominates the noise-driven loss (making `D -> 1` and `Phi`
large) only in the last fraction

    w ~ sigma_1 / sum_i sigma_i

of the way to the edge: the share of gradient noise along the sharpest direction. For
Gaussian design, `w = lambda_1 / tr(H)`. If `w` is small, the run reaches the
(adaptive) edge along a "quiet" direction and diverges or stalls without large
noise-driven spikes. On PLRF, spectra with `w >= 0.12` spike and the spectrum with
`w = 0.02` diverges first. Adam's per-coordinate whitening can destroy the alignment:
on the tiny LM, `w` was 0.1-0.16 in SGD geometry but about 1e-3 in Adam geometry at
the same checkpoint.

**Scope of the Kesten term (section 3).** The heavy-tail amplification assumes that
the noise along the sharp mode is proportional to the error along it (Gaussian
design). On the tiny LM, where the sharpest direction carries only 0.4-4% of the
noise, the Kesten-based severity over-predicted by 1-5 orders of magnitude, while `Phi`
alone was within about 1-8x. Use `xi` only when that structure holds.

## 6. How to use these to plan an experiment

On PLRF, large excursions (severity of order 1 to 10, i.e. 100%-1000% above the
trailing baseline) appear at `Phi ~ 0.15-0.7` (Adam calibration:
`severity ~ 15 Phi^1.38`). A configuration is worth an expensive spike experiment only
if, in the *optimizer's own* geometry and at a learning rate below its edge:

1. `w >~ 0.1`: the sharpest direction carries at least 10% of the gradient noise;
2. `Phi >~ 0.15`: the noise-driven, concentrated part of the loss is large;
3. `kappa < 1`, or `D ~ 1`: otherwise the run diverges instead of spiking.

All three can be measured at a checkpoint in seconds to minutes (a probe plus a few
dozen GGN products). Screen with them before scaling up.

## 7. Adam's preconditioner memory (study E9)

The sections above treat `P = 1/sqrt(v_hat)` as frozen. Adam's own spikes come from
`v_hat` *changing*: it decays at rate `beta2` when the gradients it sees are smaller than
what it remembers, so the effective step `eta/sqrt(v_hat)` grows.

- **Noise floor.** In steady state `v >= mu^2 + sigma^2/B`. Evaluating the edge ratio
  with `P_f = 1/(sigma/sqrt(B))` gives `R_floor`; if `R_floor < 1`, even a fully decayed
  memory cannot reach the adaptive edge. On a noisy one-dimensional quadratic the
  effective step `eta h/sqrt(v_hat)` follows about `38 R_floor/(R_floor + 2)` (a fit),
  and any of the noise levels tested removes the memory spikes that appear without
  noise at `beta2 = 0.999`. `B_crit = B / R_floor^2` and an onset time are in
  `adam_memory_forecast`.
- **Edge amplitude.** At the edge the gradient must refill `sqrt(v)` along the top
  whitened direction `u`, which costs
  `Delta L_edge = eta^2 h_w / (2 * 38^2 * max_i P_i u_i^2)` (`adam_edge_amplitude`).
  Observed peaks on quadratics are 20-50 times this.
- **Reactivation gain.** A coordinate whose memory has decayed and whose gradient then
  returns steadily steps at most
  `A(beta1, beta2) = max_k (1-beta1^k)/sqrt(1-beta2^k) ~ 0.8 sqrt((1-beta1)/(1-beta2))`
  times `eta` (`adam_reactivation_gain`, which also handles a finite silence). A steady
  coordinate steps at most about `eta`. `A` is 1.1 for `beta2 = 0.95`, 2.1 for 0.99
  and 6.6 for 0.999 (`beta1 = 0.9`), and it depends on `(1-beta2)/(1-beta1)`, the
  variable of the empirical (beta1, beta2) spike phase diagram.

**What holds in networks.** In a tiny LM on a confident language, spikes appeared at
`beta2 = 0.999` with `R_floor` well below 1. They were carried by a few converged bias
vectors whose gradient *and* gradient noise vanish, so a network-level noise floor does
not protect them. `A` bounded the observed overshoot, but at equal `A` a high `beta1`
kept the realized overshoot far below it (the returning gradient is not coherent for
`1/(1-beta1)` steps). The realized overshoot `max_i |m_i|/sqrt(v_hat_i)`, read from the
optimizer state, separated spiking from quiet runs with AUROC 0.97. None of the
curvature-based indices of sections 1-6 respond to this mechanism.
