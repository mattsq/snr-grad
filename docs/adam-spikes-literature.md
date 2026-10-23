# Adam loss spikes: literature notes and the angle taken in study E9

A short survey made after study E8 found that tiny Adam-trained LMs show no noise-driven
spikes. The aim was to understand Adam-specific mechanisms. Claims below are as reported
by the cited papers, mostly from abstracts and HTML full texts, and have not been
independently re-derived except where noted.

## Mechanisms in the literature

- **Adaptive edge of stability.** Cohen et al., *Adaptive Gradient Methods at the Edge of
  Stability* ([arXiv:2207.14484](https://arxiv.org/abs/2207.14484)). Adam is stable iff
  `lambda_max(P H) < 2(1+beta1) / ((1-beta1) eta) = 38/eta` for `beta1 = 0.9`, with
  `P = diag(1/(sqrt(v)+eps))`. In full-batch training the preconditioned sharpness
  equilibrates at that threshold, and the effect is stronger at larger batch sizes.

- **Second-moment decay ("adaptive preconditioners trigger spikes").** Bai et al.
  ([arXiv:2506.04805](https://arxiv.org/abs/2506.04805)). When gradients become small
  relative to `sqrt(v_hat)`, `v_hat` decays at rate `beta2`, so the effective step
  `eta/sqrt(v_hat)` grows. A spike starts when the *gradient-directional* preconditioned
  curvature `g^T H~ g / |g|^2` crosses the threshold and stays there. The gradient then
  grows, refills `v_hat`, and stability returns.
  - This happens even on a fixed quadratic, at learning rates far below the classical
    threshold.
  - Reported mitigations: lower `beta2`, larger `eps`, higher `beta1`.
  - Most experiments are full-batch, or batch 2048 on synthetic data.

- **Time-correlated vanishing gradients and eps.** Molybog et al., *A Theory on Adam
  Instability in Large-Scale Machine Learning*
  ([arXiv:2304.09871](https://arxiv.org/abs/2304.09871); runs of 7B-546B parameters).
  1. Early-layer gradients fall to the `eps` scale (four layers of the 546B model sat
     below `1e-8`).
  2. Large batches make the gradients temporally correlated.
  3. The update ratio `m/sqrt(v)` becomes bimodal near `+/- beta1/sqrt(beta2)`.
  4. A rare batch then triggers a chain reaction.

  Spikes are more likely with depth and large batches; lowering the learning rate or
  the batch size and skipping batches help.

- **(beta1, beta2) phase diagram and "core-wall" geometry.** *Beyond Quadratic Loss: The
  Stability Phase Diagram of Adam* ([arXiv:2609.18314](https://arxiv.org/abs/2609.18314)).
  Spiky iff `(1-beta2)/(1-beta1) < C`, with `C` about 2-4 in six model-task settings.
  - Confident cross-entropy gives a flat quadratic core and a steep superquadratic wall
    whose width is comparable to an Adam step.
  - On a pure quadratic the boundary is cubic: `1-beta2 ~ 15.6 (1-beta1)^3`.
  - Batch noise was not varied; most runs are full-batch.

- **Practical reports.**
  - AdaGC ([arXiv:2502.11034](https://arxiv.org/abs/2502.11034)): `beta2 = 0.999`
    caused recurrent spikes and 0.95 helped.
  - SPAM ([arXiv:2501.06842](https://arxiv.org/abs/2501.06842)): gradient spikes up to
    1000x typical; resetting the moments helps.
  - COPUS ([arXiv:2604.26687](https://arxiv.org/abs/2604.26687)): spikes at batch-size
    increases, because `v` still reflects the old gradient scale.
  - *Small Batch Size Training for Language Models*
    ([arXiv:2507.07101](https://arxiv.org/abs/2507.07101)): hold the `beta2` half-life
    fixed in tokens.
  - *Spike No More* ([arXiv:2312.16903](https://arxiv.org/abs/2312.16903)): gradient-norm
    growth.

- **Other mechanisms.**
  - Attention-logit growth and output-logit divergence, reproduced in small models at
    high learning rates: Wortsman et al.
    ([arXiv:2309.14322](https://arxiv.org/abs/2309.14322)).
  - Weight-norm criticality from normalization plus weight decay
    ([arXiv:2607.21005](https://arxiv.org/abs/2607.21005)).
  - Warmup lowers the initially high preconditioned sharpness: Kalra and Barkeshli
    ([arXiv:2406.09405](https://arxiv.org/abs/2406.09405)).

## The angle taken in E9: noise protects Adam

These accounts share one ingredient: **Adam's preconditioner memory `v_hat` decays when
the gradient energy it sees falls below what it remembers.** Gradient noise puts a floor
under that decay: in steady state `v >= mu^2 + sigma^2/B`. That gives three checkpoint
quantities:

- **Floor ratio.** `R_floor = eta * lambda_max(P_f^1/2 H P_f^1/2) / 38`, with
  `P_f = 1/(sigma/sqrt(B) + eps)`. If `R_floor < 1`, even a fully decayed `v` cannot
  reach the edge: Adam is *protected* by its own noise.
- **Critical batch.** Since `R_floor ~ sqrt(B)`, protection is lost above
  `B_crit = B / R_floor^2`.
- **Onset time.** With actual ratio `R_act < 1` and `R_floor > 1`, `v_hat` must shrink
  by `R_act^2`. That takes about `t* = 2 ln(1/R_act) / ln(1/beta2)` steps if gradients
  are negligible compared with the memory.

This explains, with one mechanism, why noisy small-batch training with high-entropy
targets (our tiny LMs) does not spike, why PLRF Adam spiked more at larger batch (E1,
E8a), and why large-batch, confident LLM training does. It also predicts a testable
`B_crit` in small models, which E9 checks.

## What E9 found

Details are in the study README (section E9).

- **On a quadratic the angle holds.** Noise caps the effective step as predicted, and
  removes the memory spikes completely. The edge-oscillation amplitude follows the
  closed form within a constant factor.
- **In a tiny LM it does not.** On a confident language (no copy rule, temperature 0.03)
  the model spikes at `beta2 = 0.999` (5 of 5 seeds, severity 2.3-24.5) with
  `R_floor` 0.2-0.55. The spikes are carried by a few converged bias vectors in the
  first block, whose gradient noise vanishes along with their gradient. The rare-token
  embedding rows are not the cause.
- **The reactivation gain** `A(beta1, beta2) = max_k (1-beta1^k)/sqrt(1-beta2^k)`
  bounds how far a reactivated coordinate overshoots. It explains why `beta2 = 0.95`
  (A = 1.1) never spiked. At equal `A`, raising `beta1` keeps the realized overshoot
  well below the bound, which matches the mitigations reported by Bai et al. and the
  phase diagram's dependence on `(1-beta2)/(1-beta1)`.
- **A free sensor.** The realized overshoot `max |m|/sqrt(v_hat)` per tensor separates
  spiking from quiet runs with AUROC 0.97. Among E9's runs no run below 1.85 lr
  spiked, but E10 found two smaller events (severity 1.3 and 2.2) at about 1 lr, so it
  marks the large spikes, not every event. It is not a precise timer either.
- **Mitigation (E10).** Bounding the per-coordinate update ratio at 1, by clipping or
  by refilling `v_hat`, removed the large spikes at no loss cost; `beta1 = 0.98` and
  `beta2 = 0.95` also worked, AMSGrad only partly, and `eps` on biases not at all. The
  guard is a per-coordinate relative of the update clipping in Adafactor and
  StableAdamW.
