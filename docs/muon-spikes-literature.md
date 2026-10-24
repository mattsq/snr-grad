# Muon loss spikes: literature notes and how they relate to the Adam memory guard

A short survey made before extending `GuardedOptimizer` to Muon-style optimizers (study
E11). It follows [`adam-spikes-literature.md`](adam-spikes-literature.md). Claims are as
reported by the cited papers, mostly from abstracts and full texts, and were not
re-derived except where noted. The study's findings are in
[`studies/stability_study/README.md`](../studies/stability_study/README.md) (E11).

## Background: what a Muon step is

For a weight matrix with momentum `M`, Muon steps by `lr_eff * NS(M / ||M||_F)`, where
`NS` is a few Newton-Schulz iterations that map each singular value `s` of the normalized
momentum (all in `(0, 1]`) towards 1, keeping the singular vectors. So the step is
approximately the polar factor `U V^T`. Bernstein and Newhouse showed that this is
steepest descent under the spectral norm: the step's spectral norm is the lr, whatever
the gradient's size. `lr_eff` carries a shape-dependent factor:
`sqrt(max(1, rows/cols))` in Jordan's implementation, or `0.2 sqrt(max(rows, cols))`
(Moonlight), which matches AdamW's update RMS. Embeddings, the output head, biases and
norms are usually trained by AdamW alongside.

## Mechanisms reported for Muon

- **Weight-norm growth from spectral flattening.** Musec
  ([arXiv:2609.11655](https://arxiv.org/abs/2609.11655); NanoGPT, 491M-1.63B
  parameters).
  - Setting every singular value of the momentum to about 1 gives weakly represented
    (small-singular-value) directions the same step as dominant ones.
  - In NanoGPT-Medium at lr 0.2, Muon's weight spectral norms reached 300-400 in QK,
    value-output and MLP matrices, with large oscillations. Spectral clipping kept them
    below 10.
  - Muon and MuonClip showed "large loss spikes and sustained oscillations" from lr 0.1
    in the 613M and 1.63B models, and diverged at the largest learning rates.
  - Their fix replaces the polar factor by a clipped momentum,
    `H(M, D) = D (M M^T + D^2 I)^{-1/2} M`, which keeps small singular values small
    (soft clipping, computed by coupled Newton-Schulz). The clipped momentum is fed back
    into the momentum EMA.

- **Attention-logit explosion.** Kimi K2
  ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534)).
  - In a 9B-active/53B-total MoE trained with vanilla Muon, max attention logits
    "quickly exceeded 1000", which led to loss spikes and occasional divergence. The
    report says this occurred more often with Muon than with AdamW.
  - Appendix E argues that Muon updates have full effective rank (all singular values
    equal), unlike Adam's low-rank ones. Singular directions of the weights and of the
    update then align more often, and the weight's singular values grow additively.
  - The fix, QK-Clip, applies after each step: where a head's max logit `S` exceeds
    `tau = 100`, its query and key weights are scaled by `sqrt(tau / S)`. K2 was
    pretrained on 15.5T tokens without a loss spike.
  - Musec notes that QK-Clip leaves value-output and MLP norms elevated.

- **Weight decay and update scale are needed at scale.** Moonlight
  ([arXiv:2502.16982](https://arxiv.org/abs/2502.16982)) found two changes essential for
  scaling Muon: decoupled weight decay, without which weight and output RMS grew too
  large, and per-matrix update scaling to AdamW's RMS. Muown
  ([arXiv:2605.10797](https://arxiv.org/abs/2605.10797)) reports that weight spectral
  norms rise over training without decoupled weight decay, driven mainly by the largest
  rows.

- **Noise in small singular directions.** The polar map is most sensitive to noise where
  singular values are small or nearly degenerate
  ([arXiv:2606.08388](https://arxiv.org/abs/2606.08388),
  [arXiv:2509.11983](https://arxiv.org/abs/2509.11983)). Orthogonalization therefore
  turns small, noise-dominated parts of the momentum into full-size steps.

- **Spectral clipping as a general wrapper.** SPECTRA
  ([arXiv:2603.14315](https://arxiv.org/abs/2603.14315), ICML 2026) clips the update's
  spectral norm after any base optimizer (AdamW, Signum, AdEMAMix), optionally clips the
  gradient before it, and reports lower loss and smaller weight norms.

- **Variants that bring back a second moment.** NorMuon
  ([arXiv:2510.05491](https://arxiv.org/abs/2510.05491)) keeps a per-row EMA of the
  squared orthogonalized update and divides by its square root, because Muon's per-neuron
  update norms are very uneven. A memory of this kind can go stale in the Adam way (for
  a row whose update fell silent).

## How this relates to the Adam memory mechanism

1. **Muon's matrices have no memory to go stale.** Its normalizer is the current
   momentum, so the E9 reactivation overshoot, `A(beta1, beta2)` up to 6.6 lr, cannot
   occur. The largest step is the largest singular value that the Newton-Schulz
   polynomial can produce, `snr_grad.muon_update_gain()`:
   - 1.2024 lr_eff for Jordan's quintic coefficients `(3.4445, -4.7750, 2.0315)`, at any
     number of iterations. The PyTorch docstring's "Uniform(0.5, 1.5)" overstates the
     top.
   - 1 for a convergent iteration.

   In units of `lr_eff`, every Muon step is about the same size; Muon behaves like a
   sign method in the spectral norm.
2. **So the per-coordinate guard does not fit Muon's matrices.** With
   `adjust_lr_fn="match_rms_adamw"`, a typical Muon coordinate already moves about 1.1 lr
   per step (E11), so a per-coordinate cap at 1 lr would clip Muon on every step.
   - The natural cap for Muon is on the step's spectral norm, the norm Muon's lr is
     defined in.
   - For plain Muon that cap is a safety net. It matters for variants whose step can
     exceed the polar factor: NorMuon-style row normalization with a stale row,
     element-wise Adam on the orthogonalized update, gates, or bugs.
3. **Muon's documented failure is cumulative, not a single-step overshoot.**
   - Flattening keeps pushing weak directions at full size, so weight norms and
     attention logits grow over many bounded steps.
   - A per-step guard cannot see this. The literature's remedies act on the state:
     weight decay, QK-Clip, or spectral clipping of the momentum (Musec) or of the
     update (SPECTRA).
4. **The Adam half of a Muon setup keeps the Adam mechanism.** Embeddings, the head,
   biases and norms are trained by AdamW with its usual `beta2`. E9 located the tiny-LM
   spikes in the first block's biases, so the existing guard still applies there.

## What E11 found

Details are in the study README (section E11). Tiny LM, E10's spiking setting, Muon on
the hidden matrices and AdamW on the rest:

- **Muon removes the memory spikes even though its Adam half still overshoots.** 1 of 6
  runs had an event (severity 0.36), against 6 of 6 for AdamW. The biases and norms
  still stepped 3.4-5.3 lr. Under AdamW, the matrices themselves had taken steps of up
  to 97 times Muon's spectral step, so the spikes needed those matrix overshoots.
- **Guarding the Adam half removes the rest** (0 of 6). Muon's matrix steps sat at
  exactly the Newton-Schulz bound of 1.20 lr_eff, and the spectral cap never
  triggered.
- **Muon's own failure appears at high lr and is cumulative.** From lr 0.01 to 0.3 the
  weight spectral norm grew from 22 to about 550, and the attention logits from
  hundreds to about 6e5 (3x AdamW's already at lr 0.01). The loss degraded at lr 0.3
  without isolated spikes. Weight decay bounded the growth but did not restore that
  loss at this scale.
