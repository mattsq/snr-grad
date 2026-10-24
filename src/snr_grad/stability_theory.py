"""
Closed-form stability indices for stochastic training on (locally) quadratic losses.

Two mechanisms can produce loss excursions below the classical deterministic edge
``lr * lambda_max < 2``. Each has a cheap index here; both are validated in
``studies/stability_study`` (E8).

**SGD: stochastic (multiplicative-noise) edge.** For least squares with Gaussian
features ``z ~ N(0, H)`` (exactly the PLRF setting) and minibatch size ``B``, write
``e = theta - theta*`` and ``c_i = E[e_i^2]`` in the eigenbasis of ``H``. Using
``E[S A S] = (1 + 1/B) H A H + tr(H A) H / B`` for the Wishart minibatch
second-moment ``S``, the mean dynamics are exactly linear::

    c_i' = a_i c_i + (lr^2 / B) lambda_i (sum_j lambda_j c_j + 2 L*)
    a_i  = 1 - 2 lr lambda_i + lr^2 lambda_i^2 (1 + 1/B)

The rank-one feedback term makes the mean loss unstable exactly when::

    kappa = (lr / B) sum_i lambda_i / (2 - lr lambda_i (1 + 1/B))   reaches 1

(or a denominator reaches 0, the deterministic edge). Below it, the stationary
expected excess loss is exactly ``L* kappa / (1 - kappa)``. The feedback is carried
by the terms ``w_i = lambda_i / (2 - lr lambda_i (1 + 1/B))``; their concentration
``D_K = (sum w)^2 / sum w^2`` measures how many modes carry it. A small ``D_K``
means the feedback does not self-average, so the loss has intermittent bursts
rather than a smooth plateau.

**Adam: adaptive edge.** Linearizing Adam at a frozen preconditioner
``P = diag(1 / (sqrt(v) + eps))``, a mode of ``h = lambda(P^1/2 H P^1/2)`` follows
heavy-ball dynamics with EMA momentum ``beta1``, stable iff::

    lr * h < 2 (1 + beta1) / (1 - beta1)      (= 38 for beta1 = 0.9)

(Cohen et al. 2022). Noise matters through ``v``: once the second moment
equilibrates, ``v_i ~ mu_i^2 + sigma_i^2 / B``. As the loss and per-example
variance ``sigma^2`` fall, or as ``B`` grows, ``v`` shrinks and the effective
sharpness rises. That predicts spikes at low loss and *large* batch, with ``beta2``
setting how long ``v`` lags. :func:`adam_edge_ratio` evaluates the ratio for any
preconditioner; :func:`adam_stationary_preconditioner` builds the equilibrium one
from probe statistics.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor

from snr_grad.curvature import CurvatureProxy, flatten_params, unflatten_like

__all__ = [
    "sgd_stochastic_edge",
    "dominant_mode_tail",
    "noise_fluctuation_index",
    "adam_edge_threshold",
    "adam_stationary_preconditioner",
    "preconditioned_top_eigenvalue",
    "adam_edge_ratio",
    "adam_memory_forecast",
    "adam_edge_amplitude",
]


def sgd_stochastic_edge(eigenvalues: Tensor | Sequence[float], lr: float, batch_size: int) -> dict:
    """Exact mean-loss stability of plain SGD on a Gaussian-design quadratic.

    Args:
        eigenvalues: Hessian eigenvalues (non-negative).
        lr: Learning rate (no momentum).
        batch_size: Independent examples per step.

    Returns:
        ``kappa`` (mean-loss feedback gain; >= 1 or inf = unstable),
        ``deterministic_margin`` ``2 - lr lambda_max (1 + 1/B)`` (<= 0 = unstable),
        ``amplification`` ``1 / (1 - kappa)``, ``excess_over_floor`` = stationary
        ``E[L] - L*`` divided by ``L*`` (``kappa / (1 - kappa)``), and the feedback
        concentration ``D_K``.
    """
    lam = torch.as_tensor(eigenvalues, dtype=torch.float64).clamp_min(0.0).reshape(-1)
    if batch_size < 1 or lr <= 0:
        raise ValueError("lr must be > 0 and batch_size >= 1.")
    c = 1.0 + 1.0 / batch_size
    denom = 2.0 - lr * lam * c
    margin = float(denom.min())
    if margin <= 0:
        return {"kappa": float("inf"), "deterministic_margin": margin, "amplification": float("inf"),
                "excess_over_floor": float("inf"), "D_K": float("nan")}
    w = lam / denom
    kappa = lr / batch_size * float(w.sum())
    stable = kappa < 1
    return {
        "kappa": kappa,
        "deterministic_margin": margin,
        "amplification": 1.0 / (1.0 - kappa) if stable else float("inf"),
        "excess_over_floor": kappa / (1.0 - kappa) if stable else float("inf"),
        "D_K": float(w.sum() ** 2 / (w * w).sum()) if float((w * w).sum()) > 0 else float("nan"),
    }


def dominant_mode_tail(
    lr_lambda: float,
    batch_size: int,
    *,
    samples: int = 40_000,
    seed: int = 0,
) -> dict:
    """Kesten tail of SGD along a dominant curvature mode (concentrated spectra).

    With one mode carrying the noise feedback (small ``D_K``), the error along it obeys
    ``x' = (1 - lr lambda chi) x + additive noise`` with ``chi ~ chi^2_B / B``. Such a
    random linear recursion has a power-law stationary tail ``P(|x| > u) ~ u^-a`` with
    ``E|1 - lr lambda chi|^a = 1`` (Kesten), so the loss (``~ x^2``) has tail exponent
    ``a / 2``. Regimes:

    * ``lyapunov >= 0``: typical trajectories grow (divergence);
    * ``loss_tail_exponent <= 1``: infinite mean loss, i.e. large intermittent bursts
      (matches ``kappa >= 1`` for a single mode);
    * ``1 < loss_tail_exponent <= 2``: infinite variance, i.e. visible spikes;
    * larger exponents: light tails.

    Monte Carlo over ``chi`` with a private generator (deterministic for a given seed).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    gen = torch.Generator().manual_seed(seed)
    chunks = []
    left = samples
    while left > 0:
        n = min(left, max(1, 2_000_000 // batch_size))
        chunks.append(torch.randn(n, batch_size, generator=gen, dtype=torch.float64).pow(2).mean(1))
        left -= n
    chi = torch.cat(chunks)
    logm = (1.0 - lr_lambda * chi).abs().clamp_min(1e-300).log()
    lyap = float(logm.mean())
    out = {"lyapunov": lyap, "tail_index": float("nan"), "loss_tail_exponent": float("nan")}
    if lyap >= 0:
        out.update(tail_index=0.0, loss_tail_exponent=0.0)
        return out

    def log_moment(a: float) -> float:
        return float(torch.logsumexp(a * logm, 0) - math.log(logm.numel()))

    lo, hi = 0.0, 1.0
    while log_moment(hi) < 0 and hi < 1e4:
        lo, hi = hi, hi * 2
    if hi >= 1e4:
        out.update(tail_index=float("inf"), loss_tail_exponent=float("inf"))
        return out
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if log_moment(mid) < 0:
            lo = mid
        else:
            hi = mid
    out.update(tail_index=0.5 * (lo + hi), loss_tail_exponent=0.25 * (lo + hi))
    return out


def noise_fluctuation_index(
    curvature_eigs: Tensor | Sequence[float],
    noise_var: Tensor | Sequence[float],
    lr: float,
    batch_size: int,
    loss: float,
    *,
    edge: float = 2.0,
    horizon: Optional[int] = None,
    rest: Optional[tuple[float, float, int]] = None,
) -> dict:
    """Noise-driven loss fraction and its relative fluctuation, ``Phi = kappa sqrt(2 / D)``.

    Per (preconditioned) curvature mode ``i`` with eigenvalue ``h_i`` and per-example
    gradient-noise variance ``sigma_i`` along its eigenvector, the stationary excess
    loss of (momentum-)SGD is ``E_i = lr sigma_i / (2 B (2 - 2 lr h_i / edge))``. For
    Gaussian-design least squares (``sigma_i = 2 L h_i``) ``sum E_i / L`` reduces to
    :func:`sgd_stochastic_edge`'s ``kappa``. Then:

    * ``kappa_hat = sum_i E_i / loss``: fraction of the loss that is noise-driven;
    * ``D_hat = (sum E)^2 / sum E^2``: how many modes carry it;
    * ``Phi = kappa_hat sqrt(2 / D_hat)``: relative SD of the loss if each mode's
      excess is an independent chi-square (Gaussian approximation; heavy tails near
      the edge come on top, see :func:`dominant_mode_tail`).

    Args:
        curvature_eigs: ``h_i`` (e.g. eigenvalues of ``P^1/2 H P^1/2``).
        noise_var: ``sigma_i`` (per-example variance along the same eigenvectors).
        lr, batch_size: Step size and batch.
        loss: Current loss level ``L``.
        edge: Stability limit of ``lr h`` (2 for SGD, ``adam_edge_threshold(beta1)``
            for Adam with ``h`` from ``1/sqrt(v)``).
        horizon: If given, weight modes by ``1 - exp(-2 lr h_i horizon)`` so that
            slow modes that cannot equilibrate within the window do not count.
        rest: Optional ``(h_rest, sigma_rest_total, n_rest)`` for modes beyond the
            resolved ones (e.g. outside a low-rank curvature): ``n_rest`` modes of
            curvature ``h_rest`` sharing total noise variance ``sigma_rest_total``.
    """
    h = torch.as_tensor(curvature_eigs, dtype=torch.float64).reshape(-1).clamp_min(0.0)
    sig = torch.as_tensor(noise_var, dtype=torch.float64).reshape(-1).clamp_min(0.0)
    if h.shape != sig.shape:
        raise ValueError("curvature_eigs and noise_var must have the same length.")
    margin = 1.0 - lr * h / edge
    if bool((margin <= 0).any()):
        return {"kappa_hat": float("inf"), "D_hat": float("nan"), "Phi": float("inf")}
    e = lr * sig / (4.0 * batch_size * margin)
    if horizon is not None:
        e = e * (1.0 - torch.exp(-2.0 * lr * h * horizon))
    tot = float(e.sum())
    sq = float((e * e).sum())
    if rest is not None:
        h_r, sig_r, n_r = rest
        if n_r > 0 and sig_r > 0:
            m_r = 1.0 - lr * h_r / edge
            if m_r <= 0:
                return {"kappa_hat": float("inf"), "D_hat": float("nan"), "Phi": float("inf")}
            e_r = lr * sig_r / (4.0 * batch_size * m_r)
            if horizon is not None:
                e_r *= 1.0 - math.exp(-2.0 * lr * h_r * horizon)
            tot += e_r
            sq += e_r * e_r / n_r
    kappa = tot / loss if loss > 0 else float("inf")
    d = tot * tot / sq if sq > 0 else float("nan")
    return {"kappa_hat": kappa, "D_hat": d,
            "Phi": kappa * math.sqrt(2.0 / d) if math.isfinite(d) and d > 0 else float("nan")}


def adam_edge_threshold(beta1: float) -> float:
    """``2 (1 + beta1) / (1 - beta1)``: stability limit of ``lr * lambda(P H)`` for Adam."""
    if not 0.0 <= beta1 < 1.0:
        raise ValueError(f"beta1 must be in [0, 1), got {beta1}.")
    return 2.0 * (1.0 + beta1) / (1.0 - beta1)


def adam_stationary_preconditioner(
    per_example_var: Mapping[Tensor, Tensor],
    batch_size: int,
    *,
    mean_grad: Optional[Mapping[Tensor, Tensor]] = None,
    eps: float = 1e-8,
) -> dict[Tensor, Tensor]:
    """``1 / (sqrt(mu^2 + sigma^2 / B) + eps)``: Adam's preconditioner once ``v`` has
    equilibrated to the *current* gradient statistics.

    ``per_example_var`` is the per-example (not batch-mean) gradient variance per
    coordinate, e.g. ``microbatch_size * Var_k(g_k)`` from a probe. Omit
    ``mean_grad`` for the pure noise floor.
    """
    out = {}
    for p, s2 in per_example_var.items():
        v = s2 / batch_size
        if mean_grad is not None and p in mean_grad:
            v = v + mean_grad[p] ** 2
        out[p] = 1.0 / (v.clamp_min(0.0).sqrt() + eps)
    return out


def preconditioned_top_eigenvalue(
    curvature: CurvatureProxy,
    params: Sequence[Tensor],
    precond: Mapping[Tensor, Tensor],
    *,
    iters: int = 50,
    tol: float = 1e-4,
    seed: int = 0,
) -> float:
    """Top eigenvalue of ``P^1/2 H P^1/2`` by power iteration on ``curvature.matvec``.

    ``precond`` maps each parameter to its (positive) diagonal ``P``; parameters not
    in it are treated as frozen (``P = 0``). Does not touch the global RNG.
    """
    params = list(params)
    root = [precond[p].clamp_min(0.0).sqrt().reshape(-1).to(torch.float64) if p in precond
            else torch.zeros(p.numel(), dtype=torch.float64) for p in params]
    root_flat = torch.cat(root)
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(root_flat.numel(), generator=gen, dtype=torch.float64)
    x = x / x.norm()
    lam = 0.0
    dtype = params[0].dtype
    for _ in range(iters):
        hv = curvature.matvec(unflatten_like((root_flat * x).to(dtype), params))
        y = root_flat * flatten_params(hv, params).to(torch.float64)
        new = float(x @ y)
        norm = y.norm()
        if float(norm) == 0.0:
            return 0.0
        x = y / norm
        if abs(new - lam) <= tol * max(abs(new), 1e-30):
            lam = new
            break
        lam = new
    return lam


def adam_edge_ratio(
    curvature: CurvatureProxy,
    params: Sequence[Tensor],
    precond: Mapping[Tensor, Tensor],
    lr: float,
    beta1: float,
    **kwargs,
) -> float:
    """``lr * lambda_max(P^1/2 H P^1/2) / (2 (1 + beta1) / (1 - beta1))``; > 1 = unstable.

    ``precond`` is ``1 / (sqrt(v_hat) + eps)`` without the momentum factor (e.g. from
    the optimizer state, or :func:`adam_stationary_preconditioner`).
    """
    lam = preconditioned_top_eigenvalue(curvature, params, precond, **kwargs)
    return lr * lam / adam_edge_threshold(beta1)


def adam_memory_forecast(
    ratio_actual: float,
    ratio_floor: float,
    beta2: float,
    batch_size: Optional[int] = None,
) -> dict:
    """Second-moment-decay ("memory") spike forecast for Adam.

    Adam's preconditioner memory ``v_hat`` decays at rate ``beta2`` whenever the
    gradient energy it sees falls below what it remembers. Gradient noise floors that
    decay at ``sigma^2 / B`` (see ``docs/adam-spikes-literature.md``).

    Args:
        ratio_actual: ``adam_edge_ratio`` with the current ``v_hat``.
        ratio_floor: ``adam_edge_ratio`` with the noise-floor preconditioner
            (:func:`adam_stationary_preconditioner` without ``mean_grad``).
        beta2: Adam's second-moment decay.
        batch_size: Current batch, to report the critical batch.

    Returns:
        ``protected`` (``ratio_floor < 1``: even a fully decayed ``v`` stays below the
        edge); ``critical_batch`` ``B / ratio_floor^2`` (protection lost above it, since
        ``ratio_floor`` grows as ``sqrt(B)``); and ``onset_steps``, the decay time
        ``2 ln(1/ratio_actual) / ln(1/beta2)`` to reach the edge (0 if already past it,
        ``inf`` if protected).
    """
    if not 0.0 < beta2 < 1.0:
        raise ValueError(f"beta2 must be in (0, 1), got {beta2}.")
    protected = ratio_floor < 1.0
    if ratio_actual >= 1.0:
        onset = 0.0
    elif protected or ratio_actual <= 0.0:
        onset = float("inf")
    else:
        onset = 2.0 * math.log(1.0 / ratio_actual) / math.log(1.0 / beta2)
    out = {"protected": protected, "onset_steps": onset}
    if batch_size is not None and ratio_floor > 0:
        out["critical_batch"] = batch_size / ratio_floor ** 2
    return out


def adam_edge_amplitude(
    whitened_eig: float,
    whitened_vec: Tensor,
    precond: Tensor,
    lr: float,
    beta1: float = 0.9,
) -> float:
    """Loss scale of Adam's edge oscillation: ``lr^2 h_w / (2 edge^2 max_i P_i u_i^2)``.

    When Adam reaches its adaptive edge along the top eigenvector ``u`` of
    ``P^1/2 H P^1/2`` (eigenvalue ``h_w``), the instability grows until the gradient
    refills the second moment of the first coordinate to saturate, i.e. until
    ``|g_i| ~ 1/P_i`` for the coordinate maximizing ``P_i u_i^2``. The loss excursion
    this costs is ``Delta L ~ lr^2 h_w / (2 edge^2 max_i P_i u_i^2)``. It is invariant to
    a uniform rescaling of ``v``, so it can be evaluated with the current ``P``, and it
    reduces to ``lr^2 h / (2 * 38^2)`` in one dimension. A direction spread over ``k``
    comparable coordinates gives ``k`` times more, so at a fixed per-coordinate lr the
    amplitude grows with model size. Observed peaks on quadratics are about 20x this
    scale (study E9a).

    Args:
        whitened_eig: ``h_w``, top eigenvalue of ``P^1/2 H P^1/2``.
        whitened_vec: Its unit eigenvector ``u`` (flat).
        precond: Diagonal ``P`` (flat), e.g. ``1/(sqrt(v_hat)+eps)``.
        lr, beta1: Adam step size and momentum.
    """
    u2p = (whitened_vec.reshape(-1).to(torch.float64) ** 2) * precond.reshape(-1).to(torch.float64)
    m = float(u2p.max())
    if m <= 0:
        return float("inf")
    edge = adam_edge_threshold(beta1)
    return lr * lr * whitened_eig / (2.0 * edge * edge * m)


def adam_reactivation_gain(
    beta1: float,
    beta2: float,
    silence: float = math.inf,
    return_ratio: float = 1.0,
    horizon: int = 100_000,
) -> float:
    """Largest per-coordinate Adam step, in units of lr, after a gradient returns.

    A coordinate whose gradient was zero for ``silence`` steps keeps only ``beta2^silence``
    of its second moment ``v0`` (its first moment decays much faster and is taken as zero).
    If a constant gradient ``g`` with ``g^2 = return_ratio * v0`` then returns, step ``k``
    of the return has ``m_k = (1 - beta1^k) g`` and
    ``v_k = beta2^(silence + k) v0 + (1 - beta2^k) g^2``. This returns ``max_k m_k/sqrt(v_k)``
    (late training, bias corrections ~ 1). For an infinite silence it is
    ``A = max_k (1 - beta1^k) / sqrt(1 - beta2^k)``, about ``0.8 sqrt((1-beta1)/(1-beta2))``
    when ``1 - beta2 << 1 - beta1``: 1.1 for ``beta2 = 0.95``, 2.1 for 0.99 and 6.6 for
    0.999 (``beta1 = 0.9``). A steady coordinate steps at most about 1 lr, so ``A``
    measures how far a reactivated coordinate overshoots. It depends on
    ``(1 - beta2)/(1 - beta1)``, the variable of the empirical (beta1, beta2) spike phase
    diagram.

    Args:
        beta1, beta2: Adam moment decay rates.
        silence: Steps without gradient before the return (``inf``: fully decayed memory).
        return_ratio: ``g^2 / v0``, the returning gradient's energy relative to the
            remembered one.
        horizon: Number of return steps scanned.
    """
    k = torch.arange(1, horizon + 1, dtype=torch.float64)
    decay = 0.0 if math.isinf(silence) else beta2 ** silence
    v = decay * beta2 ** k / max(return_ratio, 1e-300) + (1 - beta2 ** k)
    return float(((1 - beta1 ** k) / v.sqrt()).max())


def muon_update_gain(
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    steps: int = 5,
    grid: int = 1_000_001,
) -> float:
    """Largest singular value of a Muon step, in units of its (adjusted) lr.

    Muon divides the momentum ``M`` by its Frobenius norm, so every singular value lies in
    ``(0, 1]``, and then applies ``steps`` Newton-Schulz iterations
    ``x <- a x + b x^3 + c x^5`` to each singular value. This returns the maximum of the
    iterated polynomial over ``(0, 1]``: the most a Muon step can exceed its nominal
    spectral size ``lr``, whatever the momentum or its history. It is Muon's counterpart of
    :func:`adam_reactivation_gain`, with two differences: it does not depend on the
    momentum or on a silence, because Muon keeps no second-moment memory, and it is
    small: 1.2024 for the default quintic coefficients of Jordan et al. (which trade exact
    convergence for speed) and 1 for a convergent iteration such as the cubic
    ``(1.5, -0.5, 0)``. Rounding in bfloat16 can add a little to it.

    Args:
        ns_coefficients: ``(a, b, c)`` of the quintic iteration.
        steps: Number of iterations.
        grid: Number of points scanned on ``(0, 1]``.
    """
    a, b, c = (float(x) for x in ns_coefficients)
    x = torch.linspace(1.0 / grid, 1.0, grid, dtype=torch.float64)
    for _ in range(steps):
        x2 = x * x
        x = x * (a + x2 * (b + c * x2))
    return float(x.max())
