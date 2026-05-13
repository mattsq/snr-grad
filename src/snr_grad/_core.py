"""
SNR / population-risk preconditioner for AdamW.

Implements the per-parameter gate described in arXiv:2605.01172:
    s_t = rho * s_{t-1} + (1 - rho) * (g_t - m_{t-1})^2

with Adam moments:
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2

and gated AdamW update:
    w <- w - lr * q * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * w

Supported gates:
    hard: q = 1[m_hat^2 > alpha * s_hat]
    soft: q = relu(m_hat^2 - alpha * s_hat) /
              (relu(m_hat^2 - alpha * s_hat) + lambda_pop * s_hat + gate_eps)
    snr:  q = m_hat^2 / (m_hat^2 + lambda_pop * s_hat + gate_eps)

The "soft" gate is the paper's Algorithm 1 default. The "snr" gate is the smoother SNR
shrinker. The "hard" gate is useful for ablations and debugging.

This file is intentionally self-contained: drop it into a project and use SNRAdamW
where you would normally use torch.optim.AdamW.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Union, Literal, Any

import torch
from torch import Tensor
from torch.optim import Optimizer


GateType = Literal["soft", "snr", "hard"]
AlphaSpec = Union[float, int, Literal["online", "fresh", "fresh_batch", "finite", "finite_dataset"]]


def resolve_alpha(
    alpha: AlphaSpec,
    *,
    batch_size: Optional[int] = None,
    dataset_size: Optional[int] = None,
) -> float:
    """
    Resolve the leave-one-out coefficient alpha.

    Paper defaults:
      - online / fresh-batch boundary: alpha = 1
      - finite-dataset boundary: alpha = b / (n - b)

    Args:
        alpha:
            Numeric alpha, or one of:
              "online", "fresh", "fresh_batch" -> 1.0
              "finite", "finite_dataset" -> batch_size / (dataset_size - batch_size)
        batch_size:
            Current minibatch size, required for finite-dataset alpha unless set on optimizer group.
        dataset_size:
            Training-set size, required for finite-dataset alpha unless set on optimizer group.
    """
    if isinstance(alpha, (float, int)):
        if alpha < 0:
            raise ValueError(f"Numeric alpha must be non-negative, got {alpha}.")
        return float(alpha)

    key = alpha.lower()
    if key in {"online", "fresh", "fresh_batch"}:
        return 1.0

    if key in {"finite", "finite_dataset"}:
        if batch_size is None or dataset_size is None:
            raise ValueError(
                "alpha='finite' requires batch_size and dataset_size, either in the "
                "optimizer param group or passed to step(...)."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if dataset_size <= batch_size:
            raise ValueError(
                f"dataset_size must be larger than batch_size for finite alpha; "
                f"got dataset_size={dataset_size}, batch_size={batch_size}."
            )
        return float(batch_size) / float(dataset_size - batch_size)

    raise ValueError(f"Unknown alpha spec: {alpha!r}.")


def compute_gate(
    m_hat: Tensor,
    s_hat: Tensor,
    *,
    gate: GateType = "snr",
    alpha: float = 1.0,
    lambda_pop: float = 1.0,
    gate_eps: float = 1e-12,
) -> Tensor:
    """
    Compute the per-parameter SNR/population-risk gate q.

    m_hat and s_hat must already be bias-corrected.
    """
    m2 = m_hat.square()

    if gate == "hard":
        return (m2 > alpha * s_hat).to(dtype=m_hat.dtype)

    if gate == "soft":
        delta = torch.relu(m2 - alpha * s_hat)
        return delta / (delta + lambda_pop * s_hat + gate_eps)

    if gate == "snr":
        return m2 / (m2 + lambda_pop * s_hat + gate_eps)

    raise ValueError(f"Unknown gate: {gate!r}. Expected 'soft', 'snr', or 'hard'.")


def per_sample_variance_term(per_sample_grads: Tensor) -> Tensor:
    """
    Exact diagonal variance term for a minibatch, on the same scale as s_hat.

    Input shape is [batch, ...parameter_shape...], containing per-example gradients.
    The paper defines Sigma_B = (1 / b) * sum_i (g_i - g_bar)^2 and uses
    Sigma_B / (b - 1) in the gate. This equals the usual unbiased sample variance / b.

    Returns:
        Tensor with shape [...parameter_shape...].
    """
    if per_sample_grads.ndim < 1:
        raise ValueError("per_sample_grads must have a batch dimension.")
    b = per_sample_grads.shape[0]
    if b < 2:
        raise ValueError("Need at least two per-example gradients to estimate variance.")
    return per_sample_grads.var(dim=0, unbiased=True) / b


@dataclass
class SNRAdamWStats:
    """
    Lightweight diagnostics from the most recent optimizer step.
    """
    mean_gate: float
    min_gate: float
    max_gate: float
    mean_s_hat: float
    mean_m2: float
    parameters_seen: int


class SNRAdamW(Optimizer):
    """
    AdamW with the SNR / population-risk gate from arXiv:2605.01172.

    Main use:
        optimizer = SNRAdamW(
            model.parameters(),
            lr=3e-4,
            gate="snr",           # "snr" (default), "soft" (paper Algorithm 1), or "hard"
            lambda_pop=1.0,
            alpha="online",       # or "finite" with batch_size + dataset_size
            rho=0.99,
            weight_decay=0.01,
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    Finite-dataset correction:
        optimizer = SNRAdamW(
            model.parameters(),
            alpha="finite",
            batch_size=128,
            dataset_size=len(train_dataset),
        )

    Exact variance override:
        If you compute per-example gradient variance yourself, pass a dict mapping
        parameter objects to tensors on the same scale as s_hat:
            optimizer.step(grad_variances={param: variance_term})
        The variance term should be Sigma_B / (b - 1), equivalently
        unbiased per-example gradient variance / batch_size.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        rho: float = 0.99,
        eps: float = 1e-8,
        gate_eps: float = 1e-12,
        weight_decay: float = 0.0,
        gate: GateType = "snr",
        lambda_pop: float = 1.0,
        alpha: AlphaSpec = "online",
        batch_size: Optional[int] = None,
        dataset_size: Optional[int] = None,
        maximize: bool = False,
        track_stats: bool = True,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")
        if gate_eps <= 0:
            raise ValueError(f"Invalid gate_eps: {gate_eps}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0 <= rho < 1:
            raise ValueError(f"Invalid rho: {rho}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if lambda_pop < 0:
            raise ValueError(f"Invalid lambda_pop: {lambda_pop}")
        if gate not in {"soft", "snr", "hard"}:
            raise ValueError(f"Invalid gate: {gate!r}")

        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            eps=eps,
            gate_eps=gate_eps,
            weight_decay=weight_decay,
            gate=gate,
            lambda_pop=lambda_pop,
            alpha=alpha,
            batch_size=batch_size,
            dataset_size=dataset_size,
            maximize=maximize,
            track_stats=track_stats,
        )
        super().__init__(params, defaults)
        self.last_stats: Optional[SNRAdamWStats] = None

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Any] = None,
        *,
        batch_size: Optional[int] = None,
        dataset_size: Optional[int] = None,
        grad_variances: Optional[Mapping[Tensor, Tensor]] = None,
    ) -> Optional[float]:
        """
        Perform one optimizer step.

        Args:
            closure:
                Optional closure, as in standard PyTorch optimizers.
            batch_size, dataset_size:
                Optional per-step values used only when alpha='finite'.
            grad_variances:
                Optional mapping param -> exact variance term on the same scale as s_hat.
                When supplied for a parameter, this replaces the streaming EMA s_hat for
                that parameter in the gate. The internal EMA is still updated for continuity.

        Returns:
            The closure loss, if a closure was provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        gate_sum = 0.0
        gate_min = float("inf")
        gate_max = float("-inf")
        s_sum = 0.0
        m2_sum = 0.0
        elem_count = 0
        parameters_seen = 0

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            eps = group["eps"]
            gate_eps = group["gate_eps"]
            wd = group["weight_decay"]
            gate_type: GateType = group["gate"]
            lambda_pop = group["lambda_pop"]
            alpha_value = resolve_alpha(
                group["alpha"],
                batch_size=batch_size if batch_size is not None else group.get("batch_size"),
                dataset_size=dataset_size if dataset_size is not None else group.get("dataset_size"),
            )
            maximize = group["maximize"]
            track_stats = group["track_stats"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SNRAdamW does not support sparse gradients.")

                grad = grad.detach()
                if maximize:
                    grad = -grad

                state: MutableMapping[str, Any] = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_grad_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]
                exp_grad_var: Tensor = state["exp_grad_var"]

                state["step"] += 1
                step_num: int = state["step"]

                # Paper's variance state uses previous first moment m_{t-1}.
                m_prev = exp_avg.clone()

                exp_grad_var.mul_(rho).addcmul_(grad - m_prev, grad - m_prev, value=1.0 - rho)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step_num
                bias_correction2 = 1.0 - beta2 ** step_num
                bias_correction_s = 1.0 - rho ** step_num

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                s_hat = exp_grad_var / bias_correction_s

                if grad_variances is not None and p in grad_variances:
                    exact_s = grad_variances[p].to(device=p.device, dtype=p.dtype)
                    if exact_s.shape != p.shape:
                        raise ValueError(
                            f"grad_variances entry for parameter has shape {tuple(exact_s.shape)}, "
                            f"expected {tuple(p.shape)}."
                        )
                    s_for_gate = exact_s
                else:
                    s_for_gate = s_hat

                q = compute_gate(
                    m_hat,
                    s_for_gate,
                    gate=gate_type,
                    alpha=alpha_value,
                    lambda_pop=lambda_pop,
                    gate_eps=gate_eps,
                )

                # Decoupled weight decay, matching AdamW and the paper's update.
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

                update = q * m_hat / (v_hat.sqrt() + eps)
                p.add_(update, alpha=-lr)

                if track_stats:
                    q_detached = q.detach()
                    s_detached = s_for_gate.detach()
                    m2_detached = m_hat.detach().square()
                    n = q_detached.numel()

                    gate_sum += float(q_detached.sum().cpu())
                    gate_min = min(gate_min, float(q_detached.min().cpu()))
                    gate_max = max(gate_max, float(q_detached.max().cpu()))
                    s_sum += float(s_detached.sum().cpu())
                    m2_sum += float(m2_detached.sum().cpu())
                    elem_count += n
                    parameters_seen += 1

        if elem_count > 0:
            self.last_stats = SNRAdamWStats(
                mean_gate=gate_sum / elem_count,
                min_gate=gate_min,
                max_gate=gate_max,
                mean_s_hat=s_sum / elem_count,
                mean_m2=m2_sum / elem_count,
                parameters_seen=parameters_seen,
            )
        else:
            self.last_stats = None

        return loss


def _newton_schulz_orthogonalize(matrix: Tensor, *, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate UV^T (semi-orthogonal factor) with Newton-Schulz iteration."""
    if matrix.ndim != 2:
        raise ValueError("_newton_schulz_orthogonalize expects a 2D tensor.")

    m = matrix
    transposed = False
    if m.shape[0] < m.shape[1]:
        m = m.t()
        transposed = True

    norm = m.norm() + eps
    y = m / norm
    eye = torch.eye(y.shape[1], dtype=y.dtype, device=y.device)
    for _ in range(steps):
        yty = y.transpose(0, 1) @ y
        y = 0.5 * y @ (3.0 * eye - yty)

    if transposed:
        y = y.t()
    return y


class SNRMuon(Optimizer):
    """SNR-gated Muon-style optimizer for 2D parameters + AdamW fallback for others."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        rho: float = 0.99,
        eps: float = 1e-8,
        gate_eps: float = 1e-12,
        weight_decay: float = 0.0,
        gate: GateType = "snr",
        lambda_pop: float = 1.0,
        alpha: AlphaSpec = "online",
        batch_size: Optional[int] = None,
        dataset_size: Optional[int] = None,
        maximize: bool = False,
        muon_ns_steps: int = 5,
        muon_mode: Literal["post", "pre"] = "post",
    ):
        defaults = dict(
            lr=lr, betas=betas, rho=rho, eps=eps, gate_eps=gate_eps, weight_decay=weight_decay,
            gate=gate, lambda_pop=lambda_pop, alpha=alpha, batch_size=batch_size,
            dataset_size=dataset_size, maximize=maximize, muon_ns_steps=muon_ns_steps,
            muon_mode=muon_mode,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            eps = group["eps"]
            wd = group["weight_decay"]
            maximize = group["maximize"]
            alpha_value = resolve_alpha(group["alpha"], batch_size=group.get("batch_size"), dataset_size=group.get("dataset_size"))

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if maximize:
                    g = -g
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                    st["exp_grad_var"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]

                m = st["exp_avg"]
                v = st["exp_avg_sq"]
                s = st["exp_grad_var"]
                m_prev = m.clone()

                s.mul_(rho).addcmul_(g - m_prev, g - m_prev, value=1.0 - rho)
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                m_hat = m / (1.0 - beta1**t)
                v_hat = v / (1.0 - beta2**t)
                s_hat = s / (1.0 - rho**t)
                q = compute_gate(m_hat, s_hat, gate=group["gate"], alpha=alpha_value, lambda_pop=group["lambda_pop"], gate_eps=group["gate_eps"])

                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

                base_update = m_hat / (v_hat.sqrt() + eps)
                if p.ndim == 2:
                    if group["muon_mode"] == "pre":
                        update = _newton_schulz_orthogonalize(q * base_update, steps=group["muon_ns_steps"])
                    else:
                        update = q * _newton_schulz_orthogonalize(base_update, steps=group["muon_ns_steps"])
                else:
                    update = q * base_update
                p.add_(update, alpha=-lr)
        return loss


class RotatedSNRAdamW(Optimizer):
    """SOAP-style rotated-basis SNRAdamW for 2D parameters (AdamW fallback otherwise)."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        rho: float = 0.99,
        eps: float = 1e-8,
        gate_eps: float = 1e-12,
        weight_decay: float = 0.0,
        gate: GateType = "soft",
        lambda_pop: float = 1.0,
        alpha: AlphaSpec = "online",
        basis_beta: float = 0.95,
        basis_update_interval: int = 50,
        maximize: bool = False,
    ):
        defaults = dict(
            lr=lr, betas=betas, rho=rho, eps=eps, gate_eps=gate_eps, weight_decay=weight_decay,
            gate=gate, lambda_pop=lambda_pop, alpha=alpha, basis_beta=basis_beta,
            basis_update_interval=basis_update_interval, maximize=maximize,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            eps = group["eps"]
            wd = group["weight_decay"]
            maximize = group["maximize"]
            alpha_value = resolve_alpha(group["alpha"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if maximize:
                    g = -g
                if g.is_sparse:
                    raise RuntimeError("RotatedSNRAdamW does not support sparse gradients.")

                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    if p.ndim == 2:
                        o, i = p.shape
                        st["L_cov"] = torch.eye(o, device=p.device, dtype=torch.float32)
                        st["R_cov"] = torch.eye(i, device=p.device, dtype=torch.float32)
                        st["QL"] = torch.eye(o, device=p.device, dtype=torch.float32)
                        st["QR"] = torch.eye(i, device=p.device, dtype=torch.float32)
                        st["M_c"] = torch.zeros_like(p, dtype=torch.float32)
                        st["V_c"] = torch.zeros_like(p, dtype=torch.float32)
                        st["S_c"] = torch.zeros_like(p, dtype=torch.float32)
                    else:
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                        st["exp_grad_var"] = torch.zeros_like(p)

                st["step"] += 1
                t = st["step"]

                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

                if p.ndim != 2:
                    m, v, s = st["exp_avg"], st["exp_avg_sq"], st["exp_grad_var"]
                    m_prev = m.clone()
                    s.mul_(rho).addcmul_(g - m_prev, g - m_prev, value=1 - rho)
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    m_hat = m / (1 - beta1**t)
                    v_hat = v / (1 - beta2**t)
                    s_hat = s / (1 - rho**t)
                    q = compute_gate(m_hat, s_hat, gate=group["gate"], alpha=alpha_value, lambda_pop=group["lambda_pop"], gate_eps=group["gate_eps"])
                    p.add_(q * m_hat / (v_hat.sqrt() + eps), alpha=-lr)
                    continue

                G = g.float()
                basis_beta = group["basis_beta"]
                st["L_cov"].mul_(basis_beta).add_(G @ G.t(), alpha=1 - basis_beta)
                st["R_cov"].mul_(basis_beta).add_(G.t() @ G, alpha=1 - basis_beta)

                if t % group["basis_update_interval"] == 0:
                    QL_old, QR_old = st["QL"], st["QR"]
                    _, QL_new = torch.linalg.eigh(st["L_cov"])
                    _, QR_new = torch.linalg.eigh(st["R_cov"])
                    A = QL_new.t() @ QL_old
                    B = QR_old.t() @ QR_new
                    st["M_c"] = A @ st["M_c"] @ B
                    st["S_c"] = A.square() @ st["S_c"] @ B.square()
                    st["V_c"] = A.square() @ st["V_c"] @ B.square()
                    st["QL"], st["QR"] = QL_new, QR_new

                QL, QR = st["QL"], st["QR"]
                Gc = QL.t() @ G @ QR
                M, V, S = st["M_c"], st["V_c"], st["S_c"]
                M_prev = M.clone()
                S.mul_(rho).addcmul_(Gc - M_prev, Gc - M_prev, value=1 - rho)
                M.mul_(beta1).add_(Gc, alpha=1 - beta1)
                V.mul_(beta2).addcmul_(Gc, Gc, value=1 - beta2)

                M_hat = M / (1 - beta1**t)
                V_hat = V / (1 - beta2**t)
                S_hat = S / (1 - rho**t)
                q = compute_gate(M_hat, S_hat, gate=group["gate"], alpha=alpha_value, lambda_pop=group["lambda_pop"], gate_eps=group["gate_eps"])
                Uc = q * M_hat / (V_hat.sqrt() + eps)
                update = QL @ Uc @ QR.t()
                p.add_(update.to(dtype=p.dtype), alpha=-lr)
        return loss


class SpectralSNRMuon(Optimizer):
    """SVD-basis SNR gating with diagonal or full spectral coefficients."""

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3, momentum: float = 0.9, betas: tuple[float, float] = (0.9, 0.95), rho: float = 0.99, eps: float = 1e-8, gate_eps: float = 1e-12, weight_decay: float = 0.0, gate: GateType = "soft", lambda_pop: float = 1.0, alpha: AlphaSpec = "online", variant: Literal["muon_spectral_gate", "adam_spectral_gate"] = "adam_spectral_gate", mode: Literal["diag", "full"] = "diag"):
        defaults = dict(lr=lr, momentum=momentum, betas=betas, rho=rho, eps=eps, gate_eps=gate_eps, weight_decay=weight_decay, gate=gate, lambda_pop=lambda_pop, alpha=alpha, variant=variant, mode=mode)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("SpectralSNRMuon does not support sparse gradients.")
                if p.ndim != 2:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["M"] = torch.zeros_like(p, dtype=torch.float32)
                    if group["mode"] == "diag":
                        r = min(p.shape)
                        st["a"] = torch.zeros(r, device=p.device)
                        st["s"] = torch.zeros(r, device=p.device)
                        st["v"] = torch.zeros(r, device=p.device)
                    else:
                        r = min(p.shape)
                        st["A"] = torch.zeros((r, r), device=p.device, dtype=torch.float32)
                        st["S"] = torch.zeros((r, r), device=p.device, dtype=torch.float32)
                        st["V"] = torch.zeros((r, r), device=p.device, dtype=torch.float32)
                st["step"] += 1
                t = st["step"]
                G = g.float()
                M = st["M"]
                M.mul_(group["momentum"]).add_(G, alpha=1 - group["momentum"])
                U, _, Vh = torch.linalg.svd(M, full_matrices=False)
                V = Vh.t()
                C = U.t() @ G @ V
                b1, b2 = group["betas"]
                rho = group["rho"]
                alpha_value = resolve_alpha(group["alpha"])
                if group["mode"] == "diag":
                    c = C.diag()
                    a, s, v = st["a"], st["s"], st["v"]
                    a_prev = a.clone()
                    s.mul_(rho).addcmul_(c - a_prev, c - a_prev, value=1 - rho)
                    a.mul_(b1).add_(c, alpha=1 - b1)
                    v.mul_(b2).addcmul_(c, c, value=1 - b2)
                    a_hat = a / (1 - b1**t)
                    s_hat = s / (1 - rho**t)
                    v_hat = v / (1 - b2**t)
                    q = compute_gate(a_hat, s_hat, gate=group["gate"], alpha=alpha_value, lambda_pop=group["lambda_pop"], gate_eps=group["gate_eps"])
                    if group["variant"] == "muon_spectral_gate":
                        d = q
                    else:
                        d = q * a_hat / (v_hat.sqrt() + group["eps"])
                    D = U @ torch.diag(d) @ V.t()
                else:
                    A, S, Vst = st["A"], st["S"], st["V"]
                    A_prev = A.clone()
                    S.mul_(rho).addcmul_(C - A_prev, C - A_prev, value=1 - rho)
                    A.mul_(b1).add_(C, alpha=1 - b1)
                    Vst.mul_(b2).addcmul_(C, C, value=1 - b2)
                    A_hat = A / (1 - b1**t)
                    S_hat = S / (1 - rho**t)
                    V_hat = Vst / (1 - b2**t)
                    q = compute_gate(A_hat, S_hat, gate=group["gate"], alpha=alpha_value, lambda_pop=group["lambda_pop"], gate_eps=group["gate_eps"])
                    coeff = q if group["variant"] == "muon_spectral_gate" else q * A_hat / (V_hat.sqrt() + group["eps"])
                    D = U @ coeff @ V.t()
                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])
                p.add_(D.to(dtype=p.dtype), alpha=-group["lr"])
        return loss
