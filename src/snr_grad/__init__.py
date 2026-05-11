"""
SNR / population-risk preconditioner for AdamW (arXiv:2605.01172).
"""

from snr_grad._core import (
    AlphaSpec,
    GateType,
    SNRAdamW,
    SNRAdamWStats,
    compute_gate,
    per_sample_variance_term,
    resolve_alpha,
)

__all__ = [
    "AlphaSpec",
    "GateType",
    "SNRAdamW",
    "SNRAdamWStats",
    "compute_gate",
    "per_sample_variance_term",
    "resolve_alpha",
]
