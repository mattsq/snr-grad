"""
SNR / population-risk preconditioner for AdamW (arXiv:2605.01172).
"""

from snr_grad._core import (
    AlphaSpec,
    GateType,
    SNRAdamW,
    SNRAdamWStats,
    SNRMuon,
    RotatedSNRAdamW,
    SpectralSNRMuon,
    MARSSNRAdamW,
    SNRScheduleFreeAdamW,
    SNRScheduleFreeMuon,
    RotatedSNRScheduleFreeAdamW,
    SpectralSNRScheduleFreeMuon,
    compute_gate,
    per_sample_variance_term,
    resolve_alpha,
)
from snr_grad.adaptive import (
    AdaptiveThresholdConfig,
)
from snr_grad.activation import (
    ActivationPrecondConfig,
    ActivationPreconditioner,
    DoPr,
)
from snr_grad.variance import (
    VarianceEstimator,
    ExactVarianceEstimator,
    MicrobatchVarianceEstimator,
    per_sample_grad_variances,
    backward_with_microbatch_variance,
    compare_gate_with_external_variance,
    tree_batch_size,
    tree_split,
)

__all__ = [
    "AlphaSpec",
    "GateType",
    "SNRAdamW",
    "SNRAdamWStats",
    "SNRMuon",
    "RotatedSNRAdamW",
    "SpectralSNRMuon",
    "MARSSNRAdamW",
    "SNRScheduleFreeAdamW",
    "SNRScheduleFreeMuon",
    "RotatedSNRScheduleFreeAdamW",
    "SpectralSNRScheduleFreeMuon",
    "compute_gate",
    "per_sample_variance_term",
    "resolve_alpha",
    "AdaptiveThresholdConfig",
    "ActivationPrecondConfig",
    "ActivationPreconditioner",
    "DoPr",
    "VarianceEstimator",
    "ExactVarianceEstimator",
    "MicrobatchVarianceEstimator",
    "per_sample_grad_variances",
    "backward_with_microbatch_variance",
    "compare_gate_with_external_variance",
    "tree_batch_size",
    "tree_split",
]
