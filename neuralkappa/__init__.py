"""NeuralKappa: rate-independent irregularity metrics for neural spiking."""

from .lab_tools import compare_kappa_conditions, compute_kappa_from_timestamps, compute_trial_averaged_kappa
from .metrics import compute_cv, compute_fano, compute_si, si_to_kappa
from .simulation import generate_rate_modulated_gamma

__all__ = [
    "compute_si",
    "si_to_kappa",
    "compute_cv",
    "compute_fano",
    "compute_trial_averaged_kappa",
    "compute_kappa_from_timestamps",
    "compare_kappa_conditions",
    "generate_rate_modulated_gamma",
]
