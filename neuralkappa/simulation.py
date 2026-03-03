from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._exceptions import DomainError
from ._types import ArrayLike1D


def generate_rate_modulated_gamma(
    base_kappa: float,
    rate_profile: ArrayLike1D,
    *,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """Generate synthetic ISIs under a rate-modulated gamma process.

    `rate_profile` is interpreted as instantaneous firing rates in spikes/ms.
    One gamma-distributed ISI sample is drawn per profile point.
    """
    if not np.isfinite(base_kappa) or base_kappa <= 0:
        raise DomainError("base_kappa must be a positive finite scalar.")

    rates = np.asarray(rate_profile, dtype=np.float64)
    if rates.ndim != 1 or rates.size == 0:
        raise DomainError("rate_profile must be a non-empty 1D array.")
    if not np.all(np.isfinite(rates)):
        raise DomainError("rate_profile must contain finite values.")
    if np.any(rates <= 0):
        raise DomainError("rate_profile must contain strictly positive rates.")

    rng = np.random.default_rng(random_state)
    scale = 1.0 / (base_kappa * rates)
    return rng.gamma(shape=base_kappa, scale=scale, size=rates.shape).astype(np.float64)
