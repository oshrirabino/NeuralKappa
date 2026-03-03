## Auther: Oshrira
## Using codex agent

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._exceptions import DomainError, InsufficientDataError, InvalidISIError
from ._lut import _GLOBAL_LUT
from ._types import ArrayLike1D
from .preprocessing import validate_isis


def compute_si(isis: ArrayLike1D) -> float:
    """Compute raw spiking irregularity SI from consecutive ISIs."""
    arr = validate_isis(isis, min_length=2)
    ti = arr[:-1]
    ti1 = arr[1:]
    local = -0.5 * np.log((4.0 * ti * ti1) / ((ti + ti1) ** 2))
    return float(np.mean(local))


def si_to_kappa(si_array: ArrayLike1D) -> NDArray[np.float64]:
    """Convert SI values to kappa via a precomputed LUT interpolator."""
    si = np.asarray(si_array, dtype=np.float64)
    if si.ndim == 0:
        si = si.reshape(1)
    if si.ndim != 1:
        raise InvalidISIError("si_array must be a 1D array-like input.")
    if not np.all(np.isfinite(si)):
        raise InvalidISIError("SI values must be finite.")
    return _GLOBAL_LUT.convert(si)


def compute_cv(isis: ArrayLike1D) -> float:
    """Compute coefficient of variation of ISIs."""
    arr = validate_isis(isis, min_length=2)
    mean = np.mean(arr)
    if mean <= 0:
        raise DomainError("Mean ISI must be positive for CV.")
    return float(np.std(arr, ddof=0) / mean)


def compute_fano(isis: ArrayLike1D, window_ms: float = 500.0) -> float:
    """Compute Fano factor by binning reconstructed spike times in fixed windows."""
    arr = validate_isis(isis, min_length=2)
    if not np.isfinite(window_ms) or window_ms <= 0:
        raise DomainError("window_ms must be a positive finite scalar.")

    timestamps = np.cumsum(arr)
    t_max = float(timestamps[-1])

    if t_max <= window_ms:
        raise InsufficientDataError("Need enough duration for at least two counting windows.")

    edges = np.arange(0.0, t_max + window_ms, window_ms, dtype=np.float64)
    if edges.size < 3:
        raise InsufficientDataError("Need at least two bins to compute Fano factor.")

    counts, _ = np.histogram(timestamps, bins=edges)
    mean_counts = float(np.mean(counts))
    if mean_counts == 0.0:
        raise DomainError("Mean spike count per window is zero; Fano factor is undefined.")

    return float(np.var(counts, ddof=0) / mean_counts)
