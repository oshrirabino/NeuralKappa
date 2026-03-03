from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._exceptions import InsufficientDataError, InvalidISIError
from ._types import ArrayLike1D


def validate_isis(isis: ArrayLike1D, *, min_length: int = 2) -> NDArray[np.float64]:
    """Validate and normalize ISIs to a finite positive 1D float64 numpy array."""
    arr = np.asarray(isis, dtype=np.float64)
    if arr.ndim != 1:
        raise InvalidISIError("ISIs must be a 1D array.")
    if arr.size < min_length:
        raise InsufficientDataError(f"Need at least {min_length} ISIs, got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise InvalidISIError("ISIs must be finite.")
    if np.any(arr <= 0):
        raise InvalidISIError("ISIs must be strictly positive.")
    return arr


def timestamps_to_isis(timestamps: ArrayLike1D) -> NDArray[np.float64]:
    """Convert strictly increasing timestamps into ISIs via first differences."""
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.ndim != 1:
        raise InvalidISIError("Timestamps must be a 1D array.")
    if ts.size < 2:
        raise InsufficientDataError("Need at least 2 timestamps to compute ISIs.")
    if not np.all(np.isfinite(ts)):
        raise InvalidISIError("Timestamps must be finite.")
    if np.any(np.diff(ts) <= 0):
        raise InvalidISIError("Timestamps must be strictly increasing.")
    return np.diff(ts)
