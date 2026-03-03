from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from ._exceptions import DomainError, InsufficientDataError
from ._types import ArrayLike1D
from .metrics import compute_si, si_to_kappa


def compute_trial_averaged_kappa(list_of_isi_arrays: Sequence[ArrayLike1D]) -> float:
    """Compute trial-level SI values, average SI, then convert mean SI to kappa."""
    if len(list_of_isi_arrays) == 0:
        raise InsufficientDataError("list_of_isi_arrays must contain at least one trial.")

    si_vals = np.array([compute_si(trial) for trial in list_of_isi_arrays], dtype=np.float64)
    mean_si = float(np.mean(si_vals))
    return float(si_to_kappa(np.array([mean_si], dtype=np.float64))[0])


def build_clustering_matrix(
    kappas: ArrayLike1D,
    rates: ArrayLike1D,
    widths: ArrayLike1D,
    *,
    standardize: bool = True,
) -> NDArray[np.float64]:
    """Build a [kappa, rate, width] feature matrix for scikit-learn clustering."""
    k = np.asarray(kappas, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    w = np.asarray(widths, dtype=np.float64)

    if any(arr.ndim != 1 for arr in (k, r, w)):
        raise DomainError("kappas, rates, and widths must all be 1D arrays.")
    if not (k.size == r.size == w.size):
        raise DomainError("kappas, rates, and widths must have equal lengths.")
    if k.size == 0:
        raise InsufficientDataError("Feature arrays must not be empty.")

    matrix = np.column_stack((k, r, w)).astype(np.float64, copy=False)
    if not np.all(np.isfinite(matrix)):
        raise DomainError("All feature values must be finite.")

    if standardize:
        scaler = StandardScaler(copy=True)
        matrix = scaler.fit_transform(matrix)

    return matrix
