from __future__ import annotations

from typing import Sequence

import numpy as np

from ._exceptions import InsufficientDataError
from ._types import ArrayLike1D
from .metrics import compute_si, si_to_kappa


def compute_trial_averaged_kappa(list_of_isi_arrays: Sequence[ArrayLike1D]) -> float:
    """Compute trial-level SI values, average SI, then convert mean SI to kappa."""
    if len(list_of_isi_arrays) == 0:
        raise InsufficientDataError("list_of_isi_arrays must contain at least one trial.")

    si_vals = np.array([compute_si(trial) for trial in list_of_isi_arrays], dtype=np.float64)
    mean_si = float(np.mean(si_vals))
    return float(si_to_kappa(np.array([mean_si], dtype=np.float64))[0])
