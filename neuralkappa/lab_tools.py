from __future__ import annotations

from typing import Sequence

import numpy as np

from ._exceptions import DomainError, InsufficientDataError
from ._types import ArrayLike1D
from .metrics import compute_si, si_to_kappa
from .preprocessing import timestamps_to_isis


def compute_trial_averaged_kappa(list_of_isi_arrays: Sequence[ArrayLike1D]) -> float:
    """Compute trial-level SI values, average SI, then convert mean SI to kappa."""
    if len(list_of_isi_arrays) == 0:
        raise InsufficientDataError("list_of_isi_arrays must contain at least one trial.")

    si_vals = np.array([compute_si(trial) for trial in list_of_isi_arrays], dtype=np.float64)
    mean_si = float(np.mean(si_vals))
    return float(si_to_kappa(np.array([mean_si], dtype=np.float64))[0])


def compute_kappa_from_timestamps(timestamps: ArrayLike1D) -> float:
    """Convert timestamps to ISIs and compute kappa."""
    isis = timestamps_to_isis(timestamps)
    si = compute_si(isis)
    return float(si_to_kappa(np.array([si], dtype=np.float64))[0])


def compare_kappa_conditions(
    trials_a: Sequence[ArrayLike1D],
    trials_b: Sequence[ArrayLike1D],
    *,
    n_perm: int = 5000,
    random_state: int | None = None,
) -> dict[str, float | int]:
    """Compare trial-level kappa between two conditions using a permutation test."""
    if len(trials_a) == 0 or len(trials_b) == 0:
        raise InsufficientDataError("Both conditions must contain at least one ISI trial.")
    if n_perm <= 0:
        raise DomainError("n_perm must be a positive integer.")

    kappas_a = np.array(
        [si_to_kappa(np.array([compute_si(trial)], dtype=np.float64))[0] for trial in trials_a],
        dtype=np.float64,
    )
    kappas_b = np.array(
        [si_to_kappa(np.array([compute_si(trial)], dtype=np.float64))[0] for trial in trials_b],
        dtype=np.float64,
    )

    observed = float(np.mean(kappas_a) - np.mean(kappas_b))
    pooled = np.concatenate((kappas_a, kappas_b))
    n_a = kappas_a.size
    rng = np.random.default_rng(seed=random_state)

    extreme = 0
    for _ in range(n_perm):
        permuted = rng.permutation(pooled)
        stat = float(np.mean(permuted[:n_a]) - np.mean(permuted[n_a:]))
        if abs(stat) >= abs(observed):
            extreme += 1

    p_value = float((extreme + 1) / (n_perm + 1))
    return {
        "kappa_a_mean": float(np.mean(kappas_a)),
        "kappa_b_mean": float(np.mean(kappas_b)),
        "effect_mean_diff": observed,
        "p_value": p_value,
        "n_trials_a": int(kappas_a.size),
        "n_trials_b": int(kappas_b.size),
        "n_perm": int(n_perm),
    }
