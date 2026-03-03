import numpy as np
import pytest

from neuralkappa.metrics import compute_cv, compute_fano, compute_si, si_to_kappa


def test_compute_si_regular_train_is_zero():
    isis = np.array([10.0, 10.0, 10.0, 10.0])
    assert np.isclose(compute_si(isis), 0.0)


def test_si_to_kappa_poisson_reference_near_one():
    si_poisson = 1.0 - np.log(2.0)
    kappa = si_to_kappa(np.array([si_poisson]))[0]
    assert np.isclose(kappa, 1.0, rtol=1e-3)


def test_compute_cv_matches_definition():
    isis = np.array([1.0, 2.0, 3.0])
    expected = np.std(isis) / np.mean(isis)
    assert np.isclose(compute_cv(isis), expected)


def test_compute_fano_regular_counts_zero():
    isis = np.array([100.0] * 100)
    fano = compute_fano(isis, window_ms=500.0)
    # Edge-bin effects can yield a small non-zero value for perfectly regular trains.
    assert 0.0 <= fano < 0.05


def test_compute_fano_requires_duration():
    isis = np.array([100.0, 100.0])
    with pytest.raises(Exception):
        compute_fano(isis, window_ms=500.0)
