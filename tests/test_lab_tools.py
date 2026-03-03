import numpy as np

from neuralkappa.lab_tools import build_clustering_matrix, compute_trial_averaged_kappa
from neuralkappa.metrics import compute_si, si_to_kappa


def test_trial_averaged_kappa_definition():
    trials = [np.array([10.0, 10.0, 10.0]), np.array([8.0, 12.0, 9.0, 11.0])]
    expected = si_to_kappa(np.array([np.mean([compute_si(t) for t in trials])]))[0]
    got = compute_trial_averaged_kappa(trials)
    assert np.isclose(got, expected)


def test_build_clustering_matrix_shape_and_standardization():
    X = build_clustering_matrix([1, 2, 3], [5, 6, 7], [10, 11, 12], standardize=True)
    assert X.shape == (3, 3)
    assert np.allclose(np.mean(X, axis=0), 0.0)
