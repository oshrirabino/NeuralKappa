import numpy as np
import pytest

from neuralkappa.lab_tools import compare_kappa_conditions, compute_kappa_from_timestamps, compute_trial_averaged_kappa
from neuralkappa.metrics import compute_si, si_to_kappa
from neuralkappa.preprocessing import timestamps_to_isis


def test_trial_averaged_kappa_definition():
    trials = [np.array([10.0, 10.0, 10.0]), np.array([8.0, 12.0, 9.0, 11.0])]
    expected = si_to_kappa(np.array([np.mean([compute_si(t) for t in trials])]))[0]
    got = compute_trial_averaged_kappa(trials)
    assert np.isclose(got, expected)


def test_compute_kappa_from_timestamps_matches_manual_pipeline():
    ts = np.array([1.0, 2.5, 4.0, 6.0, 9.0])
    expected = si_to_kappa(np.array([compute_si(timestamps_to_isis(ts))]))[0]
    got = compute_kappa_from_timestamps(ts)
    assert np.isclose(got, expected)


def test_compare_kappa_conditions_uses_isi_trials():
    trials_a = [
        np.array([10.0, 10.0, 10.0, 10.0]),
        np.array([9.0, 11.0, 10.0, 10.0]),
        np.array([8.5, 10.5, 10.0, 11.0]),
    ]
    trials_b = [
        np.array([5.0, 20.0, 5.0, 20.0]),
        np.array([6.0, 18.0, 7.0, 17.0]),
        np.array([4.5, 22.0, 5.0, 19.0]),
    ]
    out = compare_kappa_conditions(trials_a, trials_b, n_perm=200, random_state=0)
    assert out["n_trials_a"] == 3
    assert out["n_trials_b"] == 3
    assert out["n_perm"] == 200
    assert 0.0 <= float(out["p_value"]) <= 1.0


def test_compare_kappa_conditions_requires_trials_in_both_conditions():
    with pytest.raises(Exception):
        compare_kappa_conditions([], [np.array([10.0, 11.0, 12.0])])
