import numpy as np

from neuralkappa.simulation import generate_rate_modulated_gamma


def test_simulation_returns_positive_1d_isis():
    rates = np.array([0.01, 0.02, 0.03])
    isis = generate_rate_modulated_gamma(2.0, rates, random_state=123)
    assert isis.shape == rates.shape
    assert np.all(np.isfinite(isis))
    assert np.all(isis > 0)
