import numpy as np

from neuralkappa._lut import KappaLUT


def test_lut_monotonic_roundtrip():
    lut = KappaLUT(resolution=50_000)
    lut.ensure_initialized()

    kappas = np.array([0.1, 1.0, 10.0, 100.0])

    from scipy.special import digamma

    si = digamma(2.0 * kappas) - digamma(kappas) - np.log(2.0)
    recovered = lut.convert(si)
    assert np.allclose(recovered, kappas, rtol=1e-3, atol=1e-5)
