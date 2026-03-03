import numpy as np
import pytest

from neuralkappa.preprocessing import timestamps_to_isis, validate_isis


def test_timestamps_to_isis():
    ts = np.array([1.0, 3.0, 6.0, 10.0])
    assert np.allclose(timestamps_to_isis(ts), np.array([2.0, 3.0, 4.0]))


def test_validate_isis_rejects_non_positive():
    with pytest.raises(Exception):
        validate_isis([1.0, 0.0, 2.0])
