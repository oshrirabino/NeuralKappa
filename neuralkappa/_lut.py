from __future__ import annotations

from dataclasses import dataclass, field
import os
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.special import digamma

from ._exceptions import DomainError

DEFAULT_KAPPA_MIN = 0.01
DEFAULT_KAPPA_MAX = 1000.0
DEFAULT_LUT_RESOLUTION = 200_000


@dataclass
class KappaLUT:
    """Lookup table and interpolator for SI -> kappa conversion."""

    kappa_min: float = 0.01
    kappa_max: float = 1000.0
    resolution: int = 200_000
    _interp: interp1d | None = field(default=None, init=False, repr=False)
    _kappa_grid: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def ensure_initialized(self) -> None:
        """Build interpolation artifacts once, lazily."""
        if self._interp is not None:
            return

        kappa_grid = np.logspace(
            np.log10(self.kappa_min),
            np.log10(self.kappa_max),
            num=self.resolution,
            dtype=np.float64,
        )
        si_grid = digamma(2.0 * kappa_grid) - digamma(kappa_grid) - np.log(2.0)

        if not np.all(np.diff(si_grid) < 0):
            raise DomainError("SI grid is not strictly monotonic decreasing.")

        # interp1d expects ascending x-values.
        si_asc = si_grid[::-1]
        kappa_asc = kappa_grid[::-1]

        self._interp = interp1d(
            si_asc,
            kappa_asc,
            kind="linear",
            bounds_error=False,
            fill_value=(kappa_asc[0], kappa_asc[-1]),
            assume_sorted=True,
        )
        self._kappa_grid = kappa_grid

    def convert(self, si_array: NDArray[np.float64]) -> NDArray[np.float64]:
        """Vectorized SI -> kappa conversion using the prepared interpolator."""
        self.ensure_initialized()
        assert self._interp is not None
        return np.asarray(self._interp(si_array), dtype=np.float64)


def _read_env_float(name: str, default: float) -> float:
    """Read a float from env with fallback to default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        warnings.warn(
            f"Invalid {name}={raw!r}; using default {default}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return default


def _read_env_int(name: str, default: int) -> int:
    """Read an int from env with fallback to default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        warnings.warn(
            f"Invalid {name}={raw!r}; using default {default}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return default


def _lut_from_env() -> KappaLUT:
    """Create the global LUT instance using optional env overrides."""
    kappa_min = _read_env_float("NEURALKAPPA_KAPPA_MIN", DEFAULT_KAPPA_MIN)
    kappa_max = _read_env_float("NEURALKAPPA_KAPPA_MAX", DEFAULT_KAPPA_MAX)
    resolution = _read_env_int("NEURALKAPPA_LUT_RESOLUTION", DEFAULT_LUT_RESOLUTION)

    valid = kappa_min > 0.0 and kappa_max > kappa_min and resolution >= 2
    if not valid:
        warnings.warn(
            "Invalid LUT env configuration; using defaults "
            f"(kappa_min={DEFAULT_KAPPA_MIN}, "
            f"kappa_max={DEFAULT_KAPPA_MAX}, "
            f"resolution={DEFAULT_LUT_RESOLUTION}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return KappaLUT(
            kappa_min=DEFAULT_KAPPA_MIN,
            kappa_max=DEFAULT_KAPPA_MAX,
            resolution=DEFAULT_LUT_RESOLUTION,
        )

    return KappaLUT(kappa_min=kappa_min, kappa_max=kappa_max, resolution=resolution)


_GLOBAL_LUT = _lut_from_env()
