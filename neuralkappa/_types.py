from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

ArrayLike1D = Union[Sequence[float], NDArray[np.floating]]
ArrayLikeInt1D = Union[Sequence[int], NDArray[np.integer]]
