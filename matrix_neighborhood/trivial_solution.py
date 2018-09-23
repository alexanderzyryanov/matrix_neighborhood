from typing import Iterator

import numpy as np

from matrix_neighborhood.a_set import get_a_set

from matrix_neighborhood.utils import check_a
from matrix_neighborhood.utils import check_d
from matrix_neighborhood.utils import hamming_distance
from matrix_neighborhood.utils import get_max_hamming_distance


__all__ = ['get_b_set']


def _get_b_set(a: np.ndarray, d: int) -> Iterator[np.ndarray]:
    n, k = a.shape
    a_set = get_a_set(n, k)
    b_set = (b for b in a_set if hamming_distance(a, b) == d)

    return b_set


def get_b_set(a: np.ndarray, d: int) -> Iterator[np.ndarray]:
    check_d(d)
    check_a(a)

    if d == 0:
        return iter((a.copy(),))

    if d > get_max_hamming_distance(a):
        return iter(())

    b_set = _get_b_set(a, d)

    return b_set
