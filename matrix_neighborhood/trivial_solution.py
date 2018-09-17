from typing import Iterator

import numpy as np

from matrix_neighborhood.a_set import get_a_set

from matrix_neighborhood.utils import is_binary
from matrix_neighborhood.utils import is_in_a_set
from matrix_neighborhood.utils import hamming_distance
from matrix_neighborhood.utils import get_max_hamming_distance


__all__ = ['get_b_set']


def _get_b_set(a: np.ndarray, d: int) -> Iterator[np.ndarray]:
    n, k = a.shape
    a_set = get_a_set(n, k)
    b_set = (b for b in a_set if hamming_distance(a, b) == d)

    return b_set


def get_b_set(a: np.ndarray, d: int) -> Iterator[np.ndarray]:
    if type(d) is not int:
        raise ValueError('Distance must be integer')

    if d < 0:
        raise ValueError('Distance must be >= 0')

    if len(a.shape) != 2:
        raise ValueError('a is not matrix')

    n, k = a.shape
    if k < 2 or n < 2:
        raise ValueError('a is vector')

    if not is_binary(a):
        raise ValueError('a is not binary')

    if not is_in_a_set(a):
        raise ValueError('a not in A')

    if d == 0:
        return iter((a.copy(),))

    if d > get_max_hamming_distance(a):
        return iter(())

    b_set = _get_b_set(a, d)

    return b_set
