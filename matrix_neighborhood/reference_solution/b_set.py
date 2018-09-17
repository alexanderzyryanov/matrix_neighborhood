from typing import Iterator

import numpy as np

from matrix_neighborhood.utils import is_binary
from matrix_neighborhood.utils import is_in_a_set

from matrix_neighborhood.reference_solution.unit_b_set import get_unit_b_set
from matrix_neighborhood.reference_solution.couple_b_set import get_couple_b_set
from matrix_neighborhood.reference_solution.full_b_set import get_full_b_set


__all__ = ['get_b_set']


def get_b_set(a: np.ndarray, d: int) -> Iterator[np.ndarray]:
    if not isinstance(d, int):
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

    if d == 1:
        return get_unit_b_set(a)

    if d == 2:
        return get_couple_b_set(a)

    # if d > get_max_hamming_distance(a):
    #     return iter(())

    return get_full_b_set(a, d)
