from typing import Iterator

import numpy as np

from matrix_neighborhood.utils import check_a
from matrix_neighborhood.utils import check_d

from matrix_neighborhood.reference_solution.unit_b_set import get_unit_b_set
from matrix_neighborhood.reference_solution.couple_b_set import get_couple_b_set
from matrix_neighborhood.reference_solution.full_b_set import get_full_b_set


__all__ = ['get_b_set']


def get_b_set(a: np.ndarray, d: int) -> Iterator[np.ndarray]:
    check_d(d)
    check_a(a)

    if d == 0:
        return iter((a.copy(),))

    if d == 1:
        return get_unit_b_set(a)

    if d == 2:
        return get_couple_b_set(a)

    # if d > get_max_hamming_distance(a):
    #     return iter(())

    return get_full_b_set(a, d)
