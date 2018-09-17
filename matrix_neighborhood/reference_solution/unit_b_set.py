from itertools import chain

from typing import Iterator

import numpy as np

from matrix_neighborhood.reference_solution.utils import get_nonzero_zero_indexes_2d


__all__ = ['get_unit_b_set']


def _get_b_set_by_v(a: np.ndarray, nonzero_rows: np.ndarray) -> Iterator[np.ndarray]:
    for v_row in nonzero_rows:
        b = a.copy()
        b[v_row] = 0

        yield b


def _get_b_set_by_w(a: np.ndarray, zero_rows: np.ndarray) -> Iterator[np.ndarray]:
    n, k = a.shape
    for w_row in zero_rows:
        for w_col in range(k):
            b = a.copy()
            b[w_row, w_col] = 1

            yield b


def get_unit_b_set(a: np.ndarray) -> Iterator[np.ndarray]:
    nonzero_rows, _, zero_rows = get_nonzero_zero_indexes_2d(a)
    b_set_by_v = _get_b_set_by_v(a, nonzero_rows)
    b_set_by_w = _get_b_set_by_w(a, zero_rows)
    b_set = chain(b_set_by_v, b_set_by_w)

    return b_set
