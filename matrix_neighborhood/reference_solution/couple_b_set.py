from itertools import chain
from itertools import combinations
from itertools import product

from typing import Iterator

import numpy as np

from matrix_neighborhood.reference_solution.utils import get_nonzero_zero_indexes_2d
from matrix_neighborhood.reference_solution.utils import range_without_dot


__all__ = ['get_couple_b_set']


def _get_b_set_by_u(a: np.ndarray,
                    nonzero_rows: np.ndarray,
                    nonzero_cols: np.ndarray):

    n, k = a.shape
    for u_row, u_original_col in zip(nonzero_rows, nonzero_cols):

        u_cols = range_without_dot(0, k, u_original_col)
        for u_col in u_cols:
            b = a.copy()
            b[u_row, u_original_col] = 0
            b[u_row, u_col] = 1

            yield b


def _get_b_set_by_v(a: np.ndarray, nonzero_rows: np.ndarray):
    v_row_combinations = combinations(nonzero_rows, 2)
    for v_rows in v_row_combinations:
        v_rows = list(v_rows)

        b = a.copy()
        b[v_rows] = 0

        yield b


def _get_b_set_by_v_w(a: np.ndarray,
                      nonzero_rows: np.ndarray,
                      nonzero_cols: np.ndarray,
                      zero_rows: np.ndarray):
    n, k = a.shape
    for v_row, v_col in zip(nonzero_rows, nonzero_cols):

        w_index_combinations = product(zero_rows, range(k))
        for w_row, w_col in w_index_combinations:
            b = a.copy()
            b[v_row, v_col] = 0
            b[w_row, w_col] = 1

            yield b


def _get_w_col_combinations(k: int):
    w_value_combinations = [range(k)] * 2
    w_value_combinations = product(*w_value_combinations)

    return w_value_combinations


def _get_b_set_by_w(a: np.ndarray, zero_rows: np.ndarray):
    n, k = a.shape
    w_row_combinations = combinations(zero_rows, 2)
    w_col_combinations = _get_w_col_combinations(k)
    w_index_combinations = product(w_row_combinations, w_col_combinations)
    for w_rows, w_cols in w_index_combinations:
        w_rows = list(w_rows)
        w_cols = list(w_cols)

        b = a.copy()
        b[w_rows, w_cols] = 1

        yield b


def get_couple_b_set(a: np.ndarray) -> Iterator[np.ndarray]:
    nonzero_rows, nonzero_cols, zero_rows = get_nonzero_zero_indexes_2d(a)
    b_set_by_u = _get_b_set_by_u(a, nonzero_rows, nonzero_cols)
    b_set_by_v = _get_b_set_by_v(a, nonzero_rows)
    b_set_by_v_w = _get_b_set_by_v_w(a, nonzero_rows, nonzero_cols, zero_rows)
    b_set_by_w = _get_b_set_by_w(a, zero_rows)
    b_set = chain(b_set_by_u, b_set_by_v, b_set_by_v_w, b_set_by_w)

    return b_set
