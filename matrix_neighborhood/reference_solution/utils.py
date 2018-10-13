from typing import Iterator
from typing import Tuple

import numpy as np

from matrix_neighborhood.a_set import build_a


__all__ = ['encode_a',
           'decode_b',
           'get_max_hamming_distance',
           'get_nonzero_zero_indexes_1d',
           'get_nonzero_zero_indexes_2d',
           'range_without_dot']


def range_without_dot(start: int, stop: int, dot: int, step: int = 1) -> Iterator[int]:
    return (i for i in range(start, stop, step) if i != dot)


def encode_a(a: np.ndarray) -> np.ndarray:
    nonzero_rows, nonzero_cols = np.nonzero(a)

    n, k = a.shape
    encoded_a = np.full(n, 0)
    encoded_a[nonzero_rows] = nonzero_cols + 1

    return encoded_a


decode_b = build_a


def get_max_hamming_distance(a: np.ndarray) -> int:
    nonzero_element_count = (a > 0).sum()
    max_distance = nonzero_element_count + len(a)

    return max_distance


def get_nonzero_zero_indexes_2d(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonzero_rows, nonzero_cols = np.nonzero(a)
    zero_rows = np.where(~a.any(axis=1))[0]

    return nonzero_rows, nonzero_cols, zero_rows


def get_nonzero_zero_indexes_1d(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    nonzero_indexes = np.nonzero(a)[0]
    zero_indexes = np.nonzero(a == 0)[0]

    return nonzero_indexes, zero_indexes
