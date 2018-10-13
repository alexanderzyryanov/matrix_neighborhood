from itertools import product

from typing import Iterable
from typing import Iterator

import numpy as np


__all__ = [
    'build_a',
    'get_a_set',
]


def _get_ones_positions(n: int, k: int) -> Iterator[Iterable[int]]:
    ones_positions = [range(k + 1)] * n
    ones_positions = product(*ones_positions)

    return ones_positions


def build_a(ones_position: Iterable[int], k: int, n: int) -> np.ndarray:
    a = np.zeros((n, k), dtype=bool)
    for row, col in enumerate(ones_position):
        if col > 0:
            a[row, col-1] = 1

    return a


def get_a_set(n: int, k: int) -> Iterator[np.ndarray]:
    ones_positions = _get_ones_positions(n, k)
    for ones_position in ones_positions:
        yield build_a(ones_position, k, n)
