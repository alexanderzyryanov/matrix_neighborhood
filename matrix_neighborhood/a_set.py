from itertools import product

from typing import Iterable
from typing import Iterator

import numpy as np


__all__ = ['get_a_set']


def _get_ones_positions(n: int, k: int) -> Iterator[Iterable[int]]:
    ones_positions = [range(-1, k)] * n
    ones_positions = product(*ones_positions)

    return ones_positions


def _create_a(n: int, k: int, ones_position: Iterable[int]) -> np.ndarray:
    a = np.zeros((n, k), dtype=bool)
    for row, col in enumerate(ones_position):
        if col > -1:
            a[row, col] = 1

    return a


def get_a_set(n: int, k: int) -> Iterator[np.ndarray]:
    ones_positions = _get_ones_positions(n, k)
    for ones_position in ones_positions:
        a = _create_a(n, k, ones_position)

        yield a
