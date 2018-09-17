from itertools import combinations
from itertools import product

from math import ceil

from typing import Iterable
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from matrix_neighborhood.reference_solution.utils import encode_a
from matrix_neighborhood.reference_solution.utils import decode_b
from matrix_neighborhood.reference_solution.utils import get_nonzero_zero_indexes_1d
from matrix_neighborhood.reference_solution.utils import range_without_dot


__all__ = ['get_full_b_set']


Indexes = Union[Tuple[int, ...], List[int]]

SetProto = List[Iterable[int]]


def _get_v_bounds(d: int,
                  nonzero_element_count: int,
                  zero_element_count: int) -> Tuple[int, int]:

    v_min = (d - zero_element_count) / 2
    v_min = ceil(v_min)
    v_min = max(v_min, 0)

    v_max = min(nonzero_element_count, d)

    return int(v_min), v_max


def _get_v_counts(d: int,
                  nonzero_element_count: int,
                  zero_element_count: int) -> Iterable[int]:

    v_min, v_max = _get_v_bounds(d, nonzero_element_count, zero_element_count)
    v_counts = range(v_min, v_max + 1)

    return v_counts


def _get_v_index_combinations(v_count: int,
                              nonzero_indexes: Indexes) -> Iterable[Indexes]:

    return combinations(nonzero_indexes, v_count)


def _get_w_count(d: int,
                 v_count: int) -> int:

    return d - v_count


def _get_w_index_combinations(w_count: int,
                              zero_indexes: Indexes) -> Iterable[Indexes]:

    return combinations(zero_indexes, w_count)


def _apply_v_indexes(v_indexes: Indexes, set_proto: SetProto) -> None:
    for v_index in v_indexes:
        set_proto[v_index] = (0,)


def _apply_w_indexes(a: np.ndarray,
                     k: int,
                     w_indexes: Indexes,
                     set_proto: SetProto) -> None:

    for w_index in w_indexes:
        set_proto[w_index] = range_without_dot(1, k + 1, a[w_index])


def _get_constant_indexes(n: int,
                          v_indexes: Indexes,
                          w_indexes: Indexes) -> Iterable[int]:
    all_indexes = range(n)
    used_indexes = v_indexes + w_indexes
    constant_indexes = set(all_indexes) - set(used_indexes)

    return constant_indexes


def _apply_constant_indexes(a: np.ndarray,
                            v_indexes: Indexes,
                            w_indexes: Indexes,
                            set_proto: SetProto) -> None:

    n = a.shape[0]
    constant_indexes = _get_constant_indexes(n, v_indexes, w_indexes)
    for constant_index in constant_indexes:
        set_proto[constant_index] = (a[constant_index],)


def _get_v_w_b_set(a: np.ndarray,
                   k: int,
                   v_indexes: Indexes,
                   w_indexes: Indexes) -> Iterator[Iterable[int]]:

    n = a.shape[0]
    set_proto = [()] * n
    _apply_v_indexes(v_indexes, set_proto)
    _apply_w_indexes(a, k, w_indexes, set_proto)
    _apply_constant_indexes(a, v_indexes, w_indexes, set_proto)

    v_w_b_set = product(*set_proto)

    return v_w_b_set


def get_full_b_set(a: np.ndarray, d: int) -> Iterator[np.ndarray]:
    n, k = a.shape
    a = encode_a(a)

    nonzero_zero_indexes = get_nonzero_zero_indexes_1d(a)
    nonzero_indexes, zero_indexes = map(tuple, nonzero_zero_indexes)
    nonzero_element_count, zero_element_count = map(len, nonzero_zero_indexes)

    v_counts = _get_v_counts(d, nonzero_element_count, zero_element_count)
    for v_count in v_counts:
        v_index_combinations = _get_v_index_combinations(v_count, nonzero_indexes)

        for v_indexes in v_index_combinations:
            _zero_indexes = v_indexes + zero_indexes
            w_count = _get_w_count(d, v_count)
            w_index_combinations = _get_w_index_combinations(w_count, _zero_indexes)

            for w_indexes in w_index_combinations:
                v_w_b_set = _get_v_w_b_set(a, k, v_indexes, w_indexes)

                for b in v_w_b_set:
                    b = decode_b(b, n, k)

                    yield b
