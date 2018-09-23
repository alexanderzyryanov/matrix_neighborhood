import numpy as np


__all__ = [
    'check_a',
    'check_d',
    'is_binary',
    'is_in_a_set',
    'hamming_distance',
    'get_max_hamming_distance',
]


def is_binary(m: np.ndarray) -> bool:
    return np.array_equal(m, m.astype(bool))


def is_in_a_set(m: np.ndarray) -> bool:
    row_one_counts = m.sum(axis=1)
    _is_in_a_set = row_one_counts <= 1
    _is_in_a_set = _is_in_a_set.all()

    return _is_in_a_set


def check_a(a: np.ndarray) -> bool:
    if not isinstance(a, np.ndarray):
        raise TypeError('a mast be numpy.ndarray')

    if len(a.shape) != 2:
        raise ValueError('a is not matrix')

    n, k = a.shape
    if k < 2 or n < 2:
        raise ValueError('a is vector')

    if not is_binary(a):
        raise ValueError('a is not binary')

    if not is_in_a_set(a):
        raise ValueError('a not in A')

    return True


def check_d(d: int) -> bool:
    if type(d) is not int:
        raise TypeError('Distance must be integer')

    if d < 0:
        raise ValueError('Distance must be >= 0')

    return True


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return (a != b).sum()


def get_max_hamming_distance(a: np.ndarray) -> int:
    n = a.shape[0]
    nonzero_rows = a.sum()
    max_distance = nonzero_rows + n

    return max_distance
