import numpy as np


__all__ = ['is_binary', 'is_in_a_set', 'hamming_distance', 'get_max_hamming_distance']


def is_binary(m: np.ndarray) -> bool:
    return np.array_equal(m, m.astype(bool))


def is_in_a_set(m: np.ndarray) -> bool:
    row_one_counts = m.sum(axis=1)
    _is_in_a_set = row_one_counts <= 1
    _is_in_a_set = _is_in_a_set.all()

    return _is_in_a_set


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return (a != b).sum()


def get_max_hamming_distance(a: np.ndarray) -> int:
    n = a.shape[0]
    nonzero_rows = a.sum()
    max_distance = nonzero_rows + n

    return max_distance
