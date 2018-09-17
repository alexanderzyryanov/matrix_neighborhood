from time import time

from typing import Callable
from typing import Iterator

import numpy as np

from scipy.special import binom

from matrix_neighborhood.utils import is_binary
from matrix_neighborhood.utils import is_in_a_set
from matrix_neighborhood.utils import hamming_distance


__all__ = ['get_a_for_test',
           'iterate_b_set',
           'test_b_set']


BSetGenerator = Callable[[np.ndarray, int], Iterator[np.ndarray]]


def _check_b(a: np.ndarray, b: np.ndarray, d: int):
    fact_d = hamming_distance(a, b)
    if fact_d != d:
        print('Error!')
        print('Fact d is not equal true d')
        print('True d: ', d)
        print('Fact d: ', fact_d)

        print('a:')
        print(a.astype(int))
        print(a.shape)

        print('b:')
        print(b.astype(int))

        raise Exception('Fact d is not equal true d')

    if not is_binary(b):
        print('Error!')
        print('b is not binary')

        print('a:')
        print(a.astype(int))
        print(a.shape)

        print('b:')
        print(b.astype(int))

        print('d: ', d)

        raise Exception('b is not binary')

    if not is_in_a_set(b):
        print('Error!')
        print('b not in A')

        print('a:')
        print(a.astype(int))
        print(a.shape)

        print('b:')
        print(b.astype(int))

        print('d: ', d)

        raise Exception('b not in A')

    n, k = b.shape
    b.put(range(n * k), np.random.randint(n * k, size=n*k))

    return True


def test_b_set(a: np.ndarray, d: int, get_b_set: BSetGenerator) -> float:
    start_time = time()

    b_set = get_b_set(a, d)
    checked_b_set = (_check_b(a, b, d) for b in b_set)
    b_power = sum(checked_b_set)

    end_time = time()
    execution_time = end_time - start_time

    n, k = a.shape
    ones_count = a.sum()
    zero_count = n - ones_count
    true_b_power = _get_b_power(ones_count, zero_count, k, d)

    if b_power != true_b_power:
        print('Error!')
        print('Fact B power not equal true B power')
        print('True B power: ', true_b_power)
        print('Fact B power: ', b_power)

        print('a:')
        print(a.astype(int))
        print(a.shape)

        print('d: ', d)

        raise Exception('Fact B power not equal true B power')

    return execution_time


def iterate_b_set(a: np.ndarray, d: int, get_b_set: BSetGenerator) -> float:
    start_time = time()

    b_set = get_b_set(a, d)
    for _ in b_set:
        pass

    end_time = time()
    execution_time = end_time - start_time

    return execution_time


def get_a_for_test(n_max: int, n_step: int,
                   k_max: int, k_step: int) -> Iterator[np.ndarray]:

    for n in range(2, n_max + 1, n_step):
        for k in range(2, k_max + 1, k_step):
            for ones_count in range(n + 1):
                ones_rows = np.random.choice(range(n), ones_count, False)
                ones_cols = np.random.randint(k, size=ones_count)

                a = np.zeros((n, k), dtype=bool)
                a[ones_rows, ones_cols] = 1

                yield a


def _get_b_power_by_v_w(ones_count: int, zeros_count: int, k: int, d: int) -> int:
    b_power_by_v_w = sum(binom(ones_count, i) * binom(zeros_count, d - i) * k ** (d - i)
                         for i in range(d + 1))

    if np.isnan(b_power_by_v_w):
        return 1

    return b_power_by_v_w


def _get_b_power(ones_count: int, zeros_count: int, k: int, d: int) -> int:
    b_power = sum(binom(ones_count, i) * (k - 1) ** i *
                  _get_b_power_by_v_w(ones_count - i, zeros_count, k, d - 2 * i) for i in range(d // 2 + 1))

    return b_power
