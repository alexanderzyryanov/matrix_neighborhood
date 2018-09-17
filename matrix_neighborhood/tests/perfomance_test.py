from collections import defaultdict

import pprint

import traceback

import numpy as np

from matrix_neighborhood.tests.utils import test_b_set
from matrix_neighborhood.tests.utils import iterate_b_set

from matrix_neighborhood.reference_solution import get_b_set as reference_solution
reference_solution.__name__ = 'reference_solution'


np.random.seed(321645)

N = 9
K = 10

HARD_COEFF = 0.7

ones_count = int(N * HARD_COEFF)

max_distance = ones_count + N
d = int(max_distance * HARD_COEFF)

a = np.zeros((N, K), dtype=bool)

ones_rows = np.random.choice(range(N), ones_count, False)
ones_cols = np.random.randint(K, size=ones_count)
a[ones_rows, ones_cols] = 1

TESTED_SOLUTIONS = [reference_solution]


if __name__ == '__main__':
    solution_times = defaultdict(int)

    for solution in TESTED_SOLUTIONS:
        solution_time = iterate_b_set(a, d, solution)
        solution_times[solution.__name__] += solution_time

    print('n: ', N, ', k: ', K, ', ones count: ', ones_count, ', d: ', d, ', max distance: ', max_distance)
    pp = pprint.PrettyPrinter()
    pp.pprint(solution_times)

    for solution in TESTED_SOLUTIONS:
        try:
            test_b_set(a, d, solution)
        except:
            print()
            print(traceback.format_exc())
            print('Fallen solution name: ', solution.__name__)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
            continue

