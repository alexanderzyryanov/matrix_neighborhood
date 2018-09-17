from collections import defaultdict

import pprint

import traceback

import numpy as np

from matrix_neighborhood.utils import get_max_hamming_distance

from matrix_neighborhood.tests.utils import get_a_for_test
from matrix_neighborhood.tests.utils import iterate_b_set
from matrix_neighborhood.tests.utils import test_b_set

from matrix_neighborhood.reference_solution import get_b_set as reference_solution
reference_solution.__name__ = 'reference_solution'


np.random.seed(321645)


# N_MAX = 7
# N_STEP = 1
#
# K_MAX = 7
# K_STEP = 1


N_MAX = 7
N_STEP = 1

K_MAX = 7
K_STEP = 1

TESTED_SOLUTIONS = [reference_solution]


if __name__ == '__main__':
    solution_times = defaultdict(float)
    fallen_solutions = []

    a_set = get_a_for_test(N_MAX, N_STEP, K_MAX, K_STEP)
    for a in a_set:
        d_max = get_max_hamming_distance(a)
        for d in range(3, d_max + 2):
            for solution in TESTED_SOLUTIONS:
                if solution.__name__ in fallen_solutions:
                    continue

                try:
                    test_b_set(a, d, solution)
                except:
                    fallen_solutions += [solution.__name__]

                    print()
                    print(traceback.format_exc())
                    print('Fallen solution name: ', solution.__name__)
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
                    continue

                solution_time = iterate_b_set(a, d, solution)
                solution_times[solution.__name__] += solution_time

    pp = pprint.PrettyPrinter()
    pp.pprint(solution_times)
