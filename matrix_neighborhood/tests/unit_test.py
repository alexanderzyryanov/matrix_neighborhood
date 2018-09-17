from collections import defaultdict

import pprint

import traceback

import numpy as np

from matrix_neighborhood.tests.utils import get_a_for_test
from matrix_neighborhood.tests.utils import iterate_b_set
from matrix_neighborhood.tests.utils import test_b_set

from matrix_neighborhood.reference_solution import get_b_set as reference_solution
reference_solution.__name__ = 'reference_solution'


np.random.seed(321645)


# D = 0
#
# N_MAX = 40
# N_STEP = 1
#
# K_MAX = 40
# K_STEP = 1


# D = 1
#
# N_MAX = 30
# N_STEP = 1
#
# K_MAX = 30
# K_STEP = 1


# D = 2
#
# N_MAX = 15
# N_STEP = 1
#
# K_MAX = 15
# K_STEP = 1


ZERO_TEST = {
    'd': 0,
    'n_max': 30,
    'n_step': 1,
    'k_max': 30,
    'k_step': 1
}

UNIT_TEST = {
    'd': 1,
    'n_max': 30,
    'n_step': 1,
    'k_max': 20,
    'k_step': 1
}

COUPLE_TEST = {
    'd': 2,
    'n_max': 15,
    'n_step': 1,
    'k_max': 15,
    'k_step': 1
}

TEST_CASES = [ZERO_TEST, UNIT_TEST, COUPLE_TEST]

TESTED_SOLUTIONS = [reference_solution]


if __name__ == '__main__':
    solution_times = defaultdict(float)
    fallen_solutions = []

    for test_case in TEST_CASES:
        d = test_case['d']
        n_max = test_case['n_max']
        n_step = test_case['n_step']
        k_max = test_case['k_max']
        k_step = test_case['k_step']
        a_set = get_a_for_test(n_max, n_step, k_max, k_step)
        for a in a_set:
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
        pp.pprint(test_case)
        pp.pprint(solution_times)
        print('================================================\n')
