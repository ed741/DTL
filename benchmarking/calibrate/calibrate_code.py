from random import Random

import numpy as np

from benchmarking.benchmark import TestCode

calibrate_code = TestCode(
    setup="""
root, (a) = lib.init_A()
lib.prepare(a)
""",
    benchmark="""
lib.func(a)
""",
    test="""
results = lib.check_A(a, ref_a, f_a)
correct = bool(results[0].value)
total_error = float(results[1].value)
consistent = bool(results[2].value)
""",
    clean="""
lib.dealloc_A(root)
""",
)

def make_dense_np_array(seed: int, i: int, j: int) -> np.ndarray:
    r = Random(seed)
    np_a = np.zeros((i, j), dtype=np.float32)
    for i_i in range(i):
        for i_j in range(j):
            num = r.random()
            np_a[i_i, i_j] = num
    return np_a