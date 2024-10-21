from random import Random

import numpy as np

from benchmarking.benchmark import TestCode

matmul_base_code = TestCode(
    setup="""
assert False
""",
    benchmark="""
lib.matmul(c, a, b)
""",
    test="""
results = lib.check_C(c, np_c, f_c)
correct = bool(results[0].value)
total_error = float(results[1].value)
consistent = bool(results[2].value)
""",
    clean="""
assert False
""",
)

matmul_triple_code = TestCode(
    setup="""
root, (a,b,c) = lib.init()
lib.setup_A(a, np_a)
lib.setup_B(b, np_b)
""",
    benchmark=matmul_base_code.benchmark,
    test=matmul_base_code.test,
    clean="""
lib.dealloc(root)
""",
)

matmul_pair_code = TestCode(
    setup="""
r_ab, (a, b) = lib.init_AB()
r_c, (c) = lib.init_C()
lib.setup_A(a, np_a)
lib.setup_B(b, np_b)
""",
    benchmark=matmul_base_code.benchmark,
    test=matmul_base_code.test,
    clean="""
lib.dealloc_AB(r_ab)
lib.dealloc_C(r_c)
""",
)

matmul_single_code = TestCode(
    setup="""
r_a, (a) = lib.init_A()
r_b, (b) = lib.init_B()
r_c, (c) = lib.init_C()
lib.setup_A(a, np_a)
lib.setup_B(b, np_b)
lib.prepare(a,b,c)
""",
    benchmark=matmul_base_code.benchmark,
    test=matmul_base_code.test,
    clean="""
lib.dealloc_A(r_a)
lib.dealloc_B(r_b)
lib.dealloc_C(r_c)
""",
)

def make_dense_np_arrays(r: Random, i: int, j: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np_a = np.zeros((i, j), dtype=np.float32)
    np_b = np.zeros((j, k), dtype=np.float32)
    for i_i in range(i):
        for i_j in range(j):
            num = r.random()
            np_a[i_i, i_j] = num
    for i_j in range(j):
        for i_k in range(k):
            num = r.random()
            np_b[i_j, i_k] = num
    np_c = np.matmul(np_a, np_b, dtype=np.float32, casting="no")
    return np_a, np_b, np_c

def make_random_sparse_np_arrays(r: Random, i: int, j: int, k: int, rate_a: float, rate_b: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np_a = np.zeros((i, j), dtype=np.float32)
    np_b = np.zeros((j, k), dtype=np.float32)
    for i_i in range(i):
        for i_j in range(j):
            if r.random() < rate_a:
                num = r.random()
                np_a[i_i, i_j] = num
    for i_j in range(j):
        for i_k in range(k):
            if r.random() < rate_b:
                num = r.random()
                np_b[i_j, i_k] = num
    np_c = np.matmul(np_a, np_b, dtype=np.float32, casting="no")
    return np_a, np_b, np_c