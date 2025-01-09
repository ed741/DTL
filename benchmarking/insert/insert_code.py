from random import Random

import numpy as np

from benchmarking.benchmark import TestCode

def get_code(shape: tuple[int,...]):

    setup_coord_args = [f"np_coord_{str(i)}" for i in range(len(shape))]
    setup_coord_args_str = ", ".join(setup_coord_args)
    coord_args = [f"np_coord_{str(i)}_b" for i in range(len(shape))]
    coord_args_str = ", ".join(coord_args)
    coord_check_args = [f"np_check_coord_{str(i)}" for i in range(len(shape))]
    coord_check_args_str = ", ".join(coord_check_args)

    insert_base_code = TestCode(
        setup=f"root, (a) = lib.init()\nlib.prepare(a)\n{coord_args_str}={setup_coord_args_str}\nnp_val_b=np_val",
        benchmark=f"lib.insert(a, {coord_args_str}, np_val_b)",
        test="""
results = lib.check(a, """ + coord_check_args_str + """, np_check_val, f_a)
correct = bool(results[0].value)
total_error = float(results[1].value)
consistent = bool(results[2].value)
""",
        clean="lib.dealloc(root)",
    )
    return insert_base_code

def make_dense_np_arrays(seed: int, i: int, j: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = Random(seed)
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

def make_random_sparse_np_arrays(seed: int, i: int, j: int, k: int, rate_a: float, rate_b: float, sparse_a: tuple[float, float] = (1.0,1.0), sparse_b: tuple[float, float] = (1.0,1.0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = Random(seed)
    np_a = np.zeros((i, j), dtype=np.float32)
    np_b = np.zeros((j, k), dtype=np.float32)

    nonzero_row = [True]*i
    nonzero_col = [True]*j
    for i_i in range(i):
        nonzero_row[i_i] = r.random() < sparse_a[0]
    for i_j in range(j):
        nonzero_col[i_j] = r.random()  < sparse_a[1]

    for i_i in range(i):
        for i_j in range(j):
            if nonzero_row[i_i] and nonzero_col[i_j]:
                if r.random() < rate_a:
                    num = r.random()
                    np_a[i_i, i_j] = num

    nonzero_row = [True]*j
    nonzero_col = [True]*k
    for i_j in range(j):
        nonzero_row[i_j] = r.random() < sparse_b[0]
    for i_k in range(k):
        nonzero_col[i_k] = r.random()  < sparse_b[1]

    for i_j in range(j):
        for i_k in range(k):
            if nonzero_row[i_j] and nonzero_col[i_k]:
                if r.random() < rate_b:
                    num = r.random()
                    np_b[i_j, i_k] = num

    np_c = np.matmul(np_a, np_b, dtype=np.float32, casting="no")
    return np_a, np_b, np_c


def make_insertion_arrays(seed: int, shape: tuple[int,...], count: int, ordered: bool =False, sort_func=None, allow_duplicates: bool =False) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    assert allow_duplicates or np.prod(np.array(shape)) >= count
    r = Random(seed)
    coords = []
    while len(coords) < count:
        coord = tuple([r.randrange(e) for e in shape])
        if not allow_duplicates and coord in coords:
            continue
        coords.append(coord)
    if ordered:
        if sort_func is None:
            coords.sort()
        else:
            coords.sort(key=sort_func)
    np_coords = tuple([np.zeros(shape=(count,), dtype=np.int32) for _ in shape])
    np_vals = np.zeros(shape=(count,), dtype=np.float32)
    for c, coord in enumerate(coords):
        for i in range(len(shape)):
            np_coords[i][c] = coord[i]
        np_vals[c] = r.random()
    return np_coords, np_vals