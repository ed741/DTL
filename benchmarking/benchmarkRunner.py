import os
import pickle
import sys
from timeit import timeit

import numpy as np

from dtl.libBuilder import DTLCLib, FuncTypeDescriptor
from xdsl.dialects.builtin import FunctionType, StringAttr

def run_benchmark(lib: DTLCLib, runs: int, np_arg_arrays: dict[str, np.ndarray], np_res_arrays: dict[str, np.ndarray], setup_code: str, benchmark_code: str, test_code: str, clean_code: str):

    benchmark_scopes = []
    arg_scopes = [{"lib": lib} | np_arg_arrays for _ in range(runs)]
    for scope in arg_scopes:
        exec(setup_code, scope)
        benchmark_scope = {k:v for k,v in scope.items() if k not in np_arg_arrays}
        benchmark_scopes.append(benchmark_scope)

    assert len(benchmark_scopes) == runs
    def benchmark():
        for scope in benchmark_scopes:
            exec(benchmark_code, scope)

    result = timeit(benchmark, number=1)

    correct = True
    consistent = True
    total_error = 0.0

    first_scope = benchmark_scopes[0]

    for scope in benchmark_scopes:
        test_scope = scope | np_res_arrays | {"f_"+k:v for k,v in first_scope.items() if k != "lib"}
        exec(test_code, test_scope)
        correct &= test_scope["correct"]
        consistent &= test_scope["consistent"]
        total_error += test_scope["total_error"]

    mean_error = total_error / runs

    for scope in benchmark_scopes:
        exec(clean_code, scope)

    return result, correct, mean_error, consistent

if __name__ == '__main__':

    assert len(sys.argv) > 3
    # called script
    # lib to load
    lib_path = sys.argv[1]
    # func_types to load
    func_types_path = sys.argv[2]
    # runs to do
    runs = int(sys.argv[3])

    assert os.path.exists(func_types_path)
    with open(func_types_path, "rb") as f:
        function_types_tuple = pickle.load(f)
        function_types, dlt_func_types = function_types_tuple
        assert isinstance(function_types, dict)
        assert isinstance(dlt_func_types, dict)
        for k, v in function_types.items():
            assert isinstance(k, StringAttr)
            assert isinstance(v, FunctionType)
        for k, v in dlt_func_types.items():
            assert isinstance(k, str)
            assert isinstance(v, FuncTypeDescriptor)

    assert os.path.exists(lib_path)
    function_types_str = {k.data: v for k, v in function_types.items()}
    lib = DTLCLib(lib_path, dlt_func_types, function_types_str)

    np_arg_paths = {}
    np_res_paths = {}
    setup_code = None
    benchmark_code = None
    test_code = None
    clean_code = None

    args = list(sys.argv[4:])
    while len(args) > 0:
        arg = args.pop(0)
        if arg.startswith("-a="):
            np_arg_paths[arg.removeprefix("-a=")] = args.pop(0)
        elif arg.startswith("-r="):
            np_res_paths[arg.removeprefix("-r=")] = args.pop(0)
        elif arg == "--setup":
            setup_code = args.pop(0).replace("\\n", "\n")
        elif arg == "--benchmark":
            benchmark_code = args.pop(0).replace("\\n", "\n")
        elif arg == "--test":
            test_code = args.pop(0).replace("\\n", "\n")
        elif arg == "--clean":
            clean_code = args.pop(0).replace("\\n", "\n")
        else:
            raise ValueError(f"sys arg {arg} not recognized. From {sys.argv}")

    assert setup_code is not None
    assert benchmark_code is not None
    assert test_code is not None
    assert clean_code is not None

    np_arg_arrays = {}
    for name, path in np_arg_paths.items():
        assert os.path.exists(path)
        loaded_array = np.loadtxt(path).astype(np.float32)
        np_arg_arrays[name] = loaded_array

    np_res_arrays = {}
    for name, path in np_res_paths.items():
        assert os.path.exists(path)
        loaded_array = np.loadtxt(path).astype(np.float32)
        np_res_arrays[name] = loaded_array

    result, correct, mean_error, consistent = run_benchmark(lib, runs, np_arg_arrays, np_res_arrays, setup_code, benchmark_code, test_code, clean_code)

    print(result)
    print(correct)
    print(mean_error)
    print(consistent)


