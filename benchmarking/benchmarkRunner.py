import os
import sys
from datetime import datetime
from timeit import timeit

import numpy as np


def inline_print(chars: int, string: str) -> int:
    print(("\b" * chars) + string, end="")
    return len(string)

def time_stamp(include: bool) -> str:
    if include:
        return f" {datetime.now()}:"
    else:
        return ""

def run_benchmark(
    lib,
    runs: int,
    np_arg_arrays: dict[str, np.ndarray],
    np_res_arrays: dict[str, np.ndarray],
    setup_code: str,
    benchmark_code: str,
    test_code: str,
    clean_code: str,
    result_variables: list[tuple[str, type[int] | type[bool] | type[float]]],
    print_updates: bool = False,
    inline_updates: bool = False,
    time_stamps: bool = False,
):
    chars = 0
    if print_updates:
        if inline_updates:
            chars = inline_print(chars, "setting up: ")
        else:
            print(f"#{time_stamp(time_stamps)} setting up")
    benchmark_scopes = []
    arg_scopes = [{"lib": lib} | np_arg_arrays for _ in range(runs)]
    i_chars = 0
    for i, scope in enumerate(arg_scopes):
        if print_updates and inline_updates:
            i_chars = inline_print(i_chars, f"{i}")
        exec(setup_code, scope)
        benchmark_scope = {k: v for k, v in scope.items() if k not in np_arg_arrays}
        benchmark_scopes.append(benchmark_scope)
    if print_updates and inline_updates:
        inline_print(i_chars, "")

    assert len(benchmark_scopes) == runs

    def benchmark():
        for scope in benchmark_scopes:
            exec(benchmark_code, scope)

    if print_updates:
        if inline_updates:
            chars = inline_print(chars, "running benchmark...")
        else:
            print(f"#{time_stamp(time_stamps)} running benchmark")
    result = timeit(benchmark, number=1)

    if print_updates:
        if inline_updates:
            chars = inline_print(chars, "testing result: ")
        else:
            print(f"#{time_stamp(time_stamps)} testing result")

    results = {}
    for n, t in result_variables:
        if t == int:
            results[n] = 0
        elif t == bool:
            results[n] = True
        elif t == float:
            results[n] = 0.0

    first_scope = benchmark_scopes[0]

    i_chars = 0
    for i, scope in enumerate(benchmark_scopes):
        if print_updates and inline_updates:
            i_chars = inline_print(i_chars, f"{i}")
        test_scope = (
            scope
            | np_res_arrays
            | {"f_" + k: v for k, v in first_scope.items() if k != "lib"}
        )
        exec(test_code, test_scope)

        for n, t in result_variables:
            if t == int:
                results[n] += t(test_scope[n])
            elif t == bool:
                results[n] &= t(test_scope[n])
            elif t == float:
                results[n] += t(test_scope[n])
    if print_updates and inline_updates:
        inline_print(i_chars, "")

    if print_updates:
        if inline_updates:
            chars = inline_print(chars, "cleaning: ")
        else:
            print(f"#{time_stamp(time_stamps)} cleaning")
    i_chars = 0
    for i, scope in enumerate(benchmark_scopes):
        if print_updates and inline_updates:
            i_chars = inline_print(i_chars, f"{i}")
        exec(clean_code, scope)
    if print_updates and inline_updates:
        inline_print(i_chars, "")

    if print_updates:
        if inline_updates:
            inline_print(chars, "")
        else:
            print(f"#{time_stamp(time_stamps)} finished")

    return result, tuple([results[n] for n, t in result_variables])


if __name__ == "__main__":
    arg_idx = 1

    # called script
    # runs to do
    runs = int(sys.argv[arg_idx])
    arg_idx += 1

    # assert os.path.exists(func_types_path)
    # with open(func_types_path, "rb") as f:
    #     function_types_tuple = pickle.load(f)
    #     function_types, dlt_func_types = function_types_tuple
    #     assert isinstance(function_types, dict)
    #     assert isinstance(dlt_func_types, dict)
    #     for k, v in function_types.items():
    #         assert isinstance(k, StringAttr)
    #         assert isinstance(v, FunctionType)
    #     for k, v in dlt_func_types.items():
    #         assert isinstance(k, str)
    #         assert isinstance(v, FuncTypeDescriptor)
    #
    # assert os.path.exists(lib_path)
    # function_types_str = {k.data: v for k, v in function_types.items()}
    # lib = DTLCLib(lib_path, dlt_func_types, function_types_str)

    np_arg_paths: dict[str, tuple[type, str]] = {}
    np_res_paths: dict[str, tuple[type, str]] = {}
    load_code = None
    setup_code = None
    benchmark_code = None
    test_code = None
    clean_code = None
    result_variables = []

    print_updates = False
    inline_updates = False
    time_stamps = False

    args = list(sys.argv[arg_idx:])
    while len(args) > 0:
        arg = args.pop(0)
        found = False
        for t, ty in [
            ("", np.float32),
            (":f32", np.float32),
            (":f64", np.float64),
            (":i32", np.int32),
            (":i64", np.int64),
        ]:
            if arg.startswith(f"-a{t}="):
                np_arg_paths[arg.removeprefix(f"-a{t}=")] = (ty, args.pop(0))
                found = True
                break
            if arg.startswith(f"-r{t}="):
                np_res_paths[arg.removeprefix(f"-r{t}=")] = (ty, args.pop(0))
                found = True
                break
        if found:
            continue
        for t, ty in [("i", int), ("b", bool), ("f", float)]:
            if arg.startswith(f"-o:{t}="):
                name = arg.removeprefix(f"-o:{t}=")
                result_variables.append((name, ty))
                found = True
                break
        if found:
            continue
        if arg == "--load":
            load_code = args.pop(0).replace("\\n", "\n")
        elif arg == "--setup":
            setup_code = args.pop(0).replace("\\n", "\n")
        elif arg == "--benchmark":
            benchmark_code = args.pop(0).replace("\\n", "\n")
        elif arg == "--test":
            test_code = args.pop(0).replace("\\n", "\n")
        elif arg == "--clean":
            clean_code = args.pop(0).replace("\\n", "\n")
        elif arg.startswith("-"):
            flags = arg.removeprefix("-")
            for f in flags:
                if f == "p":
                    print_updates = True
                elif f == "l":
                    inline_updates = True
                elif f == "t":
                    time_stamps = True
                else:
                    raise ValueError(f"Unknown flag {f} in arg '{arg}' from {sys.argv}")
        else:
            raise ValueError(f"sys arg {arg} not recognized. From {sys.argv}")

    assert load_code is not None
    assert setup_code is not None
    assert benchmark_code is not None
    assert test_code is not None
    assert clean_code is not None

    np_arg_arrays = {}
    for name, (ty, path) in np_arg_paths.items():
        assert os.path.exists(path)
        if path.endswith(".npy"):
            loaded_array = np.load(path).astype(ty)
        else:
            loaded_array = np.loadtxt(path).astype(ty)
        np_arg_arrays[name] = loaded_array
        # print(f"Loaded {name} : {ty} : {path}")

    np_res_arrays = {}
    for name, (ty, path) in np_res_paths.items():
        assert os.path.exists(path)
        if path.endswith(".npy"):
            loaded_array = np.load(path).astype(ty)
        else:
            loaded_array = np.loadtxt(path).astype(ty)
        np_res_arrays[name] = loaded_array
        # print(f"Loaded {name} : {ty} : {path}")

    scope = {}
    exec(load_code, scope)
    lib = scope["lib"]

    result_time, results = run_benchmark(
        lib,
        runs,
        np_arg_arrays,
        np_res_arrays,
        setup_code,
        benchmark_code,
        test_code,
        clean_code,
        result_variables,
        print_updates=print_updates,
        inline_updates=inline_updates,
        time_stamps=time_stamps,
    )

    print(result_time)
    for r in results:
        print(r)
