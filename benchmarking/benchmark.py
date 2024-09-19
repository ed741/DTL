import abc
import csv
import datetime
import os
import pickle
import subprocess
import time
from io import StringIO

import nptyping
import numpy as np

import benchmarkRunner
from dtl import TensorVariable
from dtl.libBuilder import DTLCLib, LibBuilder, TupleStruct
from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer
from xdsl.transforms.experimental.dlt.generate_dlt_iteration_orders import (
    IterationGenerator,
    IterationMapping,
)
from xdsl.transforms.experimental.dlt.generate_dlt_layouts import (
    LayoutGenerator,
    PtrMapping, ReifyConfig,
)
from xdsl.transforms.experimental.dlt.iteration_map import IterationMap
from xdsl.transforms.experimental.dlt.layout_graph import LayoutGraph

class Benchmark(abc.ABC):

    def __init__(
        self,
        base_dir: str,
        layout_store: str,
        order_store: str,
        runs: int,
        repeats: int,
        opt_num: int,
        epsilon: float,
        waste_of_time_threshold: float = 0.1,
        test_too_short_threshold: float = 0.0001,
        long_run_multiplier: int = 1000,
        benchmark_timeout: float = 5,
    ):
        self.base_dir = base_dir
        self.runs = runs
        self.repeats = repeats
        self.opt_num = opt_num
        self.epsilon = epsilon

        self.layout_store = layout_store
        self.order_store = order_store

        self.take_first_layouts = 0
        self.take_first_orders = 0
        self.do_not_generate = False
        self.only_generate = False
        self.do_not_lower = False
        self.do_not_compile_mlir = False
        self.only_compile_to_llvm = False
        self.skip_testing = False

        self.waste_of_time_threshold = waste_of_time_threshold
        self.test_too_short_threshold = test_too_short_threshold
        self.long_run_multiplier = long_run_multiplier
        self.benchmark_timeout = benchmark_timeout

        self.np_arg_paths: dict[str, str] = {}
        self.np_args: dict[str, np.ndarray] = {}
        self.np_res_paths: dict[str, str] = {}
        self.np_ress: dict[str, np.ndarray] = {}


        os.makedirs(self.base_dir, exist_ok=True)

    def get_extra_clang_args(self) -> list[str]:
        match self.opt_num:
            case 0:
                return []
            case 1:
                return ["-O1"]
            case 2:
                return ["-O2"]
            case 3:
                return ["-O3"]
            case 4:
                return ["-O3", "-march=native"]

    def handle_reference_array(self, array: nptyping.NDArray, name: str, is_arg: bool, is_res: bool, scope_name: str):
        if scope_name is None:
            scope_name = name
        path = f"{self.base_dir}/{name}"
        if os.path.exists(path):
            loaded_array = np.loadtxt(path)
            if not np.equal(array, loaded_array).all():
                print(
                    f"ERROR! loaded array does not match newly calculated reference array: {path}"
                )
            else:
                # print(f"Reference array consistent with {path}")
                pass
        else:
            np.savetxt(path, array)
            print(f"Reference array saved to {path}")
        if is_arg:
            assert scope_name not in self.np_arg_paths
            self.np_arg_paths[scope_name] = path
            assert scope_name not in self.np_args
            self.np_args[scope_name] = array

        if is_res:
            assert scope_name not in self.np_res_paths
            self.np_res_paths[scope_name] = path
            assert scope_name not in self.np_ress
            self.np_ress[scope_name] = array

    @abc.abstractmethod
    def define_lib_builder(self) -> LibBuilder:
        raise NotImplementedError

    @abc.abstractmethod
    def get_configs_for_DTL_tensors(self) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_setup(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_benchmark(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_test(self) -> str:
        # must define 'correct', 'total_error', 'consistent' in the scope
        raise NotImplementedError

    @abc.abstractmethod
    def get_clean(self) -> str:
        raise NotImplementedError

    def run(self):
        print(
            f"{datetime.datetime.now()} Running benchmark {type(self)} at {self.base_dir}"
        )
        print(f"{datetime.datetime.now()} Defining lib builder")
        lib_builder = self.define_lib_builder()
        print(f"{datetime.datetime.now()} Starting benchmarking...")
        self.run_benchmarking(lib_builder)

    def get_compiled_lib(
        self,
        new_layout: PtrMapping,
        new_order: IterationMapping,
        module: ModuleOp,
        lib_builder: LibBuilder,
        layout_graph: LayoutGraph,
        iteration_map: IterationMap,
    ) -> tuple[DTLCLib, str, str] | None:
        print(
            f"Getting lib for: l: {new_layout.number}, o: {new_order.number} :: ",
            end="",
        )

        test_name = f"{new_layout.number}.{new_order.number}"
        test_path = f"{self.base_dir}/tests/{test_name}"
        os.makedirs(test_path, exist_ok=True)

        lib_name = f"lib_{new_layout.number}_{new_order.number}"
        llvm_path = f"{test_path}/{lib_name}.ll"
        lib_path = f"{test_path}/{lib_name}.so"
        func_types_path = f"{test_path}/{lib_name}.ft"
        graph_path = f"{test_path}/"

        func_types_exists = os.path.exists(func_types_path)
        print("func_types: ", end="")
        if func_types_exists:
            module_clone = None
            with open(func_types_path, "rb") as f:
                function_types_tuple = pickle.load(f)
                function_types, dlt_func_types = function_types_tuple
            print("loaded, ", end="")
        elif not self.do_not_lower:
            module_clone = module.clone()
            function_types, dlt_func_types = lib_builder.lower(
                module_clone,
                layout_graph,
                new_layout.make_ptr_dict(),
                iteration_map,
                new_order.make_iter_dict(),
                graph_dir=graph_path,
                verbose=0,
            )
            with open(func_types_path, "wb") as f:
                f.write(pickle.dumps((function_types, dlt_func_types)))
            print("made, ", end="")
        else:
            print("not found, but do not lower is set")
            return None

        lib_exists = os.path.exists(lib_path)
        print("lib: ", end="")
        if lib_exists:
            function_types_str = {k.data: v for k, v in function_types.items()}
            lib = DTLCLib(lib_path, dlt_func_types, function_types_str)
            print("found.")
            return lib, lib_path, func_types_path
        else:
            print("not found, ", end="")

        llvm_exists = os.path.exists(llvm_path)
        print("llvm: ", end="")
        if llvm_exists and self.only_compile_to_llvm:
            print(f"found & Done.")
            return None
        elif not llvm_exists and self.do_not_compile_mlir:
            print("not found - but do not compile mlir is set.")
            return None
        else:
            if llvm_exists:
                print(f"found, ", end="")
                lib = lib_builder.compile_from(
                    llvm_path,
                    function_types,
                    dlt_func_map=dlt_func_types,
                    lib_path=lib_path,
                    clang_args=self.get_extra_clang_args(),
                    verbose=0,
                )
                print(f"lib compiled to: {lib_path}")
                return lib, lib_path, func_types_path
            else:
                print("not found, ", end="")
                if module_clone is None:
                    print("func_types: ", end="")
                    module_clone = module.clone()
                    function_types, dlt_func_types = lib_builder.lower(
                        module_clone,
                        layout_graph,
                        new_layout.make_ptr_dict(),
                        iteration_map,
                        new_order.make_iter_dict(),
                        graph_dir=graph_path,
                        verbose=0,
                    )
                    with open(func_types_path, "wb") as f:
                        f.write(pickle.dumps((function_types, dlt_func_types)))
                    print("remade, ", end="")

                print("lib: ", end="")
                lib = lib_builder.compile(
                    module_clone,
                    function_types,
                    dlt_func_map=dlt_func_types,
                    llvm_out=llvm_path,
                    llvm_only=self.only_compile_to_llvm,
                    lib_path=lib_path,
                    clang_args=self.get_extra_clang_args(),
                    verbose=0,
                )
                if lib is not None:
                    print(
                        f"compiled to binary. LLVM: {llvm_path}, lib: {lib._library_path}"
                    )
                    return lib, lib_path, func_types_path
                else:
                    print(f"compiled to LLVM: {llvm_path} but no lib was produced.")
                    return None

    def get_test_id_from_row(self, row) -> tuple[tuple, int, tuple[int, float]]:
        layout_num = int(row[0])
        order_num = int(row[1])
        rep = int(row[2])
        runs = int(row[3])
        time = float(row[4])
        correct = row[5] == "True"
        mean_error = float(row[6])
        consistent = row[7] == "True"
        waiting_time = float(row[8])
        finished = row[9] == "True"
        return (layout_num, order_num), rep, (runs, time)

    def get_test_id(self, layout_num, order_num) -> tuple:
        return (layout_num, order_num)

    def get_results_header(self):
        return [
            "layout_mapping",
            "iter_mapping",
            "rep",
            "runs",
            "time",
            "correct",
            "mean_error",
            "consistent",
            "waiting_time",
            "finished"
        ]

    def run_benchmarking(self, lib_builder: LibBuilder) -> None:
        module, layout_graph, iteration_map = lib_builder.prepare(verbose=0)

        print(
            f"{datetime.datetime.now()} Generating possible layouts and iteration maps"
        )
        layouts, orders = self._generate_versions(layout_graph, iteration_map, lib_builder)
        if self.only_generate:
            print("Only Generate is set, so benchmarking is ending.")
            return

        print(f"{datetime.datetime.now()} Loading from existing results")
        results_done = {}

        results_file = f"{self.base_dir}/results.csv"
        if os.path.exists(results_file):
            with open(results_file, "r", newline="") as f:
                r = csv.reader(f)
                next(r, None)  # skip header
                for row in r:
                    test_id, rep, result = self.get_test_id_from_row(row)
                    results_done[(*test_id, rep)] = result
        print(f"{datetime.datetime.now()} Found {len(results_done)} results")

        write_results_header = not os.path.exists(results_file)
        results_correct = True

        with open(results_file, "a", newline="") as csv_results:
            result_writer = csv.writer(csv_results)
            if write_results_header:
                result_writer.writerow(
                    self.get_results_header()
                )

            count = 0
            total_l_o_pairs = len(layouts) * len(orders)

            for new_layout in layouts:
                for new_order in orders:
                    count += 1
                    test_id = self.get_test_id(new_layout.number, new_order.number)
                    print(
                        f"{datetime.datetime.now()} Running Benchmarks for {test_id}. ({count}/{total_l_o_pairs})"
                    )

                    if all(
                        (*test_id, rep) in results_done
                        for rep in range(-1, self.repeats)
                    ):
                        print(
                            f"{datetime.datetime.now()} Skipping layout: {new_layout.number}, order: {new_order.number} for all repeats [0..{self.repeats}) as they are already in the results file"
                        )
                        continue

                    t_result = None
                    runs_to_do = self.runs

                    if (*test_id, -1) in results_done:
                        t_r, test_time_result = results_done[(*test_id, -1)]
                        t_result = test_time_result / t_r
                        if test_time_result < 0 or t_result > self.waste_of_time_threshold:
                            print(
                                f"{datetime.datetime.now()} Skipping {test_id} for all repeats [0..{self.repeats}) as the test time was {t_result}"
                            )
                            continue
                        elif t_result < self.test_too_short_threshold:
                            runs_to_do = self.runs * self.long_run_multiplier
                            print(
                                f"Test result found ({t_result}) and runs set to: {self.runs}*{self.long_run_multiplier}={runs_to_do}"
                            )
                        else:
                            print(
                                f"Test result found ({t_result}) and runs set to: {runs_to_do}"
                            )

                    compiled_lib = self.get_compiled_lib(
                        new_layout,
                        new_order,
                        module,
                        lib_builder,
                        layout_graph,
                        iteration_map,
                    )

                    if t_result is None:
                        print(
                            f"{datetime.datetime.now()} testing :: test id: {test_id} to check if time is reasonable: ",
                            end = ""
                        )
                        if self.skip_testing:
                            print(f"Skipping testing")
                        elif compiled_lib is None:
                            print(f"Cannot test because lib is none")
                        else:
                            t_result, t_correct, t_mean_error, t_consistent, t_waiting_time, t_finished  = (
                                self._run_benchmark(compiled_lib, runs = 1, external=True)
                            )
                            result_writer.writerow(
                                [
                                    *test_id,
                                    -1,
                                    1,
                                    t_result,
                                    t_correct,
                                    t_mean_error,
                                    t_consistent,
                                    t_waiting_time,
                                    t_finished
                                ]
                            )
                            csv_results.flush()
                            print(
                                f" ==>  result: {t_result}, correct: {t_correct}, mean error: {t_mean_error}, consistent: {t_consistent}, waiting time: {t_waiting_time}, finished: {t_finished} :: ",
                                end = ""
                            )
                            if not t_finished or t_result < 0 or t_result > self.waste_of_time_threshold:
                                print(
                                    f"Test time too long - Skipping"
                                )
                                continue
                            elif t_result < self.test_too_short_threshold:
                                runs_to_do = self.runs * self.long_run_multiplier
                                print(
                                    f"Runs set to: {self.runs}*{self.long_run_multiplier}={runs_to_do}"
                                )
                            else:
                                print(
                                    f"Runs set to: {runs_to_do}"
                                )


                    for rep in range(self.repeats):
                        if (*test_id, rep) in results_done:
                            print(
                                f"Skipping test id: {test_id}, rep: {rep} as it is already in the results file"
                            )
                            continue
                        print(
                            f"{datetime.datetime.now()} Running benchmark repeat :: test id: {test_id}, rep: {rep} ",
                            end="",
                        )

                        if self.skip_testing:
                            print(f"Skipping testing")
                        elif compiled_lib is None:
                            print(f"Cannot test because lib is none")
                        else:
                            result, correct, mean_error, consistent, waiting_time, finished = (
                                self._run_benchmark(compiled_lib, runs_to_do, external=False)
                            )
                            result_writer.writerow(
                                [
                                    *test_id,
                                    rep,
                                    runs_to_do,
                                    result,
                                    correct,
                                    mean_error,
                                    consistent,
                                    waiting_time,
                                    finished
                                ]
                            )
                            csv_results.flush()
                            results_correct &= correct
                            print(
                                f" ==>  result: {result}, correct: {correct}, mean error: {mean_error}, consistent: {consistent}, waiting time: {waiting_time}, finished {finished}"
                            )
                    if compiled_lib is not None:
                        lib = compiled_lib[0]
                        lib._close(delete=False)
        print(f"finished - results all correct: {results_correct}")


    def _run_benchmark(self, lib_paths: tuple[DTLCLib, str, str], runs: int, external: bool = False) -> tuple[
        float, bool, float, bool, float, bool]:
        lib, lib_path, func_types_path = lib_paths
        if external:
            return self._run_benchmark_external(lib_path, func_types_path, runs)
        chars = inline_print(0, "running benchmark")
        start_time = time.time()
        result, correct, mean_error, consistent =  benchmarkRunner.run_benchmark(lib, runs, self.np_args, self.np_ress, self.get_setup(), self.get_benchmark(), self.get_test(), self.get_clean())
        waiting_time = time.time() - start_time
        chars = inline_print(chars, f"")
        return result, correct, mean_error, consistent, waiting_time, True

    def _run_benchmark_external(self,  lib_path: str, func_types_path: str, runs: int) -> tuple[float, bool, float, bool, float, bool]:
        chars = inline_print(0, "setting up")
        args = []
        args.extend(["python", "benchmarking/benchmarkRunner.py"])
        args.extend([lib_path, func_types_path])
        args.append(f"{runs}")
        for np_arg_name, np_arg_path in self.np_arg_paths.items():
            args.extend([f"-a={np_arg_name}", np_arg_path])
        for np_res_name, np_res_path in self.np_res_paths.items():
            args.extend([f"-r={np_res_name}", np_res_path])
        args.extend(["--setup", self.get_setup()])
        args.extend(["--benchmark", self.get_benchmark()])
        args.extend(["--test", self.get_test()])
        args.extend(["--clean", self.get_clean()])

        current_env = os.environ.copy()

        start_time = time.time()
        process_benchmark = subprocess.Popen(
            args, env=current_env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        chars = inline_print(chars, "waiting for benchmark")

        finished = False
        try:
            out, err = process_benchmark.communicate(timeout=self.benchmark_timeout*runs)
            finished = True
            waiting_time = time.time() - start_time
            chars = inline_print(chars, "finished")
        except subprocess.TimeoutExpired as e:
            waiting_time = time.time() - start_time
            chars = inline_print(chars, "timed out")
            process_benchmark.kill()
            out, err = process_benchmark.communicate()
            chars = inline_print(chars, "killed")

        out = out.decode("utf8")
        split_out = out.split("\n")
        if len(split_out) != 5:
            if finished:
                print(f" ERROR: test finished but output is: {split_out} ")
                return (-1.0, False, -1.0, False, waiting_time, True)
            else:
                return (-1.0, True, -1.0, True, waiting_time, False)
        result = float(split_out[0])
        correct = bool(split_out[1] == "True")
        mean_error = float(split_out[2])
        consistent = bool(split_out[3] == "True")
        last_check = split_out[4] == ""
        if not last_check:
            return (-2.0, False, -2.0, False, waiting_time, finished)
        return (result, correct, mean_error, consistent, waiting_time, finished)

    def _generate_versions(
        self, layout_graph: LayoutGraph, iteration_map: IterationMap, lib_builder: LibBuilder,
    ) -> tuple[list[PtrMapping], list[IterationMapping]]:

        layout_store_keys = f"{self.layout_store}/keys"
        os.makedirs(layout_store_keys, exist_ok=True)
        layout_store_values = f"{self.layout_store}/values"
        os.makedirs(layout_store_values, exist_ok=True)

        loaded_layouts = None
        max_key = -1
        count = 0
        checked_keys = set()
        files = os.listdir(layout_store_keys)
        for file in files:
            print(".", end="")
            count += 1

            key_file_name = os.fsdecode(file)
            is_locked = key_file_name.endswith(".lock")
            key_int = int(key_file_name.removesuffix(".lock"))
            checked_keys.add(key_int)
            if not is_locked:
                is_locked = os.path.exists(f"{layout_store_keys}/{key_int}.lock")
            max_key = max(max_key, key_int)
            if not is_locked:
                with open(f"{layout_store_keys}/{key_int}", "rb") as f:
                    loaded_layout_graph = pickle.load(f)
                if layout_graph.matches(loaded_layout_graph):
                    print(
                        f"{'\b' * count}Found matching layout graph in layout store: {key_int}"
                    )
                    with open(f"{layout_store_values}/{key_int}", "rb") as f:
                        loaded_layouts = pickle.load(f)
                    with open(f"{self.base_dir}/layout", "a") as f:
                        f.write(f"{datetime.datetime.now()} Using Loaded layouts {key_int} from layouts store\n")
                    break

        if loaded_layouts is None and not self.do_not_generate:
            new_key = max_key + 1
            print("\b" * count, end="")
            new_key_str = f"{new_key}?"
            count = len(new_key_str)
            print(
                f"No matching layout graph found after checking {len(checked_keys)} graphs. Generating from scratch as {new_key_str}", end=""
            )

            file_found = False
            pid = os.getpid()
            while not file_found:
                try:
                    with open(f"{layout_store_keys}/{new_key}.lock", "x") as f:
                        f.write(f"{int(pid)}")
                    file_found = True
                except FileExistsError as e:
                    new_key = new_key + 1
                    new_key_str = f"{new_key}?"
                    print(f"{'\b'*count}{new_key_str}")
            print("\b")

            dtl_config_map: dict[TupleStruct[TensorVariable], ReifyConfig] = self.get_configs_for_DTL_tensors()
            config_map = {}
            for tensor_var, config in dtl_config_map.items():
                assert tensor_var in lib_builder.tensor_var_details
                ident = lib_builder.get_base_version_for_ptr(lib_builder.tensor_var_details[tensor_var]).identification
                closure = layout_graph.get_transitive_closure(ident)
                for i in closure:
                    assert i not in config_map
                    config_map[i] = config

            new_layouts = LayoutGenerator(
                layout_graph, config_map, plot_dir=f"{self.layout_store}/gen/{new_key}"
            ).generate_mappings(take_first=self.take_first_layouts)
            print("Writing new layouts to pickle")

            with open(f"{layout_store_keys}/{new_key}.lock", "r") as f:
                lock_pid = int(f.read())
                if lock_pid != pid:
                    print(f"ERROR: lock pid: {lock_pid} does not match out pid {pid} - something has tampered with the layouts store directory")
                    assert False, f"lock pid: {lock_pid} does not match out pid {pid} - something has tampered with the layouts store directory"
            with open(f"{layout_store_values}/{new_key}", "wb") as f:
                f.write(pickle.dumps(new_layouts))
            with open(f"{layout_store_keys}/{new_key}", "wb") as f:
                f.write(pickle.dumps(layout_graph))
            loaded_layouts = new_layouts
            os.remove(f"{layout_store_keys}/{new_key}.lock")
            with open(f"{self.base_dir}/layout", "a") as f:
                f.write(f"{datetime.datetime.now()} Generated Layouts as {new_key} in layouts store\n")

        order_store_keys = f"{self.order_store}/keys"
        os.makedirs(order_store_keys, exist_ok=True)
        order_store_values = f"{self.order_store}/values"
        os.makedirs(order_store_values, exist_ok=True)

        loaded_orders = None
        max_key = -1
        count = 0
        checked_keys = set()
        for file in os.listdir(order_store_keys):
            print(".", end="")
            count += 1

            key_file_name = os.fsdecode(file)
            is_locked = key_file_name.endswith(".lock")
            key_int = int(key_file_name.removesuffix(".lock"))
            checked_keys.add(key_int)
            if not is_locked:
                is_locked = os.path.exists(f"{order_store_keys}/{key_int}.lock")
            max_key = max(max_key, key_int)
            if not is_locked:
                with open(f"{order_store_keys}/{key_file_name}", "rb") as f:
                    loaded_iteration_map = pickle.load(f)
                if iteration_map.matches(loaded_iteration_map):
                    print(
                        f"{'\b' * count}Found matching iteration map in order store: {key_int}"
                    )
                    with open(f"{order_store_values}/{key_file_name}", "rb") as f:
                        loaded_orders = pickle.load(f)
                    with open(f"{self.base_dir}/order", "a") as f:
                        f.write(f"{datetime.datetime.now()} Using Loaded orders {key_int} from orders store\n")
                    break

        if loaded_orders is None and not self.do_not_generate:
            new_key = max_key + 1
            print("\b" * count, end="")
            new_key_str = f"{new_key}?"
            count = len(new_key_str)
            print(
                f"No matching iteration map found after checking {len(checked_keys)} maps. Generating from scratch as {new_key_str}"
            )

            file_found = False
            pid = os.getpid()
            while not file_found:
                try:
                    with open(f"{order_store_keys}/{new_key}.lock", "x") as f:
                        f.write(f"{int(pid)}")
                    file_found = True
                except FileExistsError as e:
                    new_key = new_key + 1
                    new_key_str = f"{new_key}?"
                    print(f"{'\b'*count}{new_key_str}", end="")
            print("\b")

            new_orders = IterationGenerator(
                iteration_map, plot_dir=f"{self.order_store}/gen/{new_key}"
            ).generate_mappings(take_first=self.take_first_orders)
            print("Writing new orders to pickle")

            with open(f"{order_store_keys}/{new_key}.lock", "r") as f:
                lock_pid = int(f.read())
                if lock_pid != pid:
                    print(
                        f"ERROR: lock pid: {lock_pid} does not match out pid {pid} - something has tampered with the orders store directory")
                    assert False, f"lock pid: {lock_pid} does not match out pid {pid} - something has tampered with the orders store directory"

            with open(f"{order_store_values}/{new_key}", "wb") as f:
                f.write(pickle.dumps(new_orders))
            with open(f"{order_store_keys}/{new_key}", "wb") as f:
                f.write(pickle.dumps(iteration_map))
            loaded_orders = new_orders
            os.remove(f"{order_store_keys}/{new_key}.lock")
            with open(f"{self.base_dir}/order", "a") as f:
                f.write(f"{datetime.datetime.now()} Generated orders as {new_key} in orders store\n")

        if loaded_layouts is None:
            print("loaded_layouts is none! Probably because do_not_generate is set")
            loaded_layouts = []
        if loaded_orders is None:
            print("loaded_orders is none! Probably because do_not_generate is set")
            loaded_orders = []

        new_layouts = list(loaded_layouts)
        new_orders = list(loaded_orders)
        return new_layouts, new_orders

    @staticmethod
    def _print_to_str(module: ModuleOp) -> str:
        res = StringIO()
        printer = Printer(print_generic_format=False, stream=res)
        printer.print(module)
        return res.getvalue()

def inline_print(chars: int, string: str) -> int:
    print(("\b" * chars) + string, end="")
    return len(string)