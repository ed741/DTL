import abc
import csv
import datetime
import os
import pickle
import timeit
import typing
from io import StringIO

import nptyping
import numpy as np

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

_T = typing.TypeVar("_T", bound=tuple["_CData", ...])


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
    ):
        self.base_dir = base_dir
        self.runs = runs
        self.repeats = repeats
        self.opt_num = opt_num
        self.epsilon = epsilon

        self.layout_store = layout_store
        self.order_store = order_store

        self._lib_store: dict[tuple[int, int], DTLCLib] = {}

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

    def handle_reference_array(self, array: nptyping.NDArray, name: str):
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

    @abc.abstractmethod
    def define_lib_builder(self) -> LibBuilder:
        raise NotImplementedError

    @abc.abstractmethod
    def get_configs_for_DTL_tensors(self) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        raise NotImplementedError

    @abc.abstractmethod
    def setup(self, lib: DTLCLib) -> _T:
        raise NotImplementedError

    @abc.abstractmethod
    def get_benchmark(self, lib: DTLCLib) -> typing.Callable[[_T], None]:
        raise NotImplementedError

    @abc.abstractmethod
    def test(
        self, lib: DTLCLib, args: _T, first_args: _T
    ) -> tuple[
        bool, float, bool
    ]:  # (within_epsilon, total_error, bit_wise_reapeatable)
        raise NotImplementedError

    @abc.abstractmethod
    def teardown(self, lib: DTLCLib, args: _T) -> None:
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
    ) -> DTLCLib | None:
        print(
            f"Getting lib for: l: {new_layout.number}, o: {new_order.number} :: ",
            end="",
        )
        if (key := (new_layout.number, new_order.number)) in self._lib_store:
            print("loaded from store")
            return self._lib_store[key]

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
            print("loaded.")
            self._lib_store[key] = lib
            return lib
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
                self._lib_store[key] = lib
                return lib
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
                    self._lib_store[key] = lib
                    return lib
                else:
                    print(f"compiled to LLVM: {llvm_path} but no lib was produced.")
                    return None

    def close_compiled_lib(self, lib: DTLCLib, delete: bool = False):
        keys = []
        for k, value in self._lib_store.items():
            if value == lib:
                keys.append(k)
        for k in keys:
            del self._lib_store[k]
        lib._close(delete=delete)
        del lib

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
                    layout_num = row[0]
                    order_num = row[1]
                    rep = row[2]
                    runs = row[3]
                    time = row[4]
                    results_done[(int(layout_num), int(order_num), int(rep))] = (int(runs), float(time))
        print(f"{datetime.datetime.now()} Found {len(results_done)} results")

        write_results_header = not os.path.exists(results_file)
        results_correct = True

        with open(results_file, "a", newline="") as csv_results:
            result_writer = csv.writer(csv_results)
            if write_results_header:
                result_writer.writerow(
                    [
                        "layout_mapping",
                        "iter_mapping",
                        "rep",
                        "runs",
                        "time",
                        "within_epsilon",
                        "per_run_error",
                        "bit_repeatable",
                    ]
                )

            count = 0
            total_l_o_pairs = len(layouts) * len(orders)

            for new_layout in layouts:
                for new_order in orders:
                    count += 1
                    print(
                        f"{datetime.datetime.now()} Running Benchmarks for layout: {new_layout.number}, order: {new_order.number}. ({count}/{total_l_o_pairs})"
                    )

                    if all(
                        (new_layout.number, new_order.number, rep) in results_done
                        for rep in range(-1, self.repeats)
                    ):
                        print(
                            f"{datetime.datetime.now()} Skipping layout: {new_layout.number}, order: {new_order.number} for all repeats [0..{self.repeats}) as they are already in the results file"
                        )
                        continue

                    test_time = None
                    runs_to_do = self.runs
                    if (new_layout.number, new_order.number, -1) in results_done:
                        t_r, test_time_result = results_done[(new_layout.number, new_order.number, -1)]
                        test_time = test_time_result / t_r
                        if test_time > self.waste_of_time_threshold:
                            print(
                                f"{datetime.datetime.now()} Skipping layout: {new_layout.number}, order: {new_order.number} for all repeats [0..{self.repeats}) as the test time was {test_time}"
                            )
                            continue
                        elif test_time < self.test_too_short_threshold:
                            runs_to_do = self.runs * self.long_run_multiplier
                            print(
                                f"Test result found and runs set to: {self.runs}*{self.long_run_multiplier}={runs_to_do}"
                            )
                        else:
                            print(
                                f"Test result found and runs set to: {runs_to_do}"
                            )

                    lib = self.get_compiled_lib(
                        new_layout,
                        new_order,
                        module,
                        lib_builder,
                        layout_graph,
                        iteration_map,
                    )

                    if test_time is None:
                        print(
                            f"{datetime.datetime.now()} testing :: l: {new_layout.number}, o: {new_order.number} to check if time is reasonable: ",
                            end = ""
                        )
                        if self.skip_testing:
                            print(f"Skipping testing")
                        elif lib is None:
                            print(f"Cannot test because lib is none")
                        else:
                            result, epsilon_correct, error_per_run, bit_repeatable = (
                                self._run_benchmark(lib, runs = 1)
                            )
                            result_writer.writerow(
                                [
                                    new_layout.number,
                                    new_order.number,
                                    -1,
                                    1,
                                    result,
                                    epsilon_correct,
                                    error_per_run,
                                    bit_repeatable,
                                ]
                            )
                            csv_results.flush()
                            test_time = result
                            print(
                                f" ==>  result: {result}, epsilon_correct: {epsilon_correct}, error_per_run: {error_per_run}, bit_repeatable: {bit_repeatable} :: ",
                                end = ""
                            )
                            if test_time > self.waste_of_time_threshold:
                                print(
                                    f"Test time too long - Skipping"
                                )
                                continue
                            elif test_time < self.test_too_short_threshold:
                                runs_to_do = self.runs * self.long_run_multiplier
                                print(
                                    f"Runs set to: {self.runs}*{self.long_run_multiplier}={runs_to_do}"
                                )
                            else:
                                print(
                                    f"Runs set to: {runs_to_do}"
                                )


                    for rep in range(self.repeats):
                        test_id = (new_layout.number, new_order.number, rep)
                        if test_id in results_done:
                            print(
                                f"Skipping layout: {new_layout.number}, order: {new_order.number}, rep: {rep} as it is already in the results file"
                            )
                            continue
                        print(
                            f"{datetime.datetime.now()} Running benchmark repeat :: l: {new_layout.number}, o: {new_order.number}, rep: {rep} ",
                            end="",
                        )

                        if self.skip_testing:
                            print(f"Skipping testing")
                        elif lib is None:
                            print(f"Cannot test because lib is none")
                        else:
                            result, epsilon_correct, error_per_run, bit_repeatable = (
                                self._run_benchmark(lib, runs_to_do)
                            )
                            result_writer.writerow(
                                [
                                    new_layout.number,
                                    new_order.number,
                                    rep,
                                    runs_to_do,
                                    result,
                                    epsilon_correct,
                                    error_per_run,
                                    bit_repeatable,
                                ]
                            )
                            csv_results.flush()
                            results_correct &= epsilon_correct
                            print(
                                f" ==>  result: {result}, epsilon_correct: {epsilon_correct}, error_per_run: {error_per_run}, bit_repeatable: {bit_repeatable}"
                            )
                    if lib is not None:
                        self.close_compiled_lib(lib)
        print(f"finished - results all correct: {results_correct}")

    def _run_benchmark(self, lib: DTLCLib, runs: int):
        run_args = []
        chars = 0

        val = "setting up"
        print("\b" * chars + val, end="")
        chars = len(val)

        for r in range(runs):
            args = self.setup(lib)
            run_args.append(args)
            if r % 2 == 0:
                print(".", end="")
                chars += 1
            else:
                print("\b", end="")
                chars -= 1

        val = "getting benchmark..."
        print("\b" * chars + val, end="")
        chars = len(val)

        single_benchmark = self.get_benchmark(lib)

        def benchmark():
            for args in run_args:
                single_benchmark(args)

        val = "running benchmark..."
        print("\b" * chars + val, end="")
        chars = len(val)

        result = timeit.timeit(benchmark, number=1)

        val = "testing results"
        print("\b" * chars + val, end="")
        chars = len(val)

        total_within_epsilon = True
        total_error = 0.0
        total_bitwise_consistent = True

        first_args = run_args[0]

        for r, args in enumerate(run_args):
            within_epsilon, error, bitwise_consistent = self.test(lib, args, first_args)
            total_within_epsilon &= within_epsilon
            total_error += error
            total_bitwise_consistent &= bitwise_consistent
            if r % 2 == 0:
                print(".", end="")
                chars += 1
            else:
                print("\b", end="")
                chars -= 1

        val = "tearing down"
        print("\b" * chars + val, end="")
        chars = len(val)

        for r, args in enumerate(run_args):
            self.teardown(lib, args)
            if r % 2 == 0:
                print(".", end="")
                chars += 1
            else:
                print("\b", end="")
                chars -= 1

        mean_error = total_error / len(run_args)

        val = ""
        print("\b" * chars + val, end="")
        chars = len(val)

        return result, total_within_epsilon, mean_error, total_bitwise_consistent

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
                    print(f"{'\b'*count}{new_key_str}")
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
