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

from dtl.libBuilder import DTLCLib, LibBuilder
from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer
from xdsl.transforms.experimental.dlt.generate_dlt_iteration_orders import IterationGenerator, IterationMapping
from xdsl.transforms.experimental.dlt.generate_dlt_layouts import LayoutGenerator, PtrMapping
from xdsl.transforms.experimental.dlt.iteration_map import IterationMap
from xdsl.transforms.experimental.dlt.layout_graph import LayoutGraph

_T = typing.TypeVar('_T', bound=tuple["_CData", ...])

class Benchmark(abc.ABC):

    def __init__(self, base_dir: str, layout_store: str, order_store: str, runs: int, repeats: int, epsilon: float):
        self.base_dir = base_dir
        self.runs = runs
        self.repeats = repeats
        self.epsilon = epsilon

        self.layout_store = layout_store
        self.order_store = order_store

        self._lib_store: dict[tuple[int, int], DTLCLib] = {}

        self.take_first_layouts = 0
        self.take_first_orders = 0
        self.do_not_generate = False
        self.only_generate = False
        self.do_not_compile_mlir = False
        self.only_compile_to_llvm = False
        self.skip_testing = False

        os.makedirs(self.base_dir, exist_ok=True)

    def handle_reference_array(self, array: nptyping.NDArray, name: str):
        path = f"{self.base_dir}/{name}"
        if os.path.exists(path):
            loaded_array = np.loadtxt(path)
            if not np.equal(array, loaded_array).all():
                print(f"ERROR! loaded array does not match newly calculated reference array: {path}")
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
    def setup(self, lib: DTLCLib) -> _T:
        raise NotImplementedError

    @abc.abstractmethod
    def get_benchmark(self, lib: DTLCLib) -> typing.Callable[[_T], None]:
        raise NotImplementedError

    @abc.abstractmethod
    def test(self, lib: DTLCLib, args: _T, first_args: _T) -> tuple[bool, float, bool]: # (within_epsilon, total_error, bit_wise_reapeatable)
        raise NotImplementedError

    @abc.abstractmethod
    def teardown(self, lib: DTLCLib, args: _T) -> None:
        raise NotImplementedError

    def run(self):
        print(f"{datetime.datetime.now()} Running benchmark {type(self)} at {self.base_dir}")
        print(f"{datetime.datetime.now()} Defining lib builder")
        lib_builder = self.define_lib_builder()
        print(f"{datetime.datetime.now()} Starting benchmarking...")
        self.run_benchmarking(lib_builder)

    def get_compiled_lib(self, new_layout: PtrMapping, new_order: IterationMapping, module: ModuleOp, lib_builder: LibBuilder, layout_graph: LayoutGraph, iteration_map: IterationMap) -> DTLCLib | None:
        if (key := (new_layout.number, new_order.number)) in self._lib_store:
            return self._lib_store[key]

        lib = None
        lib_name = f"lib_{new_layout.number}_{new_order.number}"
        llvm_path = f"{self.base_dir}/llvm/{lib_name}.ll"
        print(f"Getting compiled library for: layout: {new_layout.number}, iter_order: {new_order.number}: ", end="")
        llvm_exists = os.path.exists(llvm_path)

        if llvm_exists and self.only_compile_to_llvm:
            print(f"Found existing LLVM file and Skipping LLVM compilation")
            return None
        elif not llvm_exists and self.do_not_compile_mlir:
            print("Do not compile mlir is set, but no llvm ir file was found")
            return None
        else:
            module_clone = module.clone()
            function_types = lib_builder.lower(module_clone, layout_graph, new_layout.make_ptr_dict(),
                                               iteration_map,
                                               new_order.make_iter_dict(), verbose=0)
            if llvm_exists:
                print(f"Found existing LLVM file, ",  end="")
                lib = lib_builder.compile_from(llvm_path, function_types, verbose=0)
                print(f"LLVM file compiled to {lib._library_path}")
                self._lib_store[key] = lib
                return lib
            else:
                lib = lib_builder.compile(module_clone, function_types, llvm_out=llvm_path,
                                          llvm_only=self.only_compile_to_llvm, verbose=0)
                if lib is not None:
                    print(f"Compiled to LLVM: {llvm_path} and then to library: {lib._library_path}")
                    self._lib_store[key] = lib
                    return lib
                else:
                    print(f"Compiled to LLVM: {llvm_path} and but no library was produced.")
                    return None

        if os.path.exists(llvm_path):
            print(f"Found existing LLVM file, ", end="")
            if not self.only_compile_to_llvm:
                module_clone = module.clone()
                function_types = lib_builder.lower(module_clone, layout_graph, new_layout.make_ptr_dict(),
                                                   iteration_map,
                                                   new_order.make_iter_dict(), verbose=0)
                lib = lib_builder.compile_from(llvm_path, function_types, verbose=0)
                print(f"LLVM file compiled to {lib._library_path}")
            else:
                print("Skipping LLVM compilation")
                return None
        elif self.do_not_compile_mlir:
            print("Do not compile mlir is set, but no llvm ir file was found")
        else:
            module_clone = module.clone()
            function_types = lib_builder.lower(module_clone, layout_graph, new_layout.make_ptr_dict(),
                                               iteration_map,
                                               new_order.make_iter_dict(), verbose=0)
            lib = lib_builder.compile(module_clone, function_types, llvm_out=llvm_path,
                                          llvm_only=self.only_compile_to_llvm, verbose=0)
            if lib is not None:
                print(f"Compiled to LLVM: {llvm_path} and then to library: {lib._library_path}")
            else:
                print(f"Compiled to LLVM: {llvm_path} and but no library was produced.")

        if lib is not None:
            self._lib_store[key] = lib
        return lib

    def close_compiled_lib(self, lib):
        keys = []
        for k, value in self._lib_store.items():
            if value == lib:
                keys.append(k)
        for k in keys:
            del self._lib_store[k]
        lib._close(delete=True)
        del lib


    def run_benchmarking(self, lib_builder: LibBuilder) -> None:
        module, layout_graph, iteration_map = lib_builder.prepare(verbose=0)

        print(f"{datetime.datetime.now()} Generating possible layouts and iteration maps")
        layouts, orders = self._generate_versions(layout_graph, iteration_map)
        if self.only_generate:
            print("Only Generate is set, so benchmarking is ending.")
            return

        print(f"{datetime.datetime.now()} Loading from existing results")
        results_done = set()
        results_file = f"{self.base_dir}/results.csv"
        if os.path.exists(results_file):
            with open(results_file, "r", newline="") as f:
                r = csv.reader(f)
                next(r, None)  # skip header
                for row in r:
                    layout_num = row[0]
                    order_num = row[1]
                    rep = row[2]
                    results_done.add((int(layout_num), int(order_num), int(rep)))
        print(f"{datetime.datetime.now()} Found {len(results_done)} results")



        write_results_header = not os.path.exists(results_file)

        with open(results_file, "a", newline="") as csv_results:
            result_writer = csv.writer(csv_results)
            if write_results_header:
                result_writer.writerow(
                    ["layout_mapping", "iter_mapping", "rep", "time", "within_epsilon", "per_run_error",
                     "bit_repeatable"])

            lib = None
            count = 0
            total_l_o_pairs = len(layouts) * len(orders)

            for new_layout in layouts:
                for new_order in orders:
                    count += 1
                    print(f"{datetime.datetime.now()} Running Benchmarks for layout: {new_layout.number}, order: {new_order.number}. ({count}/{total_l_o_pairs})")
                    results = []
                    results_correct = True
                    if all((new_layout.number, new_order.number, rep) in results_done for rep in range(self.repeats)):
                        print(f"{datetime.datetime.now()} Skipping layout: {new_layout.number}, order: {new_order.number} for all repeats [0..{self.repeats}) as they are already in the results file")
                        continue

                    lib = self.get_compiled_lib(new_layout, new_order, module, lib_builder, layout_graph,
                                                iteration_map)
                    for rep in range(self.repeats):
                        test_id = (new_layout.number, new_order.number, rep)
                        if test_id in results_done:
                            print(
                                f"Skipping layout: {new_layout.number}, order: {new_order.number}, rep: {rep} as it is already in the results file")
                            continue
                        print(
                            f"{datetime.datetime.now()} Running benchmark repeat :: l: {new_layout.number}, o: {new_order.number}, rep: {rep} ",
                            end="")

                        if self.skip_testing:
                            print(f"Skipping testing")
                        elif lib is None:
                            print(f"Cannot test because lib is none")
                        else:
                            result, epsilon_correct, error_per_run, bit_repeatable = self._run_benchmark(lib)
                            result_writer.writerow(
                                [new_layout.number, new_order.number, rep, result, epsilon_correct, error_per_run,
                                 bit_repeatable])
                            csv_results.flush()
                            results.append(result)
                            results_correct &= epsilon_correct
                            print(
                                f" ==>  result: {result}, epsilon_correct: {epsilon_correct}, error_per_run: {error_per_run}, bit_repeatable: {bit_repeatable}")
            if lib is not None:
                self.close_compiled_lib(lib)
        print("finished")

    def _run_benchmark(self, lib: DTLCLib):
        run_args = []
        chars = 0

        val = "setting up"
        print("\b"*chars + val, end="")
        chars = len(val)

        for r in range(self.runs):
            args = self.setup(lib)
            run_args.append(args)
            print(".", end="")
            chars += 1

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
            print(".", end="")
            chars += 1

        val = "tearing down"
        print("\b" * chars + val, end="")
        chars = len(val)

        for r, args in enumerate(run_args):
            self.teardown(lib, args)
            print(".", end="")
            chars += 1

        mean_error = total_error / self.runs

        val = ""
        print("\b" * chars + val, end="")
        chars = len(val)

        return result, total_within_epsilon, mean_error, total_bitwise_consistent

    def _generate_versions(self, layout_graph: LayoutGraph, iteration_map: IterationMap) -> tuple[list[PtrMapping], list[IterationMapping]]:

        layout_store_keys = f"{self.layout_store}/keys"
        os.makedirs(layout_store_keys, exist_ok=True)
        layout_store_values = f"{self.layout_store}/values"
        os.makedirs(layout_store_values, exist_ok=True)

        loaded_layouts = None
        max_key = -1
        count = 0
        for file in os.listdir(layout_store_keys):
            print(".", end="")
            count += 1

            key = os.fsdecode(file)
            key_int = int(key)
            max_key = max(max_key, key_int)
            with open(f"{layout_store_keys}/{key}", "rb") as f:
                loaded_layout_graph = pickle.load(f)

            if layout_graph.matches(loaded_layout_graph):
                print(f"{'\b'*count}Found matching layout graph in layout store: {key_int}")
                with open(f"{layout_store_values}/{key}", "rb") as f:
                    loaded_layouts = pickle.load(f)
                break
        if loaded_layouts is None and not self.do_not_generate:
            new_key = max_key + 1
            print("\b"*count, end="")
            print(f"No matching layout graph found after checking {count} graphs. Generating from scratch as {new_key}")
            new_layouts = LayoutGenerator(layout_graph, plot_dir=f"{self.layout_store}/gen/{new_key}").generate_mappings(
                take_first=self.take_first_layouts)
            print("Writing new layouts to pickle")
            with open(f"{layout_store_values}/{new_key}", "wb") as f: f.write(pickle.dumps(new_layouts))
            with open(f"{layout_store_keys}/{new_key}", "wb") as f: f.write(pickle.dumps(layout_graph))
            loaded_layouts = new_layouts

        order_store_keys = f"{self.order_store}/keys"
        os.makedirs(order_store_keys, exist_ok=True)
        order_store_values = f"{self.order_store}/values"
        os.makedirs(order_store_values, exist_ok=True)

        loaded_orders = None
        max_key = -1
        count = 0
        for file in os.listdir(order_store_keys):
            print(".", end="")
            count += 1

            key = os.fsdecode(file)
            key_int = int(key)
            max_key = max(max_key, key_int)
            with open(f"{order_store_keys}/{key}", "rb") as f:
                loaded_iteration_map = pickle.load(f)

            if iteration_map.matches(loaded_iteration_map):
                print(f"{'\b' * count}Found matching iteration map in order store: {key_int}")
                with open(f"{order_store_values}/{key}", "rb") as f:
                    loaded_orders = pickle.load(f)
                break
        if loaded_orders is None and not self.do_not_generate:
            new_key = max_key + 1
            print("\b" * count, end="")
            print(f"No matching iteration map found after checking {count} maps. Generating from scratch as {new_key}")
            new_orders = IterationGenerator(iteration_map, plot_dir=f"{self.order_store}/gen/{new_key}").generate_mappings(
                take_first=self.take_first_orders)
            print("Writing new orders to pickle")
            with open(f"{order_store_values}/{new_key}", "wb") as f: f.write(pickle.dumps(new_orders))
            with open(f"{order_store_keys}/{new_key}", "wb") as f: f.write(pickle.dumps(iteration_map))
            loaded_orders = new_orders

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