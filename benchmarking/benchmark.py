import abc
import csv
import datetime
import os
import pickle
import timeit
import typing
from io import StringIO

from dtl.libBuilder import DTLCLib, LibBuilder
from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer
from xdsl.transforms.experimental.dlt.generate_dlt_iteration_orders import IterationGenerator, IterationMapping
from xdsl.transforms.experimental.dlt.generate_dlt_layouts import LayoutGenerator, PtrMapping
from xdsl.transforms.experimental.dlt.iteration_map import IterationMap
from xdsl.transforms.experimental.dlt.layout_graph import LayoutGraph

_T = typing.TypeVar('_T', bound=tuple["_CData", ...])

class Benchmark(abc.ABC):

    def __init__(self, base_dir: str, runs: int, repeats: int, epsilon: float):
        self.base_dir = base_dir
        self.runs = runs
        self.repeats = repeats
        self.epsilon = epsilon

        self._lib_store: dict[tuple[int, int], DTLCLib] = {}

        self.take_first_layouts = 0
        self.take_first_orders = 0
        self.do_not_generate = False
        self.only_generate = False
        self.do_not_compile_mlir = False
        self.only_compile_to_llvm = False
        self.skip_testing = False

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
        print(f"Getting compiled library for: layout={new_layout.number}, iter_order={new_order.number}")
        module_clone = module.clone()
        function_types = lib_builder.lower(module_clone, layout_graph, new_layout.make_ptr_dict(),
                                           iteration_map,
                                           new_order.make_iter_dict(), verbose=0)
        if os.path.exists(llvm_path):
            print(f"Found existing LLVM file for: layout={new_layout.number}, iter_order={new_order.number}")
            if not self.only_compile_to_llvm:
                lib = lib_builder.compile_from(llvm_path, function_types, verbose=0)
                print(f"LLVM file compiled to {lib._library_path}")
        elif self.do_not_compile_mlir:
            print("Do not compile mlir is set, but no llvm ir file was found")
        else:
            lib = lib_builder.compile(module_clone, function_types, llvm_out=llvm_path,
                                          llvm_only=self.only_compile_to_llvm, verbose=0)
            if lib is not None:
                print(f"Compiled to LLVM: {llvm_path} and then to library: {lib._library_path}")
            else:
                print(f"Compiled to LLVM: {llvm_path} and but no library was produced.")

        if lib is not None:
            self._lib_store[key] = lib
        return lib

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

            for new_layout in layouts:
                for new_order in orders:
                    print(f"{datetime.datetime.now()} Running Benchmarks for layout: {new_layout.number}, order: {new_order.number}")
                    lib = self.get_compiled_lib(new_layout, new_order, module, lib_builder, layout_graph, iteration_map)
                    results = []
                    results_correct = True
                    for rep in range(self.repeats):
                        test_id = (new_layout.number, new_order.number, rep)
                        if test_id in results_done:
                            print(
                                f"Skipping layout: {new_layout.number}, order: {new_order.number}, rep: {rep} as it is already in the results file")
                            continue


                        print(
                            f"{datetime.datetime.now()} Running benchmark repeat :: rep: {rep} ",
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

        layouts_pickle_file = f"{self.base_dir}/layouts/layouts.pickle"

        new_layouts = None
        if os.path.exists(layouts_pickle_file):
            print("loading layouts from pickle")
            loaded_layout_graph, loaded_layouts = pickle.load(open(layouts_pickle_file, "rb"))
            if layout_graph.matches(loaded_layout_graph):
                print("pickle layouts graph matches")
                new_layouts = loaded_layouts
            else:
                print("pickle loaded but layout graph does not match")

        if new_layouts is None and not self.do_not_generate:
            print("Generating layouts")
            new_layouts = LayoutGenerator(layout_graph, plot_dir=f"{self.base_dir}/layouts/").generate_mappings(take_first=self.take_first_layouts)
            print("Writing new layouts to pickle")
            open(layouts_pickle_file, "wb").write(pickle.dumps((layout_graph, new_layouts)))

        orders_pickle_file = f"{self.base_dir}/iter_orders/orders.pickle"

        new_orders = None
        if os.path.exists(orders_pickle_file):
            print("loading orders from pickle")
            loaded_iteration_map, loaded_orders = pickle.load(open(orders_pickle_file, "rb"))
            if iteration_map.matches(loaded_iteration_map):
                print("pickle orders map matches")
                new_orders = loaded_orders
            else:
                print("pickle loaded but iteration map does not match")

        if new_orders is None and not self.do_not_generate:
            print("Generating orders")
            new_orders = IterationGenerator(iteration_map, plot_dir=f"{self.base_dir}/iter_orders/").generate_mappings(
                take_first=self.take_first_orders)
            print("Writing new orders to pickle")
            open(orders_pickle_file, "wb").write(pickle.dumps((iteration_map, new_orders)))

        if new_layouts is None:
            print("new_layouts is none! Probably because do_not_generate is set")
            new_layouts = []
        if new_orders is None:
            print("new_orders is none! Probably because do_not_generate is set")
            new_orders = []

        new_layouts = list(new_layouts)
        new_orders = list(new_orders)
        return new_layouts, new_orders

    @staticmethod
    def _print_to_str(module: ModuleOp) -> str:
        res = StringIO()
        printer = Printer(print_generic_format=False, stream=res)
        printer.print(module)
        return res.getvalue()