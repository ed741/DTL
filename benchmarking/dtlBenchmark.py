import abc
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, TypeVar, Generic

from benchmarking.benchmark import (
    Benchmark,
    BenchmarkSettings,
    ID_Tuple,
    Options,
    PythonCode,
    Test,
    TestCode,
)
from dtl import TensorVariable
from dtl.libBuilder import DTLCLib, LibBuilder, TupleStruct
from xdsl.dialects.builtin import ModuleOp
from xdsl.transforms.experimental.dlt.generate_dlt_iteration_orders import (
    IterationGenerator,
    IterationMapping,
)
from xdsl.transforms.experimental.dlt.generate_dlt_layouts import (
    LayoutGenerator,
    PtrMapping,
    ReifyConfig,
)
from xdsl.transforms.experimental.dlt.iteration_map import IterationMap
from xdsl.transforms.experimental.dlt.layout_graph import LayoutGraph


@dataclass(frozen=True)
class DLTCompileContext:
    lib_builder: LibBuilder
    module: ModuleOp
    layout_graph: LayoutGraph
    iteration_map: IterationMap
    options: Options


class DLTTest(Test, abc.ABC):

    def __init__(
        self,
        code: TestCode,
        context: DLTCompileContext,
        layout: PtrMapping,
        order: IterationMapping,
    ):
        super().__init__(code)
        self.context = context
        self.layout = layout
        self.order = order

    def get_lib_name(self) -> str:
        return f"lib_{self.layout.number}_{self.order.number}.so"

    def get_func_types_name(self) -> str:
        return f"lib_{self.layout.number}_{self.order.number}.ft"

    def get_path_str(self) -> str:
        return "_".join([str(p) for p in self.get_id()])

    def get_test_path(self, tests_path: str) -> str:
        return f"{tests_path}/{self.get_path_str()}"

    def get_load(self, tests_path: str) -> PythonCode:
        test_path = self.get_test_path(tests_path)
        code = """
import pickle
from dtl.libBuilder import DTLCLib, FuncTypeDescriptor
from xdsl.dialects.builtin import FunctionType, StringAttr
with open(##func_types_path##, "rb") as f:
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
function_types_str = {k.data: v for k, v in function_types.items()}
lib = DTLCLib(##lib_path##, dlt_func_types, function_types_str)
        """.replace(
            "##func_types_path##", f"\"{test_path}/{self.get_func_types_name()}\""
        ).replace(
            "##lib_path##", f"\"{test_path}/{self.get_lib_name()}\""
        )
        return code


T_DTL = TypeVar("T_DTL", bound=DLTTest)


class DTLBenchmark(Benchmark[T_DTL, DTLCLib], abc.ABC, Generic[T_DTL]):

    def __init__(
        self,
        base_dir: str,
        layout_store: str,
        order_store: str,
        settings: BenchmarkSettings,
        opt_num: int,
        skip_layout_func: Callable[[PtrMapping], bool] = None,
        skip_order_func: Callable[[IterationMapping], bool] = None,
    ):
        super().__init__(base_dir, settings)
        self.layout_store = layout_store
        self.order_store = order_store
        self.opt_num = opt_num
        self.skip_layout = (
            skip_layout_func if skip_layout_func is not None else lambda l: False
        )
        self.skip_order = (
            skip_order_func if skip_order_func is not None else lambda l: False
        )

    @abc.abstractmethod
    def define_lib_builder(self) -> LibBuilder:
        raise NotImplementedError

    @abc.abstractmethod
    def get_configs_for_DTL_tensors(
        self,
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        raise NotImplementedError

    @abc.abstractmethod
    def make_tests_for(
        self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping
    ) -> list[T_DTL]:
        raise NotImplementedError

    def enumerate_tests(
        self,
        context: DLTCompileContext,
        layouts: list[PtrMapping],
        orders: list[IterationMapping],
    ) -> list[T_DTL]:
        return [
            t
            for l in layouts
            for o in orders
            for t in self.make_tests_for(context, l, o)
        ]

    def parse_options(self, benchmark_options: list[str] = None) -> dict[str, Any]:
        options = super().parse_options(benchmark_options)
        options["only-to-llvm"] = "--only-to-llvm" in benchmark_options
        options["no-mlir"] = "--no-mlir" in benchmark_options
        options["do-not-lower"] = "--do-not-lower" in benchmark_options
        options["do-not-generate"] = "--do-not-generate" in benchmark_options

        only_layouts = set()
        only_orders = set()
        for arg in benchmark_options:
            if arg.startswith("-l="):
                only_layouts.add(int(arg.removeprefix("-l=")))
            elif arg.startswith("-o="):
                only_orders.add(int(arg.removeprefix("-o=")))
        if len(only_layouts) == 0:
            only_layouts = None
        if len(only_orders) == 0:
            only_orders = None
        options["only-layouts"] = only_layouts
        options["only-orders"] = only_orders

        take_first_layouts = 0
        take_first_orders = 0
        for arg in benchmark_options:
            if arg.startswith("-tfl="):
                take_first_layouts = int(arg.removeprefix("-tfl="))
            elif arg.startswith("-tfo="):
                take_first_orders = int(arg.removeprefix("-tfo="))

        options["take-first-layouts"] = take_first_layouts
        options["take-first-orders"] = take_first_orders

        return options

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

    def initialise_benchmarks(self, options: Options) -> list[T_DTL]:
        lib_builder = self.define_lib_builder()
        module, layout_graph, iteration_map = lib_builder.prepare(verbose=0)
        compile_context = DLTCompileContext(
            lib_builder, module, layout_graph, iteration_map, options
        )
        layouts = self.get_layouts(layout_graph, lib_builder, options)
        layouts = [l for l in layouts if not self.skip_layout(l)]
        orders = self.get_orders(iteration_map, lib_builder, options)
        orders = [o for o in orders if not self.skip_order(o)]
        tests = self.enumerate_tests(compile_context, layouts, orders)
        return tests

    def get_layouts(
        self, layout_graph: LayoutGraph, lib_builder: LibBuilder, options: Options
    ) -> list[PtrMapping]:

        match_func = lambda a, b: a.matches(b)
        found, key_int, layouts = self.search_store(
            self.layout_store,
            layout_graph,
            match_function=match_func,
            lock=(not options["do-not-generate"]),
        )

        if not found and key_int >= 0:
            gen_path = layouts
            dtl_config_map: dict[TupleStruct[TensorVariable], ReifyConfig] = (
                self.get_configs_for_DTL_tensors()
            )
            config_map = {}
            for tensor_var, config in dtl_config_map.items():
                assert tensor_var in lib_builder.tensor_var_details
                ident = lib_builder.get_base_version_for_ptr(
                    lib_builder.tensor_var_details[tensor_var]
                ).identification
                closure = layout_graph.get_transitive_closure(ident)
                for i in closure:
                    assert i not in config_map
                    config_map[i] = config

            new_layouts = LayoutGenerator(
                layout_graph, config_map, plot_dir=gen_path
            ).generate_mappings(take_first=options["take-first-layouts"])
            new_layouts = list(new_layouts)
            new_layouts.sort(key=lambda l: l.number)
            self.log("Writing new layouts to store")
            self.write_store(self.layout_store, layout_graph, new_layouts, key_int)
            layouts = new_layouts
        else:
            if layouts is None:
                self.log("No layouts found or produced")
                layouts = []

        if options["only-layouts"] is not None:
            self.log(f"Running only layouts: {options['only-layouts']}")
            layouts = [l for l in layouts if l.number in options["only-layouts"]]
        return layouts

    def get_orders(
        self,
        iteration_map: IterationMap,
        lib_builder: LibBuilder,
        options: Options,
    ) -> list[IterationMapping]:

        match_func = lambda a, b: a.matches(b)
        found, key_int, orders = self.search_store(
            self.order_store,
            iteration_map,
            match_function=match_func,
            lock=(not options["do-not-generate"]),
        )

        if not found and key_int >= 0:
            gen_path = orders

            new_orders = IterationGenerator(
                iteration_map, plot_dir=gen_path
            ).generate_mappings(take_first=options["take-first-orders"])
            new_orders = list(new_orders)
            new_orders.sort(key=lambda o: o.number)
            self.log("Writing new orders to store")

            self.write_store(self.order_store, iteration_map, new_orders, key_int)
            orders = new_orders
        else:
            if orders is None:
                self.log("No orders found or produced")
                orders = []

        if options["only-orders"] is not None:
            self.log(f"Running only orders: {options['only-orders']}")
            orders = [o for o in orders if o.number in options["only-orders"]]
        return orders

    def unload_lib(self, lib: DTLCLib):
        lib._close(delete=False)

    def load_lib(
        self, test: T_DTL, test_path: str, options: Options
    ) -> DTLCLib | None:
        # test_path = test.get_test_path(tests_path)
        return self.get_compiled_lib(
            test.layout,
            test.order,
            test.context,
            test_path,
            test.get_lib_name(),
            test.get_func_types_name(),
            options,
        )

    def get_compiled_lib(
        self,
        new_layout: PtrMapping,
        new_order: IterationMapping,
        context: DLTCompileContext,
        test_path: str,
        lib_name: str,
        func_types_name: str,
        options: Options,
    ) -> DTLCLib | None:
        self.start_inline_log(
            f"Getting lib for: l: {new_layout.number}, o: {new_order.number} :: "
        )

        module: ModuleOp = context.module
        lib_builder: LibBuilder = context.lib_builder
        layout_graph: LayoutGraph = context.layout_graph
        iteration_map: IterationMap = context.iteration_map

        # lib_name = f"lib_{new_layout.number}_{new_order.number}"
        llvm_path = f"{test_path}/{lib_name}.ll"
        lib_path = f"{test_path}/{lib_name}"
        func_types_path = f"{test_path}/{func_types_name}"
        graph_path = f"{test_path}/"

        func_types_exists = os.path.exists(func_types_path)
        self.inline_log(0, "func_types: ")
        if func_types_exists:
            module_clone = None
            with open(func_types_path, "rb") as f:
                function_types_tuple = pickle.load(f)
                function_types, dlt_func_types = function_types_tuple
            self.inline_log(0, "loaded, ")
        elif not options["do-not-lower"]:
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
            self.inline_log(0, "made, ")
        else:
            self.end_inline_log(0, "not found, but do not lower is set")
            return None

        lib_exists = os.path.exists(lib_path)
        self.inline_log(0, "lib: ")
        if lib_exists:
            function_types_str = {k.data: v for k, v in function_types.items()}
            lib = DTLCLib(lib_path, dlt_func_types, function_types_str)
            self.end_inline_log(0, "found.")
            return lib
        else:
            self.inline_log(0, "not found, ")

        llvm_exists = os.path.exists(llvm_path)
        self.inline_log(0, "llvm: ")
        if llvm_exists and options["only-to-llvm"]:
            self.end_inline_log(0, f"found & Done.")
            return None
        elif not llvm_exists and options["no-mlir"]:
            self.end_inline_log(0, "not found - but do not compile mlir is set.")
            return None
        else:
            if llvm_exists:
                self.inline_log(0, f"found, ")
                lib = lib_builder.compile_from(
                    llvm_path,
                    function_types,
                    dlt_func_map=dlt_func_types,
                    lib_path=lib_path,
                    clang_args=self.get_extra_clang_args(),
                    verbose=0,
                )
                self.end_inline_log(0, f"lib compiled to: {lib_path}")
                return lib
            else:
                self.inline_log(0, "not found, ")
                if module_clone is None:
                    self.inline_log(0, "func_types: ")
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
                    self.inline_log(0, "remade, ")

                self.inline_log(0, "lib: ")
                lib = lib_builder.compile(
                    module_clone,
                    function_types,
                    dlt_func_map=dlt_func_types,
                    llvm_out=llvm_path,
                    llvm_only=options["only-to-llvm"],
                    lib_path=lib_path,
                    clang_args=self.get_extra_clang_args(),
                    verbose=0,
                )
                if lib is not None:
                    self.end_inline_log(
                        0,
                        f"compiled to binary. LLVM: generated, lib: {lib._library_path}",
                    )
                    return lib
                else:
                    self.end_inline_log(
                        0, f"compiled to LLVM: {llvm_path} but no lib was produced."
                    )
                    return None
