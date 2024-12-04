import abc
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, Generic

import numpy as np

from benchmarking.benchmark import (
    Benchmark,
    BenchmarkSettings,
    Options,
    PythonCode,
    Test,
    TestCode,
)
from dtl import TensorSpace, TensorVariable
from dtl.libBuilder import DTLCLib, LibBuilder, NpArrayCtype, TupleStruct
from xdsl.dialects import arith, builtin, func, llvm, printf, scf
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntegerAttr, IntegerType, ModuleOp
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import DimensionAttr, SetAttr
from xdsl.ir import Block, Operation, SSAValue
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
from xdsl.transforms.experimental.dlt.layout_llvm_semantics import SemanticsMapper, load_all_semantics


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
            "##func_types_path##", f'"{test_path}/{self.get_func_types_name()}"'
        ).replace(
            "##lib_path##", f'"{test_path}/{self.get_lib_name()}"'
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
    def define_lib_builder(self) -> tuple[LibBuilder, tuple[TensorVariable, ...]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_configs_for_DTL_tensors(
        self,
        *tensor_variables: TensorVariable,
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        raise NotImplementedError

    @abc.abstractmethod
    def make_tests_for(
        self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping
    ) -> list[T_DTL]:
        raise NotImplementedError

    def get_dlt_semantics(self, options: Options) -> SemanticsMapper | None:

        semantics_flags = options["dlt-semantics"]
        semantics = SemanticsMapper(
            print_memory_calls = "print-memory-calls" in semantics_flags,
        )
        load_all_semantics(semantics)
        return semantics

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

    def parse_options(self, benchmark_options: list[str] = None) -> Options:
        options = super().parse_options(benchmark_options)
        options["only-to-llvm"] = "--only-to-llvm" in benchmark_options
        options["no-mlir"] = "--no-mlir" in benchmark_options
        options["do-not-lower"] = "--do-not-lower" in benchmark_options
        options["do-not-generate"] = "--do-not-generate" in benchmark_options
        options["llvm-debug"] = "--llvm-debug" in benchmark_options
        options["output-dlt"] = "--output-dlt" in benchmark_options
        options["output-xdsl"] = "--output-xdsl" in benchmark_options
        options["output-mlir"] = "--output-mlir" in benchmark_options

        dlt_semantics_flags = set()
        only_layouts = set()
        only_orders = set()
        for arg in benchmark_options:
            if arg.startswith("-l="):
                only_layouts.add(int(arg.removeprefix("-l=")))
            elif arg.startswith("-o="):
                only_orders.add(int(arg.removeprefix("-o=")))
            elif arg.startswith("--dlt-semantics="):
                dlt_semantics_flags.add(arg.removeprefix("--dlt-semantics="))
        if len(only_layouts) == 0:
            only_layouts = None
        if len(only_orders) == 0:
            only_orders = None
        options["only-layouts"] = only_layouts
        options["only-orders"] = only_orders
        options["dlt-semantics"] = dlt_semantics_flags

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

    def get_extra_clang_args(self, options: Options) -> list[str]:
        if options["llvm-debug"]:
            return []
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
                return ["-O3", "-ffast-math"]
            case 5:
                return ["-O3", "-ffast-math", "-march=native"]

    def initialise_benchmarks(self, options: Options) -> list[T_DTL]:
        lib_builder, tensor_variables = self.define_lib_builder()
        module, layout_graph, iteration_map = lib_builder.prepare(verbose=0)
        compile_context = DLTCompileContext(
            lib_builder, module, layout_graph, iteration_map, options
        )
        layouts = self.get_layouts(layout_graph, lib_builder, tensor_variables, options)
        layouts = [l for l in layouts if not self.skip_layout(l)]
        orders = self.get_orders(iteration_map, lib_builder, options)
        orders = [o for o in orders if not self.skip_order(o)]
        tests = self.enumerate_tests(compile_context, layouts, orders)
        return tests

    def get_layouts(
        self, layout_graph: LayoutGraph, lib_builder: LibBuilder, tensor_variables: tuple[TensorVariable, ...], options: Options
    ) -> list[PtrMapping]:
        dtl_config_map: dict[TupleStruct[TensorVariable], ReifyConfig] = (
            self.get_configs_for_DTL_tensors(*tensor_variables)
        )
        def match_func(a, b):
            if isinstance(a, tuple):
                a_l, a_c = a
                if not isinstance(a_l, LayoutGraph):
                    return False
                if not isinstance(a_c, dict):
                    return False
                if isinstance(b, tuple):
                    b_l, b_c = b
                    if not isinstance(b_l, LayoutGraph):
                        return False
                    if not isinstance(b_c, dict):
                        return False
                    return a_c == b_c and a_l.matches(b_l)
            return False

        found, key_int, layouts = self.search_store(
            self.layout_store,
            (layout_graph, dtl_config_map),
            match_function=match_func,
            lock=(not options["do-not-generate"]),
        )

        if not found and key_int >= 0:
            gen_path = layouts
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
            self.write_store(self.layout_store, (layout_graph, dtl_config_map), new_layouts, key_int)
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
        self, test: T_DTL, test_path: str, options: Options, load: bool = True
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
            load=load,
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
        load: bool = True,
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
                output_dlt_path=f"{lib_path}.dlt.ir" if options["output-dlt"] else None,
                semantics=self.get_dlt_semantics(options),
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
            if load:
                lib = DTLCLib(lib_path, dlt_func_types, function_types_str)
            else:
                lib = None
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
                    clang_args=self.get_extra_clang_args(options),
                    enable_debug=options["llvm-debug"],
                    load=load,
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
                        output_dlt_path=f"{lib_path}.dlt.ir" if options["output-dlt"] else None,
                        semantics=self.get_dlt_semantics(options),
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
                    clang_args=self.get_extra_clang_args(options),
                    load=load,
                    enable_debug=options["llvm-debug"],
                    output_xdsl=options["output-xdsl"],
                    output_mlir=options["output-mlir"],
                    verbose=0,
                )
                if options["only-to-llvm"]:
                    self.end_inline_log(
                        0, f"compiled to LLVM: {llvm_path} but no lib was produced."
                    )
                    return None
                else:
                    self.end_inline_log(
                        0,
                        f"compiled to binary. LLVM: generated, lib: {lib._library_path}",
                    )
                    return lib

def make_setup_func_coo(
    lib_builder: LibBuilder, name: str, t_var: TensorVariable, nnz: int
):
    tensor_space: TensorSpace = t_var.tensor_space
    func_arg_types = [lib_builder.tensor_var_details[t_var]]  # DLT tensor
    for d in tensor_space.shape:
        func_arg_types.append(llvm.LLVMPointerType.opaque())  # each coord array for coo
    func_arg_types.append(llvm.LLVMPointerType.opaque())  # the val array

    setup_block = Block(arg_types=func_arg_types)
    arg_tensor = setup_block.args[0]
    args_np_ptr_coord = list(setup_block.args[1:-1])
    arg_np_ptr_val = setup_block.args[-1]

    arg_tensor_dims = lib_builder.tensor_var_dims[t_var]
    assert len(arg_tensor_dims) == len(tensor_space.shape)

    zero_op = arith.Constant(IntegerAttr(0, IndexType()))
    zero_i64_op = arith.Constant(IntegerAttr(0, builtin.i64))
    one_op = arith.Constant(IntegerAttr(1, IndexType()))
    nnz_ub_i64_op = arith.Constant(IntegerAttr(nnz, builtin.i64))
    setup_block.add_ops([zero_op, zero_i64_op, one_op, nnz_ub_i64_op])

    one_i64_op = arith.Constant(IntegerAttr(1, builtin.i64))
    setup_block.add_op(one_i64_op)

    dlt_s_coo_indexing_struct = llvm.LLVMStructType.from_type_list(
        ([llvm.LLVMPointerType.opaque()]*len(args_np_ptr_coord))
        + [llvm.LLVMPointerType.opaque(), builtin.i64, builtin.i64]
    )
    dlt_indexing_buffer_op = llvm.AllocaOp(one_i64_op.result, dlt_s_coo_indexing_struct)
    setup_block.add_op(dlt_indexing_buffer_op)

    idx_range_start_ptr_op = llvm.GEPOp(dlt_indexing_buffer_op.res, [0, len(args_np_ptr_coord) + 1], [], pointee_type=dlt_s_coo_indexing_struct)
    idx_range_end_ptr_op = llvm.GEPOp(dlt_indexing_buffer_op.res, [0, len(args_np_ptr_coord) + 2], [], pointee_type=dlt_s_coo_indexing_struct)
    idx_range_start_store_op = llvm.StoreOp(zero_i64_op.result, idx_range_start_ptr_op)
    idx_range_end_store_op = llvm.StoreOp(nnz_ub_i64_op.result, idx_range_end_ptr_op)
    setup_block.add_ops([idx_range_start_ptr_op, idx_range_end_ptr_op, idx_range_start_store_op, idx_range_end_store_op])

    for i, ptr in enumerate(list(args_np_ptr_coord)+[arg_np_ptr_val]):
        current_indexing_buffer_ptr_op = llvm.GEPOp(dlt_indexing_buffer_op.res, [0,i], [], pointee_type=dlt_s_coo_indexing_struct)
        current_buffer_store_op = llvm.StoreOp(ptr, current_indexing_buffer_ptr_op.result)
        setup_block.add_ops([current_indexing_buffer_ptr_op, current_buffer_store_op])

    dlt_ref_ptr_llvm_struct = llvm.LLVMStructType.from_type_list([llvm.LLVMPointerType.opaque()])
    dlt_ref_ptr_undef_op = llvm.UndefOp(dlt_ref_ptr_llvm_struct)
    dlt_ref_ptr_op = llvm.InsertValueOp(builtin.DenseArrayBase.from_list(builtin.i64, [0]), dlt_ref_ptr_undef_op.res, dlt_indexing_buffer_op.res)
    setup_block.add_ops([dlt_ref_ptr_undef_op, dlt_ref_ptr_op])

    dlt_coo_dims = [dlt.DimensionAttr(f"ref_{name}_{i}", d.extent) for i, d in enumerate(arg_tensor_dims)]
    dlt_ref_ptr_layout = dlt.IndexingLayoutAttr(dlt.PrimitiveLayoutAttr(dlt.IndexRangeType()), dlt.SeparatedCOOLayoutAttr(dlt.PrimitiveLayoutAttr(builtin.f32), dlt_coo_dims, buffer_scaler=0, index_buffer_types=[builtin.i32 for _ in dlt_coo_dims]))
    dlt_ptr_type = dlt.PtrType(dlt_ref_ptr_layout.contents_type, dlt_ref_ptr_layout, identity=f"ref_{name}_ptr")
    dlt_ptr_cast_op = builtin.UnrealizedConversionCastOp.get([dlt_ref_ptr_op.res], [dlt_ptr_type])
    setup_block.add_op(dlt_ptr_cast_op)
    dlt_ref_ptr = dlt_ptr_cast_op.outputs[0]

    dlt_inner_ptr_type = dlt.PtrType(dlt_ref_ptr_layout.contents_type.select_dimensions(dlt_coo_dims), dlt_ref_ptr_layout, SetAttr([]), ArrayAttr(dlt_coo_dims), identity=f"ref_{name}_ptr_inner")
    iterate_block = Block(arg_types=[IndexType() for _ in dlt_coo_dims] + [dlt_inner_ptr_type] + [IndexType()])
    extents = iterate_block.args[:len(dlt_coo_dims)]
    dlt_ref_inner_ptr = iterate_block.args[len(dlt_coo_dims)]
    inc_idx = iterate_block.args[len(dlt_coo_dims)+1]


    dlt_ref_get_op = dlt.GetSOp(dlt_ref_inner_ptr, builtin.f32)
    iterate_block.add_op(dlt_ref_get_op)
    ref_value = dlt_ref_get_op.res
    # print_op = printf.PrintFormatOp(f"# {name}: " + "{} Ref " + ', '.join(["{}" for _ in extents]) + " to {} ({})", inc_idx, *extents, ref_value, dlt_ref_get_op.found)
    # iterate_block.add_op(print_op)

    dlt_sel_ptr_op = dlt.SelectOp(arg_tensor, [], arg_tensor_dims, extents)
    iterate_block.add_op(dlt_sel_ptr_op)
    dlt_set_ptr_op = dlt.SetOp(dlt_sel_ptr_op.res, builtin.f32, ref_value)
    iterate_block.add_op(dlt_set_ptr_op)

    inc_add_op = arith.Addi(inc_idx, one_op.result)
    iterate_block.add_op(inc_add_op)
    iterate_block.add_op(dlt.IterateYieldOp(inc_add_op.result))

    order = dlt.NonZeroIterationOrderAttr(list(range(len(dlt_coo_dims))), 0, dlt.BodyIterationOrderAttr())
    # order = dlt.BodyIterationOrderAttr()
    # for i in reversed(list(range(len(dlt_coo_dims)))):
    #     order = dlt.NestedIterationOrderAttr(i, order)
    iterate_op = dlt.IterateOp(
        [d.extent for d in dlt_coo_dims],
        [],
        [[[d] for d in dlt_coo_dims]],
        [dlt_ref_ptr],
        [zero_op.result],
        order,
        f"{name}_ref_loop",
        iterate_block,
    )
    setup_block.add_op(iterate_op)

    setup_block.add_ops([func.Return()])

    np_ptr_type_val = NpArrayCtype((nnz,))
    np_ptr_type_dim_list = []
    for d in tensor_space.shape:
        np_ptr_type_dim_list.append(NpArrayCtype((nnz,), np.int32))
    lib_builder.make_custom_function(
        name, setup_block, [t_var, *np_ptr_type_dim_list, np_ptr_type_val]
    )


def make_setup_func_dense(
    lib_builder: LibBuilder, name: str, t_var: TensorVariable, dims: list[int]
):
    setup_block = Block(
        arg_types=[
            lib_builder.tensor_var_details[t_var],
            llvm.LLVMPointerType.opaque(),
        ]
    )
    arg_tensor, arg_np_ptr_val = setup_block.args
    arg_tensor_dims = lib_builder.tensor_var_dims[t_var]
    assert len(arg_tensor_dims) == len(dims), f"{len(arg_tensor_dims)} != {dims}"

    def make_loop(
        dims_to_loop: list[int],
        tensor_dims: list[DimensionAttr],
        indices: list[SSAValue],
        dlt_ptr: SSAValue,
    ) -> list[Operation]:
        assert len(dims_to_loop) == len(tensor_dims)
        if len(dims_to_loop) == 0:
            ops = []
            i64_indices = []
            for index in indices:
                index_cast_op = arith.IndexCastOp(
                    index, builtin.i64
                )
                i64_indices.append(index_cast_op.result)
                ops.append(index_cast_op)
            llvm_ptr_type = builtin.f32
            for dim in reversed(dims):
                llvm_ptr_type = llvm.LLVMArrayType.from_size_and_type(
                    dim, llvm_ptr_type
                )
            val_ptr_arith_op = llvm.GEPOp(
                arg_np_ptr_val,
                [0, *([llvm.GEP_USE_SSA_VAL] * len(i64_indices))],
                i64_indices,
                pointee_type=llvm_ptr_type,
            )
            val_load_op = llvm.LoadOp(val_ptr_arith_op.result, builtin.f32)
            ops.extend([val_ptr_arith_op, val_load_op])
            set_op = dlt.SetOp(dlt_ptr, builtin.f32, val_load_op.dereferenced_value)
            ops.append(set_op)
            return ops
        else:
            dim = dims_to_loop.pop(0)
            dlt_dim = tensor_dims.pop(0)
            ops = []
            zero_op = arith.Constant(IntegerAttr(0, IndexType()))
            one_op = arith.Constant(IntegerAttr(1, IndexType()))
            ub_op = arith.Constant(IntegerAttr(dim, IndexType()))
            ops.extend([zero_op, one_op, ub_op])
            block = Block(arg_types=[IndexType()])
            select_op = dlt.SelectOp(dlt_ptr, [], [dlt_dim], [block.args[0]])
            block.add_op(select_op)
            block.add_ops(
                make_loop(
                    dims_to_loop,
                    tensor_dims,
                    indices + [block.args[0]],
                    select_op.res,
                )
            )
            block.add_op(scf.Yield())
            for_op = scf.For(zero_op.result, ub_op.result, one_op.result, [], block)
            ops.append(for_op)
            return ops

    loop_ops = make_loop(list(dims), list(arg_tensor_dims), [], arg_tensor)
    setup_block.add_ops(loop_ops)

    setup_block.add_op(func.Return())

    np_ptr_type = NpArrayCtype(tuple(dims))
    lib_builder.make_custom_function(name, setup_block, [t_var, np_ptr_type])


def make_check_func_coo(
    lib_builder: LibBuilder,
    name: str,
    t_var: TensorVariable,
    nnz: int,
    epsilon: float,
):
    tensor_space: TensorSpace = t_var.tensor_space
    func_arg_types = [lib_builder.tensor_var_details[t_var]]  # DLT tensor
    for d in tensor_space.shape:
        func_arg_types.append(llvm.LLVMPointerType.opaque())  # each coord array for coo
    func_arg_types.append(llvm.LLVMPointerType.opaque())  # the val array
    func_arg_types.append(lib_builder.tensor_var_details[t_var])  # the first DLT tensor

    check_block = Block(arg_types=func_arg_types)
    arg_tensor = check_block.args[0]
    args_np_ptr_coord = list(check_block.args[1:-2])
    arg_np_ptr_val = check_block.args[-2]
    arg_tensor_first = check_block.args[-1]

    arg_tensor_dims = lib_builder.tensor_var_dims[t_var]
    assert len(arg_tensor_dims) == len(tensor_space.shape)

    zero_op = arith.Constant(IntegerAttr(0, IndexType()))
    one_op = arith.Constant(IntegerAttr(1, IndexType()))
    false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
    true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
    f_zero_op = arith.Constant(builtin.FloatAttr(0.0, builtin.f32))
    f_epsilon_op = arith.Constant(builtin.FloatAttr(epsilon, builtin.f32))
    check_block.add_ops([zero_op, one_op, false_op, true_op, f_zero_op, f_epsilon_op])

    nnz_ub_op = arith.Constant(IntegerAttr(nnz, IndexType()))
    check_block.add_ops([nnz_ub_op])

    lb = zero_op.result
    step = one_op.result
    ub = nnz_ub_op.result

    loop_block = Block(
        arg_types=[IndexType(), IntegerType(1), builtin.f32, IntegerType(1)]
    )
    nnz_index, correct_arg, total_error_arg, consistent_arg = loop_block.args
    nnz_index_cast_op = arith.IndexCastOp(nnz_index, builtin.i64)
    loop_block.add_op(nnz_index_cast_op)
    nnz_idx_i64 = nnz_index_cast_op.result

    coord_indices = []
    coord_i64_indices = []
    for arg_np_ptr_coord in args_np_ptr_coord:
        ptr_op = llvm.GEPOp(
            arg_np_ptr_coord,
            [0, llvm.GEP_USE_SSA_VAL],
            [nnz_idx_i64],
            pointee_type=llvm.LLVMArrayType.from_size_and_type(nnz, builtin.i32),
        )
        load_op = llvm.LoadOp(ptr_op.result, builtin.i32)
        coord_i64_op = arith.ExtUIOp(load_op.dereferenced_value, builtin.i64)
        coord_index_op = arith.IndexCastOp(
            coord_i64_op.result, IndexType()
        )
        loop_block.add_ops([ptr_op, load_op, coord_i64_op, coord_index_op])
        coord_i64_indices.append(coord_i64_op.result)
        coord_indices.append(coord_index_op.result)

    val_ptr_arith_op = llvm.GEPOp(
        arg_np_ptr_val,
        [0, llvm.GEP_USE_SSA_VAL],
        [nnz_idx_i64],
        pointee_type=llvm.LLVMArrayType.from_size_and_type(nnz, builtin.f32),
    )
    val_load_op = llvm.LoadOp(val_ptr_arith_op.result, builtin.f32)
    ref_value = val_load_op.dereferenced_value
    loop_block.add_ops([val_ptr_arith_op, val_load_op])

    select_op = dlt.SelectOp(arg_tensor, [], arg_tensor_dims, coord_indices)
    get_op = dlt.GetOp(select_op.res, builtin.f32)
    value = get_op.res
    loop_block.add_ops([select_op, get_op])

    select_first_op = dlt.SelectOp(arg_tensor_first, [], arg_tensor_dims, coord_indices)
    get_first_op = dlt.GetOp(select_first_op.res, builtin.f32)
    value_first = get_first_op.res
    loop_block.add_ops([select_first_op, get_first_op])

    error_op = arith.Subf(value, ref_value)
    neg_error_op = arith.Negf(error_op.result)
    abs_error_op = arith.Maximumf(error_op.result, neg_error_op.result)
    loop_block.add_ops([error_op, neg_error_op, abs_error_op])

    norm_signed_epsilon_op = arith.Mulf(ref_value, f_epsilon_op.result)
    norm_neg_signed_epsilon_op = arith.Negf(norm_signed_epsilon_op.result)
    norm_epsilon_op = arith.Maximumf(
        norm_signed_epsilon_op.result, norm_neg_signed_epsilon_op.result
    )
    loop_block.add_ops(
        [
            norm_signed_epsilon_op,
            norm_neg_signed_epsilon_op,
            norm_epsilon_op,
        ]
    )

    new_total_error_op = arith.Addf(total_error_arg, abs_error_op.result)
    new_total_error_arg = new_total_error_op.result
    loop_block.add_op(new_total_error_op)

    cmp_error_op = arith.Cmpf(abs_error_op.result, norm_epsilon_op.result, "ogt")
    new_correct_op = arith.Select(cmp_error_op, false_op.result, correct_arg)
    new_correct_arg = new_correct_op.result
    loop_block.add_ops([cmp_error_op, new_correct_op])

    if_error_op = scf.If(
        cmp_error_op,
        [],
        [
            printf.PrintFormatOp(
                f"# Result miss match at {', '.join([d.dimensionName.data + ': {}' for d in arg_tensor_dims])} : reference = {{}}, result = {{}}, error = {{}} > {{}}",
                *coord_i64_indices,
                ref_value,
                value,
                abs_error_op.result,
                norm_epsilon_op.result,
            ),
            scf.Yield(),
        ],
    )
    loop_block.add_op(if_error_op)

    cmp_first_op = arith.Cmpf(value, value_first, "one")
    new_consistent_op = arith.Select(cmp_first_op, false_op.result, consistent_arg)
    new_consistent_arg = new_consistent_op.result
    loop_block.add_ops([cmp_first_op, new_consistent_op])

    if_bit_wise_op = scf.If(
        cmp_first_op,
        [],
        [
            printf.PrintFormatOp(
                f"# Result does not match previous result at {', '.join([d.dimensionName.data + ': {}' for d in arg_tensor_dims])}: first result = {{}}, this result = {{}}",
                *coord_i64_indices,
                value_first,
                value,
            ),
            scf.Yield(),
        ],
    )
    loop_block.add_op(if_bit_wise_op)
    loop_block.add_op(
        scf.Yield(new_correct_arg, new_total_error_arg, new_consistent_arg)
    )

    loop_op = scf.For(
        lb, ub, step, [true_op.result, f_zero_op.result, true_op.result], loop_block
    )
    correct_res, total_error_res, consistent_res = tuple(loop_op.results)
    check_block.add_op(loop_op)

    output_correct_op = arith.ExtUIOp(correct_res, builtin.i64)
    output_consistent_op = arith.ExtUIOp(consistent_res, builtin.i64)
    check_block.add_ops([output_correct_op, output_consistent_op])
    check_block.add_op(
        func.Return(
            output_correct_op.result,
            total_error_res,
            output_consistent_op.result,
        )
    )

    np_ptr_type_val = NpArrayCtype((nnz,))
    np_ptr_type_dim_list = []
    for d in tensor_space.shape:
        np_ptr_type_dim_list.append(NpArrayCtype((nnz,), np.int32))
    lib_builder.make_custom_function(
        name, check_block, [t_var, *np_ptr_type_dim_list, np_ptr_type_val, t_var]
    )


def make_check_func_dense(
    lib_builder: LibBuilder,
    name: str,
    t_var: TensorVariable,
    dims: list[int],
    epsilon: float,
):
    check_block = Block(
        arg_types=[
            lib_builder.tensor_var_details[t_var],
            llvm.LLVMPointerType.opaque(),
            lib_builder.tensor_var_details[t_var],
        ]
    )
    arg_tensor, arg_np_ptr_val, arg_tensor_first = check_block.args
    arg_tensor_dims = lib_builder.tensor_var_dims[t_var]
    assert len(arg_tensor_dims) == len(dims)

    def make_loop(
        dims_to_loop: list[int],
        tensor_dims: list[DimensionAttr],
        indices: list[SSAValue],
        dlt_ptr: SSAValue,
        dlt_ptr_first: SSAValue,
        correct_arg: SSAValue,
        total_error_arg: SSAValue,
        consistent_arg: SSAValue,
    ) -> tuple[list[Operation], tuple[SSAValue, SSAValue, SSAValue]]:
        assert len(dims_to_loop) == len(tensor_dims)
        if len(dims_to_loop) == 0:
            ops = []
            false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
            f_epsilon_op = arith.Constant(builtin.FloatAttr(epsilon, builtin.f32))
            ops.extend([false_op, f_epsilon_op])
            i64_indices = []
            for index in indices:
                index_cast_op = arith.IndexCastOp(index, builtin.i64)
                i64_indices.append(index_cast_op.result)
                ops.append(index_cast_op)
            llvm_ptr_type = builtin.f32
            for dim in reversed(dims):
                llvm_ptr_type = llvm.LLVMArrayType.from_size_and_type(
                    dim, llvm_ptr_type
                )
            val_ptr_arith_op = llvm.GEPOp(
                arg_np_ptr_val,
                [0, *([llvm.GEP_USE_SSA_VAL] * len(i64_indices))],
                i64_indices,
                pointee_type=llvm_ptr_type,
            )
            val_load_op = llvm.LoadOp(val_ptr_arith_op.result, builtin.f32)
            ops.extend([val_ptr_arith_op, val_load_op])
            get_op = dlt.GetOp(dlt_ptr, builtin.f32)
            ops.append(get_op)
            get_first_op = dlt.GetOp(dlt_ptr_first, builtin.f32)
            ops.append(get_first_op)

            ref_value = val_load_op.dereferenced_value
            value = get_op.res
            value_first = get_first_op.res

            error_op = arith.Subf(value, ref_value)
            neg_error_op = arith.Negf(error_op.result)
            abs_error_op = arith.Maximumf(error_op.result, neg_error_op.result)
            ops.extend([error_op, neg_error_op, abs_error_op])

            norm_signed_epsilon_op = arith.Mulf(ref_value, f_epsilon_op.result)
            norm_neg_signed_epsilon_op = arith.Negf(norm_signed_epsilon_op.result)
            norm_epsilon_op = arith.Maximumf(
                norm_signed_epsilon_op.result, norm_neg_signed_epsilon_op.result
            )
            ops.extend(
                [
                    norm_signed_epsilon_op,
                    norm_neg_signed_epsilon_op,
                    norm_epsilon_op,
                ]
            )

            new_total_error_op = arith.Addf(total_error_arg, abs_error_op.result)
            new_total_error_arg = new_total_error_op.result
            ops.append(new_total_error_op)

            cmp_error_op = arith.Cmpf(
                abs_error_op.result, norm_epsilon_op.result, "ogt"
            )
            new_correct_op = arith.Select(cmp_error_op, false_op.result, correct_arg)
            new_correct_arg = new_correct_op.result
            ops.extend([cmp_error_op, new_correct_op])

            if_error_op = scf.If(
                cmp_error_op,
                [],
                [
                    printf.PrintFormatOp(
                        f"# Result miss match at {', '.join([d.dimensionName.data +': {}' for d in arg_tensor_dims])} : reference = {{}}, result = {{}}, error = {{}} > {{}}",
                        *i64_indices,
                        ref_value,
                        value,
                        abs_error_op.result,
                        norm_epsilon_op.result,
                    ),
                    scf.Yield(),
                ],
            )
            ops.append(if_error_op)

            cmp_first_op = arith.Cmpf(value, value_first, "one")
            new_consistent_op = arith.Select(
                cmp_first_op, false_op.result, consistent_arg
            )
            new_consistent_arg = new_consistent_op.result
            ops.extend([cmp_first_op, new_consistent_op])

            if_bit_wise_op = scf.If(
                cmp_first_op,
                [],
                [
                    printf.PrintFormatOp(
                        f"# Result does not match previous result at {', '.join([d.dimensionName.data +': {}' for d in arg_tensor_dims])}: first result = {{}}, this result = {{}}",
                        *i64_indices,
                        value_first,
                        value,
                    ),
                    scf.Yield(),
                ],
            )
            ops.append(if_bit_wise_op)

            return ops, (new_correct_arg, new_total_error_arg, new_consistent_arg)
        else:
            dim = dims_to_loop.pop(0)
            dlt_dim = tensor_dims.pop(0)
            ops = []
            zero_op = arith.Constant(IntegerAttr(0, IndexType()))
            one_op = arith.Constant(IntegerAttr(1, IndexType()))
            ub_op = arith.Constant(IntegerAttr(dim, IndexType()))
            ops.extend([zero_op, one_op, ub_op])
            block = Block(
                arg_types=[IndexType(), IntegerType(1), builtin.f32, IntegerType(1)]
            )
            (
                index,
                inner_correct_arg,
                inner_total_error_arg,
                inner_consistent_arg,
            ) = block.args
            select_op = dlt.SelectOp(dlt_ptr, [], [dlt_dim], [index])
            select_first_op = dlt.SelectOp(dlt_ptr_first, [], [dlt_dim], [index])
            block.add_ops([select_op, select_first_op])
            inner_ops, (
                new_inner_correct_arg,
                new_inner_total_error_arg,
                new_inner_consistent_arg,
            ) = make_loop(
                dims_to_loop,
                tensor_dims,
                indices + [index],
                select_op.res,
                select_first_op.res,
                inner_correct_arg,
                inner_total_error_arg,
                inner_consistent_arg,
            )
            block.add_ops(inner_ops)
            block.add_op(
                scf.Yield(
                    new_inner_correct_arg,
                    new_inner_total_error_arg,
                    new_inner_consistent_arg,
                )
            )
            for_op = scf.For(
                zero_op.result,
                ub_op.result,
                one_op.result,
                [correct_arg, total_error_arg, consistent_arg],
                block,
            )
            correct_arg_res, total_error_arg_res, consistent_arg_res = for_op.results
            ops.append(for_op)
            return ops, (correct_arg_res, total_error_arg_res, consistent_arg_res)

    true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
    f_zero_op = arith.Constant(builtin.FloatAttr(0.0, builtin.f32))
    check_block.add_ops([true_op, f_zero_op])
    loop_ops, (correct_res, total_error_res, consistent_res) = make_loop(
        list(dims),
        list(arg_tensor_dims),
        [],
        arg_tensor,
        arg_tensor_first,
        true_op.result,
        f_zero_op.result,
        true_op.result,
    )
    check_block.add_ops(loop_ops)
    output_correct_op = arith.ExtUIOp(correct_res, builtin.i64)
    output_consistent_op = arith.ExtUIOp(consistent_res, builtin.i64)
    check_block.add_ops([output_correct_op, output_consistent_op])
    check_block.add_op(
        func.Return(
            output_correct_op.result,
            total_error_res,
            output_consistent_op.result,
        )
    )

    np_ptr_type = NpArrayCtype(tuple(dims))
    lib_builder.make_custom_function(name, check_block, [t_var, np_ptr_type, t_var])
