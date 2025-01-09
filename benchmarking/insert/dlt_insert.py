import sys
from typing import Generic

import numpy as np

from benchmarking.dlt_base_test import BasicDTLTest
from benchmarking.benchmark import BenchmarkSettings
from benchmarking.insert import insert_code
from dtl import *
from dtl.dag import RealVectorSpace

from dtl.libBuilder import LibBuilder, NpArrayCtype, StructType, TupleStruct
from benchmarking.dtlBenchmark import DLTCompileContext, DTLBenchmark, T_DTL, make_check_func_coo
from xdsl.dialects import arith, builtin, func, llvm, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr
from xdsl.dialects.experimental import dlt
from xdsl.ir import Block
from xdsl.transforms.experimental.dlt.generate_dlt_iteration_orders import (
    IterationMapping,
)
from xdsl.transforms.experimental.dlt.generate_dlt_layouts import (
    PtrMapping,
    ReifyConfig,
)

_Epsilon = 0.00001

_Args = tuple[Any, StructType, StructType, StructType]


class TensorInsertDLT(DTLBenchmark[T_DTL], abc.ABC, Generic[T_DTL]):
    def __init__(
        self,
        shape: tuple[int,...],
        insertions: int,
        use_scope_vars: bool,
        seed: int,
        base_dir: str,
        opt_num: int,
        epsilon: float,
        settings: BenchmarkSettings,
    ):
        var_name = "scope" if use_scope_vars else "static"
        new_base_dir = f"{base_dir}/insert"
        shape_str = ".".join([str(i) for i in shape])
        results_base_dir = f"{new_base_dir}/{self.get_self_name()}_{var_name}_O{opt_num}_{shape_str}_{insertions}_{seed}"

        self.shape = tuple(shape)
        self.insertions = insertions
        self.seed = seed
        self.use_scope_vars = use_scope_vars
        self.epsilon = epsilon
        super().__init__(
            results_base_dir,
            f"{new_base_dir}/layouts",
            f"{new_base_dir}/orders",
            settings,
            opt_num,
            skip_layout_func=self.skip_layout_func,
            skip_order_func=self.skip_order_func,
        )

        np_coords, np_val = self.make_values(seed)
        assert len(np_coords) == len(self.shape)
        assert len({c.shape for c in np_coords} | {np_val.shape}) == 1
        assert len(np_val.shape) == 1
        assert np_val.dtype == np.float32

        self.np_coords = np_coords
        self.np_val = np_val
        self.np_nnz = np_val.shape[-1]
        assert self.np_nnz == self.insertions

        np_check_coords, np_check_val = self.make_check_values(self.np_coords, self.np_val)
        assert len(np_check_coords) == len(self.shape)
        assert len({c.shape for c in np_check_coords} | {np_check_val.shape}) == 1
        assert len(np_check_val.shape) == 1
        assert np_check_val.dtype == np.float32
        self.np_check_coords = np_check_coords
        self.np_check_val = np_check_val
        self.np_check_nnz = np_check_val.shape[-1]
        assert self.np_check_nnz <= self.np_nnz


        data_variant_name = self.test_data_variant_name()
        for i, np_coord in enumerate(self.np_coords):
            self.handle_reference_array(
                np_coord, f"arrays/{data_variant_name}/np_coord_{str(i)}", True, False, f"np_coord_{i}",dtype=np.int32
            )
        self.handle_reference_array(
            np_val, f"arrays/{data_variant_name}/np_val", True, False, "np_val"
        )

        for i, np_check_coord in enumerate(self.np_check_coords):
            self.handle_reference_array(
                np_check_coord, f"arrays/{data_variant_name}/np_check_coord_{str(i)}", False, True, f"np_check_coord_{i}",dtype=np.int32
            )
        self.handle_reference_array(
            np_check_val, f"arrays/{data_variant_name}/np_check_val", False, True, "np_check_val"
        )

    @abc.abstractmethod
    def get_self_name(self) -> str:
        raise NotImplementedError

    def skip_layout_func(self, layout: PtrMapping) -> bool:
        return False

    def skip_order_func(self, order: IterationMapping) -> bool:
        return False

    def skip_layout_order_func(self, layout_mapping: PtrMapping, order_mapping: IterationMapping, context: DLTCompileContext) -> bool:
        return False

    def make_tests_for(
        self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping
    ) -> list[T_DTL]:
        if self.skip_layout_order_func(layout, order, context):
            return []
        return self.make_test_for(context, layout, order)

    @abc.abstractmethod
    def make_test_for(
        self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping
    ) -> list[T_DTL]:
        raise NotImplementedError

    def test_data_variant_name(self) -> str:
        return f"{str(self.seed)}"

    @abc.abstractmethod
    def make_values(self, seed: int) -> tuple[tuple[np.ndarray,...], np.ndarray]:
        raise NotImplementedError

    def make_check_values(self, np_coords: tuple[np.ndarray,...], np_val: np.ndarray) -> tuple[tuple[np.ndarray,...], np.ndarray]:
        ts = {}
        assert len({c.shape for c in np_coords} | {np_val.shape}) == 1
        for i in range(np_val.shape[-1]):
            ts[tuple([c[i] for c in np_coords])] = np_val[i]

        new_coords = tuple([[] for _ in np_coords])
        new_values = []
        for coord in sorted(ts):
            for i in range(len(np_coords)):
                new_coords[i].append(coord[i])
            new_values.append(ts[coord])

        new_np_coords = tuple([np.array(c, dtype=np.float32) for c in new_coords])
        new_np_val = np.array(new_values, dtype=np.float32)
        return new_np_coords, new_np_val

    @abc.abstractmethod
    def get_configs_for_tensors(
        self, a: TensorVariable
    ) -> ReifyConfig:
        raise NotImplementedError

    def get_configs_for_DTL_tensors(
        self,
        *tensor_variables: TensorVariable,
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        assert len(tensor_variables) == 1
        return {tensor_variables[0] : self.get_configs_for_tensors(tensor_variables[0])}

    def define_lib_builder(self) -> tuple[LibBuilder, tuple[TensorVariable, ...]]:

        if self.use_scope_vars:
            VSs = []
            scope_var_map = {}
            for i, e in enumerate(self.shape):
                vs = UnknownSizeVectorSpace(f"v{str(i)}")
                VSs.append(vs)
                scope_var_map[vs] = e
        else:
            VSs = []
            scope_var_map = {}
            for i, e in enumerate(self.shape):
                vs = RealVectorSpace(e)
                VSs.append(vs)

        A = TensorVariable(TensorSpace(VSs), "A")

        lib_builder = LibBuilder(scope_var_map)
        lib_builder.make_init("init", A, [], free_name="dealloc")
        # lib_builder.make_setter("set_A", (A), {}, [0, 1])
        self._make_prepare_func(lib_builder, A)

        self.make_insert_func(lib_builder, "insert", A, self.np_nnz)
        make_check_func_coo(lib_builder, "check", A, self.np_check_nnz, self.epsilon)
        return lib_builder, (A,)

    def make_insert_func(self,
            lib_builder: LibBuilder, name: str, t_var: TensorVariable, nnz: int
    ):
        tensor_space: TensorSpace = t_var.tensor_space
        func_arg_types = [lib_builder.tensor_var_details[t_var]]  # DLT tensor
        for d in tensor_space.shape:
            func_arg_types.append(llvm.LLVMPointerType.opaque())  # each coord array for coo
        func_arg_types.append(llvm.LLVMPointerType.opaque())  # the val array

        insert_block = Block(arg_types=func_arg_types)
        arg_tensor = insert_block.args[0]
        args_np_ptr_coord = list(insert_block.args[1:-1])
        arg_np_ptr_val = insert_block.args[-1]

        arg_tensor_dims = lib_builder.tensor_var_dims[t_var]
        assert len(arg_tensor_dims) == len(tensor_space.shape)
        assert len(arg_tensor_dims) == len(self.shape)

        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        zero_i64_op = arith.Constant(IntegerAttr(0, builtin.i64))
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        one_i64_op = arith.Constant(IntegerAttr(1, builtin.i64))
        nnz_op = arith.Constant(IntegerAttr(nnz, IndexType()))
        nnz_i64_op = arith.Constant(IntegerAttr(nnz, builtin.i64))
        insert_block.add_ops([zero_op, zero_i64_op, one_op, one_i64_op, nnz_op, nnz_i64_op])

        loop_block = Block(arg_types=[IndexType()])
        idx = loop_block.args[0]
        idx_i64_op = arith.IndexCastOp(idx, builtin.i64)
        loop_block.add_op(idx_i64_op)
        idx_i64 = idx_i64_op.result
        coords = []
        for coord in args_np_ptr_coord:
            get_coord_ptr_op = llvm.GEPOp.from_mixed_indices(coord, [idx_i64], llvm.LLVMPointerType.opaque(), pointee_type=builtin.i32)
            load_coord_op = llvm.LoadOp(get_coord_ptr_op.result, builtin.i32)
            load_coord_conv_op = arith.IndexCastOp(load_coord_op.dereferenced_value, IndexType())
            loop_block.add_ops([get_coord_ptr_op, load_coord_op, load_coord_conv_op])
            coords.append(load_coord_conv_op.result)
        get_val_ptr_op = llvm.GEPOp.from_mixed_indices(arg_np_ptr_val, [idx_i64], llvm.LLVMPointerType.opaque(), pointee_type=builtin.f32)
        load_val_op = llvm.LoadOp(get_val_ptr_op.result, builtin.f32)
        loop_block.add_ops([get_val_ptr_op, load_val_op])
        val = load_val_op.dereferenced_value

        select_op = dlt.SelectOp(arg_tensor, [], arg_tensor_dims, coords)
        set_op = dlt.SetOp(select_op.res, builtin.f32, val)
        loop_block.add_ops([select_op, set_op])

        loop_block.add_op(scf.Yield())

        loop_op = scf.For(zero_op, nnz_op, one_op, [], loop_block)

        insert_block.add_op(loop_op)

        insert_block.add_ops([func.Return()])

        np_ptr_type_val = NpArrayCtype((nnz,))
        np_ptr_type_dim_list = []
        for d in tensor_space.shape:
            np_ptr_type_dim_list.append(NpArrayCtype((nnz,), np.int32))
        lib_builder.make_custom_function(
            name, insert_block, [t_var, *np_ptr_type_dim_list, np_ptr_type_val]
        )

    @staticmethod
    def _make_prepare_func(lib_builder: LibBuilder, A: TensorVariable):
        block = Block(arg_types=[
            lib_builder.tensor_var_details[A]
        ], ops=[func.Return()])
        lib_builder.make_custom_function("prepare", block, [A])

class SimpleInsert(TensorInsertDLT[BasicDTLTest]):

    def __init__(self, shape: tuple[int, ...], insertions: int, use_scope_vars: bool, seed: int, base_dir: str,
                 opt_num: int, epsilon: float, settings: BenchmarkSettings, reify_config: ReifyConfig, ordered: bool, allow_duplicates: bool, name:str):
        self.reify_config = reify_config
        self.ordered = ordered
        self.allow_duplicates = allow_duplicates
        self.name = name
        super().__init__(shape, insertions, use_scope_vars, seed, base_dir, opt_num, epsilon, settings)

    def get_self_name(self) -> str:
        return f"{'ordered.' if self.ordered else 'unordered.'}{'random.' if self.allow_duplicates else 'unique.'}{self.name}"

    def make_test_for(
        self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping
    ) -> list[BasicDTLTest]:
        return [BasicDTLTest(insert_code.get_code(self.shape), context, layout, order)]

    def get_configs_for_tensors(
        self, a: TensorVariable
    ) -> ReifyConfig:
        return self.reify_config

    def make_values(self, seed: int) -> tuple[tuple[np.ndarray,...], np.ndarray]:
        return insert_code.make_insertion_arrays(self.seed, self.shape, self.insertions, ordered=self.ordered, allow_duplicates=self.allow_duplicates)

if __name__ == "__main__":

    repeats = 3
    runs = 10
    benchmarks = []

    print(f"Args: {sys.argv}")
    benchmark_names = [a for a in sys.argv[1:] if not a.startswith("-")]
    run_all = len(benchmark_names) == 0

    settings = BenchmarkSettings(
        runs=100,
        repeats=3,
        waste_of_time_threshold=0.01,
        test_too_short_threshold=0.0005,
        long_run_multiplier=10,
        setup_timeout=2.0,
        benchmark_timeout=1.0,
        testing_timeout=2.0,
        tear_down_timeout=2.0,
        benchmark_trial_child_process=True,
        benchmark_in_child_process=True,
    )


    if run_all or "ou1" in benchmark_names:
        benchmarks.append(
            SimpleInsert(
                (128,128,128),
                128*128,
                False,
                0,
                "./results",
                3,
                _Epsilon,
                settings,
                ReifyConfig(
                    dense=True,
                    unpacked_coo_buffer_options=frozenset([0]),
                    separated_coo_buffer_options=frozenset([]),
                    separated_coo_buffer_index_options=frozenset([]),
                    coo_minimum_dims=1,
                    arith_replace=False,
                    force_arith_replace_immediate_use=False,
                    permute_structure_size_threshold=-1,
                    members_first=True,
                    all_sorted_members=True,
                ),
                True,
                False,
                "UpCOO_Dense"
            )
        )

    if run_all or "CmpCOO" in benchmark_names:
        benchmarks.append(
            SimpleInsert(
                (128,128),
                128*128,
                False,
                0,
                "./results",
                3,
                _Epsilon,
                settings,
                ReifyConfig(
                    dense=True,
                    unpacked_coo_buffer_options=frozenset([0, 2, 8, -8]),
                    separated_coo_buffer_options=frozenset([0, 2, 8, -8]),
                    separated_coo_buffer_index_options=frozenset([builtin.i32, builtin.i64]),
                    coo_minimum_dims=1,
                    arith_replace=False,
                    force_arith_replace_immediate_use=False,
                    permute_structure_size_threshold=-1,
                    members_first=True,
                    all_sorted_members=True,
                ),
                False,
                True,
                "CmpCOO"
            )
        )

    benchmark_options = [a for a in sys.argv if a.startswith("-")]
    for benchmark in benchmarks:
        benchmark.run(benchmark_options)

    if "run" in sys.argv:
        for benchmark in benchmarks:
            benchmark.run()
