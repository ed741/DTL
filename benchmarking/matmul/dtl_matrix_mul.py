import sys
from random import Random
from typing import Generic

import numpy as np

from benchmarking.matmul.matrix_mul_code import make_dense_np_arrays, matmul_pair_code, matmul_single_code, matmul_triple_code
from benchmarking.benchmark import BenchmarkSettings, ID_Tuple, TestCode
from dtl import *
from dtl.dag import RealVectorSpace, Index

from dtl.libBuilder import LibBuilder, NpArrayCtype, StructType, TupleStruct
from benchmarking.dtlBenchmark import DLTCompileContext, DLTTest, DTLBenchmark, T_DTL
from xdsl.dialects import arith, builtin, func, llvm, printf, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
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



class MatMulDenseDTLTest(DLTTest):

    @classmethod
    def get_id_headings(cls) -> list[tuple[str, type[str] | type[int] | type[bool]]]:
        return [("layout", int), ("order", int)]

    @classmethod
    def get_result_headings(
        cls,
    ) -> list[tuple[str, type[int] | type[bool] | type[float]]]:
        return [("correct", bool), ("total_error", float), ("consistent", bool)]

    def get_id(self) -> ID_Tuple:
        return self.layout.number, self.order.number

class MatMulDenseDTL(DTLBenchmark[T_DTL], abc.ABC, Generic[T_DTL]):
    def __init__(
        self,
        i: int,
        j: int,
        k: int,
        use_scope_vars: bool,
        seed: int,
        base_dir: str,
        opt_num: int,
        epsilon: float,
        settings: BenchmarkSettings,
    ):
        var_name = "scope" if use_scope_vars else "static"
        new_base_dir = f"{base_dir}/matmul"
        results_base_dir = f"{new_base_dir}/{self.get_self_name()}_{var_name}_O{opt_num}_{i}.{j}.{k}_{seed}"

        self.i, self.j, self.k = i, j, k
        self.seed = seed
        self.use_scope_vars = use_scope_vars
        self.epsilon = epsilon
        super().__init__(
            results_base_dir,
            f"{new_base_dir}/layouts",
            f"{new_base_dir}/orders",
            settings,
            opt_num,
        )

        r = Random(seed)
        np_a, np_b, np_c = self.make_abc(r)

        self.np_a = np_a
        self.np_b = np_b
        self.np_c = np_c

        self.tensor_variables = None

        data_variant_name = self.test_data_variant_name()
        self.handle_reference_array(
            np_a, f"arrays/{data_variant_name}/np_a", True, False, "np_a"
        )
        self.handle_reference_array(
            np_b, f"arrays/{data_variant_name}/np_b", True, False, "np_b"
        )
        self.handle_reference_array(
            np_c, f"arrays/{data_variant_name}/np_c", False, True, "np_c"
        )

    @abc.abstractmethod
    def get_self_name(self) -> str:
        raise NotImplementedError

    def test_data_variant_name(self) -> str:
        return f"{str(self.seed)}"

    def make_abc(self, r: Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return make_dense_np_arrays(r, self.i, self.j, self.k)

    @abc.abstractmethod
    def construct_lib_builder(
        self,
        lib_builder: LibBuilder,
        a: TensorVariable,
        b: TensorVariable,
        c: TensorVariable,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def get_configs_for_tensors(
        self, a: TensorVariable, b: TensorVariable, c: TensorVariable
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        raise NotImplementedError

    def get_configs_for_DTL_tensors(
        self,
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        assert self.tensor_variables is not None
        return self.get_configs_for_tensors(*self.tensor_variables)

    def define_lib_builder(self) -> LibBuilder:
        if self.use_scope_vars:
            vi = UnknownSizeVectorSpace("vi")
            vj = UnknownSizeVectorSpace("vj")
            vk = UnknownSizeVectorSpace("vk")
            scope_var_map = {vi: self.i, vj: self.j, vk: self.k}
        else:
            vi = RealVectorSpace(self.i)
            vj = RealVectorSpace(self.j)
            vk = RealVectorSpace(self.k)
            scope_var_map = {}
        A = TensorVariable(vi * vj, "A")
        B = TensorVariable(vj * vk, "B")
        C = TensorVariable(vi * vk, "C")

        self.tensor_variables = (A, B, C)

        _i = Index("i")
        _j = Index("j")
        _k = Index("k")
        matmul = (A[_i, _j] * B[_j, _k]).sum(_j).forall(_i, _k)

        lib_builder = LibBuilder(scope_var_map)
        self.construct_lib_builder(lib_builder, A, B, C)

        lib_builder.make_setter("set_A", (A), {}, [0, 1])
        lib_builder.make_setter("set_B", (B), {}, [0, 1])

        lib_builder.make_getter("get_C", (C), {}, [0, 1])
        lib_builder.make_function("matmul", matmul, [C], [A, B], [], [])

        self._make_setup_func(lib_builder, "setup_A", A, self.i, self.j)
        self._make_setup_func(lib_builder, "setup_B", B, self.j, self.k)
        self._make_check_func(lib_builder, "check_C", C, self.i, self.k, self.epsilon)
        return lib_builder

    @staticmethod
    def _make_setup_func(
        lib_builder: LibBuilder,
        name: str,
        t_var: TensorVariable,
        dim_1: int,
        dim_2: int,
    ):
        setup_a_block = Block(
            arg_types=[
                lib_builder.tensor_var_details[t_var],
                llvm.LLVMPointerType.opaque(),
            ]
        )
        arg_a, arg_np_ptr = setup_a_block.args
        arg_a_dims = lib_builder.tensor_var_dims[t_var]

        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        setup_a_block.add_ops([zero_op, one_op])

        i_ub_op = arith.Constant(IntegerAttr(dim_1, IndexType()))
        j_ub_op = arith.Constant(IntegerAttr(dim_2, IndexType()))
        setup_a_block.add_ops([i_ub_op, j_ub_op])

        lb = zero_op.result
        step = one_op.result
        ub_i = i_ub_op.result
        ub_j = j_ub_op.result

        i_loop_block = Block(arg_types=[IndexType()])
        j_loop_block = Block(arg_types=[IndexType()])
        select_op = dlt.SelectOp(
            arg_a, [], arg_a_dims, [i_loop_block.args[0], j_loop_block.args[0]]
        )
        i_index_cast_op = builtin.UnrealizedConversionCastOp.get(
            [i_loop_block.args[0]], [builtin.i64]
        )
        j_index_cast_op = builtin.UnrealizedConversionCastOp.get(
            [j_loop_block.args[0]], [builtin.i64]
        )
        ptr_arith_op = llvm.GEPOp(
            arg_np_ptr,
            [0, llvm.GEP_USE_SSA_VAL, llvm.GEP_USE_SSA_VAL],
            [i_index_cast_op.outputs[0], j_index_cast_op.outputs[0]],
            pointee_type=llvm.LLVMArrayType.from_size_and_type(
                dim_1, llvm.LLVMArrayType.from_size_and_type(dim_2, builtin.f32)
            ),
        )
        load_op = llvm.LoadOp(ptr_arith_op, builtin.f32)
        set_op = dlt.SetOp(select_op.res, builtin.f32, load_op.dereferenced_value)
        j_loop_block.add_ops(
            [
                select_op,
                i_index_cast_op,
                j_index_cast_op,
                ptr_arith_op,
                load_op,
                set_op,
                scf.Yield(),
            ]
        )
        j_loop_op = scf.For(lb, ub_j, step, [], j_loop_block)
        i_loop_block.add_ops([j_loop_op, scf.Yield()])
        i_loop_op = scf.For(lb, ub_i, step, [], i_loop_block)
        setup_a_block.add_ops([i_loop_op, func.Return()])

        np_ptr_type = NpArrayCtype((dim_1, dim_2))
        lib_builder.make_custom_function(name, setup_a_block, [t_var, np_ptr_type])

    @staticmethod
    def _make_check_func(
        lib_builder: LibBuilder,
        name: str,
        t_var: TensorVariable,
        dim_1: int,
        dim_2: int,
        epsilon: float,
    ):
        setup_a_block = Block(
            arg_types=[
                lib_builder.tensor_var_details[t_var],
                llvm.LLVMPointerType.opaque(),
                lib_builder.tensor_var_details[t_var],
            ]
        )
        arg_a, arg_np_ptr, first_arg_a = setup_a_block.args
        arg_a_dims = lib_builder.tensor_var_dims[t_var]

        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        f_zero_op = arith.Constant(builtin.FloatAttr(0.0, builtin.f32))
        f_epsilon_op = arith.Constant(builtin.FloatAttr(epsilon, builtin.f32))
        setup_a_block.add_ops(
            [zero_op, one_op, false_op, true_op, f_zero_op, f_epsilon_op]
        )

        i_ub_op = arith.Constant(IntegerAttr(dim_1, IndexType()))
        j_ub_op = arith.Constant(IntegerAttr(dim_2, IndexType()))
        setup_a_block.add_ops([i_ub_op, j_ub_op])

        epsilon_correct = true_op.result
        total_error = f_zero_op.result
        bit_wise_consistent = true_op.result

        lb = zero_op.result
        step = one_op.result
        ub_i = i_ub_op.result
        ub_j = j_ub_op.result

        i_loop_block = Block(
            arg_types=[IndexType(), IntegerType(1), builtin.f32, IntegerType(1)]
        )
        j_loop_block = Block(
            arg_types=[IndexType(), IntegerType(1), builtin.f32, IntegerType(1)]
        )
        select_op = dlt.SelectOp(
            arg_a, [], arg_a_dims, [i_loop_block.args[0], j_loop_block.args[0]]
        )
        get_op = dlt.GetOp(select_op.res, builtin.f32)
        j_loop_block.add_ops([select_op, get_op])

        i_index_cast_op = builtin.UnrealizedConversionCastOp.get(
            [i_loop_block.args[0]], [builtin.i64]
        )
        j_index_cast_op = builtin.UnrealizedConversionCastOp.get(
            [j_loop_block.args[0]], [builtin.i64]
        )
        ptr_arith_op = llvm.GEPOp(
            arg_np_ptr,
            [0, llvm.GEP_USE_SSA_VAL, llvm.GEP_USE_SSA_VAL],
            [i_index_cast_op.outputs[0], j_index_cast_op.outputs[0]],
            pointee_type=llvm.LLVMArrayType.from_size_and_type(
                dim_1, llvm.LLVMArrayType.from_size_and_type(dim_2, builtin.f32)
            ),
        )
        np_load_op = llvm.LoadOp(ptr_arith_op, builtin.f32)
        j_loop_block.add_ops(
            [i_index_cast_op, j_index_cast_op, ptr_arith_op, np_load_op]
        )

        error_op = arith.Subf(get_op.res, np_load_op.dereferenced_value)
        neg_error_op = arith.Negf(error_op.result)
        abs_error_op = arith.Maximumf(error_op.result, neg_error_op.result)
        norm_epsilon_op = arith.Mulf(np_load_op.dereferenced_value, f_epsilon_op.result)
        new_total_error_op = arith.Addf(j_loop_block.args[2], abs_error_op.result)
        cmp_error_op = arith.Cmpf(abs_error_op.result, norm_epsilon_op.result, "ogt")
        new_epsilon_correct_op = arith.Select(
            cmp_error_op, false_op.result, j_loop_block.args[1]
        )
        if_error_op = scf.If(
            cmp_error_op,
            [],
            [
                printf.PrintFormatOp(
                    "Result miss match at {}, {}: reference = {}, result = {}, error = {} >= {}",
                    i_index_cast_op.outputs[0],
                    j_index_cast_op.outputs[0],
                    np_load_op.dereferenced_value,
                    get_op.res,
                    abs_error_op.result,
                    norm_epsilon_op.result,
                ),
                scf.Yield(),
            ],
        )
        j_loop_block.add_ops(
            [
                error_op,
                neg_error_op,
                abs_error_op,
                norm_epsilon_op,
                new_total_error_op,
                cmp_error_op,
                new_epsilon_correct_op,
                if_error_op,
            ]
        )

        select_first_op = dlt.SelectOp(
            first_arg_a, [], arg_a_dims, [i_loop_block.args[0], j_loop_block.args[0]]
        )
        get_first_op = dlt.GetOp(select_first_op.res, builtin.f32)
        cmp_first_op = arith.Cmpf(get_op.res, get_first_op.res, "one")
        new_bit_wise_consistent_op = arith.Select(
            cmp_first_op, false_op.result, j_loop_block.args[3]
        )
        if_bit_wise_op = scf.If(
            cmp_first_op,
            [],
            [
                printf.PrintFormatOp(
                    "Result does not match previous result at {}, {}: first result = {}, this result = {}",
                    i_index_cast_op.outputs[0],
                    j_index_cast_op.outputs[0],
                    get_first_op.res,
                    get_op.res,
                ),
                scf.Yield(),
            ],
        )
        j_loop_block.add_ops(
            [
                select_first_op,
                get_first_op,
                cmp_first_op,
                new_bit_wise_consistent_op,
                if_bit_wise_op,
            ]
        )

        j_loop_block.add_op(
            scf.Yield(
                new_epsilon_correct_op.result,
                new_total_error_op.result,
                new_bit_wise_consistent_op.result,
            )
        )

        j_loop_op = scf.For(lb, ub_j, step, i_loop_block.args[1:], j_loop_block)
        i_loop_block.add_ops([j_loop_op, scf.Yield(*j_loop_op.res)])
        i_loop_op = scf.For(
            lb,
            ub_i,
            step,
            [epsilon_correct, total_error, bit_wise_consistent],
            i_loop_block,
        )
        output_epsilon_check_op = arith.ExtUIOp(i_loop_op.res[0], builtin.i64)
        output_total_error = i_loop_op.res[1]
        output_bit_consistent_op = arith.ExtUIOp(i_loop_op.res[2], builtin.i64)
        setup_a_block.add_ops(
            [
                i_loop_op,
                output_epsilon_check_op,
                output_bit_consistent_op,
                func.Return(
                    output_epsilon_check_op,
                    output_total_error,
                    output_bit_consistent_op,
                ),
            ]
        )

        np_ptr_type = NpArrayCtype((dim_1, dim_2))
        lib_builder.make_custom_function(
            name, setup_a_block, [t_var, np_ptr_type, t_var]
        )


class Triple(MatMulDenseDTL[MatMulDenseDTLTest]):
    def get_self_name(self) -> str:
        return "triple"

    def make_tests_for(
        self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping
    ) -> list[MatMulDenseDTLTest]:
        return [MatMulDenseDTLTest(matmul_triple_code, context, layout, order)]

    def construct_lib_builder(
        self,
        lib_builder: LibBuilder,
        a: TensorVariable,
        b: TensorVariable,
        c: TensorVariable,
    ):
        lib_builder.make_init("init", (a, b, c), [], free_name="dealloc")

    def get_configs_for_tensors(
        self, a: TensorVariable, b: TensorVariable, c: TensorVariable
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        return {(a, b, c): ReifyConfig()}


class Pair(MatMulDenseDTL[MatMulDenseDTLTest]):

    def get_self_name(self) -> str:
        return "pair"

    def make_tests_for(
        self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping
    ) -> list[MatMulDenseDTLTest]:
        return [MatMulDenseDTLTest(matmul_pair_code, context, layout, order)]

    def construct_lib_builder(
        self,
        lib_builder: LibBuilder,
        a: TensorVariable,
        b: TensorVariable,
        c: TensorVariable,
    ):
        lib_builder.make_init("init_AB", (a, b), [], free_name="dealloc_AB")
        lib_builder.make_init("init_C", (c), [], free_name="dealloc_C")

    def get_configs_for_tensors(
        self, a: TensorVariable, b: TensorVariable, c: TensorVariable
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        return {
            (a, b): ReifyConfig(coo_buffer_options=frozenset([0])),
            c: ReifyConfig(),
        }


class Single(MatMulDenseDTL[MatMulDenseDTLTest]):

    def get_self_name(self) -> str:
        return "single"

    def make_tests_for(
        self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping
    ) -> list[MatMulDenseDTLTest]:
        return [MatMulDenseDTLTest(matmul_single_code, context, layout, order)]

    def construct_lib_builder(
        self,
        lib_builder: LibBuilder,
        a: TensorVariable,
        b: TensorVariable,
        c: TensorVariable,
    ):
        lib_builder.make_init("init_A", (a), [], free_name="dealloc_A")
        lib_builder.make_init("init_B", (b), [], free_name="dealloc_B")
        lib_builder.make_init("init_C", (c), [], free_name="dealloc_C")

    def get_configs_for_tensors(
        self, a: TensorVariable, b: TensorVariable, c: TensorVariable
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        return {
            a: ReifyConfig(coo_buffer_options=frozenset([0])),
            b: ReifyConfig(coo_buffer_options=frozenset([0])),
            c: ReifyConfig(),
        }


class MatMulSparseDTLTest(DLTTest):

    def __init__(
        self,
        code: TestCode,
        context: DLTCompileContext,
        layout: PtrMapping,
        order: IterationMapping,
        rate_a: float,
        rate_b: float,
    ):
        super().__init__(code, context, layout, order)
        self.rate_a = rate_a
        self.rate_b = rate_b

    @classmethod
    def get_id_headings(cls) -> list[tuple[str, type[str] | type[int] | type[bool]]]:
        return [("layout", int), ("order", int), ("rate_a", str), ("rate_b", str)]

    @classmethod
    def get_result_headings(
        cls,
    ) -> list[tuple[str, type[int] | type[bool] | type[float]]]:
        return [("correct", bool), ("total_error", float), ("consistent", bool)]

    def get_id(self) -> ID_Tuple:
        return self.layout.number, self.order.number, str(self.rate_a), str(self.rate_b)


class RandomSparseSingle(MatMulDenseDTL[MatMulSparseDTLTest]):
    def __init__(self, *args, rate_a: float = 0.0, rate_b: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_a = rate_a
        self.rate_b = rate_b

    def get_self_name(self) -> str:
        return "random_sparse"

    def make_abc(self, r: Random) -> tuple[np.ndarray, np.ndarray]:
        np_a = np.zeros((self.i, self.j), dtype=np.float32)
        np_b = np.zeros((self.j, self.k), dtype=np.float32)
        for i_i in range(self.i):
            for i_j in range(self.j):
                if r.random() < self.rate_a:
                    num = r.random()
                    np_a[i_i, i_j] = num
        for i_j in range(self.j):
            for i_k in range(self.k):
                if r.random() < self.rate_b:
                    num = r.random()
                    np_b[i_j, i_k] = num
        return np_a, np_b

    def test_data_variant_name(self) -> str:
        return super().test_data_variant_name() + f"{self.rate_a}_{self.rate_b}"

    def make_tests_for(
        self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping
    ) -> list[MatMulSparseDTLTest]:
        return [
            MatMulSparseDTLTest(
                matmul_single_code, context, layout, order, self.rate_a, self.rate_b
            )
        ]

    def construct_lib_builder(
        self,
        lib_builder: LibBuilder,
        a: TensorVariable,
        b: TensorVariable,
        c: TensorVariable,
    ):
        lib_builder.make_init("init_A", (a), [], free_name="dealloc_A")
        lib_builder.make_init("init_B", (b), [], free_name="dealloc_B")
        lib_builder.make_init("init_C", (c), [], free_name="dealloc_C")

    def get_configs_for_tensors(
        self, a: TensorVariable, b: TensorVariable, c: TensorVariable
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        return {
            a: ReifyConfig(coo_buffer_options=frozenset([0])),
            b: ReifyConfig(coo_buffer_options=frozenset([0])),
            c: ReifyConfig(),
        }


class RowSparseSingles(RandomSparseSingle):
    def get_self_name(self) -> str:
        return super().get_self_name() + "_row"

    def sparsify(self, np_a, np_b, r: Random) -> tuple[np.ndarray, np.ndarray]:
        for i_i in range(np_a.shape[0]):
            if r.random() < self.rate_a:
                for i_j in range(np_a.shape[1]):
                    np_a[i_i, i_j] = 0
        for i_j in range(np_b.shape[0]):
            if r.random() < self.rate_b:
                for i_k in range(np_b.shape[1]):
                    np_b[i_j, i_k] = 0
        return np_a, np_b


if __name__ == "__main__":

    repeats = 3
    runs = 10
    benchmarks = []

    print(f"Args: {sys.argv}")

    settings_128 = BenchmarkSettings(
        runs=10,
        repeats=3,
        waste_of_time_threshold=0.1,
        test_too_short_threshold=0.001,
        long_run_multiplier=100,
        benchmark_timeout=3.0,
        benchmark_child_process=True,
    )
    settings_8 = BenchmarkSettings(
        runs=10,
        repeats=3,
        waste_of_time_threshold=0.01,
        test_too_short_threshold=0.001,
        long_run_multiplier=100,
        benchmark_timeout=3.0,
        benchmark_child_process=False,
    )

    # if len(sys.argv) == 1 or "1" in sys.argv:
    #     benchmarks.append(StaticTriple(128,128,128,True, 0, "./results", repeats=repeats, runs=runs, opt_level=3, epsilon=_Epsilon))
    if len(sys.argv) == 1 or "2" in sys.argv:

        benchmarks.append(
            Pair(
                128,
                128,
                128,
                True,
                0,
                "./results",
                3,
                _Epsilon,
                settings_128
            )
        )
    if len(sys.argv) == 1 or "3" in sys.argv:
        benchmarks.append(
            Single(
                128,
                128,
                128,
                True,
                0,
                "./results",
                3,
                _Epsilon,
                settings_128
            )
        )
    # if len(sys.argv) == 1 or "4" in sys.argv:
    #     benchmarks.append(StaticTriple(8, 8, 8, True, 0, "./results", repeats=repeats, runs=runs, opt_level=3, epsilon=_Epsilon))
    if len(sys.argv) == 1 or "5" in sys.argv:
        benchmarks.append(
            Pair(
                8,
                8,
                8,
                True,
                0,
                "./results",
                3,
                _Epsilon,
                settings_8
            )
        )
    if len(sys.argv) == 1 or "6" in sys.argv:
        benchmarks.append(
            Single(
                8,
                8,
                8,
                True,
                0,
                "./results",
                3,
                _Epsilon,
                settings_8
            )
        )

    # if len(sys.argv) == 1 or "7" in sys.argv:
    #     benchmarks.append(StaticTriple(128, 128, 128, False, 0, "./results", repeats=repeats, runs=runs, opt_level=3, epsilon=_Epsilon))
    if len(sys.argv) == 1 or "8" in sys.argv:
        benchmarks.append(
            Pair(
                128,
                128,
                128,
                False,
                0,
                "./results",
                3,
                _Epsilon,
                settings_128
            )
        )
    if len(sys.argv) == 1 or "9" in sys.argv:
        benchmarks.append(
            Single(
                128,
                128,
                128,
                False,
                0,
                "./results",
                3,
                _Epsilon,
                settings_128
            )
        )
    # if len(sys.argv) == 1 or "10" in sys.argv:
    #     benchmarks.append(StaticTriple(8, 8, 8, False, 0, "./results", repeats=repeats, runs=runs, opt_level=3, epsilon=_Epsilon))
    if len(sys.argv) == 1 or "11" in sys.argv:
        benchmarks.append(
            Pair(
                8,
                8,
                8,
                False,
                0,
                "./results",
                3,
                _Epsilon,
                settings_8
            )
        )
    if len(sys.argv) == 1 or "12" in sys.argv:
        benchmarks.append(
            Single(
                8,
                8,
                8,
                False,
                0,
                "./results",
                3,
                _Epsilon,
                settings_8
            )
        )

    settings_sparse = BenchmarkSettings(
        runs=10,
        repeats=3,
        waste_of_time_threshold=0.1,
        test_too_short_threshold=0.001,
        long_run_multiplier=100,
        benchmark_timeout=3.0,
        benchmark_child_process=True,
    )

    for rate in ["0.1", "0.01", "0.001", "0.0001", "0.00001"]:
        if len(sys.argv) == 1 or f"13-{rate}" in sys.argv:
            benchmarks.append(
                RandomSparseSingle(
                    1024,
                    1024,
                    1024,
                    False,
                    0,
                    "./results",
                    3,
                    _Epsilon,
                    rate_a = float(rate),
                    rate_b = float(rate),
                    settings=settings_sparse
                )
            )

    # benchmarks.append(
    #     RandomSparseSingles(1024, 1024, 1024, False, 0, 1, 1,"./results", "", repeats=repeats, runs=runs, opt_level=3, epsilon=_Epsilon))

    benchmark_options = [a for a in sys.argv if a.startswith("-")]
    for benchmark in benchmarks:
        benchmark.run(benchmark_options)

    if "run" in sys.argv:
        for benchmark in benchmarks:
            benchmark.run()
