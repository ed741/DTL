import sys
from typing import Generic

import numpy as np

from benchmarking.dlt_base_test import BasicDTLTest
from benchmarking.matmul.matrix_mul_code import make_dense_np_arrays, make_random_sparse_np_arrays, matmul_pair_code, \
    matmul_single_code, matmul_triple_code
from benchmarking.benchmark import BenchmarkSettings, ID_Tuple, TestCode
from dtl import *
from dtl.dag import RealVectorSpace, Index

from dtl.libBuilder import LibBuilder, StructType, TupleStruct
from benchmarking.dtlBenchmark import DLTCompileContext, DTLBenchmark, T_DTL, make_check_func_dense, \
    make_setup_func_dense
from xdsl.dialects import builtin, func
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import Layout
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
            skip_layout_func=self.skip_layout_func,
            skip_order_func=self.skip_order_func,
        )

        np_a, np_b, np_c = self.make_abc(seed)

        self.np_a = np_a
        self.np_b = np_b
        self.np_c = np_c

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

    def skip_layout_func(self, layout: PtrMapping) -> bool:
        return False

    def skip_order_func(self, order: IterationMapping) -> bool:
        return False

    def skip_layout_order_func(self, layout_mapping: PtrMapping, order_mapping: IterationMapping, context: DLTCompileContext) -> bool:
        layouts = layout_mapping.make_ptr_dict()
        def f(l: Layout) -> list[Layout]:
            return [l] + [c for cl in l.get_children() for c in f(cl)]
        for iter_name, order in order_mapping.make_iter_dict().items():
            iter_op = context.iteration_map.iteration_ops[iter_name]
            for i in range(len(iter_op.tensors)):
                sparse_dims = order.non_zero_loop_for(i, iter_op)
                if len(sparse_dims) > 0:
                    tensor_type = typing.cast(dlt.PtrType, iter_op.tensors[i].type)
                    assert isinstance(tensor_type, dlt.PtrType)
                    ptr_type = layouts[tensor_type.identification]
                    layout = ptr_type.layout
                    nodes = f(layout)
                    sparse_nodes = [n for n in nodes if isinstance(n, dlt.SeparatedCOOLayoutAttr | dlt.UnpackedCOOLayoutAttr)]
                    sparse_stored = {d for n in sparse_nodes for d in n.dimensions}
                    if len(sparse_dims & sparse_stored) == 0:
                        return True
                    # if not any(isinstance(l, dlt.IndexingLayoutAttr) for l in nodes):
                    #     return True
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

    def make_abc(self, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return make_dense_np_arrays(seed, self.i, self.j, self.k)

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
        *tensor_variables: TensorVariable,
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        return self.get_configs_for_tensors(*tensor_variables)

    def define_lib_builder(self) -> tuple[LibBuilder, tuple[TensorVariable, ...]]:
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

        _i = Index("i")
        _j = Index("j")
        _k = Index("k")
        matmul = (A[_i, _j] * B[_j, _k]).sum(_j).forall(_i, _k)

        lib_builder = LibBuilder(scope_var_map)
        self.construct_lib_builder(lib_builder, A, B, C)

        # lib_builder.make_setter("set_A", (A), {}, [0, 1])
        # lib_builder.make_setter("set_B", (B), {}, [0, 1])
        #
        # lib_builder.make_getter("get_C", (C), {}, [0, 1])
        self._make_prepare_func(lib_builder, A, B, C)
        lib_builder.make_function("matmul", matmul, [C], [A, B], [], [])

        make_setup_func_dense(lib_builder, "setup_A", A, [self.i, self.j])
        make_setup_func_dense(lib_builder, "setup_B", B, [self.j, self.k])
        make_check_func_dense(lib_builder, "check_C", C, [self.i, self.k], self.epsilon)
        return lib_builder, (A, B, C)

    @staticmethod
    def _make_prepare_func(lib_builder: LibBuilder, A: TensorVariable, B: TensorVariable, C: TensorVariable):
        block = Block(arg_types=[
            lib_builder.tensor_var_details[A],
            lib_builder.tensor_var_details[B],
            lib_builder.tensor_var_details[C],
        ], ops=[func.Return()])
        lib_builder.make_custom_function("prepare", block, [A,B,C])

class Triple(MatMulDenseDTL[BasicDTLTest]):
    def get_self_name(self) -> str:
        return "triple"

    def make_test_for(
        self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping
    ) -> list[BasicDTLTest]:
        return [BasicDTLTest(matmul_triple_code, context, layout, order)]

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
        return {(a, b, c): ReifyConfig(
            dense = True,
            unpacked_coo_buffer_options = frozenset([]),
            separated_coo_buffer_options = frozenset([]),
            separated_coo_buffer_index_options = frozenset([]),
            coo_minimum_dims = 2,
            arith_replace = True,
            force_arith_replace_immediate_use = True,
            permute_structure_size_threshold = -1,
            members_first = True,
        )}


class Pair(MatMulDenseDTL[BasicDTLTest]):

    def get_self_name(self) -> str:
        return "pair"

    def make_test_for(
        self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping
    ) -> list[BasicDTLTest]:
        return [BasicDTLTest(matmul_pair_code, context, layout, order)]

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
            (a, b): ReifyConfig(
                dense=True,
                unpacked_coo_buffer_options=frozenset([]),
                separated_coo_buffer_options=frozenset([]),
                separated_coo_buffer_index_options=frozenset([]),
                coo_minimum_dims=2,
                arith_replace=True,
                force_arith_replace_immediate_use=True,
                permute_structure_size_threshold=-1,
                members_first=True,
            ),
            c: ReifyConfig(
                dense=True,
                unpacked_coo_buffer_options=frozenset([2]),
                separated_coo_buffer_options=frozenset([2]),
                separated_coo_buffer_index_options=frozenset([builtin.i32]),
                coo_minimum_dims=1,
                arith_replace=True,
                force_arith_replace_immediate_use=True,
                permute_structure_size_threshold=-1,
                members_first=True,
            ),
        }


class Single(MatMulDenseDTL[BasicDTLTest]):

    def get_self_name(self) -> str:
        return "single"

    def make_test_for(
        self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping
    ) -> list[BasicDTLTest]:
        return [BasicDTLTest(matmul_single_code, context, layout, order)]

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
            a: ReifyConfig(
                dense=True,
                unpacked_coo_buffer_options=frozenset([]),
                separated_coo_buffer_options=frozenset([]),
                separated_coo_buffer_index_options=frozenset([]),
                coo_minimum_dims=2,
                arith_replace=True,
                force_arith_replace_immediate_use=True,
                permute_structure_size_threshold=-1,
                members_first=True,
            ),
            b: ReifyConfig(
                dense=True,
                unpacked_coo_buffer_options=frozenset([]),
                separated_coo_buffer_options=frozenset([]),
                separated_coo_buffer_index_options=frozenset([]),
                coo_minimum_dims=2,
                arith_replace=True,
                force_arith_replace_immediate_use=True,
                permute_structure_size_threshold=-1,
                members_first=True,
            ),
            c: ReifyConfig(
                dense=True,
                unpacked_coo_buffer_options=frozenset([0, 2]),
                separated_coo_buffer_options=frozenset([0, 2]),
                separated_coo_buffer_index_options=frozenset([builtin.i32]),
                coo_minimum_dims=1,
                arith_replace=True,
                force_arith_replace_immediate_use=True,
                permute_structure_size_threshold=-1,
                members_first=True,
            ),
        }


class MatMulSparseDTLTest(BasicDTLTest):

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
        return super().get_id_headings() + [("rate_a", str), ("rate_b", str)]

    def get_id(self) -> ID_Tuple:
        return *super().get_id(), str(self.rate_a), str(self.rate_b)


class RandomSparseSingle(MatMulDenseDTL[MatMulSparseDTLTest]):
    def __init__(self, *args, rate_a: float = 0.0, rate_b: float = 0.0, **kwargs):
        self.rate_a = rate_a
        self.rate_b = rate_b
        super().__init__(*args, **kwargs)

    def get_self_name(self) -> str:
        return "random_sparse"

    def make_abc(self, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return make_random_sparse_np_arrays(seed, self.i, self.j, self.k, self.rate_a, self.rate_b)

    def test_data_variant_name(self) -> str:
        return super().test_data_variant_name() + f"_{self.rate_a}_{self.rate_b}"

    def make_test_for(
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
            a: ReifyConfig(
                dense=True,
                unpacked_coo_buffer_options=frozenset([]),
                separated_coo_buffer_options=frozenset([2]),
                separated_coo_buffer_index_options=frozenset([builtin.i32]),
                coo_minimum_dims=1,
                arith_replace=True,
                force_arith_replace_immediate_use=True,
                permute_structure_size_threshold=-1,
                members_first=True,
            ),
            b: ReifyConfig(
                dense=True,
                unpacked_coo_buffer_options=frozenset([]),
                separated_coo_buffer_options=frozenset([2]),
                separated_coo_buffer_index_options=frozenset([builtin.i32]),
                coo_minimum_dims=1,
                arith_replace=True,
                force_arith_replace_immediate_use=True,
                permute_structure_size_threshold=-1,
                members_first=True,
            ),
            c: ReifyConfig(
                dense=True,
                unpacked_coo_buffer_options=frozenset([]),
                separated_coo_buffer_options=frozenset([2]),
                separated_coo_buffer_index_options=frozenset([builtin.i32]),
                coo_minimum_dims=1,
                arith_replace=True,
                force_arith_replace_immediate_use=True,
                permute_structure_size_threshold=-1,
                members_first=True,
            ),
        }


class RowSparseSingle(RandomSparseSingle):
    def get_self_name(self) -> str:
        return super().get_self_name() + "_row"

    def make_abc(self, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return make_random_sparse_np_arrays(seed, self.i, self.j, self.k, 1.0, 1.0, sparse_a=(self.rate_a, 1.0), sparse_b=(self.rate_b, 1.0))



if __name__ == "__main__":

    repeats = 3
    runs = 10
    benchmarks = []

    print(f"Args: {sys.argv}")
    benchmark_names = [a for a in sys.argv[1:] if not a.startswith("-")]
    run_all = len(benchmark_names) == 0

    settings_128 = BenchmarkSettings(
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
    settings_8 = BenchmarkSettings(
        runs=100,
        repeats=3,
        waste_of_time_threshold=0.01,
        test_too_short_threshold=0.0005,
        long_run_multiplier=10,
        setup_timeout=1.0,
        benchmark_timeout=1.0,
        testing_timeout=1.0,
        tear_down_timeout=1.0,
        benchmark_trial_child_process=True,
        benchmark_in_child_process=True,
    )

    if run_all or "triple128" in benchmark_names:
        benchmarks.append(
            Triple(
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

    if run_all or "pair128" in benchmark_names:
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

    if run_all or "single128" in benchmark_names:
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

    if run_all or "triple8" in benchmark_names:
        benchmarks.append(
            Triple(
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

    if run_all or "pair8" in benchmark_names:
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

    if run_all or "single8" in benchmark_names:
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

    if run_all or "pair128-O5" in benchmark_names:
        benchmarks.append(
            Pair(
                128,
                128,
                128,
                False,
                0,
                "./results",
                5,
                _Epsilon,
                settings_128
            )
        )

    settings_sparse = BenchmarkSettings(
        runs=10,
        repeats=3,
        waste_of_time_threshold=0.01,
        test_too_short_threshold=0.0005,
        long_run_multiplier=10,
        setup_timeout=3.0,
        benchmark_timeout=2.0,
        testing_timeout=2.0,
        tear_down_timeout=2.0,
        benchmark_trial_child_process=True,
        benchmark_in_child_process=True,
    )

    for rate in ["0.1", "0.01", "0.001", "0.0001", "0.00001"]:
        if len(sys.argv) == 1 or f"sparse1024-{rate}" in sys.argv:
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

    for rate in ["0.1", "0.01", "0.001", "0.0001", "0.00001"]:
        if len(sys.argv) == 1 or f"sparse-row1024-{rate}" in sys.argv:
            benchmarks.append(
                RowSparseSingle(
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

    benchmark_options = [a for a in sys.argv if a.startswith("-")]
    for benchmark in benchmarks:
        benchmark.run(benchmark_options)

    if "run" in sys.argv:
        for benchmark in benchmarks:
            benchmark.run()
