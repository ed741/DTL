import sys
from random import Random
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
from xdsl.dialects import func
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

    def make_tests_for(
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
        return {(a, b, c): ReifyConfig()}


class Pair(MatMulDenseDTL[BasicDTLTest]):

    def get_self_name(self) -> str:
        return "pair"

    def make_tests_for(
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
            (a, b): ReifyConfig(coo_buffer_options=frozenset([])),
            c: ReifyConfig(coo_buffer_options=frozenset([])),
        }


class Single(MatMulDenseDTL[BasicDTLTest]):

    def get_self_name(self) -> str:
        return "single"

    def make_tests_for(
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
            a: ReifyConfig(coo_buffer_options=frozenset([0])),
            b: ReifyConfig(coo_buffer_options=frozenset([0])),
            c: ReifyConfig(),
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
        super().__init__(*args, **kwargs)
        self.rate_a = rate_a
        self.rate_b = rate_b

    def get_self_name(self) -> str:
        return "random_sparse"

    def make_abc(self, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return make_random_sparse_np_arrays(seed, self.i, self.j, self.k, self.rate_a, self.rate_b)

    def test_data_variant_name(self) -> str:
        return super().test_data_variant_name() + f"_{self.rate_a}_{self.rate_b}"

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
        runs=100,
        repeats=3,
        waste_of_time_threshold=0.1,
        test_too_short_threshold=0.001,
        long_run_multiplier=100,
        benchmark_timeout=3.0,
        benchmark_trial_child_process=True,
    )
    settings_8 = BenchmarkSettings(
        runs=10,
        repeats=3,
        waste_of_time_threshold=0.01,
        test_too_short_threshold=0.001,
        long_run_multiplier=100,
        benchmark_timeout=3.0,
        benchmark_trial_child_process=False,
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
                5,
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
        benchmark_trial_child_process=True,
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
