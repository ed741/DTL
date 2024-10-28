import abc
import sys
from typing import Generic

import scipy
from numpy import ndarray

from benchmarking.benchmark import BenchmarkSettings
from benchmarking.dlt_base_test import BasicDTLTest
from benchmarking.dtlBenchmark import (
    DLTCompileContext,
    DTLBenchmark,
    T_DTL,
    make_check_func_dense,
    make_setup_func_coo,
    make_setup_func_dense,
)
from benchmarking.sparsesuite.sparse_suite import TensorWrapper, handle_tensor
from benchmarking.sparsesuite.spmv_code import spmv_single_code
from dtl import Index, RealVectorSpace, TensorVariable
from dtl.libBuilder import LibBuilder, TupleStruct
from xdsl.dialects import func
from xdsl.dialects.builtin import StringAttr
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



class SparseSuite(DTLBenchmark[T_DTL], abc.ABC, Generic[T_DTL]):
    def __init__(
        self,
        tensor_paths: dict[str, tuple[bool, bool, str, tuple[int,...]|None]],
        base_dir: str,
        opt_num: int,
        epsilon: float,
        settings: BenchmarkSettings,
    ):
        new_base_dir = f"{base_dir}/sparseSuite"
        results_base_dir = f"{new_base_dir}/{self.get_self_name()}_O{opt_num}"
        self.epsilon = epsilon
        self.tensor_paths = tensor_paths
        super().__init__(
            results_base_dir,
            f"{new_base_dir}/layouts",
            f"{new_base_dir}/orders",
            settings,
            opt_num,
            skip_layout_func=self.skip_layout_func,
            skip_order_func=self.skip_order_func,
        )

        tensors: dict[str, TensorWrapper] = {}
        for tensor_name, (is_arg, is_res, path, reshape) in tensor_paths.items():
            self.log(f"loading {tensor_name} from {path}")
            ref_tensor = scipy.io.mmread(path)
            if reshape is not None:
                ref_tensor = ref_tensor.reshape(reshape)
            tensors[tensor_name] = handle_tensor(self, self.test_data_variant_name(),
                tensor_name, is_arg, is_res, ref_tensor
            )
        results = self.make_reference_result(tensors)
        for tensor_name, ref_tensor in results.items():
            tensors[tensor_name] = handle_tensor(self, self.test_data_variant_name(),
                tensor_name, False, True, ref_tensor
            )
        self.tensors = tensors

    @abc.abstractmethod
    def skip_layout_func(self, layout: PtrMapping) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def skip_order_func(self, order: IterationMapping) -> bool:
        raise NotImplementedError


    @abc.abstractmethod
    def make_reference_result(
        self,
        tensors: dict[str, TensorWrapper],
    ) -> dict[str, ndarray | scipy.sparse.coo_array]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_self_name(self) -> str:
        raise NotImplementedError

    def test_data_variant_name(self) -> str:
        return f"{self.get_self_name()}"


class SpMV(SparseSuite[BasicDTLTest], abc.ABC):

    def __init__(
        self,
        matrix_path: str,
        vector_path: str,
        base_dir: str,
        opt_num: int,
        epsilon: float,
        settings: BenchmarkSettings,
    ):
        tensor_paths: dict[str, tuple[bool, bool, str, tuple[int,...]|None]] = {
            "ref_a": (True, False, matrix_path, None),
            "ref_b": (True, False, vector_path, (-1)),
        }
        super().__init__(tensor_paths, base_dir, opt_num, epsilon, settings)
        self.i, self.j = self.tensors["ref_a"].shape
        assert (self.i,) == self.tensors["ref_b"].shape, f"shape of ref_b: {self.tensors["ref_b"].shape} != {(self.i,)}"
        assert (self.j,) == self.tensors["ref_c"].shape, f"shape of ref_c: {self.tensors["ref_c"].shape} != {(self.j,)}"
        assert self.tensors["ref_a"].non_zeros >= 0
        assert self.tensors["ref_a"].is_sparse
        assert not self.tensors["ref_b"].is_sparse
        assert not self.tensors["ref_c"].is_sparse
        self.nnz = self.tensors["ref_a"].non_zeros

    def make_reference_result(
        self, tensors: dict[str, TensorWrapper]
    ) -> dict[str, ndarray | scipy.sparse.coo_array]:
        return {"ref_c": tensors["ref_a"].tensor @ tensors["ref_b"].tensor}


    def make_tests_for(
        self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping
    ) -> list[BasicDTLTest]:
        return [BasicDTLTest(spmv_single_code, context, layout, order)]


    def define_lib_builder(self) -> tuple[LibBuilder, tuple[TensorVariable, ...]]:
        vi = RealVectorSpace(self.i)
        vj = RealVectorSpace(self.j)

        A = TensorVariable(vi * vj, "A")
        B = TensorVariable(vj, "B")
        C = TensorVariable(vi, "C")

        _i = Index("i")
        _j = Index("j")
        spmv = (A[_i, _j] * B[_j]).sum(_j).forall(_i)

        lib_builder = LibBuilder({})
        lib_builder.make_init("init_A", A, [], free_name="dealloc_A")
        lib_builder.make_init("init_B", B, [], free_name="dealloc_B")
        lib_builder.make_init("init_C", C, [], free_name="dealloc_C")


        # lib_builder.make_setter("set_A", (A), {}, [0, 1])
        # lib_builder.make_setter("set_B", (B), {}, [0])
        #
        # lib_builder.make_getter("get_C", (C), {}, [0])

        block = Block(arg_types=[
            lib_builder.tensor_var_details[A],
            lib_builder.tensor_var_details[B],
            lib_builder.tensor_var_details[C],
        ], ops=[func.Return()])
        lib_builder.make_custom_function("prepare", block, [A, B, C])

        lib_builder.make_function("spmv", spmv, [C], [A, B], [], [])

        make_setup_func_coo(lib_builder, "setup_A", A, self.nnz)
        make_setup_func_dense(lib_builder, "setup_B", B, [self.j])
        make_check_func_dense(lib_builder, "check_C", C, [self.i], self.epsilon)
        return lib_builder, (A,B,C)

class SCircuit(SpMV):

    def __init__(
        self,
        base_dir: str,
        opt_num: int,
        epsilon: float,
        settings: BenchmarkSettings,
    ):
        super().__init__(
            "benchmarking/sparse_suite/scircuit/scircuit.mtx",
            "benchmarking/sparse_suite/scircuit/scircuit_b.mtx",
            base_dir, opt_num, epsilon, settings)

    def get_self_name(self) -> str:
        return "SCircuit"

    def get_configs_for_DTL_tensors(self, a: TensorVariable, b: TensorVariable, c: TensorVariable) -> dict[
        TupleStruct[TensorVariable], ReifyConfig]:
        return {
            a: ReifyConfig(
                coo_buffer_options=frozenset([0]), dense=True, coo_minimum_dims=1
            ),
            b: ReifyConfig(
                coo_buffer_options=frozenset([0]), dense=True, coo_minimum_dims=1
            ),
            c: ReifyConfig(dense=True, coo_minimum_dims=1),
        }

    def skip_layout_func(self, l: PtrMapping) -> bool:
        a_layout = l.make_ptr_dict()[StringAttr("R_0")]
        if isinstance(a_layout.layout, dlt.DenseLayoutAttr):
            if isinstance(a_layout.layout.child, dlt.DenseLayoutAttr):
                return True
        return False

    def skip_order_func(self, o: IterationMapping) -> bool:
        return all(isinstance(i, dlt.NestedIterationOrderAttr) for i in o.values)


if __name__ == "__main__":

    repeats = 1
    runs = 1
    benchmarks = []

    print(f"Args: {sys.argv}")

    # if len(sys.argv) == 1 or "1" in sys.argv:
    #     benchmarks.append(StaticTriple(128,128,128,True, 0, "./results", repeats=repeats, runs=runs, opt_level=3, epsilon=_Epsilon))
    if len(sys.argv) == 1 or "scircuit" in sys.argv:
        scircuit_settings = BenchmarkSettings(
            runs=10,
            repeats=3,
            waste_of_time_threshold=0.1,
            test_too_short_threshold=0.001,
            long_run_multiplier=100,
            benchmark_timeout=5.0,
            benchmark_trial_child_process=True,
            benchmark_in_child_process=True,
        )
        benchmarks.append(
            SCircuit(
                "./results",
                5,
                _Epsilon,
                scircuit_settings
            )
        )

    benchmark_options = [a for a in sys.argv if a.startswith("-")]
    for benchmark in benchmarks:
        benchmark.run(benchmark_options)

    if "run" in sys.argv:
        for benchmark in benchmarks:
            benchmark.run()
