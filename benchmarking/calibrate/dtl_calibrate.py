import sys

from benchmarking.calibrate.calibrate_code import calibrate_code, make_dense_np_array
from benchmarking.dlt_base_test import BasicDTLTest
from benchmarking.benchmark import BenchmarkSettings
from dtl import *
from dtl.dag import RealVectorSpace

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


class CalibrateDTL(DTLBenchmark[BasicDTLTest]):
    def __init__(
        self,
        base_dir: str,
        opt_num: int,
        epsilon: float,
        settings: BenchmarkSettings,
    ):
        new_base_dir = f"{base_dir}/calibrate"
        results_base_dir = f"{new_base_dir}/_O{opt_num}"

        self.i, self.j = 16, 16
        self.epsilon = epsilon
        super().__init__(
            results_base_dir,
            f"{new_base_dir}/layouts",
            f"{new_base_dir}/orders",
            settings,
            opt_num,
        )

        ref_a = make_dense_np_array(0, self.i, self.j)

        self.ref_a = ref_a
        self.handle_reference_array(
            ref_a, f"arrays/ref_a", True, True, "ref_a"
        )

    def get_configs_for_DTL_tensors(
        self,
        a: TensorVariable,
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        return {a:ReifyConfig()}

    def define_lib_builder(self) -> tuple[LibBuilder, tuple[TensorVariable, ...]]:
        vi = RealVectorSpace(self.i)
        vj = RealVectorSpace(self.j)
        scope_var_map = {}

        A = TensorVariable(vi * vj, "A")

        lib_builder = LibBuilder(scope_var_map)
        lib_builder.make_init("init_A", (A), [], free_name="dealloc_A")

        block = Block(arg_types=[
            lib_builder.tensor_var_details[A],
        ], ops=[func.Return()])
        lib_builder.make_custom_function("prepare", block, [A])

        block = Block(arg_types=[
            lib_builder.tensor_var_details[A],
        ], ops=[func.Return()])
        lib_builder.make_custom_function("func", block, [A])

        make_setup_func_dense(lib_builder, "setup_A", A, [self.i, self.j])
        make_check_func_dense(lib_builder, "check_A", A, [self.i, self.j], self.epsilon)
        return lib_builder, (A,)

    def make_tests_for(self, context: DLTCompileContext, layout: PtrMapping, order: IterationMapping) -> list[T_DTL]:
        return [BasicDTLTest(calibrate_code, context, layout, order)]


if __name__ == "__main__":

    benchmarks = []

    print(f"Args: {sys.argv}")
    settings = BenchmarkSettings(
        runs=10,
        repeats=3,
        waste_of_time_threshold=0.1,
        test_too_short_threshold=0.001,
        long_run_multiplier=100,
        setup_timeout=2.0,
        benchmark_timeout=3.0,
        testing_timeout=2.0,
        tear_down_timeout=2.0,
        benchmark_trial_child_process=True,
    )

    # if len(sys.argv) == 1 or "1" in sys.argv:
    #     benchmarks.append(StaticTriple(128,128,128,True, 0, "./results", repeats=repeats, runs=runs, opt_level=3, epsilon=_Epsilon))
    if len(sys.argv) == 1 or "1" in sys.argv:
        benchmarks.append(
            CalibrateDTL(
                "./results",
                3,
                _Epsilon,
                settings
            )
        )

    benchmark_options = [a for a in sys.argv if a.startswith("-")]
    for benchmark in benchmarks:
        benchmark.run(benchmark_options)

    if "run" in sys.argv:
        for benchmark in benchmarks:
            benchmark.run()
