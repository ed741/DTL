import abc
import os
import sys
from typing import Any, Generic

import numpy as np

from benchmarking.calibrate.calibrate_code import calibrate_code, make_dense_np_array
from benchmarking.benchmark import BenchmarkSettings, ID_Tuple, TestCode
from dtl.libBuilder import FuncTypeDescriptor, NpArrayCtype, StructType
from benchmarking.tacoBenchmark import (
    BasicTacoTest,
    T_Taco,
    TacoBenchmark,
    TacoTest,
    TypeMap,
)
from xdsl.dialects import llvm
from xdsl.dialects.builtin import FunctionType, StringAttr, f32, i64
from xdsl.dialects.experimental import dlt


_Epsilon = 0.00001

_Args = tuple[Any, StructType, StructType, StructType]


class CalibrateTaco(TacoBenchmark[BasicTacoTest]):
    def __init__(
        self,
        base_dir: str,
        opt_num: int,
        epsilon: float,
        settings: BenchmarkSettings,
    ):
        new_base_dir = f"{base_dir}/calibrate_taco"
        results_base_dir = f"{new_base_dir}/_O{opt_num}"

        taco_include_path = os.environ["TACO_INCLUDE_PATH"].strip('"')
        taco_lib_path = os.environ["TACO_LIB_DIR_PATH"].strip('"')
        super().__init__(
            results_base_dir, taco_include_path, taco_lib_path, settings, opt_num
        )

        self.i, self.j = 16, 16
        self.epsilon = epsilon

        ref_a = make_dense_np_array(0, self.i, self.j)

        self.ref_a = ref_a

        self.handle_reference_array(ref_a, f"arrays/ref_a", True, True, "ref_a")

    def get_format_options(self) -> dict[str, tuple[list[str], ...]]:
        f = ["Dense", "Sparse"]
        return {
            "##A_Format##": (f, f),
        }

    def get_code_injection(self) -> list[tuple[str, str]]:
        return [
            ("##I##", str(self.i)),
            ("##J##", str(self.j)),
            ("##Epsilon##", str(self.epsilon)),
        ]

    def make_tests_for(self, type_map: TypeMap) -> list[T_Taco]:
        return [
            BasicTacoTest(
                calibrate_code,
                self.taco_path,
                f"./benchmarking/taco/calibrate/calibrate.cpp",
                type_map,
            )
        ]

    def make_func_types(
        self,
    ) -> tuple[dict[StringAttr, FunctionType], dict[str, FuncTypeDescriptor]]:
        DLT_Ptr_LLVM_Struct = llvm.LLVMStructType.from_type_list(
            [llvm.LLVMPointerType.opaque()]
        )
        dummy_dlt_ptr = dlt.PtrType(dlt.TypeType([({}, [], f32)]))
        function_types = {}
        function_type_descriptor = {}

        function_types["init_A"] = FunctionType.from_lists(
            [], [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct]
        )
        function_type_descriptor["init_A"] = FuncTypeDescriptor(
            [], [dummy_dlt_ptr, dummy_dlt_ptr], (1, 1)
        )

        function_types["setup_A"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque()], []
        )
        function_type_descriptor["setup_A"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.i, self.j))], [], None
        )

        function_types["prepare"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct], []
        )
        function_type_descriptor["prepare"] = FuncTypeDescriptor(
            [dummy_dlt_ptr], [], None
        )

        function_types["func"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct], []
        )
        function_type_descriptor["func"] = FuncTypeDescriptor(
            [dummy_dlt_ptr], [], None
        )

        function_types["check_A"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque(), DLT_Ptr_LLVM_Struct],
            [i64, f32, i64],
        )
        function_type_descriptor["check_A"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.i, self.j)), dummy_dlt_ptr],
            [i64, f32, i64],
            None,
        )

        function_types["dealloc_A"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct], [])
        function_type_descriptor["dealloc_A"] = FuncTypeDescriptor(
            [dummy_dlt_ptr], [], None
        )

        function_types = {StringAttr(n): f for n, f in function_types.items()}
        return function_types, function_type_descriptor


if __name__ == "__main__":

    benchmarks = []

    print(f"Args: {sys.argv}")

    settings = BenchmarkSettings(
        runs=10,
        repeats=3,
        waste_of_time_threshold=0.1,
        test_too_short_threshold=0.001,
        long_run_multiplier=100,
        benchmark_timeout=3.0,
        benchmark_trial_child_process=True,
    )

    if len(sys.argv) == 1 or "1" in sys.argv:
        benchmarks.append(CalibrateTaco("./results", 3, _Epsilon, settings))

    benchmark_options = [a for a in sys.argv if a.startswith("-")]
    for benchmark in benchmarks:
        benchmark.run(benchmark_options)

    if "run" in sys.argv:
        for benchmark in benchmarks:
            benchmark.run()
