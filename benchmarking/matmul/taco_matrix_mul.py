import abc
import os
import sys
from typing import Any, Generic

import numpy as np

from benchmarking.matmul.matrix_mul_code import (
    make_dense_np_arrays,
    make_random_sparse_np_arrays,
    matmul_single_code,
)
from benchmarking.benchmark import BenchmarkSettings, ID_Tuple, TestCode
from dtl.libBuilder import FuncTypeDescriptor, NpArrayCtype, StructType
from benchmarking.tacoBenchmark import BasicTacoTest, T_Taco, TacoBenchmark, TacoTest, TypeMap
from xdsl.dialects import llvm
from xdsl.dialects.builtin import FunctionType, StringAttr, f32, i64
from xdsl.dialects.experimental import dlt


_Epsilon = 0.00001

_Args = tuple[Any, StructType, StructType, StructType]


class MatMulDenseTaco(TacoBenchmark[T_Taco], abc.ABC, Generic[T_Taco]):
    def __init__(
        self,
        i: int,
        j: int,
        k: int,
        seed: int,
        base_dir: str,
        opt_num: int,
        epsilon: float,
        settings: BenchmarkSettings,
    ):
        new_base_dir = f"{base_dir}/matmul_taco"
        results_base_dir = (
            f"{new_base_dir}/{self.get_self_name()}_O{opt_num}_{i}.{j}.{k}_{seed}"
        )

        taco_include_path = os.environ["TACO_INCLUDE_PATH"].strip('"')
        taco_lib_path = os.environ["TACO_LIB_DIR_PATH"].strip('"')
        super().__init__(
            results_base_dir, taco_include_path, taco_lib_path, settings, opt_num
        )

        self.i, self.j, self.k = i, j, k
        self.seed = seed
        self.epsilon = epsilon


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

    def make_abc(self, seed:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return make_dense_np_arrays(seed, self.i, self.j, self.k)

    def get_format_options(self) -> dict[str, tuple[list[str], ...]]:
        f = ["Dense", "Sparse"]
        return {
            "##A_Format##": (f, f),
            "##B_Format##": (f, f),
            "##C_Format##": (f, f),
        }

    def get_code_injection(self) -> list[tuple[str, str]]:
        return [
            ("##I##", str(self.i)),
            ("##J##", str(self.j)),
            ("##K##", str(self.k)),
            ("##Epsilon##", str(self.epsilon)),
        ]


class Single(MatMulDenseTaco[BasicTacoTest]):
    def get_self_name(self) -> str:
        return "single"

    def make_tests_for(self, type_map: TypeMap) -> list[T_Taco]:
        return [
            BasicTacoTest(
                matmul_single_code,
                self.taco_path,
                f"./benchmarking/taco/matmul/single.cpp",
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
        function_types["init_B"] = FunctionType.from_lists(
            [], [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct]
        )
        function_type_descriptor["init_B"] = FuncTypeDescriptor(
            [], [dummy_dlt_ptr, dummy_dlt_ptr], (1, 1)
        )
        function_types["init_C"] = FunctionType.from_lists(
            [], [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct]
        )
        function_type_descriptor["init_C"] = FuncTypeDescriptor(
            [], [dummy_dlt_ptr, dummy_dlt_ptr], (1, 1)
        )

        function_types["setup_A"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque()], []
        )
        function_type_descriptor["setup_A"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.i, self.j))], [], None
        )
        function_types["setup_B"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque()], []
        )
        function_type_descriptor["setup_B"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.j, self.k))], [], None
        )

        function_types["prepare"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct], []
        )
        function_type_descriptor["prepare"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, dummy_dlt_ptr, dummy_dlt_ptr], [], None
        )

        function_types["matmul"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct], []
        )
        function_type_descriptor["matmul"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, dummy_dlt_ptr, dummy_dlt_ptr], [], None
        )

        function_types["check_C"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque(), DLT_Ptr_LLVM_Struct],
            [i64, f32, i64],
        )
        function_type_descriptor["check_C"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.i, self.k)), dummy_dlt_ptr],
            [i64, f32, i64],
            None,
        )

        function_types["dealloc_A"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct], [])
        function_type_descriptor["dealloc_A"] = FuncTypeDescriptor(
            [dummy_dlt_ptr], [], None
        )
        function_types["dealloc_B"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct], [])
        function_type_descriptor["dealloc_B"] = FuncTypeDescriptor(
            [dummy_dlt_ptr], [], None
        )
        function_types["dealloc_C"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct], [])
        function_type_descriptor["dealloc_C"] = FuncTypeDescriptor(
            [dummy_dlt_ptr], [], None
        )

        function_types = {StringAttr(n): f for n, f in function_types.items()}
        return function_types, function_type_descriptor


class MatMulSparseTacoTest(BasicTacoTest):

    def __init__(
        self,
        code: TestCode,
        taco_path: str,
        cpp_path: str,
        type_mapping: TypeMap,
        rate_a: float,
        rate_b: float,
    ):
        super().__init__(code, taco_path, cpp_path, type_mapping)
        self.rate_a = rate_a
        self.rate_b = rate_b

    @classmethod
    def get_id_headings(cls) -> list[tuple[str, type[str] | type[int] | type[bool]]]:
        return super().get_id_headings() + [("rate_a", str), ("rate_b", str)]

    def get_id(self) -> ID_Tuple:
        return *super().get_id(), str(self.rate_a), str(self.rate_b)


class RandomSparseSingle(MatMulDenseTaco[MatMulSparseTacoTest]):

    def __init__(
        self,
        i: int,
        j: int,
        k: int,
        seed: int,
        rate_a: float,
        rate_b: float,
        base_dir: str,
        opt_num: int,
        epsilon: float,
        settings: BenchmarkSettings,
    ):
        self.rate_a = rate_a
        self.rate_b = rate_b
        super().__init__(i, j, k, seed, base_dir, opt_num, epsilon, settings)


    def get_self_name(self) -> str:
        return "random_sparse"

    def make_tests_for(self, type_map: TypeMap) -> list[T_Taco]:
        return [
            MatMulSparseTacoTest(
                matmul_single_code,
                self.taco_path,
                f"./benchmarking/taco/matmul/single.cpp",
                type_map,
                self.rate_a,
                self.rate_b,
            )
        ]

    def test_data_variant_name(self) -> str:
        return super().test_data_variant_name() + f"_{self.rate_a}_{self.rate_b}"

    def make_abc(self, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return make_random_sparse_np_arrays(
            seed, self.i, self.j, self.k, self.rate_a, self.rate_b
        )

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
        function_types["init_B"] = FunctionType.from_lists(
            [], [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct]
        )
        function_type_descriptor["init_B"] = FuncTypeDescriptor(
            [], [dummy_dlt_ptr, dummy_dlt_ptr], (1, 1)
        )
        function_types["init_C"] = FunctionType.from_lists(
            [], [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct]
        )
        function_type_descriptor["init_C"] = FuncTypeDescriptor(
            [], [dummy_dlt_ptr, dummy_dlt_ptr], (1, 1)
        )

        function_types["setup_A"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque()], []
        )
        function_type_descriptor["setup_A"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.i, self.j))], [], None
        )
        function_types["setup_B"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque()], []
        )
        function_type_descriptor["setup_B"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.j, self.k))], [], None
        )

        function_types["prepare"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct], []
        )
        function_type_descriptor["prepare"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, dummy_dlt_ptr, dummy_dlt_ptr], [], None
        )

        function_types["matmul"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct], []
        )
        function_type_descriptor["matmul"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, dummy_dlt_ptr, dummy_dlt_ptr], [], None
        )

        function_types["check_C"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque(), DLT_Ptr_LLVM_Struct],
            [i64, f32, i64],
        )
        function_type_descriptor["check_C"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.i, self.k)), dummy_dlt_ptr],
            [i64, f32, i64],
            None,
        )

        function_types["dealloc_A"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct], [])
        function_type_descriptor["dealloc_A"] = FuncTypeDescriptor(
            [dummy_dlt_ptr], [], None
        )
        function_types["dealloc_B"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct], [])
        function_type_descriptor["dealloc_B"] = FuncTypeDescriptor(
            [dummy_dlt_ptr], [], None
        )
        function_types["dealloc_C"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct], [])
        function_type_descriptor["dealloc_C"] = FuncTypeDescriptor(
            [dummy_dlt_ptr], [], None
        )

        function_types = {StringAttr(n): f for n, f in function_types.items()}
        return function_types, function_type_descriptor

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
    base_directory = "./results"
    if run_all or "single128" in benchmark_names:
        benchmarks.append(
            Single(128, 128, 128, 0, base_directory, 3, _Epsilon, settings_128)
        )
    if run_all or "single8" in benchmark_names:
        benchmarks.append(Single(8, 8, 8, 0, base_directory, 3, _Epsilon, settings_128))

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
        if run_all or f"sparse1024-{rate}" in benchmark_names:
            benchmarks.append(
                RowSparseSingle(
                    1024, 1024, 1024, 0, float(rate), float(rate), base_directory, 3, _Epsilon, settings_128
                )
            )
    for rate in ["0.1", "0.01", "0.001", "0.0001", "0.00001"]:
        if run_all or f"sparse-row1024-{rate}" in benchmark_names:
            benchmarks.append(
                RowSparseSingle(
                    1024, 1024, 1024, 0, float(rate), float(rate), base_directory, 3, _Epsilon, settings_128
                )
            )

    benchmark_options = [a for a in sys.argv if a.startswith("-")]
    for benchmark in benchmarks:
        benchmark.run(benchmark_options)

    if "run" in sys.argv:
        for benchmark in benchmarks:
            benchmark.run()
