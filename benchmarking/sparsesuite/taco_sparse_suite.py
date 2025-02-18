import abc
import os
import sys
from typing import Generic

import numpy as np
import scipy
from numpy import ndarray

from benchmarking.benchmark import BenchmarkSettings
from benchmarking.sparsesuite.codes.spmspv_code import spmspv_single_code
from benchmarking.sparsesuite.sparse_suite import TensorWrapper, handle_tensor
from benchmarking.sparsesuite.codes.spmv_code import spmv_single_code
from benchmarking.tacoBenchmark import BasicTacoTest, T_Taco, TacoBenchmark, TypeMap
from dtl.libBuilder import FuncTypeDescriptor, NpArrayCtype
from xdsl.dialects import llvm
from xdsl.dialects.builtin import FunctionType, StringAttr, f32, i64
from xdsl.dialects.experimental import dlt

_Epsilon = 0.00001


class SparseSuiteTaco(TacoBenchmark[T_Taco], abc.ABC, Generic[T_Taco]):
    def __init__(
        self,
        tensor_paths: dict[str, tuple[bool, bool, str|scipy.sparse.coo_array|ndarray, tuple[int,...]|None]],
        base_dir: str,
        opt_num: int,
        epsilon: float,
        settings: BenchmarkSettings,
    ):
        new_base_dir = f"{base_dir}/sparseSuite_taco"
        results_base_dir = f"{new_base_dir}/{self.get_self_name()}_O{opt_num}"
        taco_include_path = os.environ["TACO_INCLUDE_PATH"].strip('"')
        taco_lib_path = os.environ["TACO_LIB_DIR_PATH"].strip('"')

        super().__init__(
            results_base_dir,
            taco_include_path,
            taco_lib_path,
            settings,
            opt_num,
        )

        self.epsilon = epsilon
        self.tensor_paths = tensor_paths

        tensors: dict[str, TensorWrapper] = {}
        for tensor_name, (is_arg, is_res, path, reshape) in tensor_paths.items():
            if isinstance(path, str):
                self.log(f"loading {tensor_name} from {path} ({'exists' if os.path.exists(path) else 'Not found!'})")
                ref_tensor = scipy.io.mmread(path)
            else:
                ref_tensor = path
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


class SpMV(SparseSuiteTaco[BasicTacoTest], abc.ABC):

    def __init__(
        self,
        matrix_path: str|scipy.sparse.coo_array,
        vector_path: str|ndarray,
        base_dir: str,
        opt_num: int,
        epsilon: float,
        settings: BenchmarkSettings,
    ):
        tensor_paths: dict[str, tuple[bool, bool, str, tuple[int,...]|None]] = {
            "ref_a": (True, False, matrix_path, None),
            "ref_b": (True, False, vector_path, (-1,)),
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
        self, type_map: TypeMap) -> list[BasicTacoTest]:
        return [BasicTacoTest(spmv_single_code, self.taco_path, "./benchmarking/taco/SpMV/single.cpp", type_map)]

    def get_format_options(self) -> dict[str, tuple[list[str], ...]]:
        f = ["Dense", "Sparse"]
        return {
            "##A_Format##": (f, f),
            "##B_Format##": (f,),
            "##C_Format##": (f,),
        }

    def get_code_injection(self) -> list[tuple[str, str]]:
        return [
            ("##I##", str(self.i)),
            ("##J##", str(self.j)),
            ("##NNZ##", str(self.nnz)),
            ("##Epsilon##", str(self.epsilon)),
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
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque(), llvm.LLVMPointerType.opaque(), llvm.LLVMPointerType.opaque()], []
        )
        function_type_descriptor["setup_A"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.nnz,), np.int32), NpArrayCtype((self.nnz,), np.int32), NpArrayCtype((self.nnz,))], [], None
        )
        function_types["setup_B"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque()], []
        )
        function_type_descriptor["setup_B"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.i,))], [], None
        )

        function_types["prepare"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct], []
        )
        function_type_descriptor["prepare"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, dummy_dlt_ptr, dummy_dlt_ptr], [], None
        )

        function_types["spmv"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct], []
        )
        function_type_descriptor["spmv"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, dummy_dlt_ptr, dummy_dlt_ptr], [], None
        )

        function_types["check_C"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque(), DLT_Ptr_LLVM_Struct],
            [i64, f32, i64],
        )
        function_type_descriptor["check_C"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.j,)), dummy_dlt_ptr],
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



class SpMSpV(SparseSuiteTaco[BasicTacoTest], abc.ABC):

    def __init__(
        self,
        matrix_path: str|scipy.sparse.coo_array,
        vector_path: str|scipy.sparse.coo_array,
        base_dir: str,
        opt_num: int,
        epsilon: float,
        settings: BenchmarkSettings,
    ):
        tensor_paths: dict[str, tuple[bool, bool, str, tuple[int,...]|None]] = {
            "ref_a": (True, False, matrix_path, None),
            "ref_b": (True, False, vector_path, (-1,)),
        }
        super().__init__(tensor_paths, base_dir, opt_num, epsilon, settings)
        self.i, self.j = self.tensors["ref_a"].shape
        assert (self.i,) == self.tensors["ref_b"].shape, f"shape of ref_b: {self.tensors["ref_b"].shape} != {(self.i,)}"
        assert (self.j,) == self.tensors["ref_c"].shape, f"shape of ref_c: {self.tensors["ref_c"].shape} != {(self.j,)}"
        assert self.tensors["ref_a"].non_zeros >= 0
        assert self.tensors["ref_a"].is_sparse
        assert self.tensors["ref_b"].is_sparse
        assert self.tensors["ref_c"].is_sparse
        self.nnz_a = self.tensors["ref_a"].non_zeros
        self.nnz_b = self.tensors["ref_b"].non_zeros
        self.nnz_c = self.tensors["ref_c"].non_zeros

    def make_reference_result(
        self, tensors: dict[str, TensorWrapper]
    ) -> dict[str, ndarray | scipy.sparse.coo_array]:
        return {"ref_c": (tensors["ref_a"].tensor @ tensors["ref_b"].tensor.reshape((-1, 1))).reshape(-1).tocoo()}


    def make_tests_for(
        self, type_map: TypeMap) -> list[BasicTacoTest]:
        return [BasicTacoTest(spmspv_single_code, self.taco_path, "./benchmarking/taco/SpMSpV/single.cpp", type_map)]

    def get_format_options(self) -> dict[str, tuple[list[str], ...]]:
        f = ["Dense", "Sparse"]
        return {
            "##A_Format##": (f, f),
            "##B_Format##": (f,),
            "##C_Format##": (f,),
        }

    def get_code_injection(self) -> list[tuple[str, str]]:
        return [
            ("##I##", str(self.i)),
            ("##J##", str(self.j)),
            ("##NNZ_A##", str(self.nnz_a)),
            ("##NNZ_B##", str(self.nnz_b)),
            ("##NNZ_C##", str(self.nnz_c)),
            ("##Epsilon##", str(self.epsilon)),
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
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque(), llvm.LLVMPointerType.opaque(), llvm.LLVMPointerType.opaque()], []
        )
        function_type_descriptor["setup_A"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.nnz_a,), np.int32), NpArrayCtype((self.nnz_a,), np.int32), NpArrayCtype((self.nnz_a,))], [], None
        )
        function_types["setup_B"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque(), llvm.LLVMPointerType.opaque()], []
        )
        function_type_descriptor["setup_B"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.nnz_b,), np.int32), NpArrayCtype((self.nnz_b,))], [], None
        )

        function_types["prepare"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct], []
        )
        function_type_descriptor["prepare"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, dummy_dlt_ptr, dummy_dlt_ptr], [], None
        )

        function_types["spmspv"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct], []
        )
        function_type_descriptor["spmspv"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, dummy_dlt_ptr, dummy_dlt_ptr], [], None
        )

        function_types["check_C"] = FunctionType.from_lists(
            [DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque(), llvm.LLVMPointerType.opaque(), DLT_Ptr_LLVM_Struct],
            [i64, f32, i64],
        )
        function_type_descriptor["check_C"] = FuncTypeDescriptor(
            [dummy_dlt_ptr, NpArrayCtype((self.nnz_c,), np.int32), NpArrayCtype((self.nnz_c,)), dummy_dlt_ptr],
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

class Rajat21(SpMSpV):

    def __init__(
        self,
        base_dir: str,
        opt_num: int,
        seed: int,
        vector_fill: float,
        epsilon: float,
        settings: BenchmarkSettings,
    ):
        r = np.random.default_rng(seed)
        length = 411676
        nnz = int(vector_fill * length)
        coords = np.sort(r.choice(np.arange(length, dtype=np.int32), nnz, replace=False))
        values = r.random((nnz,), dtype=np.float32)
        vector = scipy.sparse.coo_array((values, (coords,)), shape=(length,), dtype=np.float32)

        super().__init__(
            "benchmarking/sparse_suite/rajat21/rajat21.mtx",
            vector,
            base_dir, opt_num, epsilon, settings)

    def get_self_name(self) -> str:
        return "Rajat21"

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
            long_run_multiplier=1,
            setup_timeout=50.0,
            benchmark_timeout=5.0,
            testing_timeout=5.0,
            tear_down_timeout=5.0,
            benchmark_trial_child_process=True,
            benchmark_in_child_process=True,
        )
        benchmarks.append(
            SCircuit(
                "./results",
                4,
                _Epsilon,
                scircuit_settings
            )
        )
    if len(sys.argv) == 1 or "rajat21" in sys.argv:
        rajat21_settings = BenchmarkSettings(
            runs=10,
            repeats=3,
            waste_of_time_threshold=0.1,
            test_too_short_threshold=0.001,
            long_run_multiplier=1,
            setup_timeout=140.0,
            benchmark_timeout=5.0,
            testing_timeout=50.0,
            tear_down_timeout=50.0,
            benchmark_trial_child_process=True,
            benchmark_in_child_process=True,
        )
        benchmarks.append(
            Rajat21(
                "./results",
                4,
                0,
                0.001,
                _Epsilon,
                rajat21_settings
            )
        )

    benchmark_options = [a for a in sys.argv if a.startswith("-")]
    for benchmark in benchmarks:
        benchmark.run(benchmark_options)

    if "run" in sys.argv:
        for benchmark in benchmarks:
            benchmark.run()
