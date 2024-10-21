import ctypes

import benchmarkRunner
from benchmark import Benchmark
from dtl import TensorVariable
from dtl.libBuilder import DTLCLib, FuncTypeDescriptor, LibBuilder, NpArrayCtype, TupleStruct
from xdsl.dialects import llvm
from xdsl.dialects.builtin import FunctionType, StringAttr, f32, i64
from xdsl.dialects.experimental import dlt
from xdsl.transforms.experimental.dlt.generate_dlt_layouts import ReifyConfig


class TacoMatMul(Benchmark):

    def define_lib_builder(self) -> LibBuilder:
        pass

    def get_configs_for_DTL_tensors(self) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        pass

    def get_setup(self) -> str:
        pass

    def get_benchmark(self) -> str:
        pass

    def get_test(self) -> str:
        pass

    def get_clean(self) -> str:
        pass

    def make_lib(self, lib_path: str, i: int, j:int, k:int):
        DLT_Ptr_LLVM_Struct = llvm.LLVMStructType.from_type_list([llvm.LLVMPointerType.opaque()])
        dummy_dlt_ptr = dlt.PtrType(dlt.TypeType([({},[],f32)]))
        function_types = {}
        function_type_descriptor = {}

        function_types["init_A"] = FunctionType.from_lists([], [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct])
        function_type_descriptor["init_A"] = FuncTypeDescriptor([], [dummy_dlt_ptr, dummy_dlt_ptr], (1, 1))
        function_types["init_B"] = FunctionType.from_lists([], [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct])
        function_type_descriptor["init_B"] = FuncTypeDescriptor([], [dummy_dlt_ptr, dummy_dlt_ptr], (1, 1))
        function_types["init_C"] = FunctionType.from_lists([], [DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct])
        function_type_descriptor["init_C"] = FuncTypeDescriptor([], [dummy_dlt_ptr, dummy_dlt_ptr], (1, 1))

        function_types["setup_A"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque()], [])
        function_type_descriptor["setup_A"] = FuncTypeDescriptor([dummy_dlt_ptr, NpArrayCtype((i, j))], [], None)
        function_types["setup_B"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque()], [])
        function_type_descriptor["setup_B"] = FuncTypeDescriptor([dummy_dlt_ptr, NpArrayCtype((j, k))], [], None)

        function_types["matmul"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct, DLT_Ptr_LLVM_Struct], [])
        function_type_descriptor["matmul"] = FuncTypeDescriptor([dummy_dlt_ptr, dummy_dlt_ptr, dummy_dlt_ptr], [], None)

        function_types["check_c"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct, llvm.LLVMPointerType.opaque(), DLT_Ptr_LLVM_Struct], [i64, f32, i64])
        function_type_descriptor["check_c"] = FuncTypeDescriptor([dummy_dlt_ptr, NpArrayCtype((i, k)), dummy_dlt_ptr], [i64, f32, i64], None)

        function_types["dealloc_A"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct], [])
        function_type_descriptor["dealloc_A"] = FuncTypeDescriptor([dummy_dlt_ptr], [], None)
        function_types["dealloc_B"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct], [])
        function_type_descriptor["dealloc_B"] = FuncTypeDescriptor([dummy_dlt_ptr], [], None)
        function_types["dealloc_C"] = FunctionType.from_lists([DLT_Ptr_LLVM_Struct], [])
        function_type_descriptor["dealloc_C"] = FuncTypeDescriptor([dummy_dlt_ptr], [], None)

        taco_lib_path = "/home/edward/Code/taco/buildgcc/lib/libtaco.so"
        taco_lib = ctypes.cdll.LoadLibrary(taco_lib_path)
        lib = DTLCLib(lib_path, function_type_descriptor, function_types)
        return lib

if __name__ == "__main__":
    lib = make_lib("./native_debugging/taco_lib.so", 8,8,8)


    benchmarkRunner.run_benchmark(lib, 1, )

