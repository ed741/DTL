# import xdsl.dialects.arith
import ctypes

from dtl import *
from dtl.dag import RealVectorSpace, Index
import dtlpp.backends.xdsl as xdtl

from dtl.libBuilder import LibBuilder
from dtlpp.backends import native
from xdsl.dialects import builtin, llvm
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.experimental import dlt
from xdsl.ir import MLContext
from xdsl.pattern_rewriter import PatternRewriteWalker, GreedyRewritePatternApplier
from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations
from xdsl.transforms.experimental.generate_dlt_layouts import DLTLayoutRewriter
from xdsl.transforms.experimental.lower_dlt_to_ import DLTSelectRewriter, DLTGetRewriter, DLTSetRewriter, \
    DLTAllocRewriter, DLTIterateRewriter, DLTScopeRewriter, DLTPtrTypeRewriter, DLTCopyRewriter, \
    DLTExtractExtentRewriter
from xdsl.transforms.printf_to_llvm import PrintfToLLVM
from xdsl.transforms.printf_to_putchar import PrintfToPutcharPass
from xdsl.transforms.reconcile_unrealized_casts import reconcile_unrealized_casts
from xdslDTL import compilec
from xdsl.transforms.experimental.lower_dtl_to_dlt import DTLDenseRewriter

i = Index('i')
j = Index('j')
k = Index('k')
v10 = RealVectorSpace(10)
vS = UnknownSizeVectorSpace("S")
vQ = UnknownSizeVectorSpace("Q")
# v6 = RealVectorSpace(6)
# vu = UnknownSizeVectorSpace("VU")
A = TensorVariable(vQ*v10, "A")
B = TensorVariable(v10*vS, "B")
output_t_var = TensorVariable(vQ*vS, "out")
expr = (A[i,j]*B[j,k]).sum(j).forall(i,k)
# expr = A[i,None].tupled_with(A[j, k].sum(k)).deindex(((v10, i), (i, j)))
# expr = A[i,None].tupled_with(A[j, None].deindex((j, v10))).deindex(((v10, i), (i, vQ, v10)))
# expr = A[i:vS, j:v10].deindex((vQ, i, j, v10))

print("=== input DTL:")
print(expr)
print("expr type: "+str(expr.type))


print("=== python native reference implementation:")
builder = native.KernelBuilder(expr, debug_comments=False)
kernel = builder.build()
print(builder.codeNode.code())


print("=== xdsl.DTL:")
type = xdtl.DTLType_to_xdtl(expr.type)
print("type:")
print(type)



lib_builder = LibBuilder()
lib_builder.make_init("init_A", A, [vQ])
lib_builder.make_init("init_B", B, [vS])
lib_builder.make_init("init_Out", output_t_var, [vQ, vS])
lib_builder.make_dummy("test3", 3)
lib_builder.make_dummy("test5", 5)
#
# lib_builder.make_funcion("mm", expr,
#                          [output_t_var],
#                          [A,B],
#                          [],
#                          []
#                          )

malloc_func = llvm.FuncOp("malloc", llvm.LLVMFunctionType([builtin.i64], llvm.LLVMPointerType.opaque()),
                          linkage=llvm.LinkageAttr("external"))
abort_func = llvm.FuncOp("abort", llvm.LLVMFunctionType([]),
                          linkage=llvm.LinkageAttr("external"))


module = ModuleOp([malloc_func, abort_func, dlt.LayoutScopeOp([],lib_builder.funcs)])
module.verify()
# module = ModuleOp([func])
#
# print("lines:")
# p = printer.Printer()
# p.print(block)
# block.verify()
print(module)

print("=== DTL -> DLT")
dtl_to_dlt_applier = PatternRewriteWalker(DTLDenseRewriter(),
    walk_regions_first=False)

dtl_to_dlt_applier.rewrite_module(module)

print(module)
module.verify()

print("=== DLT --generate-layouts-> DLT")
dlt_to_dlt_applier = PatternRewriteWalker(DLTLayoutRewriter(),
    walk_regions_first=False)

dlt_to_dlt_applier.rewrite_module(module)

print(module)
module.verify()


print("=== DLT -> LLVM?")
dlt_to_llvm_applier = PatternRewriteWalker(GreedyRewritePatternApplier(
    [RemoveUnusedOperations(),
     DLTSelectRewriter(),
     DLTGetRewriter(),
     DLTSetRewriter(),
     DLTAllocRewriter(),
     DLTIterateRewriter(),
     DLTCopyRewriter(),
     DLTExtractExtentRewriter(),
     ]),
    walk_regions_first=False)

dlt_to_llvm_applier.rewrite_module(module)
rem_scope = PatternRewriteWalker(GreedyRewritePatternApplier(
    [DLTScopeRewriter(),
     DLTPtrTypeRewriter(recursive=True),
     # DLTIndexTypeRewriter(recursive=True),
     ])
)
rem_scope.rewrite_module(module)
PrintfToPutcharPass().apply(MLContext(True), module)
PrintfToLLVM().apply(MLContext(True), module)

reconcile_unrealized_casts(module)

print(module)
module.verify()


print("=== llvm -> compiler")


compilec.compile(module, "./tmp/libxdtl11.so")


print("load?")
from ctypes import cdll
lib = cdll.LoadLibrary("./tmp/libxdtl11.so")
print("lib loaded")
print("Try tests")
t3 = lib.test3()
print(t3)
t5 = lib.test5()
print(t5)
print("tests complete")
c_q = ctypes.c_uint64(3)
c_s = ctypes.c_uint64(5)

class A_Ptr(ctypes.Structure):
    _fields_= [('ptr', ctypes.c_void_p), ('q', ctypes.c_uint64)]
class B_Ptr(ctypes.Structure):
    _fields_= [('ptr', ctypes.c_void_p), ('s', ctypes.c_uint64)]
class Out_Ptr(ctypes.Structure):
    _fields_= [('ptr', ctypes.c_void_p), ('q', ctypes.c_uint64), ('s', ctypes.c_uint64)]

print(Out_Ptr)
print(Out_Ptr.ptr)
print(Out_Ptr.q)
print(Out_Ptr.s)
lib.init_A.restype = A_Ptr
lib.init_B.restype = B_Ptr
# lib.init_Out.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
lib.init_Out.restype = Out_Ptr
# lib.mm.argtypes = [Out_Ptr, A_Ptr, B_Ptr]
a_val = lib.init_A(c_q)
print("inited A")
print(a_val.ptr, a_val.q)
b_val = lib.init_B(c_s)
print("inited B")
print(b_val.ptr, b_val.s)
out_val = lib.init_Out(c_s, c_q)
print(out_val)
print(out_val.ptr, out_val.q, out_val.s)
print("vals inited")
# lib.mm(out_val, a_val, b_val)

print("Done")

