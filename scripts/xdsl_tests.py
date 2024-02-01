import io
import os
import subprocess
import tempfile
from io import StringIO

# import xdsl.dialects.arith
from dtl import *
from dtl.dag import RealVectorSpace, Index
import dtlpp.backends.xdsl as xdtl
import numpy as np

from dtl.dtlutils import visualise
from dtlpp.backends import native
from xdsl import printer, ir
from xdsl.dialects import memref, arith, builtin
from xdsl.dialects.builtin import Float64Type, f32, ModuleOp, StringAttr, IntegerAttr, IndexType
from xdsl.dialects.func import FuncOp, Return, Call
from xdsl.dialects.experimental import dlt
from xdsl.ir import SSAValue, Region
from xdsl.irdl import SingleBlockRegionDef
from xdsl.pattern_rewriter import PatternRewriteWalker, GreedyRewritePatternApplier
from xdsl.printer import Printer
from xdsl.transforms.experimental.generate_dlt_layouts import DLTLayoutRewriter
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

# module = ModuleOp([])
# block = module.body.block
block = ir.Block()
# a = block.insert_arg(xdsl.dialects.builtin.TensorType.from_type_and_list(f32, [5,5]), 0)
# v10_len_attr = IntegerAttr.from_int_and_width(5,64)


a = block.insert_arg(dlt.PtrType(
    dlt.TypeType([([], [("vQ", "Q"),("V10A", 10)], f32)]
    )).with_layout_name("a"), 0)
# b = block.insert_arg(xdsl.dialects.builtin.TensorType.from_type_and_list(f32, [5,6]), 1)
b = block.insert_arg(dlt.PtrType(
    dlt.TypeType([([], [("V10B", 10), ("vS", "S")], f32)]
    )).with_layout_name("b"), 1)

vQ_len_attr = IntegerAttr.from_int_and_width(5,64)
vS_len_attr = IntegerAttr.from_int_and_width(6,64)
vS_len = arith.Constant(vS_len_attr, IndexType())
vQ_len = arith.Constant(vQ_len_attr, IndexType())
# vS_len = block.insert_arg(xdsl.dialects.builtin.i32, 2)
# vQ_len = block.insert_arg(xdsl.dialects.builtin.i32, 3)
# i_val = block.insert_arg(xdsl.dialects.builtin.i32, 4)
block.add_op(vS_len)
block.add_op(vQ_len)

out = block.insert_arg(dlt.PtrType(dlt.TypeType([dlt.ElementAttr(tuple([dlt.SetAttr([]), dlt.SetAttr([dlt.DimensionAttr(tuple([StringAttr("vQ"), vQ_len_attr])), dlt.DimensionAttr(tuple([StringAttr("vS"), vS_len_attr]))]), f32]))])).with_layout_name("out"), 2)
# mem = block.insert_arg(memref.MemRefType.from_element_type_and_shape(f32, (5,6)), 4)

# lines, output = xdtl.get_xdsl_dtl_version(expr, tensorVariables={A:a})
# lines, output = xdtl.get_xdsl_dtl_exec_version(expr, space_map={vS:vS_len, vQ:vQ_len}, arg_map={i:i_val}, tensor_variables={A:a,B:b}, output=mem)
exec, output = xdtl.get_xdsl_dtl_exec_version(expr, space_map={vS:SSAValue.get(vS_len), vQ:SSAValue.get(vQ_len)}, arg_map={}, tensor_variables={A:(a,["vQ", "V10A"]),B:(b,["V10B", "vS"])}, outputs=[(out,["vQ", "vS"])])
ret = Return()

#TEST BLOCK
# block = ir.Block()
# const_int32_1 = xdsl.dialects.arith.Constant.from_int_and_width(1,32)
# lines = [const_int32_1]
# ret = Return(const_int32_1)


block.add_ops(exec)
init_inner = dlt.AllocOp(operands=[[],[]], attributes={"dynamic_dimensions":builtin.ArrayAttr([])}, result_types=[a.type.as_base()])
block.add_op(init_inner)
callinner = Call("foo", [init_inner, b, out],[])
block.add_op(callinner)

block.add_ops([output, ret])




region = Region([block])
argTypes = [arg.type for arg in block.args]
retTypes = [r.type for r in ret.operands]
# retTypes = []
func = FuncOp.from_region("foo", argTypes, retTypes, region)

inits = [dlt.AllocOp(operands=[[],[]], attributes={"dynamic_dimensions":builtin.ArrayAttr([])}, result_types=[t.as_base()]) for t in argTypes]

call = Call("foo", inits,[])
call2 = Call("foo", inits,[])
module = ModuleOp([dlt.LayoutScopeOp([func] + inits + [call, call2])])
module.verify()
# module = ModuleOp([func])
#
# print("lines:")
# p = printer.Printer()
# p.print(block)
# block.verify()
print(module)

print("=== DTL -> DLT")
dtl_to_dlt_applier = PatternRewriteWalker(GreedyRewritePatternApplier(
    [DTLDenseRewriter()]),
    walk_regions_first=False)

dtl_to_dlt_applier.rewrite_module(module)

print(module)
module.verify()

print("=== DLT->???")
dlt_to_llvm_applier = PatternRewriteWalker(GreedyRewritePatternApplier(
    [DLTLayoutRewriter()]),
    walk_regions_first=False)

dlt_to_llvm_applier.rewrite_module(module)

print(module)

print("=== llvm -> compiler")


compilec.compile(module, "./tmp/libxdtl10.o")
# print("args")
# print(func.args)
# print("results")
# print(func.get_return_op())
# print("func type")
# print(func.function_type)
#
# print("mlir output:")
# res = StringIO()
# printer = Printer(print_generic_format=False, stream=res)
# printer.print(module)
# print(res.getvalue())
#
#
# print("mlir-opt:")
# passes = ["--convert-scf-to-cf",
#           "--convert-cf-to-llvm",
#           "--convert-func-to-llvm",
#           "--convert-arith-to-llvm",
#           "--expand-strided-metadata",
#           "--normalize-memrefs",
#           "--memref-expand",
#           "--fold-memref-alias-ops",
#           "--finalize-memref-to-llvm",
#           "--reconcile-unrealized-casts"]
#
# process_opt = subprocess.Popen(['mlir-opt'] + passes, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
# out, err = process_opt.communicate(res.getvalue().encode('utf8'))
# process_opt.wait()
# print(out)
#
# print("mlir-translate")
# process_translate = subprocess.Popen(['mlir-translate', '--mlir-to-llvmir'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
# out, err = process_translate.communicate(out)
# process_translate.wait()
# print(out)
#
# fd, path = tempfile.mkstemp()
# try:
#     with os.fdopen(fd, 'wb') as tmp:
#         tmp.write(out)
#
#         print("clang")
#         process_clang = subprocess.Popen(['clang', '-c', '-o', './bin/dtlLib.ll', path], stdin=subprocess.PIPE,
#                                              stdout=subprocess.PIPE)
#         process_clang.wait()
# finally:
#     os.remove(path)
#
# print("done")


