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
from xdsl.dialects import memref, arith
from xdsl.dialects.builtin import Float64Type, f32, ModuleOp, StringAttr, IntegerAttr
from xdsl.dialects.func import FuncOp, Return
from xdsl.dialects.experimental import dlt
from xdsl.ir import SSAValue, Region
from xdsl.irdl import SingleBlockRegionDef
from xdsl.pattern_rewriter import PatternRewriteWalker, GreedyRewritePatternApplier
from xdsl.printer import Printer
from xdslDTL import compilec
from xdsl.transforms.experimental.lower_dtl import DTLDenseRewriter

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
expr = A[i,None].tupled_with(A[j, None]).deindex(((v10, i), (i, j, v10)))
# expr = A[i,None].tupled_with(A[j, None].deindex((j, v10))).deindex(((v10, i), (i, vQ, v10)))

builder = native.KernelBuilder(expr, debug_comments=False)
kernel = builder.build()
print(kernel)
# expr = ((A[i,j,k])).sum(j).forall(i,k)
# expr, _ = ExprTuple.tupleOf(expr, Literal(10)[i:v5]).tuple()
print("expr: "+str(expr))
print("expr type: "+str(expr.type))

type = xdtl.DTLType_to_xdtl(expr.type)
print("type")
print(type)

# module = ModuleOp([])
# block = module.body.block
block = ir.Block()
# a = block.insert_arg(xdsl.dialects.builtin.TensorType.from_type_and_list(f32, [5,5]), 0)
v10_len_attr = IntegerAttr.from_int_and_width(5,64)
vQ_len_attr = IntegerAttr.from_int_and_width(5,64)
vS_len_attr = IntegerAttr.from_int_and_width(6,64)

a = block.insert_arg(dlt.PtrType(dlt.TypeType([dlt.ElementAttr(tuple([dlt.SetAttr([]), dlt.SetAttr([dlt.DimensionAttr(tuple([StringAttr("vQ"), vQ_len_attr])), dlt.DimensionAttr(tuple([StringAttr("V10"), v10_len_attr]))]), f32]))])), 0)
# b = block.insert_arg(xdsl.dialects.builtin.TensorType.from_type_and_list(f32, [5,6]), 1)
b = block.insert_arg(dlt.PtrType(dlt.TypeType([dlt.ElementAttr(tuple([dlt.SetAttr([]), dlt.SetAttr([dlt.DimensionAttr(tuple([StringAttr("V10"), v10_len_attr])), dlt.DimensionAttr(tuple([StringAttr("vS"), vS_len_attr]))]), f32]))])), 1)
vS_len = arith.Constant(vS_len_attr)
vQ_len = arith.Constant(vQ_len_attr)
# vS_len = block.insert_arg(xdsl.dialects.builtin.i32, 2)
# vQ_len = block.insert_arg(xdsl.dialects.builtin.i32, 3)
# i_val = block.insert_arg(xdsl.dialects.builtin.i32, 4)
block.add_op(vS_len)
block.add_op(vQ_len)

out = block.insert_arg(dlt.PtrType(dlt.TypeType([dlt.ElementAttr(tuple([dlt.SetAttr([]), dlt.SetAttr([dlt.DimensionAttr(tuple([StringAttr("vQ"), vQ_len_attr])), dlt.DimensionAttr(tuple([StringAttr("vS"), vS_len_attr]))]), f32]))])), 2)
# mem = block.insert_arg(memref.MemRefType.from_element_type_and_shape(f32, (5,6)), 4)

# lines, output = xdtl.get_xdsl_dtl_version(expr, tensorVariables={A:a})
# lines, output = xdtl.get_xdsl_dtl_exec_version(expr, space_map={vS:vS_len, vQ:vQ_len}, arg_map={i:i_val}, tensor_variables={A:a,B:b}, output=mem)
exec, output = xdtl.get_xdsl_dtl_exec_version(expr, space_map={vS:SSAValue.get(vS_len), vQ:SSAValue.get(vQ_len)}, arg_map={}, tensor_variables={A:(a,["vQ", "V10"]),B:(b,["V10", "vS"])}, outputs=[(out,["vQ", "vS"])])
ret = Return()

#TEST BLOCK
# block = ir.Block()
# const_int32_1 = xdsl.dialects.arith.Constant.from_int_and_width(1,32)
# lines = [const_int32_1]
# ret = Return(const_int32_1)


block.add_ops(exec)
block.add_ops([output, ret])




region = Region([block])
argTypes = [arg.type for arg in block.args]
retTypes = [r.type for r in ret.operands]
# retTypes = []
func = FuncOp.from_region("foo", argTypes, retTypes, region)
# module = ModuleOp([func])
#
# print("lines:")
# p = printer.Printer()
# p.print(block)
# block.verify()
#
print("Module???")



print(func)

applier = PatternRewriteWalker(GreedyRewritePatternApplier(
    [DTLDenseRewriter()]),
    walk_regions_first=False)

applier.rewrite_module(func)

compilec.compile([func], "./tmp/libxdtl10.o")
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


