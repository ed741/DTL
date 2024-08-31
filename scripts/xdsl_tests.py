# import xdsl.dialects.arith
import ctypes
import time
import timeit

from dtl import *
from dtl.dag import RealVectorSpace, Index
import dtlpp.backends.xdsl as xdtl

from dtl.libBuilder import LibBuilder
from dtlpp.backends import native
from xdsl.dialects import arith, scf, func
from xdsl.dialects.builtin import f32, IntegerAttr, IntegerType
from xdsl.dialects.experimental import dlt
from xdsl.ir import Block, Region

i = Index('i')
j = Index('j')
k = Index('k')
v10 = UnknownSizeVectorSpace("10")
vS = UnknownSizeVectorSpace("S")
vQ = UnknownSizeVectorSpace("Q")
# v6 = RealVectorSpace(6)
# vu = UnknownSizeVectorSpace("VU")
A = TensorVariable(vQ*v10, "A")
B = TensorVariable(v10*vS, "B")
output_t_var = TensorVariable(vQ*vS, "out")

set_AB = dag.ExprTuple((Expr.exprInputConversion(2)[i:vQ, j:v10].forall(i,j), Expr.exprInputConversion(3)[j:v10, k:vS].forall(j,k)))
matMul = (A[i,j]*(B[j,k].forall(k,j)[k,j])).sum(j).forall(i).forall(k)
# matMul = (A[i,j]*(B[j,k])).sum(j).forall(i).forall(k)
# t1, t2, t3 = dag.ExprTuple([5, 4, 3]).tuple()
# q_vec = (t1+t2)[j:vQ].forall(j)
# matMul = (A[i,j]*B[j,k]+1+q_vec[i]).sum(j).forall(i).forall(k)

lib_builder = LibBuilder({v10: 10})
lib_builder.make_dummy("test", 3)
lib_builder.make_init("init_AB", (A,B), [vQ, vS])
# lib_builder.make_init("init_A", A, [vQ])
# lib_builder.make_init("init_B", B, [vS])
lib_builder.make_setter("set_Out", output_t_var, {}, [0, 1])
lib_builder.make_init("init_Out", output_t_var, [vQ, vS])
lib_builder.make_print_tensorVar("print_A", A, [])
lib_builder.make_print_tensorVar("print_B", B, [])
lib_builder.make_print_tensorVar("print_Out", output_t_var, [])
lib_builder.make_function("set_AB", set_AB,
                         [A, B],
                         [],
                         [],
                         []
                         )
lib_builder.make_function("mm", matMul,
                         [output_t_var],
                         [A,B],
                         [],
                         []
                         )
# block = Block()
# block.add_op(alloc_1 := dlt.AllocOp(dlt.TypeType([dlt.ElementAttr([],[], f32)]), {}))
# block.add_op(sel1 := dlt.SelectOp(alloc_1, [],[],[]))
# block.add_op(alloc_2 := dlt.AllocOp(dlt.TypeType([dlt.ElementAttr([],[], f32)]), {}))
# block.add_op(sel2 := dlt.SelectOp(alloc_2, [],[],[]))
# block.add_op(const := arith.Constant(IntegerAttr(0, IntegerType(1))))
# block.add_op(if_op := scf.If(const, [sel1.res.type], [scf.Yield(sel1.res)],[scf.Yield(sel2.res)]))
# block.add_op(func.Return(if_op.output[0]))
# func_op = func.FuncOp("test2", ([],[sel1.res.type]), Region(block))
# lib_builder.funcs.append(func_op)
lib = lib_builder.build(verbose=3)

print("Try Tests")
print(lib.test())
print("Done lib.test()")
# a_root, a = lib.init_A(3)
# b_root, b = lib.init_B(5)

ab_root, (a, b) = lib.init_AB(3, 5)
print("inited A")
out_root, out = lib.init_Out(3, 5)
print("inited a,b,out")

print("a:")
lib.print_A(a)
print("b:")
lib.print_B(b)
print("out:")
lib.print_Out(out)

lib.set_AB(a,b)
print("A&B set")

print("a:")
lib.print_A(a)
print("b:")
lib.print_B(b)
print("out:")
# lib.print_Out(out)


def benchmark():
    lib.mm(out, a, b)


result = timeit.timeit(benchmark, number=1)
print("Matrix Mul")

print("a:")
# lib.print_A(a)
print("b:")
# lib.print_B(b)
print("out:")
lib.print_Out(out)

print("timeit result:")
print(result)
