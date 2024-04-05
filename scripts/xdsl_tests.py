# import xdsl.dialects.arith
import ctypes

from dtl import *
from dtl.dag import RealVectorSpace, Index
import dtlpp.backends.xdsl as xdtl

from dtl.libBuilder import LibBuilder
from dtlpp.backends import native

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

set_AB = dag.ExprTuple((Expr.exprInputConversion(2)[i:vQ, j:v10].forall(i,j), Expr.exprInputConversion(3)[j:v10, k:vS].forall(j,k)))
matMul = (A[i,j]*B[j,k]).sum(j).forall(i,k)

lib_builder = LibBuilder()
lib_builder.make_dummy("test", 3)
lib_builder.make_init("init_A", A, [vQ])
lib_builder.make_init("init_B", B, [vS])
# lib_builder.make_init("init_AB", (A,B), [vQ, vS])
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
lib = lib_builder.build()

print(lib.test())
a_root, a = lib.init_A(3)
b_root, b = lib.init_B(5)

# ab_root, (a, b) = lib.init_AB(3,5)
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
lib.print_Out(out)

lib.mm(out, a, b)
print("Matrix Mul")


print("a:")
lib.print_A(a)
print("b:")
lib.print_B(b)
print("out:")
lib.print_Out(out)

