# import xdsl.dialects.arith
import ctypes
import time
import timeit
from random import Random, random

import numpy as np
from numpy.random import random

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
# matMul = (A[i,j]*(B[j,k].forall(k,j)[k,j])).sum(j).forall(i).forall(k)
matMul = (A[i,j]*(B[j,k])).sum(j).forall(i).forall(k)
# t1, t2, t3 = dag.ExprTuple([5, 4, 3]).tuple()
# q_vec = (t1+t2)[j:vQ].forall(j)
# matMul = (A[i,j]*B[j,k]+1+q_vec[i]).sum(j).forall(i).forall(k)


len_i, len_j, len_k = 1024, 1024, 1024
lib_builder = LibBuilder({v10: len_j})
lib_builder.make_dummy("test", 3)
lib_builder.make_init("init_AB", (A,B), [vQ, vS])
# lib_builder.make_init("init_A", A, [vQ])
# lib_builder.make_init("init_B", B, [vS])
lib_builder.make_setter("set_Out", output_t_var, {}, [0, 1])
lib_builder.make_setter("set_A", A, {}, [0, 1])
lib_builder.make_setter("set_B", B, {}, [0, 1])
lib_builder.make_getter("get_Out", output_t_var, {}, [0,1])
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

ab_root, (a, b) = lib.init_AB(len_i, len_k)
print("inited A")
out_root, out = lib.init_Out(len_i, len_k)
print("inited a,b,out")

print("a:")
# lib.print_A(a)
print("b:")
# lib.print_B(b)
print("out:")
# lib.print_Out(out)

np_a = np.zeros((len_i, len_j))
np_b = np.zeros((len_j, len_k))
r = Random(1)
# lib.set_AB(a,b)
a_non_zeros = 0
b_non_zeros = 0
for i in range(len_i):
    for j in range(len_j):
        if r.random() < 0.001:
            r_val = r.random()
            lib.set_A(a, i, j, r_val)
            np_a[i,j] = r_val
            a_non_zeros += 1
for j in range(len_j):
    for k in range(len_k):
        if r.random() < 0.001:
            r_val = r.random()
            lib.set_B(b , j, k, r_val)
            np_b[j, k] = r_val
            b_non_zeros += 1
print("A&B set")
print(np_a)
print(np_b)
print(f"a non_zeros: {a_non_zeros}")
print(f"b non_zeros: {b_non_zeros}")

print("a:")
# lib.print_A(a)
print("b:")
# lib.print_B(b)
print("out:")
# lib.print_Out(out)
np_out = np.zeros((len_i, len_k))
def np_bench():
    np.matmul(np_a, np_b, out=np_out)
np_result = timeit.timeit(np_bench, number=1)
print(np_out)
print(f"np_time: {np_result}")

def benchmark():
    lib.mm(out, a, b)

print("Start benchmark")
result = timeit.timeit(benchmark, number=1)
print("Done Matrix Mul")

print("a:")
# lib.print_A(a)
print("b:")
# lib.print_B(b)
print("out:")
# lib.print_Out(out)

total_error = 0
epsilon = 1e-5
within_epsilon = True
for i_i in range(len_i):
    for i_k in range(len_k):
        np_num = np_out[i_i, i_k]
        res = lib.get_Out(out, i_i, i_k).value
        error = abs(res - np_num)
        total_error += error
        normalised_epsilon = np_num * epsilon
        if error > normalised_epsilon:
            print(
                f"Result miss match! at i: {i_i}, k: {i_k}, np_c = {np_num}, c = {res}, error = {res - np_num}, epsilon(abs) = {epsilon}, epsilon(norm) = {normalised_epsilon}"
            )
            within_epsilon = False


print(f"within_epsilon: {within_epsilon}")
print(f"total_error: {total_error}")
print("timeit result:")
print(result)
