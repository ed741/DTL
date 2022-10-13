from dtl import *
from dtl.dag import RealVectorSpace, Index, Lambda
from dtlpp.backends.native import KernelBuilder
import numpy as np

from dtlutils import visualize

v5 = RealVectorSpace(5)
v9 = RealVectorSpace(9)
v7 = RealVectorSpace(7)
v3 = RealVectorSpace(3)
vu = UnknownSizeVectorSpace("u1")
i = Index("i")
j = Index("j")
k = Index("k")
p = Index("p")
print("native_test.1")
# expr = Lambda((B:= (vs*vs).new("B"),), B[i,j].forall(i))
A = (v5*v9).new("A")
B = (v7*v5).new("B")
C = (v7*v3).new("C")
C_TT1 = C[i,j].forall(i,j)
C_TT2 = C[i,j].forall(i,j)
expr = (((A[i,j] + (B[k,i] * (C_TT1)[k,p]).sum(k)).forall(j,p))[i,j]*(C_TT2)[k,j]).forall(i,k)
# expr = (A[i,j] * B[k,i]).forall(j,k)
print("native_test.2")
# visualize.plot_dag(expr, view=True, label_edges=True)
visualize.plot_network(expr, view=True)

builder = KernelBuilder(expr)
print("native_test.3")
kernel = builder.build()
print("native_test.4")
print(kernel)
print("native_test.5")
print("native_test.6")
visualize.plot_dag(builder._expr, view=True, label_edges=True)
print("native_test.7")

input_tensor_A = np.ones(A.space.shape)
input_tensor_B = np.ones(B.space.shape)
input_tensor_C = np.ones(C.space.shape)
output_tensor = np.zeros(expr.space.shape)
out = kernel(A=input_tensor_A, B=input_tensor_B, C=input_tensor_C)
print(out)
# print(output_tensor)

# print(func(np.ones([5,9]), np.ones([7,3]), np.ones([7,5])))