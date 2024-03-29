from dtl import *
from dtl.dag import RealVectorSpace, Index
from dtlpp.backends.loopy import KernelBuilder
import loopy as lp
import numpy as np

from dtl.dtlutils import visualise

v5 = RealVectorSpace(5)
v9 = RealVectorSpace(9)
v7 = RealVectorSpace(7)
v3 = RealVectorSpace(3)
vu = UnknownSizeVectorSpace("u1")
i = Index("i")
j = Index("j")
k = Index("k")
p = Index("p")
print("loopy_test.1")
# expr = Lambda((B:= (vs*vs).new("B"),), B[i,j].forall(i))
A = (v5*v9).new("A")
B = (v7*v5).new("B")
C = (v7*v3).new("C")
C_TT1 = C[i,j].forall(i,j)
C_TT2 = C[i,j].forall(i,j)
expr = (((A[i,j] + (B[k,i] * (C_TT1)[k,p]).sum(k)).forall(j,p))[p,j]*(C_TT2)[k,j]).forall(i,k)
expr = (C[i,j]+4).forall(i,j)
print("loopy_test.2")
visualise.plot_dag(expr, view=True, label_edges=True)

builder = KernelBuilder(expr)
print("loopy_test.3")
kernel = builder.build()
print("loopy_test.4")
print(kernel)
print("loopy_test.5")
print(lp.generate_code_v2(kernel).device_code())  # the C code
print("loopy_test.6")
visualise.plot_dag(builder._expr, view=True, label_edges=True)
print("loopy_test.7")

input_tensor_A = np.ones(A.space.shape)
input_tensor_B = np.ones(B.space.shape)
input_tensor_C = np.ones(C.space.shape)
output_tensor = np.zeros(expr.space.shape)
# kernel(P_=output_tensor, A=input_tensor_A, B=input_tensor_B, C=input_tensor_C)
kernel(P_=output_tensor, C=input_tensor_C)

print(output_tensor)