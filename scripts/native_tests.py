from dtl import *
from dtl.dag import RealVectorSpace, Index
from dtlpp.backends.native import KernelBuilder, InstantiationExprNode, SequenceExprNode
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
print("native_test.1")
# expr = Lambda((B:= (vs*vs).new("B"),), B[i,j].forall(i))
A = (v5*v9).new("A")
B = (v7*v5).new("B")
C = (v7*v3).new("C")
C_TT1 = C[i,j].forall(i,j)
C_TT2 = C[i,j].forall(i,j)
# expr = (((A[i,j] + (B[k,i] * (C_TT1)[k,p]).sum(k)).forall(j,p))[i,j]*(C_TT2)[k,j]).forall(i,k)

# expr = (A[i,j] * B[k,i]).forall(k,i)
# expr = (expr[k,i] * expr[k,i]).forall(i, k)
# expr = (A[i,j] * B[k,i]).sum(i).forall(j,k)
#[i,j].forall(i,j)
# t = ExprTuple((A[None,j].forall(j),B))[[i,j], [k,i]]
# a,b = InstantiationExprNode(t).tuple()
# expr = (a*b).sum(i).forall(j,k)

ts = InstantiationExprNode((A[None,i][j:i][k]*A[k,i]).forall(i,j))
t = ts
expr = ExprTuple((ts, ((t[i,j] * t[i,j] * B[p,k]).sum(k) * t[i,j]).forall(p,j,i).forall(k))).tuple()[1]

# expr = SequenceExprNode(t, ((t[i,j] * t[i,j] * B[p,k]).sum(k) * t[i,j]).forall(p,j,i)).forall(k)
# expr = IndexRebinding(A[i,j], [i,j], [k,p])

print(expr)
print(expr.type)
print("native_test.2")
visualise.plot_dag(expr, view=True, label_edges=True, short_strs=True)
# visualise.plot_dag(expr, view=True, label_edges=True, coalesce_duplicates=False)

builder = KernelBuilder(expr, debug_comments=True)
print("native_test.3")
kernel = builder.build()
print("native_test.4")
print(kernel)
print("native_test.5")
print("native_test.6")
# visualise.plot_dag(builder._expr, view=True, label_edges=True)
print("native_test.7")

input_tensor_A = np.ones([s.dim for s in A.type.result.dims])
input_tensor_B = np.ones([s.dim for s in B.type.result.dims])
input_tensor_C = np.ones([s.dim for s in C.type.result.dims])
output_tensor = np.zeros([s.dim for s in expr.type.result.dims])
out = kernel(A=input_tensor_A, B=input_tensor_B, C=input_tensor_C, ex_i=0)
print(out)
# print(output_tensor)

# print(func(np.ones([5,9]), np.ones([7,3]), np.ones([7,5])))