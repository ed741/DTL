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

# ts = InstantiationExprNode((A[None,i][j:i][k]*A[k,i]).forall(i,j))
# t = ts
# expr = ExprTuple((ts, ((t[i,j] * t[i,j] * B[p,k]).sum(k) * t[i,j]).forall(p,j,i).forall(k))).tuple()[1]

# expr = SequenceExprNode(t, ((t[i,j] * t[i,j] * B[p,k]).sum(k) * t[i,j]).forall(p,j,i)).forall(k)
# expr = IndexRebinding(A[i,j], [i,j], [k,p])
N = RealVectorSpace(5)
n = Index("n")
F = RealVectorSpace(4)
f = Index("f")
C = RealVectorSpace(2)
c = Index("c")
s = Index("s")
sp = Index("sp")
cp = Index("cp")
E = RealVectorSpace(3)
e = Index("e")
l1 = Index("l1")
l2 = Index("l2")

vfm = (F*C*E*E).new("vfm")
bf = (F*C*E*E).new("bf")
ID = InstantiationExprNode(Literal(1)[i:C].forall(i,i))
D = InstantiationExprNode((Literal(1)[s:C,c:C] - ID[s,c]).forall(s,c))
D = SequenceExprNode(ID, D)
expr = SequenceExprNode(D, (vfm[f,cp,l1,l2] * ID[sp,s]*ID[cp,s]*D[c,cp]).sum(cp).forall(f,c,s,sp,l1,l2))


print(expr)
print(expr.type)
print("native_test.2")
# visualise.plot_dag(expr, view=True, label_edges=True, short_strs=True)
# visualise.plot_dag(expr, view=True, label_edges=True, coalesce_duplicates=False)

builder = KernelBuilder(expr, debug_comments=False)
print("native_test.3")
kernel = builder.build()


in_vfm = np.array([[[[11,12,13],[21,22,23],[31,32,33]],[[41,42,43],[51,52,53],[61,62,63]]],
                   [[[111,112,113],[121,122,123],[131,132,133]],[[141,142,143],[151,152,153],[161,162,163]]],
                   [[[211,212,213],[221,222,223],[231,232,233]],[[241,242,243],[251,252,253],[261,262,263]]],
                   [[[311,312,313],[321,322,323],[331,332,333]],[[341,342,343],[351,352,353],[361,362,363]]]])
in_D = np.array([[0,1],[1,0]])
out = kernel(vfm=in_vfm)
print(out)
(_f, _c, _s, _sp, _l1, l2) = out.shape
for i_f in range(_f):
    for i_c in range(_c):
        print(f"======= f:{i_f} c:{i_c} =======")
        for i_s in range(_s):
            for i_sp in range(_sp):
                print(f"f:{i_f} c:{i_c} s:{i_s} sp:{i_sp}")
                print(out[i_f, i_c, i_s, i_sp, :,:])
        
    
# print(output_tensor)

# print(func(np.ones([5,9]), np.ones([7,3]), np.ones([7,5])))