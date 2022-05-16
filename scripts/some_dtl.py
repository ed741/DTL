from dtl import *
from dtlutils import visualize


def func1():
    i = Index("i")
    j = Index("j")
    k = Index("k")
    vsR1 = RealVectorSpace(1)
    vsA = UnknownSizeVectorSpace("A")
    vsB = UnknownSizeVectorSpace("B")
    print(vsR1)
    print(vsA)
    s1 = TensorSpace([vsR1, vsA])
    print(s1)
    # T1 = Lambda([A := TensorVariable(s1, "A")], deIndex(deIndex(A[j, i], [i])[i,] * deIndex(A[j, i], [j,])[i,], [i,]))
    # T2 = Lambda([A := TensorVariable("A"),B := TensorVariable("B")], (A[j, i]*Abs(B[j,i])|[i, j])[k]|[k])
    T1 = Lambda([A := TensorVariable(vsA * vsR1, "A"), B := TensorVariable(vsR1 * vsB, "B")], deIndex(MulBinOp(A[i,j], B[j,k], a="yes"), [i,k]))
    C = TensorVariable(vsR1 * vsB, "C")
    # A = TensorVariable(None, "A")
    # B = TensorVariable(None, "B")
    # T2 = Lambda([A, B], A[k].forall(k))
    print(str([j, i]))
    # print(str(T1))
    # print(str(T2))
    print(str(T1))
    print(T1.sub.space)
    print(T1.sub.scalar_expr.attributes)
    visualize.plot_dag(T1, view=True)


func1()