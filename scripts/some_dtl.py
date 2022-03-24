from dtl import *


def func1():
    i = Index("i")
    j = Index("j")
    k = Index("k")
    T1 = Lambda([A := TensorVariable("A", None)], deIndex(A[j, i], [i, j]))
    # T2 = Lambda([A := TensorVariable("A"),B := TensorVariable("B")], (A[j, i]*Abs(B[j,i])|[i, j])[k]|[k])
    A = TensorVariable("A", None)
    B = TensorVariable("B", None)
    T2 = Lambda([A, B], A[k].forall(k))
    print(str([j, i]))
    # print(str(T1))
    print(str(T2))
