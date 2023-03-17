import math

from dtl import *
from dtlutils import visualise
from dtlutils.names import make_Index_names_unique
from dtlutils.traversal import postOrder


def func1():
    ii, i = Index("i"), Index("i")
    j = Index("j")
    k = Index("k")
    vsR1 = RealVectorSpace(1)
    vsA = UnknownSizeVectorSpace("A")
    vsB = UnknownSizeVectorSpace("B")
    vsC = UnknownSizeVectorSpace("C")
    print(vsR1)
    print(vsA)
    s1 = TensorSpace([vsR1, vsA])
    print(s1)
    # T1 = Lambda([A := TensorVariable(s1, "A")], deIndex(deIndex(A[j, i], [i])[i,] * deIndex(A[j, i], [j,])[i,], [i,]))
    # T2 = Lambda([A := TensorVariable("A"),B := TensorVariable("B")], (A[j, i]*Abs(B[j,i])|[i, j])[k]|[k])
    T1 = Lambda([A := TensorVariable(vsA * vsR1, "A"), B := TensorVariable(vsR1 * vsB, "B")], deIndex(MulBinOp(A[i,j], B[j,k], attrs={"check":"yes"}), [i,k]))
    C = TensorVariable(vsR1 * vsB, "C")
    # A = TensorVariable(None, "A")
    # B = TensorVariable(None, "B")
    # T2 = Lambda([A, B], A[k].forall(k))
    print(str([j, i]))
    # print(str(T1))
    # print(str(T2))
    print(str(T1))
    print(T1.tensor_expr.space)
    print(T1.tensor_expr.scalar_expr.attrs)

    def printNodeName(node):
        # print(str(node))
        if isinstance(node, MulBinOp):
            if isinstance(node.lhs, IndexedTensor) and isinstance(node.lhs.tensor_expr, TensorVariable):
                print("waaaaaaaaaaa")
                return node.copy(lhs=node.rhs, rhs=node.lhs)
        return node.copy()
    print("Post order:")
    T1 = postOrder(T1, printNodeName)
    visualise.plot_dag(T1, view=True, coalesce_duplicates=True)
    # visualise.plot_dag(T1, view=True, coalesce_duplicates=True)

def mttkrp():
    vsI = UnknownSizeVectorSpace("I")
    vsJ = UnknownSizeVectorSpace("J")
    vsK = UnknownSizeVectorSpace("K")
    vsL = UnknownSizeVectorSpace("L")
    i = Index("i")
    j = Index("j")
    k = Index("k")
    l = Index("l")
    
    mttkrp_expr = Lambda([TB:=(vsI*vsK*vsL).new("TB"), TC:= (vsL*vsJ).new("TC"), TD:= (vsK*vsJ).new("TD")],
                         (TB[i,k,l]*TC[l,j]*TD[k,j]).forall(i,j))
    visualise.plot_dag(mttkrp_expr, view=True, coalesce_duplicates=True)
    visualise.plot_network(mttkrp_expr, view=True)


def tucker():
    vsI = UnknownSizeVectorSpace("I")
    vsJ = UnknownSizeVectorSpace("J")
    vsK = UnknownSizeVectorSpace("K")
    vsP = UnknownSizeVectorSpace("P")
    vsQ = UnknownSizeVectorSpace("Q")
    vsR = UnknownSizeVectorSpace("R")
    i = Index("i")
    j = Index("j")
    k = Index("k")
    p = Index("p")
    q = Index("q")
    r = Index("r")
    
    mttkrp_expr = Lambda([TG := (vsP * vsQ * vsR).new("TG"), TA := (vsI * vsP).new("TA"), TB := (vsJ * vsQ).new("TB"), TC := (vsK * vsR).new("TC")],
                         (TG[p, q, r] * TA[i, p] * TB[j, q] * TC[k, r]).forall(i, j, k))
    visualise.plot_dag(mttkrp_expr, view=True, coalesce_duplicates=True)
    visualise.plot_network(mttkrp_expr, view=True)


def func2():
    vsI = UnknownSizeVectorSpace("I")
    vsJ = UnknownSizeVectorSpace("J")
    vsK = UnknownSizeVectorSpace("K")
    # vsL = UnknownSizeVectorSpace("L")
    i = Index("i")
    j = Index("j")
    k = Index("k")
    
    l = Index("l")
    
    expr = Lambda([TA := (vsI * vsJ * vsK).new("TA"), TB := (vsI * vsJ).new("TB"), TC := (vsK * vsJ).new("TC")],
                         ((TA[i, j, k] * TB[i, j] * TC[k, j]).forall(i, j)[l].forall(l)[i] * TA[i,j,k]).forall(j,k))
    # visualise.plot_dag(expr, view=True, coalesce_duplicates=True)
    visualise.plot_network(expr, view=True)


def func3():
    vsI = UnknownSizeVectorSpace("I")
    vsJ = UnknownSizeVectorSpace("J")
    # vsK = UnknownSizeVectorSpace("K")
    # vsL = UnknownSizeVectorSpace("L")
    i = Index("i")
    j = Index("j")
    k = Index("k")
    
    l = Index("l")
    
    expr = Lambda([TA := (vsI * vsJ * vsJ).new("TA"), TB := (vsI * vsJ * vsJ).new("TB"), TC := (vsJ * vsJ).new("TC")],
                  ((IndexSum(TA[l,j,k] * TA[l,j,k], [l]) * IndexSum(TA[i,j,k] * TA[i,j,k], [i])).forall(k)))
    expr2 = make_Index_names_unique(expr)
    visualise.plot_dag(expr2, view=True, coalesce_duplicates=True)
    # visualise.plot_network(expr.tensor_expr.scalar_expr.scalar_expr.lhs.tensor_expr, view=True)
    # visualise.plot_network(expr.tensor_expr, view=True)


# func1()
# mttkrp()
# tucker()
# func2()
func3()

