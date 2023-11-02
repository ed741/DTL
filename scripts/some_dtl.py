from dtl import *
from dtl.dtlutils import visualise
from dtl.passes.names import make_Index_names_unique
from dtl.dtlutils.traversal import postOrder
from dtlpp.backends import native


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
    A = TensorVariable(vsA * vsR1, "A")
    B = TensorVariable(vsR1 * vsB, "B")
    # T1 = DeindexExpr((A[i,j] * B[j,k]), (i,k))
    T1 = (A[i,j] * B[j,k]).forall(i,k)
    C = TensorVariable(vsR1 * vsB, "C")
    # A = TensorVariable(None, "A")
    # B = TensorVariable(None, "B")
    # T2 = Lambda([A, B], A[k].forall(k))
    print(str([j, i]))
    # print(str(T1))
    # print(str(T2))
    print(str(T1))

    def printNodeName(node):
        # print(str(node))
        if isinstance(node, MulBinOp):
            if isinstance(node.lhs, IndexedExprTuple) and isinstance(node.lhs.expr, TensorVariable):
                print("waaaaaaaaaaa")
                return node.copy(lhs=node.rhs, rhs=node.lhs)
        return node.copy()
    print("Post order:")
    T1 = postOrder(T1, printNodeName)
    # visualise.plot_dag(T1, view=True, coalesce_duplicates=True)
    visualise.plot_network(T1, view=True)

def mttkrp():
    vsI = UnknownSizeVectorSpace("I")
    vsJ = UnknownSizeVectorSpace("J")
    vsK = UnknownSizeVectorSpace("K")
    vsL = UnknownSizeVectorSpace("L")
    i = Index("i")
    j = Index("j")
    k = Index("k")
    l = Index("l")
    TB = (vsI * vsK * vsL).new("TB")
    TC = (vsL * vsJ).new("TC")
    TD = (vsK * vsJ).new("TD")
    mttkrp_expr = (TB[i,k,l]*TC[l,j]*TD[k,j]).forall(i,j)
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

    TG = (vsP * vsQ * vsR).new("TG")
    TA = (vsI * vsP).new("TA")
    TB = (vsJ * vsQ).new("TB")
    TC = (vsK * vsR).new("TC")
    mttkrp_expr = (TG[p, q, r] * TA[i, p] * TB[j, q] * TC[k, r]).forall(i, j, k)
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
    
    TA = (vsI * vsJ * vsK).new("TA")
    TB = (vsI * vsJ).new("TB")
    TC = (vsK * vsJ).new("TC")
    expr = ((TA[i, j, k] * TB[i, j] * TC[k, j]).forall(i, j)[l].forall(l)[i] * TA[i,j,k]).forall(j,k)
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
    
    TA = (vsI * vsJ * vsJ).new("TA")
    TB = (vsI * vsJ * vsJ).new("TB")
    TC = (vsJ * vsJ).new("TC")
    
    expr = ((IndexSum(TA[l,j,k] * TA[l,j,k], [l]) * IndexSum(TA[i,j,k] * TA[i,j,k], [i])).forall(k))
    expr2 = make_Index_names_unique(expr)
    visualise.plot_dag(expr2, view=True, coalesce_duplicates=True)
    # visualise.plot_network(expr.tensor_expr.scalar_expr.scalar_expr.lhs.tensor_expr, view=True)
    # visualise.plot_network(expr.tensor_expr, view=True)

def func4():
    i, j, k = Index('i'), Index('j'), Index('k')
    R10 = RealVectorSpace(10)
    Q, S = UnknownSizeVectorSpace('Q'), UnknownSizeVectorSpace('S')
    A, B = TensorVariable(Q * R10, 'A'), TensorVariable(R10 * S, 'B')
    output = (A[i, j] * B[j, k]).sum(j).forall(i, k)
    output = MulBinOp(A.bind({i:Q, j:R10}).index([i,j]).bind({k:S}), B.bind({j:R10, k:S}).index([j,k]).bind({i:Q})).sum(j).deindex((i,k))

    print("A:", A.type)
    print("A[i,j]:", A[i,j].type)
    print("(A[rs, j] * B[j, k]):", (A[i, j] * B[j, k]).type)
    print("(A[i, j] * B[j, k]).sum(j):", (A[i, j] * B[j, k]).sum(j).type)
    print("output:", output.type)
    print("A,B:", ExprTuple((A,B)).type)
    print("A[i, j], B:", Expr.exprInputConversion((A[i, j], B)).type)


    visualise.plot_dag(output, view=True, short_strs=True, skip_terminals=True, label_edges=True)

    builder = native.KernelBuilder(output, debug_comments=False)
    print("native_test.3")
    kernel = builder.build()


# mttkrp()
# func1()
# tucker()
# func2()
# func3()
func4()
