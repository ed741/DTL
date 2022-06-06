from dtl import *
from dtlutils import visualize
    
    
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
    visualize.plot_dag(T1, view=True, coalesce_duplicates=True)
    # visualize.plot_dag(T1, view=True, coalesce_duplicates=True)

def mttkrp():
    vsI = UnknownSizeVectorSpace("I")
    vsJ = UnknownSizeVectorSpace("J")
    vsK = UnknownSizeVectorSpace("K")
    vsL = UnknownSizeVectorSpace("L")
    i = Index("i")
    j = Index("j")
    k = Index("k")
    l = Index("l")
    
    mttkrp_expr = Lambda([TB:=(vsI*vsK*vsL).new("TB"), TC:= (vsL*vsJ).new("TC"), TD:= (vsK*vsJ).new("TD")], (TB[i,k,l]*TC[l,j]*TD[k,j]).forall(i,j))
    visualize.plot_dag(mttkrp_expr, view=True, coalesce_duplicates=True)
    
    
def postOrder(node: Node, fn):
    operands = node.operands
    new_operands = [postOrder(n, fn) for n in operands]
    return fn(node.with_operands(new_operands))


# func1()
mttkrp()