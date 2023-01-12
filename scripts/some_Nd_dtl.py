import numpy as np

from dtl.dag_ext_non_scalar import *
from dtlpp.backends.native import KernelBuilder

dag_ext_non_scalar_init()
from dtlutils import visualise, ndVisualise
from dtlutils.names import make_Index_names_unique


def func1():
    i = Index("i")
    j = Index("j")
    k = Index("k")
    l = Index("l")
    vsR1 = RealVectorSpace(10)
    vsR2 = RealVectorSpace(2)
    vsR3 = RealVectorSpace(3)
    vsA = UnknownSizeVectorSpace("A")
    vsB = UnknownSizeVectorSpace("B")
    vsC = UnknownSizeVectorSpace("C")
    
    A = TensorVariable(vsR3*vsR3*vsR2, "A")
    B = TensorVariable(vsR1 * vsR2, "B")
    
    # ai =  A[i,j,k]
    # ndExpr = IndexedNDTensor(ai, [i,j])
    ndExpr = A[i,j,k].expose([i,j])
    ndExpr = MatrixInverseNdExprOp(ndExpr)
    ndExpr = NDScalarExpr(ndExpr)
    ndExpr = B[l,k] * ndExpr
    # ndExpr = ndExpr.expose([k])
    # ndExpr = MaxNdExprOp(ndExpr)
    # ndExpr = NDScalarExpr(ndExpr)
    ndExpr = ndExpr.forall(l)
    
    
    print(ndExpr)
    visualise.plot_dag(ndExpr, view=True, coalesce_duplicates=True, label_edges=True)
    ndExpr2 = make_Index_names_unique(ndExpr)
    ndVisualise.plot_nd_network(ndExpr2, view=True)

    # visualise.plot_dag(ndExpr, view=True, coalesce_duplicates=True)

    builder = KernelBuilder(ndExpr2)
    kernel = builder.build()
    print(kernel)
    visualise.plot_dag(builder._expr, view=True, label_edges=True)

    input_tensor_A = np.ones(A.space.shape)
    input_tensor_B = np.ones(B.space.shape)
    output_tensor = np.zeros(ndExpr2.space.shape)
    out = kernel(A=input_tensor_A, B=input_tensor_B)
    print(out)
    
func1()