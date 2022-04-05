import functools

from dtl.dag import *


@functools.singledispatch
def binarise(node: Node):
    raise AssertionError


@binarise.Lambda
def binarise_Lambda(node: Lambda):
    return binarise(node.sub)


@binarise.deIndex
def binarise_deIndex(node: deIndex):
    return binarise(node.scalar_expr)


""" When we have ] A[i,j,k] * B[j,k,l] * C[k,q] [i,l,q
    we must insert a deindex-reindex pair over A[..] * B[..]
    So:
    D = ] A[i,j,k] * B[j,k,l] [X
    in ] D[X] * C[k,q] [i,l,q
    
    Here X must be all the indicies in A[..] * B[..] that are also in ] D *C[..] [..
        """
