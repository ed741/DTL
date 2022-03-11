import functools
from typing import Any

import graphviz

# from dtl.ast import (Abs, astNode, TensorExpr, IndexedTensor, Literal, Lambda, Index,
#                      TensorVariable, deIndex)
from dtl.ast import *


@functools.singledispatch
def plot(node, dot):
    raise NotImplementedError


@plot.register
def plot_node(node: astNode, dot):
    label = str(node)
    dot.node(label)
    return label


@plot.register
def plot_indexed_tensor(node: IndexedTensor, dot: Any):
    label = str(node)
    dot.node(label)

    sub1 = plot(node.tensor, dot)
    for i in node.indices:
        dot.edge(label, plot(i, dot))

    dot.edge(label, sub1)

    return label


@plot.register
def plot_deindex(node: deIndex, dot: Any):
    label = str(node)
    dot.node(label)

    dot.edge(label, plot(node.tensor, dot))
    for i in node.indices:
        dot.edge(label, plot(i, dot))
    return label


@plot.register
def plot_binop(node: BinOp, dot: Any):
    label = str(node)
    dot.node(label)

    dot.edge(label, plot(node.lhs, dot))
    dot.edge(label, plot(node.rhs, dot))

    return label


@plot.register
def plot_lambda(node: Lambda, dot):
    label = str(node)
    dot.node(label)

    sub1s = [plot(v, dot) for v in node.vars]
    sub2 = plot(node.sub, dot)

    for s in sub1s:
        dot.edge(label, s)
    dot.edge(label, sub2)

    return label


if __name__ == "__main__":
    dot = graphviz.Digraph()
    i = Index("i")
    j = Index("j")
    k = Index("k")
    T1 = Lambda([A := TensorVariable("A")], deIndex(A[j, i], [i, j]))
    T2 = Lambda([A := TensorVariable("A"),B := TensorVariable("B")], (A[j, i]*Abs(B[j,i])|[i, j])[k]|[k])
    # A = TensorVariable("A")
    # B = TensorVariable("B")
    # T2 = Lambda([A, B], A[k] | [k])
    plot(T2, dot)

    print(dot.source)
    dot.render("plot.gv", view=True)
