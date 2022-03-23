import functools
from typing import Any

import graphviz

# from dtl.ast import (Abs, astNode, TensorExpr, IndexedTensor, Literal, Lambda, Index,
#                      TensorVariable, deIndex)
from dtl.dag import *


@functools.singledispatch
def plot(node, dot):
    raise NotImplementedError


@plot.register
def plot_node(node: Node, dot):
    label = str(node)
    dot.node(label)

    for o in node.operands:
        dot.edge(label, plot(o, dot))

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


def visualize_dag(expr):
    dag = graphviz.Digraph()
    plot(expr, dag)
    return dag



# if __name__ == "__main__":
#     dot = graphviz.Digraph()
#     i = Index("i")
#     j = Index("j")
#     k = Index("k")
#     expr = Lambda(
#         [A := TensorVariable("A"), B := TensorVariable("B")],
#         (A[j, i] * Abs(B[j, i]) | [i, j])[k] | [k],
#     )
#
#     matmul = Lambda(
#         [A := TensorVariable("A")
#
#     plot(T2, dot)
#
#     print(dot.source)
#     dot.render("plot.gv", view=True)
