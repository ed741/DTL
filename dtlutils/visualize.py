import functools
import typing
from typing import Any

import graphviz

import dtl
from dtl import *
from dtlutils.traversal import postOrder, forAll, postOrderRoute, node_from_path


def plot_dag(expr: dtl.Node, *, name="expression", view=False, coalesce_duplicates=True, **kwargs):
    """Render loop expression as a DAG and write to a file.

    Parameters
    ----------
    expr : pyop3.Expression
        The loop expression.
    name : str, optional
        The name of DAG (and the save file).
    view : bool, optional
        Should the rendered result be opened with the default application?
    **kwargs : dict, optional
        Extra keyword arguments passed to the `graphviz.Digraph` constructor.
    """
    dag = graphviz.Digraph(name, **kwargs)
    seen = {}
    _plot_dag(expr, dag, seen, coalesce_duplicates=coalesce_duplicates)
    dag.render(quiet_view=view)


def _plot_dag(expr: dtl.Node, dag: graphviz.Digraph, seen: typing.Dict, coalesce_duplicates=True):
    if coalesce_duplicates and expr in seen:
        return seen[expr]
    
    name = str(id(expr))
    seen[expr] = name
    label = str(expr)
    dag.node(name, label=label)
    for o in expr.operands:
        dag.edge(name, _plot_dag(o, dag, seen, coalesce_duplicates=coalesce_duplicates))
    return name


def plot_network(expr: dtl.Lambda, *, name="expression", view=False, **kwargs):
    network = graphviz.Digraph(name, **kwargs)
    seen = {}
    _plot_network(expr, network)
    network.render(quiet_view=view)

def _get_scope(node, index, path):
    """ return node(deIndex or IndexSum) and path of the enclosing scope of the index"""
    paths = [[path[i] for i in range(k)] for k in range(len(path)+1)]
    # print("\n".join(str(p) for p in paths))
    for p in reversed(paths):
        current = node_from_path(node, p)
        if isinstance(current, deIndex) and index in current.indices:
            return p
        if isinstance(current, IndexSum) and index in current.sum_indices:
            return p
    return None
    

def _plot_network(expr: dtl.Lambda, network):
    names = {}
    def add_vars(node, path):
        if isinstance(node, Index):
            name = f"I.{path}"
            scope = tuple(_get_scope(expr.tensor_expr, node, path))
            if not (node,scope) in names:
                names[(node, scope)] = name
                network.node(name, label=node.name, shape='circle')
        if isinstance(node, TensorVariable):
            name = f"TV.{path}"
            names[(node,path)] = name
            network.node(name, label=node.name, shape='box')
        return node
    postOrderRoute(expr.tensor_expr, add_vars)
    def add_edges(node, path):
        if isinstance(node, IndexedTensor):
            operand_idx = list(node.operands).index(node.tensor_expr)
            for i, index in enumerate(node.tensor_indices):
                scope = tuple(_get_scope(expr.tensor_expr, index, path))
                if isinstance(node.tensor_expr, TensorVariable):
                    keyA = node.tensor_expr, tuple(list(path)+[operand_idx])
                    keyB = index, scope
                    network.edge(names[keyA], names[keyB], arrowhead='none')
                if isinstance(node.tensor_expr, deIndex):
                    keyA = node.tensor_expr.indices[i], tuple(list(path)+[operand_idx])
                    keyB = index, scope
                    if keyA not in names:
                        print("dam")
                    if keyB not in names:
                        print("dam")
                    network.edge(names[keyA], names[keyB], arrowhead='none')
        return node
    postOrderRoute(expr.tensor_expr, add_edges)
    if isinstance(expr.tensor_expr, deIndex):
        for index in expr.tensor_expr.indices:
            network.node(str(index)+"out", label=str(expr.tensor_expr.index_spaces[index]), shape='none')
            network.edge(names[index,tuple([])], str(index)+"out", arrowhead='none')
    elif isinstance(expr.tensor_expr, TensorVariable):
        for space in expr.tensor_expr.tensor_space:
            network.node(str(space)+"out", label=str(space), shape='none')
            network.edge(names[expr.tensor_expr,tuple([])], str(space)+"out", arrowhead='none')
            
    
    
    
