import graphviz

import dtl
from dtl import *
from dtlutils.traversal import postOrderRoute, get_scope


def plot_dag(expr: dtl.Node, *, name="expression", view=False, coalesce_duplicates=True, label_edges=False, **kwargs):
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
    _plot_dag(expr, dag, seen, coalesce_duplicates=coalesce_duplicates, label_edges=label_edges)
    dag.render(quiet_view=view)


def _plot_dag(expr: dtl.Node, dag: graphviz.Digraph, seen: typing.Dict, coalesce_duplicates=True, label_edges=False):
    if coalesce_duplicates and expr in seen:
        return seen[expr]
    
    name = str(id(expr))
    seen[expr] = name
    label = str(expr)
    dag.node(name, label=label)
    for i, o in enumerate(expr.operands):
        dag.edge(name, _plot_dag(o, dag, seen, coalesce_duplicates=coalesce_duplicates, label_edges=label_edges), label=str(i) if label_edges else "")
    return name


def plot_network(expr: Union[dtl.TensorExpr, dtl.Lambda], *, name="expression", view=False, **kwargs):
    if isinstance(expr, dtl.Lambda):
        expr = expr.tensor_expr
    network = graphviz.Digraph(name, **kwargs)
    _plot_network(expr, network)
    network.render(quiet_view=view)


def _plot_network(expr: dtl.TensorExpr, network):
    names = {}
    def add_vars(node, path):
        if isinstance(node, Index):
            name = f"I.{path}"
            scope = tuple(get_scope(expr, node, path))
            if not (node,scope) in names:
                names[(node, scope)] = name
                network.node(name, label=node.name, shape='circle')
        if isinstance(node, TensorVariable):
            name = f"TV.{path}"
            names[(node,path)] = name
            network.node(name, label=node.name, shape='box')
        return node
    postOrderRoute(expr, add_vars)
    def add_edges(node, path):
        if isinstance(node, IndexedTensor):
            operand_idx = list(node.operands).index(node.tensor_expr)
            for i, index in enumerate(node.tensor_indices):
                scope = tuple(get_scope(expr, index, path))
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
    postOrderRoute(expr, add_edges)
    if isinstance(expr, deIndex):
        for index in expr.indices:
            network.node(str(index)+"out", label=str(expr.index_spaces[index]), shape='none')
            network.edge(names[index,tuple([])], str(index)+"out", arrowhead='none')
    elif isinstance(expr, TensorVariable):
        for space in expr.tensor_space:
            network.node(str(space)+"out", label=str(space), shape='none')
            network.edge(names[expr,tuple([])], str(space)+"out", arrowhead='none')
            
    
    
    
