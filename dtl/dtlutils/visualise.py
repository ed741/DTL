import graphviz

import dtl
from dtl import *
from dtl.dtlutils import traversal
from dtl.dtlutils.traversal import postOrderPath, get_scope, allOperandsLabelled


def plot_dag(expr: typing.Union[dtl.Node, typing.Iterable[dtl.Node]], *, name="expression",
             view=False,
             coalesce_duplicates=True,
             label_edges=False,
             short_strs=False,
             skip_terminals=False,
             show_types=False, **kwargs):
    """Render loop expression as a DAG and write to a file.

    Parameters
    ----------
    expr : dtl.Node
        The tensor expression.
    name : str, optional
        The name of DAG (and the save file).
    view : bool, optional
        Should the rendered result be opened with the default application?
    **kwargs : dict, optional
        Extra keyword arguments passed to the `graphviz.Digraph` constructor.
    """
    dag = graphviz.Digraph(name, **kwargs)
    seen = {}
    if not isinstance(expr, Iterable):
        expr = [expr]
    for e in expr:
        _plot_dag(e, dag, seen,
                  coalesce_duplicates=coalesce_duplicates,
                  label_edges=label_edges,
                  short_strs=short_strs,
                  skip_terminals=skip_terminals,
                  show_types=show_types)
    dag.render(view=view)


def _plot_dag(expr: dtl.Node, dag: graphviz.Digraph, seen: typing.Dict,
              coalesce_duplicates=True,
              label_edges=False,
              short_strs=False,
              skip_terminals=False,
              show_types=False):
    if coalesce_duplicates and expr in seen:
        return seen[expr]
    
    name = str(id(expr))
    seen[expr] = name
    label = expr.shortStr() if short_strs else str(expr)
    if show_types and isinstance(expr, dtl.Expr):
        label += ' :: ' + str(expr.type)
    dag.node(name, label=label)
    for l, o in allOperandsLabelled(expr.operands):
        if skip_terminals and isinstance(o, Terminal):
            continue
        childName = _plot_dag(o, dag, seen,
                              coalesce_duplicates=coalesce_duplicates,
                              label_edges=label_edges,
                              short_strs=short_strs,
                              skip_terminals=skip_terminals,
                              show_types=show_types)
        dag.edge(name, childName, label=str(l) if label_edges else "")
    return name


def plot_network(expr: dtl.Expr, *, name="expression", view=False, **kwargs):
    network = graphviz.Digraph(name, **kwargs)
    _plot_network(expr, network)
    network.render(view=view)


def _plot_network(expr: dtl.Expr, network):
    names = {}
    def add_vars(node, path):
        path = tuple(path)
        if isinstance(node, Index):
            name = f"I.{path}"
            scope = get_scope(expr, node, path)
            if scope is None:
                scope = tuple()
            else:
                scope = tuple(scope)
            if scope is None or (not (node,tuple(scope)) in names):
                network.node(name, label=node.name, shape='diamond' if scope is None else 'circle')
                if scope is not None:
                    names[(node, tuple(scope))] = name
        if isinstance(node, TensorVariable):
            name = f"TV.{path}"
            names[(node,path)] = name
            network.node(name, label=node.name, shape='box')
        return node
    postOrderPath(expr, add_vars)
    def add_edges(node, path):
        if isinstance(node, IndexExpr):
            operand_idx = 'expr' #list(node.operands).index(node.expr)
            for i, index in enumerate(node.indices):
                s = get_scope(expr, index, path)
                scope = tuple(s) if s is not None else tuple([])
                node_expr = node.expr
                while(isinstance(node_expr, IndexBinding)): node_expr = node_expr.expr
                if isinstance(node_expr, TensorVariable):
                    keyA = node_expr, tuple(list(path)+[operand_idx])
                    keyB = index, scope
                    network.edge(names[keyA], names[keyB], arrowhead='none')
                elif isinstance(node_expr, DeindexExpr):
                    keyA = node_expr.indices[i], tuple(list(path)+[operand_idx])
                    keyB = index, scope
                    if keyA not in names:
                        print("dam")
                    if keyB not in names:
                        print("dam")
                    network.edge(names[keyA], names[keyB], arrowhead='none')
                else:
                    raise NotImplementedError(f"Unsupported Indexing of {type(node_expr)} node found")
        return node
    postOrderPath(expr, add_edges)
    if isinstance(expr, DeindexExpr):
        for index in DeindexExpr.getIndices(expr.output_shape):
            network.node(str(index)+"out", label=str(index), shape='none')
            network.edge(names[index,tuple([])], str(index)+"out", arrowhead='none')
    elif isinstance(expr, TensorVariable):
        for space in expr.tensor_space:
            network.node(str(space)+"out", label=str(space), shape='none')
            network.edge(names[expr,tuple([])], str(space)+"out", arrowhead='none')
            
    
    
    
