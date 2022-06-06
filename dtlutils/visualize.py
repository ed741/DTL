import functools
import typing
from typing import Any

import graphviz

import dtl


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

