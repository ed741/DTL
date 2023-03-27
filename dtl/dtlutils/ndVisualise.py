import random

from dtl.dag_ext_non_scalar import *
from dtl.dtlutils.traversal import prepostOrderRoute


def plot_nd_network(expr: Union[dtl.TensorExpr, dtl.Lambda], *, name="expression", view=False, **kwargs):
    if isinstance(expr, dtl.Lambda):
        expr = expr.tensor_expr
    network = graphviz.Digraph(name, **kwargs)
    _plot_nd_network(expr, network)
    network.render(quiet_view=view)


def _plot_nd_network(expr: dtl.TensorExpr, network):
    names = {}
    clusters = {}
    cluster_stack= [network]
    
    def add_vars_pre(node, path):
        if isinstance(node, NDScalarExpr):
            name = f"cluster_{path}"
            if node in clusters:
                name, c = clusters[node]
            else:
                c = graphviz.Digraph(name=name)
                c.attr(label=str(node))
                c.attr(style="filled", color=f"{random.random()*0.8+0.2} {random.random()*0.8+0.2} {random.random()*0.8+0.2}")
                clusters[node] = (name,c)
            cluster_stack.append(c)
            
            print(f"new cluster: {name}")
        return cluster_stack[-1]
            
    def add_vars_post(node, subgraph, path):
        if isinstance(node, Index):
            name = f"I.{path}"
            s = get_scope(expr, node, path)
            if s is None:
                raise ValueError(f"index: {node.name}, cannot find scope from path: {path} in {str(expr)}")
            scope = tuple(s)
            if not (node, scope) in names:
                names[(node, scope)] = name
                subgraph.node(name, label=node.name, shape='circle')
        if isinstance(node, TensorVariable):
            name = f"TV.{path}"
            names[(node, path)] = name
            subgraph.node(name, label=node.name, shape='box')
        if isinstance(node, NDScalarExpr):
            if cluster_stack[-1] != subgraph:
                raise ValueError("cluster_stack not working right")
            name, c = clusters[node]
            if c != subgraph:
                raise ValueError("Passed cluster is wrong")
            cluster_stack.pop()
            cluster_stack[-1].subgraph(subgraph)
            
        return node

    prepostOrderRoute(expr, add_vars_pre, add_vars_post)
    
    def add_edges(node, path):
        if isinstance(node, IndexedTensor):
            operand_idx = list(node.operands).index(node.tensor_expr)
            for i, index in enumerate(node.tensor_indices):
                scope = tuple(get_scope(expr, index, path))
                if isinstance(node.tensor_expr, TensorVariable):
                    keyA = node.tensor_expr, tuple(list(path) + [operand_idx])
                    keyB = index, scope
                    network.edge(names[keyA], names[keyB], arrowhead='none')
                if isinstance(node.tensor_expr, deIndex):
                    keyA = node.tensor_expr.indices[i], tuple(list(path) + [operand_idx])
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
            network.node(str(index) + "out", label=str(expr.index_spaces[index]), shape='none')
            network.edge(names[index, tuple([])], str(index) + "out", arrowhead='none')
    elif isinstance(expr, TensorVariable):
        for space in expr.tensor_space:
            network.node(str(space) + "out", label=str(space), shape='none')
            network.edge(names[expr, tuple([])], str(space) + "out", arrowhead='none')
