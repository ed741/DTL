import typing

import dtl
from dtl.dtlutils import traversal


def use_common_subexpressions(root: dtl.Node, ignoring: typing.Sequence[str] = None, ignore_fn=None):
    if ignoring is None: ignoring = []
    if ignore_fn is None: ignore_fn = lambda attrs: attrs
    seen = {} # map stripped node to original node
    def po(node: dtl.Node, path):
        attrs = node.attrs
        attrs = [(k,v) for k,v in attrs if (not k in ignoring)]
        attrs = ignore_fn(attrs)
        new_node = node.copy(attrs=attrs)
        if new_node in seen:
            return seen[new_node]
        else:
            seen[new_node] = node
            return node
    return  traversal.postOrderPath(root, po)
    
    