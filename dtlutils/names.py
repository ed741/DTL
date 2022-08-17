import itertools

from dtl import Node, Index
from dtlutils.traversal import postOrderRoute, get_scope, node_from_path


class NameGenerator:
    def __init__(self, prefix="", suffix=""):
        if not (prefix or suffix):
            raise ValueError

        self._prefix = prefix
        self._suffix = suffix
        self._counter = itertools.count()

    def __iter__(self):
        return self

    def __next__(self):
        return f"{self._prefix}{next(self._counter)}{self._suffix}"


def make_Index_names_unique(root: Node):
    names = set()
    new_index_node_map = {}
    name_count = 0
    
    def _get_new_name(old_name):
        i = 0
        while (name := f"{old_name}_{str(i)}") in names:
            i += 1
        names.add(name)
        return name
    
    def _make_Index_names_unique(node, path):
        if isinstance(node, Index):
            scopePath = tuple(get_scope(root, node, path))
            if scopePath is None:
                raise NotImplementedError
            # scopeNode = node_from_path(root, scopePath)
            if (scopePath, node) in new_index_node_map:
                return new_index_node_map[(scopePath, node)]
            else:
                name = _get_new_name(node.name)
                newNode = node.copy(name=name)
                new_index_node_map[(scopePath, node)] = newNode
                return newNode
        return node
    return postOrderRoute(root, _make_Index_names_unique)
    