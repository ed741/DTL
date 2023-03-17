import itertools
import typing
from typing import Iterable, List

import dtl
from dtl import Node, Index
from dtlutils.traversal import postOrderPath, get_scope, node_from_path, prepostOrderPathPassback, allOperands


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

class AugmentedNameGenerator:
    def __init__(self):
        self._names = set()
    
    def registerName(self, name:str):
        self._names.add(name)
    
    def getNewName(self, old_name:str):
        if old_name not in self._names:
            self._names.add(old_name)
            return old_name
        i = 0
        while (name := f"{old_name}_{str(i)}") in self._names:
            i += 1
        self._names.add(name)
        return name
    

def make_Index_names_unique(root: Node):
    new_index_node_map = {}
    nameGen = AugmentedNameGenerator()
    def _make_Index_names_unique(node, path):
        if isinstance(node, Index):
            scope = get_scope(root, node, path)
            if scope is None:
                # if the index is free in an expression then there is no scope - this should hopefully allow us to
                # still ensure unique indices.
                scope = [-1]
            scopePath = tuple(scope)
            if (scopePath, node) in new_index_node_map:
                return new_index_node_map[(scopePath, node)]
            else:
                name = nameGen.getNewName(node.name)
                newNode = node.copy(name=name)
                new_index_node_map[(scopePath, node)] = newNode
                return newNode
        else:
            pass
        return node
    return postOrderPath(root, _make_Index_names_unique)


def make_Index_names_unique_CSE(root: typing.Union[Node, List[Node]]):
    return_List = True
    if not isinstance(root, List):
        root = [root]
        return_List = False
    new_index_node_map = {}
    nameGen = AugmentedNameGenerator()
    cses = {}
    
    def _pre_func(node, path):
        return node
    def _make_Index_names_unique(node, preNode, path, passback):
        """
        :param node: The current node (with its new child operands based on the already passed sub-dags)
        :param preNode: The current node from before it was re-made with new children
        :param path: The path from the root taken to get to the current node on this pass
        :param passback: Sets of paths for each operand
        :return:
        """
        if isinstance(node, dtl.Terminal):
            if len(passback)>0:
                raise ValueError("passback cannot be anything but empty set for a terminal")
        passback = set([p for sl in allOperands(passback) for p in sl])
        if isinstance(node, Index):
            scope = get_scope(root[path[0]], node, path[1:])
            if scope is None:
                # if the index is free in an expression then there is no scope - this should hopefully allow us to
                # still ensure unique indices.
                scope = [-1]
            scopePath = tuple([path[0], *scope])
            if (scopePath, node) in new_index_node_map:
                return new_index_node_map[(scopePath, node)], {scopePath}
            else:
                name = nameGen.getNewName(node.name)
                newNode = node.copy(name=name)
                new_index_node_map[(scopePath, node)] = newNode
                return newNode, {scopePath}
        else:
            passback = set([p for p in passback if p != path])
            if len(passback) == 0:
                if preNode in cses:
                    return cses[preNode], set()
                else:
                    cses[preNode] = node
                    return node, set()
            else:
                return node, passback
    
    if return_List:
        return [prepostOrderPathPassback(n, _pre_func, _make_Index_names_unique, path=[i])[0] for i,n in enumerate(root)]
    else:
        return prepostOrderPathPassback(root[0], _pre_func, _make_Index_names_unique, path=[0])[0]
