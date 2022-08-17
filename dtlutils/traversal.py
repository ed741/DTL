from dtl import Node, deIndex, IndexSum


def postOrder(node: Node, fn):
    operands = node.operands
    new_operands = [postOrder(n, fn) for n in operands]
    return fn(node.with_operands(new_operands))


def postOrderRoute(node: Node, fn, path=None):
    if path is None:
        path = []
    operands = node.operands
    new_operands = [postOrderRoute(n, fn, path=(path+[i])) for i, n in enumerate(operands)]
    return fn(node.with_operands(new_operands), tuple(path))


def forAll(node: Node, fn, seen=None):
    if seen is None:
        seen = set()
    if node in seen:
        return
    else:
        fn(node)
        seen.add(node)
        for child in node.operands:
            forAll(child, fn, seen=seen)


def node_from_path(node, path):
    if len(path) == 0:
        return node
    elif len(path) == 1:
        return node.operands[path[0]]
    else:
        index = path[0]
        new_node = node.operands[index]
        new_path = path[1:]
        return node_from_path(new_node, new_path)
    
    
def path_id(node, child):
    for i, c in enumerate(node.operands):
        if child == c:
            return i
    return -1


def get_scope(node, index, path):
    """ return path of the enclosing scope of the index at given path
    node: root node of expression
    index: the index for which you want to find the scope
    path: path from node to the occurrence of the index you want the scope for"""
    paths = [[path[i] for i in range(k)] for k in range(len(path)+1)]
    # print("\n".join(str(p) for p in paths))
    for p in reversed(paths):
        current = node_from_path(node, p)
        if current.makes_scope(index):
            return p
    return None
