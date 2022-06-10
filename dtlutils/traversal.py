from dtl import Node


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
    