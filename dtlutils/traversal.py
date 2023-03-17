import functools
import typing

from dtl import Node



def __operandsToList(operands, label=None):
    if isinstance(operands, Node):
        return [operands] if label is None else [(label, operands)]
    if isinstance(operands, typing.List):
        if label is None:
            return [n for l in operands for n in __operandsToList(l)]
        else:
            return [n for i, l in enumerate(operands) for n in __operandsToList(l, label=label + '.' + str(i))]
    if isinstance(operands, tuple):
        if label is None:
            return [n for l in operands for n in __operandsToList(l)]
        else:
            return [n for i, l in enumerate(operands) for n in __operandsToList(l, label=label + '.' + str(i))]
    if isinstance(operands, typing.Dict):
        if label is None:
            return [n for k, l in operands.items() for n in __operandsToList(l)]
        else:
            return [n for k, l in operands.items() for n in __operandsToList(l, label=label + '.' + k)]
    
    ## we can search for results in an 'operands' structure that are not nodes: but these results must not be list, tuple, or dict
    return [operands] if label is None else [(label, operands)]

def allOperands(operands) -> typing.Iterable:
    return __operandsToList(operands)


def allOperandsLabelled(operands) -> typing.Iterable:
    return __operandsToList(operands, label="")

def flattenTupleTreeToList(tree: tuple)->list:
    if isinstance(tree, tuple):
        return [e for ts in tree for e in flattenTupleTreeToList(ts)]
    else:
        return [tree]
    
def forallTupleTreeFold(tree: tuple, fn, fold):
    if isinstance(tree, tuple):
        return fold([forallTupleTreeFold(t, fn, fold) for t in tree])
    else:
        return fn(tree)
    
def operandLabelled(node: Node, operandLabel: typing.List[typing.Union[str, int]]):
    operands = node.operands
    for a in operandLabel:
        operands = operands[a]
    return ''.join([f".{str(a)}" for a in operandLabel]), operands

@functools.singledispatch
def forAllOperands(operands, fn, label=None, args=None):
    kwargs = {}
    if args is not None:
        kwargs["args"] = args
    if label is not None:
        kwargs["label"] = label
    return fn(operands, **kwargs)

@forAllOperands.register
def _(operands: Node, fn, label=None, args=None):
    kwargs = {}
    if args is not None:
        kwargs["args"] = args
    if label is not None:
        kwargs["label"]=label
    return fn(operands, **kwargs)

@forAllOperands.register
def _(operands: list, fn, label=None, args=None):
    if label is not None:
        return [forAllOperands(r, fn, label=label+'.'+str(i), args=args) for i, r in enumerate(operands)]
    else:
        return [forAllOperands(r, fn, args=args) for r in operands]

@forAllOperands.register
def _(operands: tuple, fn: typing.Callable[[Node], typing.Any], label=None, args=None):
    if label is not None:
        return tuple([forAllOperands(r, fn, label=label + '.' + str(i), args=args) for i, r in enumerate(operands)])
    else:
        return tuple([forAllOperands(r, fn, args=args) for r in operands])

@forAllOperands.register
def _(operands: dict, fn: typing.Callable[[Node], typing.Any], label=None, args=None):
    if label is not None:
        return {k:forAllOperands(v, fn, label=label+'.'+k, args=args) for k,v in operands.items()}
    else:
        return {k:forAllOperands(v, fn, args=args) for k,v in operands.items()}


def postOrder(node: Node, fn):
    def po(n: Node):
        operands = n.operands
        new_operands = forAllOperands(operands, po)
        new_n = n.with_operands(new_operands)
        out = fn(new_n)
        if out is None: out = new_n
        return out
    return po(node)

def postOrderPath(node: Node, fn):
    def po(n: Node, label=None, args=None):
        if args is None:
            raise ValueError("Args should never be None")
        path = list(args)
        if label is not None:
            path.append(label)
        operands = n.operands
        new_operands = forAllOperands(operands, po, label="", args=path)
        new_n = n.with_operands(new_operands)
        out = fn(new_n, path)
        if out is None: out = new_n
        return out
    return po(node, label=None, args=[])

# def postOrderRoute(node: Node, fn, path=None):
#     def po(n: Node, label=""):
#         operands = n.operands
#         new_operands = forAllOperands(operands, po, label="")
#     if path is None:
#         path = []
#     operands = node.operands
#     new_operands = [postOrderRoute(n, fn, path=(path+[i])) for i, n in enumerate(operands)]
#     return fn(node.with_operands(new_operands), tuple(path))

def prepostOrderPath(node: Node, pre_fn, post_fn):
    def po(n: Node, label=None, args=None):
        if args is None:
            raise ValueError("Args should never be None")
        path = list(args)
        if label is not None:
            path.append(label)
        pre_fn_result = pre_fn(n, tuple(path))
        operands = n.operands
        new_operands = forAllOperands(operands, po, label="", args=path)
        new_n = n.with_operands(new_operands)
        out = post_fn(new_n, pre_fn_result, path)
        if out is None: out = new_n
        return out
    return po(node, label=None, args=[])

# def prepostOrderRoute(node: Node, pre_fn, post_fn, path=None):
#     if path is None:
#         path = []
#     pre_fn_result = pre_fn(node, tuple(path))
#     operands = node.operands
#     new_operands = [prepostOrderRoute(n, pre_fn, post_fn, path=(path+[i])) for i, n in enumerate(operands)]
#     return post_fn(node.with_operands(new_operands), pre_fn_result, tuple(path))

class __Entry:
    def __init__(self, node: Node, passback):
        self.node = node
        self.passback = passback
    def getNode(self):
        return self.node
    
    def getPassback(self):
        return self.passback
def prepostOrderPathPassback(node: Node, pre_fn, post_fn, path=None):
    if path is None:
        path = []
    def po(n: Node, label=None, args=None):
        if args is None:
            raise ValueError("Args should never be None")
        path = list(args)
        if label is not None:
            path.append(label)
        pre_fn_result = pre_fn(n, tuple(path))
        operands = n.operands
        new_operands_and_passin = forAllOperands(operands, po, label="", args=path)
        new_operands = forAllOperands(new_operands_and_passin, __Entry.getNode)
        new_passback = forAllOperands(new_operands_and_passin, __Entry.getPassback)
        new_n = n.with_operands(new_operands)
        out, passback = post_fn(new_n, pre_fn_result, path, new_passback)
        return __Entry(out, passback)
    entry = po(node, label=None, args=path)
    return entry.getNode(), entry.getPassback()

# def prepostOrderRoutePassback(node: Node, pre_fn, post_fn, path=None):
#     if path is None:
#         path = []
#     pre_fn_result = pre_fn(node, tuple(path))
#     operands = node.operands
#     results = [prepostOrderRoutePassback(n, pre_fn, post_fn, path=(path + [i])) for i, n in enumerate(operands)]
#     new_operands = [n for n,p in results]
#     passback = [p for n, p in results]
#     return post_fn(node.with_operands(new_operands), pre_fn_result, tuple(path), passback)


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
    # if len(path) == 0:
    #     return node
    # elif len(path) == 1:
    #     return node.operands[path[0]]
    # else:
    #     index = path[0]
    #     new_node = node.operands[index]
    #     new_path = path[1:]
    #     return node_from_path(new_node, new_path)
    #
    current_node = node
    for label in path:
        if label == '.': continue
        found = False
        for l, n in allOperandsLabelled(current_node.operands):
            if l == label:
                current_node = n
                found = True
                break
        if not found:
            raise ValueError(f"Cannot find label \"{label}\" in node {str(current_node)}")
    return current_node
    
    
    
def path_id(node, child):
    for l, c in allOperandsLabelled(node.operands):
        if child == c:
            return l
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
