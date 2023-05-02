import typing

import dtl
from dtl.dtlutils import traversal
from xdsl.dialects import arith, builtin
from xdsl.ir import SSAValue
from xdslDTL import dialect as xdtl

import functools

@functools.singledispatch
def get_xdsl_dtl_version(node: dtl.Node, nodeMap=None, tensorVariables=None):
    nodeMap = {} if nodeMap is None else nodeMap
    tensorVariables = {} if tensorVariables is None else tensorVariables
    raise ValueError

@get_xdsl_dtl_version.register
def _(node: dtl.Literal, nodeMap=None, tensorVariables=None):
    nodeMap = {} if nodeMap is None else nodeMap
    tensorVariables = {} if tensorVariables is None else tensorVariables
    if node in nodeMap: return [], nodeMap[node]
    
    f = arith.Constant.from_float_and_width(node.f, builtin.f32)
    type = DTLType_to_xdtl(node.type)
    out = xdtl.ScalarConstOp.build(operands=[f], result_types=[type])
    
    nodeMap[node] = out
    return [f, out], out

@get_xdsl_dtl_version.register
def _(node: dtl.TensorVariable, nodeMap=None, tensorVariables=None):
    nodeMap = {} if nodeMap is None else nodeMap
    tensorVariables = {} if tensorVariables is None else tensorVariables
    if node in nodeMap: return [], nodeMap[node]
    
    out = tensorVariables[node]
    
    nodeMap[node] = out
    return [], out


@get_xdsl_dtl_version.register
def _(node: dtl.IndexBinding, nodeMap=None, tensorVariables=None):
    nodeMap = {} if nodeMap is None else nodeMap
    tensorVariables = {} if tensorVariables is None else tensorVariables
    if node in nodeMap: return [], nodeMap[node]
    
    bindingPairs = []
    for i, s in sorted(zip(node.indices, node.spaces), key=lambda p: p[0].name):
        idx = xdtl.Index.new([builtin.StringAttr(i.name)])
        vs = vectorSpace_to_xdtl(s)
        bindingPairs.append(xdtl.IndexToVectorSpaceMapPair.new([idx, vs]))
    binding = xdtl.IndexToVectorSpaceMap.new([builtin.ArrayAttr(bindingPairs)])
    
    lines, expr = get_xdsl_dtl_version(node.expr, nodeMap, tensorVariables)
    type = DTLType_to_xdtl(node.type)
    out = xdtl.IndexBindingOp.build(operands=[expr], attributes={"indices_map":binding}, result_types=[type])
    
    nodeMap[node] = out
    return lines+[out], out


@get_xdsl_dtl_version.register
def _(node: dtl.IndexExpr, nodeMap=None, tensorVariables=None):
    nodeMap = {} if nodeMap is None else nodeMap
    tensorVariables = {} if tensorVariables is None else tensorVariables
    if node in nodeMap: return [], nodeMap[node]
    
    def DTLIndex_to_xdtl(idx: typing.Union[dtl.Index, dtl.NoneIndex]):
        return xdtl.Index.new([builtin.StringAttr(idx.name)])

    def fold_DTLIndex_to_xdtl(children):
        if len(children) == 0:
            return xdtl.IndexShapeStruct.new(builtin.ArrayAttr([]))
        elif all(isinstance(child, xdtl.Index) for child in children):
            return xdtl.IndexShapeStruct.new(builtin.ArrayAttr(children))
        elif all(isinstance(child, xdtl.IndexShapeStruct) or isinstance(child, xdtl.IndexTupleStruct) for child in children):
            return xdtl.IndexTupleStruct.new(builtin.ArrayAttr(children))
        else:
            raise ValueError
    
    indexStruct = traversal.forallTupleTreeFold(node.indices, DTLIndex_to_xdtl, fold_DTLIndex_to_xdtl)

    lines, expr = get_xdsl_dtl_version(node.expr, nodeMap, tensorVariables)
    type = DTLType_to_xdtl(node.type)
    out = xdtl.IndexBindingOp.build(operands=[expr], attributes={"indices": indexStruct}, result_types=[type])
    
    nodeMap[node] = out
    return lines + [out], out


# @get_xdsl_dtl_version.register
# def _(node: dtl.Literal):
#     f = arith.Constant.from_float_and_width(node.f, builtin.f32)
#     out = xdtl.ScalarConstOp.get(f)
#     return [f, out], out

def DTLType_to_xdtl(type: dtl.DTLType):
    pairs = []
    for i in sorted([i for i in type.indices]):
        idx = xdtl.Index.new([builtin.StringAttr(i.name)])
        space = type.spaceOf(i)
        vs = vectorSpace_to_xdtl(space)
        pairs.append(xdtl.IndexToVectorSpaceMapPair.new([idx, vs]))
    binding = xdtl.IndexToVectorSpaceMap.new([builtin.ArrayAttr(pairs)])
    indexStruct = resultType_to_xdtl(type.result)
    return xdtl.TensorExprType([binding, indexStruct])

def resultType_to_xdtl(result: dtl.ResultType):
    if isinstance(result, dtl.ShapeType):
        return xdtl.IndexShapeStruct.new([builtin.ArrayAttr([vectorSpace_to_xdtl(s) for s in result.dims])])
    elif isinstance(result, dtl.ResultTupleType):
        return xdtl.IndexTupleStruct.new([builtin.ArrayAttr([resultType_to_xdtl(r) for r in result.results])])
    
def vectorSpace_to_xdtl(space: dtl.VectorSpaceVariable):
    if isinstance(space, dtl.UnknownSizeVectorSpace):
        return xdtl.UnknownVectorSpace.new([builtin.StringAttr(space.name)])
    elif isinstance(space, dtl.VectorSpace):
        return xdtl.KnownVectorSpace.new([builtin.IntAttr(10)])
    else:
        raise NotImplementedError