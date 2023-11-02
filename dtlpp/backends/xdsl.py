import typing

import dtl
import xdsl.dialects.builtin
from dtl.dtlutils import traversal
from xdsl.dialects import arith, builtin
from xdsl.ir import SSAValue, BlockArgument
from xdsl.dialects.experimental import dtl as xdtl

import functools


def get_xdsl_dtl_exec_version(expression: dtl.Expr,
                              space_map: typing.Dict[dtl.UnknownSizeVectorSpace, SSAValue],
                              arg_map: typing.Dict[dtl.Index, SSAValue],
                              tensor_variables: typing.Dict[dtl.TensorVariable, BlockArgument],
                              output
                              ):
    lines, expr = get_xdsl_dtl_version(expression, nodeMap=None, tensorVariables=tensor_variables)

    spaces = []
    space_lengths = []
    for s, l in space_map.items():
        spaces.append(vectorSpace_to_xdtl(s))
        space_lengths.append(l)
    context_type = xdtl.ExecuteContextType.new([builtin.ArrayAttr(spaces)])
    execContext = xdtl.ExecuteContextOp.build(operands=[space_lengths], result_types=[context_type])
    lines += [execContext]

    arg_names = []
    arg_values = []
    for n, v in arg_map.items():
        arg_names.append(xdtl.Index.new([builtin.StringAttr(n.name)]))
        arg_values.append(v)
    arg_type = xdtl.ExecuteArgsType.new([builtin.ArrayAttr(arg_names)])
    execArgs = xdtl.ExecuteArgsOp.build(operands=[arg_values], result_types=[arg_type])
    lines += [execArgs]

    execOp = xdtl.DenseExecuteTensorOp.build(operands=[expr, execContext, execArgs, output])
    lines += [execOp]
    return lines, execOp


pass


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

    type = DTLType_to_xdtl(node.type)
    out = xdtl.DenseBackedTensorOp.build(operands=[tensorVariables[node]], result_types=[type])
    nodeMap[node] = out
    return [out], out


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
    out = xdtl.IndexBindingOp.build(operands=[expr], attributes={"indices_map": binding}, result_types=[type])

    nodeMap[node] = out
    return lines + [out], out


@get_xdsl_dtl_version.register
def _(node: dtl.IndexExpr, nodeMap=None, tensorVariables=None):
    nodeMap = {} if nodeMap is None else nodeMap
    tensorVariables = {} if tensorVariables is None else tensorVariables
    if node in nodeMap: return [], nodeMap[node]

    indexStruct = DTLIndexing_to_xdtl(node.indices)

    lines, expr = get_xdsl_dtl_version(node.expr, nodeMap, tensorVariables)
    type = DTLType_to_xdtl(node.type)
    out = xdtl.IndexOp.build(operands=[expr], attributes={"indices": indexStruct}, result_types=[type])

    nodeMap[node] = out
    return lines + [out], out


@get_xdsl_dtl_version.register
def _(node: dtl.DeindexExpr, nodeMap=None, tensorVariables=None):
    nodeMap = {} if nodeMap is None else nodeMap
    tensorVariables = {} if tensorVariables is None else tensorVariables
    if node in nodeMap: return [], nodeMap[node]

    indexStruct = DTLIndexing_to_xdtl(node.output_shape)

    lines, expr = get_xdsl_dtl_version(node.expr, nodeMap, tensorVariables)
    type = DTLType_to_xdtl(node.type)
    out = xdtl.DeIndexOp.build(operands=[expr], attributes={"indices": indexStruct}, result_types=[type])

    nodeMap[node] = out
    return lines + [out], out


@get_xdsl_dtl_version.register
def _(node: dtl.IndexSum, nodeMap=None, tensorVariables=None):
    nodeMap = {} if nodeMap is None else nodeMap
    tensorVariables = {} if tensorVariables is None else tensorVariables
    if node in nodeMap: return [], nodeMap[node]

    indices = builtin.ArrayAttr([xdtl.Index.new([builtin.StringAttr(idx.name)]) for idx in node.sum_indices])

    lines, expr = get_xdsl_dtl_version(node.expr, nodeMap, tensorVariables)
    type = DTLType_to_xdtl(node.type)
    out = xdtl.SumOp.build(operands=[expr], attributes={"indices": indices}, result_types=[type])

    nodeMap[node] = out
    return lines + [out], out


@get_xdsl_dtl_version.register
def _(node: dtl.AddBinOp, nodeMap=None, tensorVariables=None):
    nodeMap = {} if nodeMap is None else nodeMap
    tensorVariables = {} if tensorVariables is None else tensorVariables
    if node in nodeMap: return [], nodeMap[node]

    lhs_lines, lhs_expr = get_xdsl_dtl_version(node.lhs, nodeMap, tensorVariables)
    rhs_lines, rhs_expr = get_xdsl_dtl_version(node.rhs, nodeMap, tensorVariables)
    type = DTLType_to_xdtl(node.type)
    out = xdtl.ScalarAddOp.build(operands=[lhs_expr, rhs_expr], result_types=[type])

    nodeMap[node] = out
    return lhs_lines + rhs_lines + [out], out


@get_xdsl_dtl_version.register
def _(node: dtl.SubBinOp, nodeMap=None, tensorVariables=None):
    nodeMap = {} if nodeMap is None else nodeMap
    tensorVariables = {} if tensorVariables is None else tensorVariables
    if node in nodeMap: return [], nodeMap[node]

    lhs_lines, lhs_expr = get_xdsl_dtl_version(node.lhs, nodeMap, tensorVariables)
    rhs_lines, rhs_expr = get_xdsl_dtl_version(node.rhs, nodeMap, tensorVariables)
    type = DTLType_to_xdtl(node.type)
    out = xdtl.ScalarSubOp.build(operands=[lhs_expr, rhs_expr], result_types=[type])

    nodeMap[node] = out
    return lhs_lines + rhs_lines + [out], out


@get_xdsl_dtl_version.register
def _(node: dtl.MulBinOp, nodeMap=None, tensorVariables=None):
    nodeMap = {} if nodeMap is None else nodeMap
    tensorVariables = {} if tensorVariables is None else tensorVariables
    if node in nodeMap: return [], nodeMap[node]

    lhs_lines, lhs_expr = get_xdsl_dtl_version(node.lhs, nodeMap, tensorVariables)
    rhs_lines, rhs_expr = get_xdsl_dtl_version(node.rhs, nodeMap, tensorVariables)
    type = DTLType_to_xdtl(node.type)
    out = xdtl.ScalarMulOp.build(operands=[lhs_expr, rhs_expr], result_types=[type])

    nodeMap[node] = out
    return lhs_lines + rhs_lines + [out], out


@get_xdsl_dtl_version.register
def _(node: dtl.ExprTuple, nodeMap=None, tensorVariables=None):
    nodeMap = {} if nodeMap is None else nodeMap
    tensorVariables = {} if tensorVariables is None else tensorVariables
    if node in nodeMap: return [], nodeMap[node]

    line_parts = []
    expr_parts = []
    for expr in node.exprs:
        xdtl_lines, xdtl_expr = get_xdsl_dtl_version(expr, nodeMap, tensorVariables)
        line_parts.extend(xdtl_lines)
        expr_parts.append(xdtl_expr)

    type = DTLType_to_xdtl(node.type)
    out = xdtl.TupleOp.build(operands=[expr_parts], result_types=[type])

    nodeMap[node] = out
    return line_parts + [out], out


@get_xdsl_dtl_version.register
def _(node: dtl.IndexedExprTuple, nodeMap=None, tensorVariables=None):
    nodeMap = {} if nodeMap is None else nodeMap
    tensorVariables = {} if tensorVariables is None else tensorVariables
    if node in nodeMap: return [], nodeMap[node]

    lines, expr = get_xdsl_dtl_version(node.expr, nodeMap, tensorVariables)
    type = DTLType_to_xdtl(node.type)
    out = xdtl.IndexedTupleOp.build(operands=[expr], attributes={"index": builtin.IntAttr(node.n)}, result_types=[type])

    nodeMap[node] = out
    return lines + [out], out


# @get_xdsl_dtl_version.register
# def _(node: dtl.Literal):
#     f = arith.Constant.from_float_and_width(node.f, builtin.f32)
#     out = xdtl.ScalarConstOp.get(f)
#     return [f, out], out
def DTLIndexing_to_xdtl(indices: tuple) -> xdtl.IndexStruct:
    def DTLIndex_to_xdtl(idx: typing.Union[dtl.Index, dtl.NoneIndex]):
        if idx is dtl.NoneIndex:
            return xdtl.NoneIndex()
        elif isinstance(idx, dtl.Index):
            return xdtl.Index.new([builtin.StringAttr(idx.name)])
        elif isinstance(idx, dtl.VectorSpaceVariable):
            return vectorSpace_to_xdtl(idx)

    def fold_DTLIndex_to_xdtl(children):
        if len(children) == 0:
            return xdtl.IndexShapeStruct.new([builtin.ArrayAttr([])])
        elif all(isinstance(child, xdtl.Index | xdtl.NoneIndex | xdtl.VectorSpace) for child in children):
            return xdtl.IndexShapeStruct.new([builtin.ArrayAttr(children)])
        elif all((isinstance(child, xdtl.IndexShapeStruct) or isinstance(child, xdtl.IndexTupleStruct)) for child in
                 children):
            return xdtl.IndexTupleStruct.new([builtin.ArrayAttr(children)])
        else:
            raise ValueError

    return traversal.forallTupleTreeFold(indices, DTLIndex_to_xdtl, fold_DTLIndex_to_xdtl)


def DTLType_to_xdtl(type: dtl.DTLType):
    pairs = []
    for i in sorted([i for i in type.indices], key=lambda i: i.name):
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
        return xdtl.KnownVectorSpace.new([builtin.IntAttr(space.dim)])
    else:
        raise NotImplementedError
