import typing

import dtl
import xdsl.dialects.builtin
from dtl.dtlutils import traversal
from xdsl.builder import Builder
from xdsl.dialects import arith, builtin
from xdsl.ir import SSAValue, BlockArgument, Block
from xdsl.dialects.linalg import Generic
from xdsl.dialects.experimental import dtl as xdtl

import functools


def get_xdsl_dtl_exec_version(expression: dtl.Expr,
                              space_map: typing.Dict[dtl.UnknownSizeVectorSpace, SSAValue],
                              arg_map: typing.Dict[dtl.Index, SSAValue],
                              tensor_variables: typing.Dict[dtl.TensorVariable, tuple[SSAValue,list[str]]],
                              outputs: list[tuple[SSAValue, list[str]]]
                              ):
    block = Block()



    extent_names = []
    extent_args = []
    for s, l in space_map.items():
        space = vectorSpace_to_xdtl(s)
        assert isinstance(space, xdtl.UnknownVectorSpace)
        extent_names.append(space)
        extent_args.append(l)
    context_type = xdtl.ExecuteContextType.new([builtin.ArrayAttr(extent_names)])
    execContext = xdtl.ExecuteContextOp.build(operands=[extent_args], result_types=[context_type])

    index_names = []
    index_values = []
    for n, v in arg_map.items():
        index_names.append(xdtl.Index.new([builtin.StringAttr(n.name)]))
        index_values.append(v)
    arg_type = xdtl.ExecuteArgsType.new([builtin.ArrayAttr(index_names)])
    execArgs = xdtl.ExecuteArgsOp.build(operands=[index_values], result_types=[arg_type])


    output_tensors = []
    output_tensors_dim_names = []
    for ssa, idx_names in outputs:
        dims = builtin.ArrayAttr([builtin.StringAttr(i) for i in idx_names])
        output_tensors_dim_names.append(dims)
        output_tensors.append(ssa)
    output_tensors_type = xdtl.ExecuteOutputType.new([builtin.ArrayAttr(output_tensors_dim_names)])
    execOutputs = xdtl.ExecuteOutputOp.build(operands=[output_tensors], result_types=[output_tensors_type])


    tensor_arg_indices = []
    tensor_arg_base_types = []
    tensor_args = []
    internal_tensor_variables = {}
    for i, (tensor_var, (ssa_val, idxs)) in enumerate(tensor_variables.items()):
        assert tensor_var.type.nResults == 1
        assert len(idxs) == tensor_var.type.result.nDims
        tensor_args.append(ssa_val)
        tensor_arg_indices.append(builtin.ArrayAttr([builtin.StringAttr(i) for i in idxs]))
        tensor_arg_base_types.append(ssa_val.type.contents_type.get_single_element().base_type)
        arg = block.insert_arg(ssa_val.type, i)
        internal_tensor_variables[tensor_var] = (arg, idxs)


    lines, expr = get_xdsl_dtl_version(expression, nodeMap=None, tensorVariables=internal_tensor_variables)
    block.add_ops(lines)
    yield_op = xdtl.ExecuteYieldOp.build(operands=[expr])
    block.add_op(yield_op)

    execOp = xdtl.DenseExecuteTensorOp.build(regions=[[block]],
                                             attributes={"tensor_arg_indices": builtin.ArrayAttr(tensor_arg_indices),
                                                         "tensor_arg_base_types": builtin.ArrayAttr(tensor_arg_base_types)
                                                         },
                                             operands=[execContext, execArgs, execOutputs, tensor_args],)

    execOp.verify()
    return [execContext, execArgs, execOutputs], execOp


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
    out = xdtl.DenseBackedTensorOp.build(operands=[tensorVariables[node][0]], result_types=[type])
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
        else:
            raise NotImplementedError()

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
