import abc
import inspect
import typing
from abc import ABC
from typing import Dict

import dtl
import itertools
import functools
import numpy as np

from dtl import Node, DTLType, dtlMatrix
from dtl.dtlutils import visualise, traversal

class ExpressionNode(abc.ABC):
    @abc.abstractmethod
    def code(self) -> str:
        pass

    @abc.abstractmethod
    def do(self, args: typing.Dict[str, typing.Any]):
        pass
    
    def __str__(self):
        return self.code()

class ExprIndexed(ExpressionNode):
    def __init__(self, expr: ExpressionNode, indices: typing.List[str]):
        self.expr = expr
        self.indices = indices
    
    def code(self):
        return f"{self.expr.code()}[{','.join(self.indices)}]"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        idx = tuple(slice(None) if i == ':' else args[i] for i in self.indices)
        return self.expr.do(args)[idx]

# class ExprTensor(ExpressionNode):
#     def __init__(self, name: str, indices: typing.List[str]):
#         self.name = name
#         self.indices = indices
#
#     def code(self):
#         return f"{self.name}[{','.join(self.indices)}]"
#
#     def do(self, args: typing.Dict[str, typing.Any]):
#         idx = tuple(args[i] for i in self.indices)
#         return args[self.name][idx]
#

class ExprAdd(ExpressionNode):
    def __init__(self, lhs: ExpressionNode, rhs: ExpressionNode):
        self.lhs = lhs
        self.rhs = rhs
    
    def code(self):
        return f"({self.lhs} + {self.rhs})"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        return self.lhs.do(args) + self.rhs.do(args)


class ExprSub(ExpressionNode):
    def __init__(self, lhs: ExpressionNode, rhs: ExpressionNode):
        self.lhs = lhs
        self.rhs = rhs
    
    def code(self):
        return f"({self.lhs} - {self.rhs})"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        return self.lhs.do(args) - self.rhs.do(args)


class ExprMul(ExpressionNode):
    def __init__(self, lhs: ExpressionNode, rhs: ExpressionNode):
        self.lhs = lhs
        self.rhs = rhs
    
    def code(self):
        return f"({self.lhs} * {self.rhs})"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        return self.lhs.do(args) * self.rhs.do(args)


class ExprConst(ExpressionNode):
    def __init__(self, val: str):
        self.val = val
    
    def code(self):
        return f"({self.val})"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        if self.val.isnumeric():
            return int(self.val)
        else:
            return args[self.val]


class ExprScalarLiteral(ExpressionNode):
    def __init__(self, val: float):
        self.val = val
    
    def code(self):
        return f"(np.full((),{self.val}))"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        return np.float32(self.val)
        return np.full((), self.val)

class ExprNdMax(ExpressionNode):
    def __init__(self, expr: ExpressionNode):
        self.expr = expr
    
    def code(self):
        return f"(np.max({self.expr}))"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        return np.max(self.expr.do(args))

class ExprNdMatInv(ExpressionNode):
    def __init__(self, expr: ExpressionNode, skip_zero=False):
        self.expr = expr
        self.skip_zero = skip_zero
    
    def code(self):
        if self.skip_zero:
            return f"(np.linalg.inv(T_if) if (T_if:={self.expr})[0,0]!=0.0 else T_if)"
        else:
            return f"(np.linalg.inv({self.expr}))"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        if self.skip_zero:
            return np.linalg.inv(base) if (base:=self.expr.do(args))[0,0]!=0.0 else base
        else:
            return np.linalg.inv(self.expr.do(args))
    
class ExprPrint(ExpressionNode):
    def __init__(self, expr: ExpressionNode, comment: str):
        self.expr = expr
        self.comment = comment
    
    def code(self) -> str:
        return self.expr.code()
    
    def do(self, args: typing.Dict[str, typing.Any]):
        subExpression = self.expr.do(args)
        print(f"{self.comment}:{subExpression}")
        


class CodeNode(abc.ABC):
    @abc.abstractmethod
    def code(self, indent: int) -> str:
        pass
    
    @abc.abstractmethod
    def do(self, args: typing.Dict[str, typing.Any]):
        pass
    
    def __str__(self):
        return self.code(0)


class Func():
    def __init__(self, args, arg_shapes, vector_space_args, child: CodeNode, results):
        self.args = args
        self.arg_shapes = arg_shapes
        self.vector_space_args = vector_space_args
        self.child = child
        self.results = results
    
    def do(self, **kwargs):
        if any(a not in kwargs for a in self.args) or any(a not in kwargs for a in self.vector_space_args):
            raise ValueError(f"Arguments not satisfied.\n Provided: {str(list(kwargs.keys()))}\n needed: {self.args + self.vector_space_args}")
        for arg, shape in zip(self.args, self.arg_shapes):
            input = kwargs[arg]
            tshape = tuple(v.dim for v in shape)
            if isinstance(shape, tuple) and input.shape != tshape:
                raise ValueError(f"Tensor Argument: {arg} should have shape {tshape} but {input.shape} was provided.")
            elif isinstance(shape, int) and input >= shape:
                raise ValueError(f"Index Argument: {arg} should be in range [0,{shape}) but {input} was provided.")
        self.child.do(kwargs)
        return traversal.forAllOperands(self.results, lambda e: e.do(kwargs))
    
    def code(self):
        def ret(e):
            if isinstance(e, ExpressionNode):
                return e.code()
            elif isinstance(e, tuple):
                return f"({','.join([ret(c) for c in e])})"
            else: raise ValueError("Internal Compiler Error: Return type is not ExpressionNode or tupleTree")
            
        instructions = []
        instructions.append(f"def func({','.join(self.args+[vs+':int' for vs in self.vector_space_args])}):")
        instructions.append(self.child.code(1))
        instructions.append(f"    return {ret(self.results)}")
        return "\n".join(instructions)
        
        
class Loop(CodeNode):
    def __init__(self, index: str, extent: typing.Union[int, str], child: CodeNode):
        self.index = index
        self.extent = extent
        self.child = child
    
    def code(self, indent:int) -> str:
        return f"{'    '*indent}for {self.index} in range({self.extent}):\n" \
               f"{self.child.code(indent+1)}\n" \
               f"{'    '*indent}#Done {self.index} loop"

    def do(self, args: typing.Dict[str, typing.Any]):
        for i in range(self.extent if isinstance(self.extent, int) else args[self.extent]):
            args[self.index] = i
            self.child.do(args)
            args.pop(self.index)


class SeqNode(CodeNode):
    def __init__(self, children: typing.List[CodeNode]):
        self.childs = []
        for c in children:
            if not (isinstance(c, SeqNode) and len(c.childs) == 0):
                self.childs.append(c)
            
    def code(self, indent: int) -> str:
        if len(self.childs)>0:
            return "\n".join(c.code(indent) for c in self.childs)
        else:
            return f"{'    ' * indent}pass"

    def do(self, args: typing.Dict[str, typing.Any]):
        for c in self.childs:
            c.do(args)


class AssignTensor(CodeNode):
    def __init__(self, lhs_name: str, lhs_indices: typing.List[str], rhs: ExpressionNode):
        self.lhs_name = lhs_name
        self.lhs_indices = lhs_indices
        self.rhs = rhs
    
    def code(self, indent: int) -> str:
        return f"{'    '*indent}{self.lhs_name}[{','.join(self.lhs_indices)}]={self.rhs.code()}"

    def do(self, args: typing.Dict[str, typing.Any]):
        args[self.lhs_name][tuple(slice(None) if i == ':' else args[i] for i in self.lhs_indices)] = self.rhs.do(args)


class AssignTemp(CodeNode):
    def __init__(self, lhs_name: str, rhs: ExpressionNode):
        self.lhs_name = lhs_name
        self.rhs = rhs
    
    def code(self, indent: int) -> str:
        return f"{'    ' * indent}{self.lhs_name}={self.rhs.code()}"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        args[self.lhs_name]= self.rhs.do(args)


class Accumulate(CodeNode):
    def __init__(self, lhs_name: str, lhs_indices: typing.List[str], rhs: ExpressionNode):
        self.lhs_name = lhs_name
        self.lhs_indices = lhs_indices
        self.rhs = rhs
    
    def code(self, indent: int) -> str:
        indexing = f"[{','.join(str(i) for i in self.lhs_indices)}]" if len(self.lhs_indices)>0 else ""
        return f"{'    ' * indent}{self.lhs_name}{indexing}+={self.rhs.code()}"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        idx = tuple(args[i] for i in self.lhs_indices)
        if len(idx)>0:
            args[self.lhs_name][idx] += self.rhs.do(args)
        else:
            args[self.lhs_name] += self.rhs.do(args)


class InitTensor(CodeNode):
    def __init__(self, name: str, indices: typing.List[typing.Union[int, str]]):
        self.name = name
        self.indices = indices
    
    def code(self, indent: int) -> str:
        return f"{'    ' * indent}{self.name}=np.zeros([{','.join(str(i) for i in self.indices)}])"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        indices = [i if isinstance(i, int) else args[i] for i in self.indices]
        args[self.name]= np.zeros(indices)

class CodeComment(CodeNode):
    def __init__(self, comment: str):
        self.comment = comment
    
    def code(self, indent: int) -> str:
        return f"{'    ' * indent}#{self.comment}"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        pass


class NameGenerator:
    def __init__(self, prefix="", suffix=""):
        if not (prefix or suffix):
            raise ValueError
        
        self._prefix = prefix
        self._suffix = suffix
        self._counter = itertools.count()
    
    def next(self, include=""):
        return f"{self._prefix}{include}{next(self._counter)}{self._suffix}"

class PythonDtlNode(Node, ABC):
    def get_expression(self, kernelBuilder: "KernelBuilder", indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        pass # return CodeNode, ExpressionNode
    
    def get_inputs(self, kernelBuilder: "KernelBuilder"):
        tv = frozenset()
        vs = frozenset()
        for child in traversal.allOperands(self.operands):
            child_tv, child_vs = kernelBuilder.get_inputs(child)
            tv = tv.union(child_tv)
            vs = vs.union(child_vs)
        return tv, vs

            
class InstantiationExprNode(dtl.Expr):
    fields = dtl.Expr.fields | {"expr"}
    
    def __init__(self, expr: dtl.ExprTypeHint, **kwargs):
        expr = dtl.Expr.exprInputConversion(expr)
        super().__init__(expr=expr, _NonUserFileNames_= [inspect.getframeinfo(inspect.currentframe()).filename], **kwargs)
    @property
    def type(self) -> DTLType:
        return self.expr.type

    @property
    def operands(self):
        return {"expr":self.expr}

    def with_operands(self, operands: Dict):
        return self.copy(expr=operands["expr"])
    
    def __str__(self):
        return "{{"+str(self.expr)+"}}"

    def shortStr(self) -> str:
        return "{{" + self.expr.terminalShortStr() + "}}"


class SequenceExprNode(dtl.Expr):
    fields = dtl.Expr.fields | {"expr", "pre"}
    
    def __init__(self, pre: dtl.ExprTypeHint, expr: dtl.ExprTypeHint, **kwargs):
        pre = dtl.Expr.exprInputConversion(pre)
        expr = dtl.Expr.exprInputConversion(expr)
        DTLType.checkCommonIndices([expr.type, pre.type])
        if len(pre.type.indices - expr.type.indices)>0:
            raise ValueError(f"pre-expr {pre.type} must not use indices that are not in use in expr: {expr.type}")
        super().__init__(expr=expr, pre=pre, _NonUserFileNames_= [inspect.getframeinfo(inspect.currentframe()).filename], **kwargs)
    
    @property
    def type(self) -> DTLType:
        return self.expr.type
    
    @property
    def operands(self):
        return {"expr": self.expr, "pre": self.pre}
    
    def with_operands(self, operands: Dict):
        return self.copy(expr=operands["expr"], pre=operands["pre"])
    
    def __str__(self):
        return "{|" + str(self.pre) + "|" + str(self.expr) + "|}"
    
    def shortStr(self) -> str:
        return "{|" + self.pre.terminalShortStr() + "|" + self.expr.terminalShortStr() + "|}"


class IndexRebinding(dtl.Expr):  # [...] -> <...>... , {b:B} => [...b:B] -> <...>...
    fields = dtl.Expr.fields | {"expr", "outer_indices", "inner_indices"}
    
    def __init__(self, expr: dtl.ExprTypeHint, outer_indices: typing.Sequence[dtl.Index], inner_indices: typing.Sequence[dtl.Index], **kwargs):
        expr = dtl.Expr.exprInputConversion(expr)
        if len(outer_indices) != len(inner_indices):
            raise ValueError(
                f"IndexReBinding, every outer index given must map to an inner index - outer indices Sequence and inner indices Sequence have different lengths!:\n"
                f"  outer indices:[{','.join([str(i) for i in outer_indices])}]\n"
                f"  inner indices:[{','.join([str(i) for i in inner_indices])}]\n")
        if len(inner_indices) != len(set(inner_indices)):
            raise ValueError(
                f"IndexReBinding, cannot bind indices multiple times! there are duplicates in inner indices: [{','.join([str(i) for i in inner_indices])}]"
            )
        exprType = expr.type
        for i_out, i_in in zip(outer_indices, inner_indices):
            if exprType.spaceOf(i_in) != None and not i_in in outer_indices:
                raise ValueError(
                    f"IndexReBinding cannot declare index {str(i_in)}:{str(exprType.spaceOf(i_in))} already found in expr with type {str(exprType)}")
            if exprType.spaceOf(i_out) == None:
                raise ValueError(
                    f"IndexReBinding cannot redeclare index {str(i_out)}:{str(exprType.spaceOf(i_out))} not found in expr with type {str(exprType)}")
        
        if "spaces" in kwargs:
            print("huh")
        super().__init__(expr=expr, outer_indices=tuple(outer_indices), inner_indices=tuple(inner_indices), _NonUserFileNames_= [inspect.getframeinfo(inspect.currentframe()).filename], **kwargs)
    
    @property
    def type(self) -> DTLType:
        exprType = self.expr.type
        args = exprType.args
        spaces = []
        for i_out in self.outer_indices:
            spaces.append(exprType.spaceOf(i_out))
            args.pop(i_out)
        for i_in, space in zip(self.inner_indices, spaces):
            args[i_in] = space
        return exprType.withArgs(args)
    
    @property
    def operands(self):
        return {"expr": self.expr, "outer_indices": self.outer_indices, "inner_indices": self.inner_indices}
    
    def with_operands(self, operands: Dict):
        return self.copy(expr=operands["expr"], outer_indices=operands["outer_indices"], inner_indices=operands["inner_indices"])
    
    def __str__(self):
        return f"{str(self.expr)}{{{','.join([str(i) + '->' + str(s) for i, s, in zip(self.outer_indices, self.inner_indices)])}}}"
    
    def shortStr(self) -> str:
        return f"{self.expr.terminalShortStr()}{{{','.join([str(i) + '->' + str(s) for i, s, in zip(self.outer_indices, self.inner_indices)])}}}"


class PrintExprNode(dtl.Expr):
    fields = dtl.Expr.fields | {"expr", "string"}
    
    def __init__(self, expr: dtl.ExprTypeHint, string, **kwargs):
        expr = dtl.Expr.exprInputConversion(expr)
        super().__init__(expr=expr, string=string)
    
    @property
    def type(self) -> DTLType:
        return self.expr.type
    
    @property
    def operands(self):
        return {"expr": self.expr}
    
    def with_operands(self, operands: Dict):
        return self.copy(expr=operands["expr"])
    
    def __str__(self):
        return str(self.expr) + f"/*{self.string}*/"
    
    def shortStr(self) -> str:
        return self.expr.terminalShortStr() + f"/*{self.string}*/"


class KernelBuilder:
    def __init__(self, expr: dtl.Expr, debug_comments=0):
        self._expr = expr
        self.codeNode = None
        self.registered_tensor_variables = {}
        self.registered_tensor_instantiations = {}
        self._namer = NameGenerator(prefix="t_")
        self._index_namer = NameGenerator(prefix="_")
        self.debug_comments = debug_comments
    
    def build(self):
        # self._expr = make_Index_names_unique_CSE(self._expr)
        # visualise.plot_dag(self._expr, view=True, label_edges=True, short_strs=True, skip_terminals=False, show_types=True)
        # print(str(self._expr))
        
        tensorInputs = frozenset()
        indexInputs = {}
        inputIndexSpaces = {}

        tensorInputs, vectorSpaceInputs = self.get_inputs(self._expr)
        tensorInputs = [i for i in tensorInputs]
        vectorSpaceInputs = [vs for vs in vectorSpaceInputs]
        tensorInputNames = [e.name for e in tensorInputs]
        vectorSpaceInputNames = [e.name for e in vectorSpaceInputs]
        if any(vs in tensorInputNames for vs in vectorSpaceInputNames):
            raise ValueError(f"TensorVariable names and UnknownVectorSpaceVariable Names must be unique\n"
                             f"found: Tensor Variables: {tensorInputNames}\n"
                             f" Vector Space Variables: {vectorSpaceInputNames}")
        tensorInputShapes = [e.type.result.dims for e in tensorInputs]
        etype  = self._expr.type
        for i in etype.indices:
            if i not in indexInputs:
                indexInputs[i] = i.name #self._index_namer.next("ex_"+i.name+"_")
                inputIndexSpaces[i] = etype.spaceOf(i)
            elif etype.spaceOf(i) != inputIndexSpaces[i]:
                raise ValueError(f"Argument index {i} must act on the same space in all expressions given")

        exprCode, expression = self._get_expression((None,self._expr), indexInputs, tuple())
        inputNames = tensorInputNames + list(indexInputs.values())
        inputShapes = tensorInputShapes + [inputIndexSpaces[i] for i in indexInputs]
        
        self.codeNode = Func(inputNames, inputShapes, vectorSpaceInputNames, exprCode, expression)
        # print(self.codeNode.code())
        return self.codeNode.do
        


    @functools.singledispatchmethod
    def get_inputs(self, expr) -> tuple[frozenset, frozenset]:
        tv = frozenset()
        vs = frozenset()
        for child in traversal.allOperands(expr.operands):
            child_tv, child_vs = self.get_inputs(child)
            tv = tv.union(child_tv)
            vs = vs.union(child_vs)
        return tv, vs
    
    @get_inputs.register
    def _(self, expr: dtl.TensorVariable):
        tvar_name = expr.name
        if expr not in self.registered_tensor_variables:
            shape = expr.type.result
            self.registered_tensor_variables[expr] = (tvar_name, shape)

        tv = frozenset([expr])
        vs = frozenset()
        for child in traversal.allOperands(expr.operands):
            child_tv, child_vs = self.get_inputs(child)
            tv = tv.union(child_tv)
            vs = vs.union(child_vs)
        return tv, vs

    @get_inputs.register
    def _(self, expr: dtl.UnknownSizeVectorSpace):
        tv = frozenset()
        vs = frozenset([expr])
        for child in traversal.allOperands(expr.operands):
            child_tv, child_vs = self.get_inputs(child)
            tv = tv.union(child_tv)
            vs = vs.union(child_vs)
        return tv, vs

    @get_inputs.register
    def _(self, expr: PythonDtlNode):
        return expr.get_inputs(self)

    
    def _get_expression(self, labelledExpr: typing.Tuple[typing.Union[str, None], dtl.Expr], indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        l, e = labelledExpr
        if l is not None:
            path = (*path, l)
        if frozenset(indexMap.keys()) != e.type.indices:
            raise ValueError(
                f"Internal Compiler Error: Indices map {str(indexMap)} does not match type {str(e.type)} in {str(e)} at path {path}")
        inst, expression = self._get_expression_r(e, indexMap, path)
        if self.debug_comments >0:
            comments = [CodeComment(f"Inst for: {e.shortStr()}")]
            if self.debug_comments > 1:
                comments.append(
                        CodeComment(f"        : {str(e)}"))
            if self.debug_comments > 2:
                comments.append(
                        CodeComment(f"        : {e.attributes['frame_info']}"))
            inst = SeqNode(comments+[inst])
        return inst, expression
    
    @functools.singledispatchmethod
    def _get_expression_r(self, expr: dtl.Expr, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        """ return tuple:
            (instructions required to produce output. expression that produces output)
            (CodeNode, ExpressionNode)
            """
        raise TypeError
    
    @_get_expression_r.register
    def _(self, expr: dtl.TensorVariable, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        if expr not in self.registered_tensor_variables:
            raise ValueError(f"Internal Compiler Error: Cannot use unregistered TensorVariable {str(expr)} at {path}")
        var, shape = self.registered_tensor_variables[expr]
        return SeqNode([]), ExprConst(var)

    @_get_expression_r.register
    def _(self, expr: dtl.IndexBinding, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        newMap = {i:v for i,v in indexMap.items() if i not in expr.indices}
        return self._get_expression(traversal.operandLabelled(expr, ["expr"]), newMap, path)

    def _match_indices_and_subexprs(self, indices, subexpr, indexMap):
        if isinstance(subexpr, tuple):
            return tuple([self._match_indices_and_subexprs(i,e, indexMap) for i,e in zip(indices, subexpr)])
        elif isinstance(subexpr, ExpressionNode):
            if not isinstance(indices, tuple):
                raise ValueError("Internal Compiler Error: IndexExpr indices do not match result of subExpr")
            if len(indices)>0 and not all([isinstance(i, dtl.Index) or i == dtl.NoneIndex for i in indices]):
                raise ValueError("Internal Compiler Error: IndexExpr indices do not match result of subExpr")
            return ExprIndexed(subexpr, [':' if i == dtl.NoneIndex else indexMap[i] for i in indices])
    @_get_expression_r.register
    def _(self, expr: dtl.IndexExpr, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        inst, subexpr = self._get_expression(traversal.operandLabelled(expr, ["expr"]), indexMap, path)
        newSubexpr = self._match_indices_and_subexprs(expr.indices, subexpr, indexMap)
        return inst, newSubexpr

    @_get_expression_r.register
    def _(self, expr: dtl.Literal, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        return SeqNode([]), ExprScalarLiteral(expr.f)
    
    @_get_expression_r.register
    def _(self, expr: dtl.MulBinOp, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        # lhsIndices = expr.lhs.type.indices
        # lhsIndexMap = {i: v for i, v in indexMap.items() if i in lhsIndices}
        # rhsIndices = expr.rhs.type.indices
        # rhsIndexMap = {i: v for i, v in indexMap.items() if i in rhsIndices}
        l_sub_inst, l_sub_expression = self._get_expression(traversal.operandLabelled(expr, ["lhs"]), indexMap, path)
        r_sub_inst, r_sub_expression = self._get_expression(traversal.operandLabelled(expr, ["rhs"]), indexMap, path)
        return SeqNode([l_sub_inst, r_sub_inst]), ExprMul(l_sub_expression, r_sub_expression)
    
    @_get_expression_r.register
    def _(self, expr: dtl.AddBinOp, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        # lhsIndices = expr.lhs.type.indices
        # lhsIndexMap = {i: v for i, v in indexMap.items() if i in lhsIndices}
        # rhsIndices = expr.rhs.type.indices
        # rhsIndexMap = {i: v for i, v in indexMap.items() if i in rhsIndices}
        l_sub_inst, l_sub_expression = self._get_expression(traversal.operandLabelled(expr, ["lhs"]), indexMap, path)
        r_sub_inst, r_sub_expression = self._get_expression(traversal.operandLabelled(expr, ["rhs"]), indexMap, path)
        return SeqNode([l_sub_inst, r_sub_inst]), ExprAdd(l_sub_expression, r_sub_expression)

    @_get_expression_r.register
    def _(self, expr: dtl.SubBinOp, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        # lhsIndices = expr.lhs.type.indices
        # lhsIndexMap = {i: v for i, v in indexMap.items() if i in lhsIndices}
        # rhsIndices = expr.rhs.type.indices
        # rhsIndexMap = {i: v for i, v in indexMap.items() if i in rhsIndices}
        l_sub_inst, l_sub_expression = self._get_expression(traversal.operandLabelled(expr, ["lhs"]), indexMap, path)
        r_sub_inst, r_sub_expression = self._get_expression(traversal.operandLabelled(expr, ["rhs"]), indexMap, path)
        return SeqNode([l_sub_inst, r_sub_inst]), ExprSub(l_sub_expression, r_sub_expression)

    @_get_expression_r.register
    def _(self, expr: dtl.IndexSum, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        t_var = self._namer.next()
        newMap = dict(indexMap)
        for i in expr.sum_indices:
            newMap[i]=self._index_namer.next(i.name+"_")
        sub_inst, sub_expression = self._get_expression(traversal.operandLabelled(expr, ["expr"]), newMap, path)
        loop = SeqNode([sub_inst,
                        Accumulate(t_var, [], sub_expression)])
        exprType :DTLType = expr.expr.type
        for i in expr.sum_indices:
            space = exprType.spaceOf(i)
            if isinstance(space, dtl.VectorSpace):
                extent = space.dim
            elif isinstance(space, dtl.UnknownSizeVectorSpace):
                extent = space.name
            else:
                raise TypeError("Unsupported space type")
            loop = Loop(newMap[i], extent, loop)

        if not exprType.result.isSingular or not isinstance(exprType.result, dtl.ShapeType):
            raise NotImplementedError(f"Index Sum over tuple tensors ({exprType.result}) is not supported: {str(expr)}")
        make_temp = AssignTemp(t_var, ExprConst('0')) if exprType.result.isScalar  else InitTensor(t_var,  [d.dim for d in exprType.result.dims])
        sum_inst = SeqNode([make_temp, loop])
        return sum_inst, ExprConst(t_var)
    
    def _init_tensor_for_all_tuple(self, exprs, result:dtl.ResultType, output_shape:dtl.DeindexFormatTypeHint, newMap: typing.Dict[dtl.Index, str]):
        if isinstance(result, dtl.ShapeType):
            if not isinstance(output_shape, tuple):
                raise ValueError("Internal Compiler Error: mismatched result type from DeindexExpr to its child")
            if not isinstance(exprs, ExpressionNode):
                raise ValueError("Internal Compiler Error: mismatched result type from DeindexExpr to expressionNode produced")
            tvar_name = self._namer.next()
            return InitTensor(tvar_name, [d.name if isinstance(d, dtl.UnknownSizeVectorSpace) else d.dim for d in result.dims]), AssignTensor(tvar_name, [newMap[o] if isinstance(o, dtl.Index) else ':' for o in output_shape], exprs), ExprConst(tvar_name)
        
        elif isinstance(result, dtl.ResultTupleType):
            if not isinstance(output_shape, tuple):
                raise ValueError("Internal Compiler Error: mismatched result type from DeindexExpr to its child")
            if not isinstance(exprs, tuple):
                raise ValueError("Internal Compiler Error: mismatched result type from DeindexExpr to expressionNode tuple produced")
            return zip(*[self._init_tensor_for_all_tuple(e,r,s,newMap) for e,r,s in zip(exprs, result.results, output_shape)])
    @_get_expression_r.register
    def _(self, expr: dtl.DeindexExpr, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        newMap = dict(indexMap)
        for i in expr.indices:
            newMap[i]=self._index_namer.next(i.name+"_")
        sub_inst, sub_expression = self._get_expression(traversal.operandLabelled(expr, ["expr"]), newMap, path)
        inits, acc, exprs = self._init_tensor_for_all_tuple(sub_expression, expr.type.result, expr.output_shape, newMap)
        allInits = traversal.flattenTupleTreeToList(inits)
        allAcc = traversal.flattenTupleTreeToList(acc)
        
        inst = list(allInits)
        loop = SeqNode([sub_inst]+allAcc)
        childtype = expr.expr.type
        for i in expr.indices:
            vs = childtype.spaceOf(i)
            if isinstance(vs, dtl.UnknownSizeVectorSpace):
                loop = Loop(newMap[i], childtype.spaceOf(i).name, loop)
            else:
                loop = Loop(newMap[i], childtype.spaceOf(i).dim, loop)

        inst.append(loop)
        code = SeqNode(inst)
        return code, exprs


    @_get_expression_r.register
    def _(self, expr: dtl.ExprTuple, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        # tIndices = [e.type.indices for e in expr.exprs]
        # tIndexMaps = [{i: v for i,v in indexMap.items() if i in idxs} for idxs in tIndices]
        tSubInsts, tSubExprs = zip(*[self._get_expression(traversal.operandLabelled(expr, ["exprs", i]), indexMap.copy(), path) for (i, e) in enumerate(expr.exprs)])
        return SeqNode(tSubInsts), tuple(tSubExprs)

    @_get_expression_r.register
    def _(self, expr: dtl.IndexedExprTuple, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        subInst, subExprs = self._get_expression(traversal.operandLabelled(expr, ["expr"]), indexMap, path)
        return subInst, subExprs[expr.n]
    
    @_get_expression_r.register
    def _(self, expr:PythonDtlNode, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        return expr.get_expression(self, indexMap, path)

    def _make_InstantiationExprNode_init_tensors(self, result: dtl.ResultType):
        if isinstance(result, dtl.ShapeType):
            tvar_name = self._namer.next()
            return tvar_name, InitTensor(tvar_name, [d.dim for d in result.dims])
        elif isinstance(result, dtl.ResultTupleType):
            names, codes = zip(*[self._make_InstantiationExprNode_init_tensors(r) for r in result.results])
            return tuple(names), SeqNode(codes)

    def _make_InstantiationExprNode_assign_tensors(self, names, expressions):
        if isinstance(expressions, ExpressionNode):
            return AssignTemp(names, expressions), ExprConst(names)
        elif isinstance(expressions, tuple):
            assigns, subNames = zip(
                *[self._make_InstantiationExprNode_assign_tensors(n, e) for n, e in zip(names, expressions)])
            return SeqNode(assigns), tuple(subNames)

    @_get_expression_r.register
    def _(self, expr: InstantiationExprNode, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str, ...],
          **kwargs):
        exprType = expr.type
        tvar_comment_name = f"P_{''.join(str(p) for p in path)}"
        print(f"build Instantiation:    {tvar_comment_name}")
        freeIdxs = list(exprType.indices)
        scopes = [traversal.get_scope(self._expr, i, path) for i in freeIdxs]
        scopePairs = frozenset([(idx, tuple(s)) for idx, s in zip(freeIdxs, scopes)])
        key = (expr, scopePairs)
        if key in self.registered_tensor_instantiations:
            name, shape, expressions = self.registered_tensor_instantiations[key]
            info = f"Skipping Instantiation: {tvar_comment_name} already built as {name} : {shape} : {traversal.forallTupleTreeFold(expressions, lambda e: e.code(), tuple)}"
            print(info)
            return SeqNode([CodeComment(info)]), expressions
        else:
            # Init Np array for output
            names, inst = self._make_InstantiationExprNode_init_tensors(exprType.result)
        
            print(f"building Instantiation: {tvar_comment_name} into {names}")
            # generate the inner expressions
            expr_inst, expression = self._get_expression(traversal.operandLabelled(expr, ["expr"]), indexMap, path)
        
            assigns, outputExpressions = self._make_InstantiationExprNode_assign_tensors(names, expression)
        
            code = SeqNode(
                [CodeComment(f"Init for {tvar_comment_name} :: {exprType} as {names}"), inst, CodeComment("do Subexp"), expr_inst,
                 CodeComment(f"assign {names}"), assigns, CodeComment(f"Done {tvar_comment_name} -> {names}")])
            # self.instructions.append(code.code(1))
            # self.codeNodes.append(code)
            self.registered_tensor_instantiations[key] = (tvar_comment_name, exprType.result, outputExpressions)
            print(
                f"built Instantiation:    {tvar_comment_name} : {str(exprType)} : {traversal.forallTupleTreeFold(outputExpressions, lambda e: e.code(), tuple)}")
            return code, outputExpressions
    
    @_get_expression_r.register
    def _(self, expr: SequenceExprNode, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str, ...],
          **kwargs):
        preIndices = expr.pre.type.indices
        preIndexMap = indexMap#{i: v for i, v in indexMap.items() if i in preIndices}
        pre_expr_inst, pre_expression = self._get_expression(traversal.operandLabelled(expr, ["pre"]), preIndexMap, path)
        expr_inst, expression = self._get_expression(traversal.operandLabelled(expr, ["expr"]), indexMap, path)
        throwing = ','.join([e.code() for e in traversal.flattenTupleTreeToList(pre_expression)])
        return SeqNode([pre_expr_inst, expr_inst, CodeComment("Throwing away: "+throwing)]), expression
    
    @_get_expression_r.register
    def _(self, expr: IndexRebinding, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        newMap = {i:v for i,v in indexMap.items() if i not in expr.inner_indices}
        for i_out, i_in in zip(expr.outer_indices, expr.inner_indices):
            newMap[i_out] = indexMap[i_in]
        return self._get_expression(traversal.operandLabelled(expr, ["expr"]), newMap, path)
    
    @_get_expression_r.register
    def _(self, expr: PrintExprNode, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        expr_inst, expression = self._get_expression(traversal.operandLabelled(expr, ["expr"]), indexMap, path)
        return SeqNode([CodeComment(expr.string),expr_inst]), ExprPrint(expression, expr.string)
    
    @_get_expression_r.register
    def _(self, expr: dtlMatrix.Invert, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        expr_inst, expression = self._get_expression(traversal.operandLabelled(expr, ["expr"]), indexMap, path)
        return expr_inst, ExprNdMatInv(expression, skip_zero=expr.skip_zero)
    