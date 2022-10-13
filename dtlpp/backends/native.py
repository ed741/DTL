import abc
import typing

import dtl
import itertools
import functools
import numpy as np

from dtlutils import visualize
from dtlutils.names import make_Index_names_unique

from dtlutils.traversal import path_id


class ExpressionNode(abc.ABC):
    @abc.abstractmethod
    def code(self) -> str:
        pass

    @abc.abstractmethod
    def do(self, args: typing.Dict[str, typing.Any]):
        pass
    
    def __str__(self):
        return self.code()


class ExprTensor(ExpressionNode):
    def __init__(self, name: str, indices: typing.List[str]):
        self.name = name
        self.indices = indices
    
    def code(self):
        return f"{self.name}[{','.join(self.indices)}]"

    def do(self, args: typing.Dict[str, typing.Any]):
        idx = tuple(args[i] for i in self.indices)
        return args[self.name][idx]


class ExprAdd(ExpressionNode):
    def __init__(self, lhs: ExpressionNode, rhs: ExpressionNode):
        self.lhs = lhs
        self.rhs = rhs
    
    def code(self):
        return f"({self.lhs} + {self.rhs})"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        return self.lhs.do(args) + self.rhs.do(args)


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
    def __init__(self, child: CodeNode):
        self.child = child
    
    def do(self, **kwargs):
        self.child.do(kwargs)
        return kwargs['P_']
        
        
class Loop(CodeNode):
    def __init__(self, index: str, extent: int, child: CodeNode):
        self.index = index
        self.extent = extent
        self.child = child
    
    def code(self, indent:int) -> str:
        return f"{'    '*indent}for {self.index} in range({self.extent}):\n" \
               f"{self.child.code(indent+1)}\n" \
               f"{'    '*indent}#Done {self.index} loop"

    def do(self, args: typing.Dict[str, typing.Any]):
        for i in range(self.extent):
            args[self.index] = i
            self.child.do(args)


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
        args[self.lhs_name][tuple(args[i] for i in self.lhs_indices)] = self.rhs.do(args)


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
    def __init__(self, name: str, indices: typing.List[int]):
        self.name = name
        self.indices = indices
    
    def code(self, indent: int) -> str:
        return f"{'    ' * indent}{self.name}=np.zeros([{','.join(str(i) for i in self.indices)}])"
    
    def do(self, args: typing.Dict[str, typing.Any]):
        args[self.name]= np.zeros(self.indices)



class NameGenerator:
    def __init__(self, prefix="", suffix=""):
        if not (prefix or suffix):
            raise ValueError
        
        self._prefix = prefix
        self._suffix = suffix
        self._counter = itertools.count()
    
    def next(self):
        return f"{self._prefix}{next(self._counter)}{self._suffix}"


class KernelBuilder:
    def __init__(self, expr):
        self._expr = expr
        self.domains = []
        self.instructions = []
        self.codeNodes = []
        self.kernel_data = []
        self.registered_tensor_variables = {}
        self._namer = NameGenerator(prefix="_t_")
    
    def build(self):
        print("buildA")
        self._expr = make_Index_names_unique(self._expr)
        visualize.plot_dag(self._expr, view=True, label_edges=True)
        
        inputs = self._get_inputs(self._expr)
        self.instructions.append(f"def func({','.join(inputs)}):")
        self._collect_bits(self._expr, [])
        self.instructions.append(f"    return P_")
        print("Instructions::\n")
        code = "\n".join(self.instructions)
        print(code)
        return Func(SeqNode(self.codeNodes)).do
        


    @functools.singledispatchmethod
    def _get_inputs(self, expr):
        s = frozenset()
        for child in expr.operands:
            s = s.union(self._get_inputs(child))
        return s
        # return frozenset.union(frozenset(), (self._get_inputs(child) for child in expr.operands))
    
    @_get_inputs.register
    def _(self, expr: dtl.TensorVariable):
        return frozenset(expr.name)

    @functools.singledispatchmethod
    def _get_domains(self, expr: dtl.Node):
        return frozenset()
    
    @_get_domains.register
    def _(self, expr: dtl.IndexSum):
        domains = set()
        for idx in expr.sum_indices:
            space = expr.index_spaces[idx]
            if isinstance(space, dtl.UnknownSizeVectorSpace):
                raise NotImplementedError
            size = space.dim
            domain = f"{{ [{idx.name}]: 0 <= {idx.name} < {size} }}"
            domains.add(domain)
        return frozenset(domains) | self._get_domains(expr.scalar_expr)
    
    @_get_domains.register
    def _(self, expr: dtl.ScalarExpr):
        return frozenset.union(*[self._get_domains(child) for child in expr.operands])
    
    @functools.singledispatchmethod
    def _get_expression(self, expr: dtl.ScalarExpr):
        raise TypeError
    
    @_get_expression.register
    def _(self, expr: dtl.IndexSum):
        t_var = self._namer.next()
        sub_inst, sub_expression = self._get_expression(expr.scalar_expr)
        loop = SeqNode([sub_inst,
                        Accumulate(t_var, [], sub_expression)])
        for i in expr.sum_indices:
            loop = Loop(i.name, expr.index_spaces[i].dim, loop)

        sum_inst = SeqNode([AssignTemp(t_var, ExprConst('0')), loop])
        return sum_inst, ExprConst(t_var)

    @_get_expression.register
    def _(self, expr: dtl.MulBinOp):
        l_sub_inst, l_sub_expression = self._get_expression(expr.lhs)
        r_sub_inst, r_sub_expression = self._get_expression(expr.rhs)
        return SeqNode([l_sub_inst, r_sub_inst]), ExprMul(l_sub_expression, r_sub_expression)
    
    @_get_expression.register
    def _(self, expr: dtl.AddBinOp):
        l_sub_inst, l_sub_expression = self._get_expression(expr.lhs)
        r_sub_inst, r_sub_expression = self._get_expression(expr.rhs)
        return SeqNode([l_sub_inst, r_sub_inst]), ExprAdd(l_sub_expression, r_sub_expression)
    
    @_get_expression.register
    def _(self, expr: dtl.IndexedTensor):
        return SeqNode([]), ExprTensor(self.registered_tensor_variables[expr.tensor_expr][0], [i.name for i in expr.tensor_indices])
    
    @_get_expression.register
    def _(self, expr: dtl.TensorVariable):
        return TypeError
    
    @_get_expression.register
    def _(self, expr: dtl.deIndex):
        return TypeError
    
    @functools.singledispatchmethod
    def _collect_bits(self, expr, path, **kwargs):
        for child in expr.operands:
            self._collect_bits(child, path + [path_id(expr, child)])
    
    @_collect_bits.register
    def _(self, expr: dtl.deIndex, path, **kwargs):
        tvar_name = f"P_{'_'.join(str(p) for p in path)}"
        print(f"build deIndex: {tvar_name}")
        if expr in self.registered_tensor_variables:
            print(f"{tvar_name} already built as {self.registered_tensor_variables[expr]}")
        else:
            self.registered_tensor_variables[expr] = (tvar_name, shape_from_TensorExpr(expr))
            
            # ensure tensors we rely on exist
            self._collect_bits(expr.scalar_expr, path + [path_id(expr, expr.scalar_expr)])
            
            # Init Np array for output
            inst = [
                InitTensor(tvar_name, [expr.index_spaces[k].dim for k in expr.indices])
            ]
            # generate the inner expressions
            expr_inst, expression = self._get_expression(expr.scalar_expr)
            loop = SeqNode([expr_inst,
                            AssignTensor(tvar_name, [i.name for i in expr.indices], expression)])
            # generate loops to fill the output
            for i in expr.indices:
                loop = Loop(i.name, expr.index_spaces[i].dim, loop)
            
            inst.append(loop)
            code = SeqNode(inst)
            self.instructions.append(code.code(1))
            self.codeNodes.append(code)
    
    @_collect_bits.register
    def _(self, tensor_variable: dtl.TensorVariable, path, **kwargs):
        tvar_name = tensor_variable.name
        if tensor_variable not in self.registered_tensor_variables:
            shape = shape_from_TensorExpr(tensor_variable)
            self.registered_tensor_variables[tensor_variable] = (tvar_name, shape)
    
    

def shape_from_TensorExpr(expr: dtl.TensorExpr):
    shape = []
    for v_space in expr.space.spaces:
        if isinstance(v_space, dtl.UnknownSizeVectorSpace):
            raise NotImplementedError
        shape.append(v_space.dim)
    shape = tuple(shape)
    return shape
    
    # @_collect_bits.register
    # def _(self, index_sum: dtl.IndexSum, path, **kwargs):
    #     print("build IndexSum")
    #     # for index in index_sum.sum_indices:
    #     #     #iname = f"{index.name}.{_get_scope(index, expr, path, root)}"
    #     #     iname = index_name(index, path)  # placeholder
    #     #
    #     #
    #     #     if isinstance(index_sum.index_spaces[index], dtl.UnknownSizeVectorSpace):
    #     #         raise NotImplementedError
    #     #
    #     #     size = index_sum.index_spaces[index].dim #if isinstance(expr.tensor_spaces[index], RealVectorSpace)
    #     #     # if small
    #     #         # size = "5"
    #     #     # else:
    #     #         # param_name = "myuniqueparam"
    #     #         # size = param_name
    #     #         # param = lp.ValueArg(param_name, dtype=np.int32)
    #     #         # self.kernel_data.append(param)
    #     #
    #     #     domain = f"{{ [{iname}]: 0 <= {iname} < {size} }}"
    #     #     self.domains.append(domain)
    #
    #     # B[i, j]
    
    #     #Assume:
    #     #index_sum.scalar_expr is IndexedTensor
    #     #index_sum.scalar_expr.tensor_expr is TensorVariable
    #     indexed_tensor = index_sum.scalar_expr
    #     tvar_name = indexed_tensor.tensor_expr.name  # B
    #     indices = tuple(pym.var(index_name(idx, get_scope(self._expr, idx, path))) for idx in indexed_tensor.tensor_indices)  # (i, j)
    #     expression = pym.subscript(pym.var(tvar_name), indices) #B[i,j]
    #
    #     # register B
    
    #     shape = tuple(indexed_tensor.index_spaces[idx].dim for idx in indexed_tensor.indices)
    #     tvar = lp.GlobalArg(tvar_name, dtype=np.float64, shape=shape)
    #     self.kernel_data.append(tvar)
    #
    #
    #     within_inames = frozenset({index_name(idx, get_scope(self._expr, idx, path)) for idx in indexed_tensor.tensor_indices})  # (i, j)indexed_tensor.tensor_  # {i, j}
    #     return expression, within_inames

