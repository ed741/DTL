from typing import Tuple, Dict

import dtl
import itertools
import loopy as lp
import functools
import pymbolic as pym
import numpy as np

from dtl.dtlutils import visualise, traversal
from dtl.passes.names import make_Index_names_unique

from dtl.dtlutils.traversal import path_id


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
        self.kernel_data = []
        self.registered_tensor_variables = {}
        #self._namer = 

    def build(self):
        print("buildA")
        self._expr = make_Index_names_unique(self._expr)
        visualise.plot_dag(self._expr, view=True, label_edges=True)
        # self._collect_bits(self._expr, [])
        expr = self._get_expression(self._expr, [])
        print(expr)
        print("buildB")

        return lp.make_kernel(
            self.domains,
            self.instructions,
            self.kernel_data,
            target=lp.ExecutableCTarget(),
            name="epic_kernel",
            lang_version=(2018, 2),
        )

    @functools.singledispatchmethod
    def _get_domains(self, expr: dtl.Node):
        return frozenset()
    
    @_get_domains.register
    def _(self, expr: dtl.IndexSum):
        domains = set()
        exprType = expr.expr.type
        for idx in expr.sum_indices:
            space = exprType.spaceOf(idx)
            if not isinstance(space, dtl.RealVectorSpace):
                raise NotImplementedError
            size = space.dim
            domain = f"{{ [{idx.name}]: 0 <= {idx.name} < {size} }}"
            domains.add(domain)
        return frozenset(domains) | self._get_domains(expr.scalar_expr)
    
    @_get_domains.register
    def _(self, expr: dtl.Expr):
        s = frozenset()
        for o in traversal.allOperands(expr):
            s = s | self._get_domains(o)
        return s
    
    @functools.singledispatchmethod
    def _get_expression(self, expr: dtl.Expr, path: Tuple[str,...]):
        raise TypeError
    
    @_get_expression.register
    def _(self, expr: dtl.IndexSum, path: Tuple[str,...]):
        inames = tuple(pym.var(idx.name) for idx in expr.sum_indices)
        # return f"sum({','.join(idx.name for idx in expr.sum_indices)}, {self._get_expression(expr.scalar_expr)})"
        return lp.Reduction(lp.library.reduction.SumReductionOperation(), inames, self._get_expression(expr.expr, (*path,".expr")))
        # return lp.Reduction("sum", inames, self._get_expression(expr.scalar_expr))

    @_get_expression.register
    def _(self, expr: dtl.Literal, path: Tuple[str,...]):
        # return f"({self._get_expression(expr.lhs)} * {self._get_expression(expr.rhs)})"
        return expr.f


    @_get_expression.register
    def _(self, expr: dtl.MulBinOp, path: Tuple[str,...]):
        # return f"({self._get_expression(expr.lhs)} * {self._get_expression(expr.rhs)})"
        return self._get_expression(expr.lhs, (*path,".expr")) * self._get_expression(expr.rhs, (*path,".expr"))

    @_get_expression.register
    def _(self, expr: dtl.AddBinOp, path: Tuple[str,...]):
        # return f"({self._get_expression(expr.lhs)} + {self._get_expression(expr.rhs)})"
        return self._get_expression(expr.lhs, (*path,".lhs")) + self._get_expression(expr.rhs, (*path,".rhs"))

    @_get_expression.register
    def _(self, expr: dtl.IndexExpr, path: Tuple[str,...]):
        # if not isinstance(expr.tensor_expr, dtl.TensorVariable):
        #     raise NotImplementedError
        # return f"({self._get_expression(expr.tensor_expr)}[{','.join(idx.name for idx in expr.tensor_indices)}])"
        return pym.subscript(self._get_expression(expr.expr, (*path,".expr")), tuple(pym.var(idx.name) for idx in expr.tensor_indices))

    @_get_expression.register
    def _(self, expr: dtl.TensorVariable, path: Tuple[str,...]):
        return pym.var(expr.name)

    def _init_tensor_for_all_tuple(self, exprs, result: dtl.ResultType, output_shape: dtl.DeindexFormatTypeHint,
                                   newMap: Dict[dtl.Index, str]):
        if isinstance(result, dtl.ShapeType):
            if not isinstance(output_shape, tuple):
                raise ValueError("Internal Compiler Error: mismatched result type from DeindexExpr to its child")
            if not isinstance(exprs, ExpressionNode):
                raise ValueError(
                    "Internal Compiler Error: mismatched result type from DeindexExpr to expressionNode produced")
            tvar_name = self._namer.next()
            return InitTensor(tvar_name, [d.dim for d in result.dims]), AssignTensor(tvar_name, [
                newMap[o] if isinstance(o, dtl.Index) else ':' for o in output_shape], exprs), ExprConst(tvar_name)
    
        elif isinstance(result, dtl.ResultTupleType):
            if not isinstance(output_shape, tuple):
                raise ValueError("Internal Compiler Error: mismatched result type from DeindexExpr to its child")
            if not isinstance(exprs, tuple):
                raise ValueError(
                    "Internal Compiler Error: mismatched result type from DeindexExpr to expressionNode tuple produced")
            return zip(*[self._init_tensor_for_all_tuple(e, r, s, newMap) for e, r, s in
                         zip(exprs, result.results, output_shape)])
    @_get_expression.register
    def _(self, expr: dtl.DeindexExpr, path: Tuple[str,...]):
        # if expr in self.registered_tensor_variables:
        #     return pym.var(self.registered_tensor_variables[expr])
        # else:
        #     raise NotImplementedError
        newMap = dict()
        for i in expr.indices:
            iname = i.name
            vs = expr.expr.type.spaceOf(i)
            if not isinstance(vs, dtl.RealVectorSpace):
                raise NotImplementedError
            size = vs.dim
            domain = f"{{ [{iname}]: 0 <= {iname} < {size} }}"
            self.domains.append(domain)
        sub_inst, sub_expression = self._get_expression(traversal.operandLabelled(expr, ["expr"]), newMap, path)
        inits, acc, exprs = self._init_tensor_for_all_tuple(sub_expression, expr.type.result, expr.output_shape, newMap)
        allInits = traversal.flattenTupleTreeToList(inits)
        allAcc = traversal.flattenTupleTreeToList(acc)

    @functools.singledispatchmethod
    def _collect_bits(self, expr, path, **kwargs):
        for label, child in traversal.allOperandsLabelled(expr.operands):
            self._collect_bits(child, path+[label])

    @_collect_bits.register
    def _(self, expr: dtl.DeindexExpr, path, **kwargs):
        tvar_name = f"P_{'_'.join(str(p) for p in path)}"
        print(f"build deIndex: {tvar_name}")
        if expr in self.registered_tensor_variables:
            print(f"{tvar_name} already built as {self.registered_tensor_variables[expr]}")
        else:
            self.registered_tensor_variables[expr] = tvar_name
            
            self._collect_bits(expr.expr, path + [".expr"])
            
            for index in (expr.indices):
                iname = index.name
                print(f"registering index: {iname}")
                if isinstance(expr.index_spaces[index], dtl.UnknownSizeVectorSpace):
                    raise NotImplementedError
                    
                size = expr.index_spaces[index].dim #if isinstance(expr.tensor_spaces[index], RealVectorSpace)
                # if small
                    # size = "5"
                # else:
                    # param_name = "myuniqueparam"
                    # size = param_name
                    # param = lp.ValueArg(param_name, dtype=np.int32)
                    # self.kernel_data.append(param)
    
                domain = f"{{ [{iname}]: 0 <= {iname} < {size} }}"
                self.domains.append(domain)
            inner_domains = self._get_domains(expr.scalar_expr)
            self.domains.extend(inner_domains)
    
            # A[i]
            # deIndex_at_[0,0,1,2,3,0]
            pymtvarname = pym.var(tvar_name)
            # indices = tuple(pym.var(idx.name) for idx in expr.indices)
            indices = tuple(pym.var(idx.name) for idx in expr.indices)  # (i, j)
            assignee = pym.subscript(pymtvarname, indices) #A[i]
            # assignee = "out[j_0, p_0]"
    
            shape = tuple(expr.index_spaces[idx].dim for idx in expr.indices)
            #temp = lp.TemporaryVariable(tvar_name, dtype=np.float64, shape=shape)
            temp = lp.GlobalArg(tvar_name, dtype=np.float64, shape=shape)
            self.kernel_data.append(temp)
    
            # loopInames = (idx for idx, space in expr.index_spaces)
            loop_inames = frozenset({idx.name for idx in expr.index_spaces.keys()})  # (i, j)indexed_tensor.tensor_  # {i, j}
            expression = self._get_expression(expr.scalar_expr, (*path,".expr"))
            print(assignee)
            print("=")
            print(expression)
            # self._collect_bits(expr.scalar_expr, path+[path_id(expr, expr.scalar_expr)])
            #maybe within_inames can be worked out from here, not passed back
            #A[i] = A[i] + B[i,j]
            insn = lp.Assignment(
                assignee, expression,
                depends_on=frozenset(),
                within_inames=loop_inames,
            )
            self.instructions.append(insn)


    @_collect_bits.register
    def _(self, tensor_variable: dtl.TensorVariable, path, **kwargs):
        tvar_name = tensor_variable.name
        if tensor_variable not in self.registered_tensor_variables:
            self.registered_tensor_variables[tensor_variable] = tvar_name
            shape = []
            for v_space in tensor_variable.tensor_space.spaces:
                if isinstance(v_space, dtl.UnknownSizeVectorSpace):
                    raise NotImplementedError
                shape.append(v_space.dim)
            shape = tuple(shape)
            tvar = lp.GlobalArg(tvar_name, dtype=np.float64, shape=shape)
            self.kernel_data.append(tvar)
            
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

