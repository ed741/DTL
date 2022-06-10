import dtl
import loopy as lp
import functools
import pymbolic as pym
import numpy as np

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
        #self._namer = 

    def build(self):
        print("buildA")
        self._collect_bits(self._expr)
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
    def _collect_bits(self, expr, **kwargs):
        raise TypeError

    @_collect_bits.register
    def _(self, expr: dtl.deIndex, **kwargs):
        print("build deIndex")
        for index in expr.indices:
            #iname = f"{index.name}.{_get_scope(index, expr, path, root)}"
            iname = index.name  # placeholder

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

        # A[i]
        # deIndex_at_[0,0,1,2,3,0]
        #tvar_name = pym.var(expr.name)
        tvar_name = "mytemporarytemporary" #FIXME
        pymtvarname = pym.var(tvar_name)
        indices = tuple(pym.var(idx.name) for idx in expr.indices)
        assignee = pym.subscript(pymtvarname, indices) #A[i]

        shape = tuple(expr.index_spaces[idx].dim for idx in expr.indices)
        #temp = lp.TemporaryVariable(tvar_name, dtype=np.float64, shape=shape)
        temp = lp.GlobalArg(tvar_name, dtype=np.float64, shape=shape)
        self.kernel_data.append(temp)

        expression, within_inames = self._collect_bits(expr.scalar_expr)
        #maybe within_inames can be worked out from here, not passed back

        #A[i] = A[i] + B[i,j]
        insn = lp.Assignment(
            assignee, assignee+expression,
            depends_on=frozenset(),
            within_inames=within_inames,
        )
        self.instructions.append(insn)

    @_collect_bits.register
    def _(self, index_sum: dtl.IndexSum, **kwargs):
        print("build IndexSum")
        for index in index_sum.sum_indices:
            #iname = f"{index.name}.{_get_scope(index, expr, path, root)}"
            iname = index.name  # placeholder

            if isinstance(index_sum.index_spaces[index], dtl.UnknownSizeVectorSpace):
                raise NotImplementedError
                
            size = index_sum.index_spaces[index].dim #if isinstance(expr.tensor_spaces[index], RealVectorSpace)
            # if small
                # size = "5"
            # else:
                # param_name = "myuniqueparam"
                # size = param_name
                # param = lp.ValueArg(param_name, dtype=np.int32)
                # self.kernel_data.append(param)

            domain = f"{{ [{iname}]: 0 <= {iname} < {size} }}"
            self.domains.append(domain)

        # B[i, j]
        # FIXME
        #Assume:
        #index_sum.scalar_expr is IndexedTensor
        #index_sum.scalar_expr.tensor_expr is TensorVariable
        indexed_tensor = index_sum.scalar_expr
        tvar_name = indexed_tensor.tensor_expr.name  # B
        indices = tuple(pym.var(idx.name) for idx in indexed_tensor.tensor_indices)  # (i, j)
        expression = pym.subscript(pym.var(tvar_name), indices) #B[i,j]

        # register B
        # FIXME
        shape = tuple(indexed_tensor.index_spaces[idx].dim for idx in indexed_tensor.indices)
        tvar = lp.GlobalArg(tvar_name, dtype=np.float64, shape=shape)
        self.kernel_data.append(tvar)

        within_inames = frozenset({idx.name for idx in indexed_tensor.tensor_indices})  # {i, j}
        return expression, within_inames