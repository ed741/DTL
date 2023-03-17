from dtl import *

raise ValueError("This is dead code")

def __scalarExpr_as(self: ScalarExpr, indices: Iterable[Index]):
    if not isinstance(indices, collections.abc.Iterable):
        indices = (indices,)
    indices = tuple(indices)
    return IndexedNDTensor(self, indices)
    
def dag_ext_non_scalar_init():
    print("INIT: Overwriting 'expose' method for ScalarExpr")
    setattr(ScalarExpr, "expose", __scalarExpr_as)
# dtl.ScalarExpr.__dict__.update({"__contains__": scalarExpr_in})


class NdExpr(TensorExpr, abc.ABC):
    
    @property
    @abc.abstractmethod
    def indices(self) -> FrozenSet[Index]:
        """Return all the indices used in this Nd_expression
        """
        pass
    
    @property
    @abc.abstractmethod
    def nd_free_indices(self) -> Iterable[Index]:
        """Return all the indices used in this Nd_expression
         that have not been explicitly contracted or used to produce the inner nd_space or are exposed as the ND_space
        """
        pass
    
    @property
    @abc.abstractmethod
    def index_spaces(self):
        """Return a dict containing the expression indices
        coupled with their spaces.
        """
        pass

    @property
    @abc.abstractmethod
    def nd_space_indices(self):
        """Return a list of the indices producing the nd_space.
            It's possible there are none
        """
        pass

    @property
    @abc.abstractmethod
    def nd_space(self):
        """The Nd `TensorSpace` of the inner expression.
        """
        pass
    
    @property
    def space(self):
        """The Nd `TensorSpace` of the inner expression.
        """
        return self.nd_space
    
    def as_scalar(self):
        return NDScalarExpr(self)
    

class IndexedNDTensor(NdExpr):
    fields = ScalarExpr.fields | {"scalar_expr", "tensor_indices"}
    
    def __init__(self, scalar_expr: ScalarExpr, tensor_indices: Iterable[Index], **kwargs):
        if not all((idx in scalar_expr.free_indices) for idx in tensor_indices): raise ValueError(
                "Indices producing the inner space must be free in the sub expression"
            )
        super().__init__(scalar_expr=scalar_expr, tensor_indices=tuple(tensor_indices), **kwargs)
    
    @property
    def indices(self) -> FrozenSet[Index]:
        return self.scalar_expr.indices
    
    @property
    def nd_free_indices(self):
        return frozenset(self.scalar_expr.free_indices).difference(self.tensor_indices)
    
    @property
    def index_spaces(self):
        return self.scalar_expr.index_spaces
    
    @property
    def nd_space_indices(self):
        return self.tensor_indices
    
    @property
    def nd_space(self):
        return TensorSpace([self.index_spaces[idx] for idx in self.tensor_indices])

    def makes_scope(self, index):
        return index in self.tensor_indices
        
    @property
    def operands(self) -> Iterable[Node]:
        return self.scalar_expr, *self.tensor_indices
    
    def with_operands(self, operands: List):
        return self.copy(scalar_expr=operands[0], tensor_indices=tuple(operands[1:]))
        # i = 2
        # while i < len(operands):
        #     if operands[i] in operands[1:i]:
        #         break
        #     i += 1
        # return self.copy(tensor_expr=operands[0], tensor_indices=tuple(operands[1:i]), expr_indices=tuple(operands[i:]))
    
    def __str__(self) -> str:
        # return f"{self.scalar_expr}|{','.join(str(idx)+('*' if idx in self.tensor_indices else '') for idx in self.indices)}|"
        return f"{self.scalar_expr}/{','.join(str(idx) for idx in self.tensor_indices)}/"


class MatrixInverseNdExprOp(NdExpr):
    fields = NdExpr.fields | {"nd_expr"}
    
    def __init__(self, nd_expr: NdExpr, **kwargs):
        if len(nd_expr.nd_space.spaces) != 2 and (nd_expr.nd_space.spaces[0] == nd_expr.nd_space.spaces[1]):
            raise ValueError("Matrix Inverse requires a square 2-Tensor input")
        super().__init__(nd_expr=nd_expr, **kwargs)
    
    @property
    def indices(self) -> FrozenSet[Index]:
        return self.nd_expr.indices

    @property
    def nd_free_indices(self) -> Iterable[Index]:
        return self.nd_expr.nd_free_indices

    @property
    def index_spaces(self):
        return self.nd_expr.index_spaces

    @property
    def nd_space_indices(self):
        return self.nd_expr.nd_space_indices

    @property
    def nd_space(self):
        return self.nd_expr.nd_space

    @property
    def operands(self):
        return [self.nd_expr]

    def with_operands(self, operands: List):
        return self.copy(nd_expr=operands[0])
    
    # def makes_scope(self, index):
    #     return index in self.nd_expr.nd_space_indices
    
    def __str__(self) -> str:
        return f"Inv({str(self.nd_expr)})"


class MaxNdExprOp(NdExpr):
    fields = ScalarExpr.fields | {"nd_expr"}
    
    def __init__(self, nd_expr: NdExpr, **kwargs):
        super().__init__(nd_expr=nd_expr, **kwargs)
    
    @property
    def indices(self) -> FrozenSet[Index]:
        return self.nd_expr.indices
    
    @property
    def nd_free_indices(self) -> Iterable[Index]:
        return self.nd_expr.nd_free_indices
    
    @property
    def index_spaces(self):
        return self.nd_expr.index_spaces
    
    @property
    def nd_space_indices(self):
        return tuple()
    
    @property
    def nd_space(self):
        return TensorSpace([])
    
    @property
    def operands(self):
        return [self.nd_expr]
    
    def with_operands(self, operands: List):
        return self.copy(nd_expr=operands[0])
    
    # def makes_scope(self, index):
    #     return index in self.nd_expr.nd_space_indices
    
    def __str__(self) -> str:
        return f"Max({str(self.nd_expr)})"


class NDScalarExpr(ScalarExpr):
    fields = ScalarExpr.fields | {"nd_expr", "exposed_indices"}
    
    def __init__(self, nd_expr: NdExpr, **kwargs):
        _exposed_indices = nd_expr.nd_space_indices
        if 'exposed_indices' in kwargs:
            _exposed_indices = tuple(kwargs.pop('exposed_indices'))
            
        if len(_exposed_indices) != len(nd_expr.nd_space_indices):
            ValueError("NDScalar Expression must expose the same number of indices as in the ND expression")
        active_index_spaces = {i:s for i,s in nd_expr.index_spaces.items() if i in nd_expr.nd_free_indices or i in _exposed_indices}
        if any(nd_expr.nd_space.spaces[i] != active_index_spaces[idx] for i, idx in enumerate(_exposed_indices) if idx in active_index_spaces):
            ValueError("NDScalar exposed_indices must act over the same space as free indices already in the subexpression if they are already free in the subexpression")
        super().__init__(nd_expr=nd_expr, exposed_indices=_exposed_indices, **kwargs)
    
    @property
    def indices(self) -> FrozenSet[Index]:
        return self.nd_expr.indices | frozenset(self.exposed_indices)

    @property
    def free_indices(self) -> Iterable[Index]:
        return frozenset(self.nd_expr.nd_free_indices) | frozenset(self.exposed_indices)

    @property
    def index_spaces(self):
        existingMapping = {i:s for i,s in self.nd_expr.index_spaces.items() if i in self.free_indices}
        newMapping = {idx:self.nd_expr.nd_space.spaces[i] for i,idx in enumerate(self.exposed_indices)}
        return {**existingMapping, **newMapping}
    

    @property
    def operands(self):
        return [self.nd_expr, *self.exposed_indices]

    def with_operands(self, operands: List):
        return self.copy(nd_expr=operands[0], exposed_indices=operands[1:])

    def __str__(self) -> str:
        return f":{str(self.nd_expr)};~{','.join([str(i) for i in self.exposed_indices])}~"
    
    