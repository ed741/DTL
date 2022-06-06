import abc
import collections
import collections.abc
import typing
import numbers
from typing import List, Iterable, FrozenSet, Union, Dict, Any, Tuple
from inspect import currentframe, getframeinfo

import pytools


class Node(pytools.ImmutableRecord, abc.ABC):
    fields = {"attrs"}
    
    def __init__(self, attrs: Dict[str, Any] = None, **kwargs):
        if attrs is None:
            attrs = {}
        if not isinstance(attrs, Dict):
            attrs = {k: v for k, v in attrs}
        if "frame_info" not in attrs:
            frame = currentframe()
            here = getframeinfo(currentframe())
            while (getframeinfo(frame).filename is here.filename):
                frame = frame.f_back
            info = getframeinfo(frame)
            attrs["frame_info"] = (info.filename, info.lineno)
        
        for kwarg in kwargs:
            if kwarg not in self.fields:
                raise ValueError(f"'{kwarg}' not in {self.fields}:: an unexpected keyword argument has been given to "
                                 f"Node constructor.")
        
        t = tuple((k, attrs[k]) for k in sorted(attrs.keys()))
        super().__init__(attrs=t, **kwargs)
    
    @property
    @abc.abstractmethod
    def operands(self):
        pass
    
    @abc.abstractmethod
    def with_operands(self, operands: List):
        pass
    
    @property
    def attributes(self) -> Dict:
        d = {k: v for k, v in self.attrs}
        return d
    
    def has_attribute(self, key):
        for k, v in self.attrs:
            if k == key:
                return True
        return False
    
    def with_attribute(self, key, value):
        return self.with_attributes(**{key: value})
    
    def with_attributes(self, **kwargs):
        d = {k: v for k, v in self.attrs}
        d.update(kwargs)
        return self.copy(attrs=d)


class Terminal(Node, abc.ABC):
    operands = ()
    
    def with_operands(self, operands: List):
        return self.copy()


class Index(Terminal):
    fields = Terminal.fields | {"name"}
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
    
    def __str__(self) -> str:
        return self.name


def make_indices(*names: str):
    """Return a tuple of `Index` instances with given names.

    Parameters
    ----------
    *names
        Iterable of names to give the new indices.

    Returns
    -------
    tuple
        The new `Index` objects.
    """
    return tuple(Index(name) for name in names)


class VectorSpaceVariable(Terminal, abc.ABC):
    def __mul__(self, other):
        if isinstance(other, VectorSpaceVariable):
            return TensorSpace([self, other])
        else:
            return NotImplemented
    
    def __pow__(self, other):
        if isinstance(other, numbers.Integral):
            return TensorSpace([self] * other)
        else:
            return NotImplemented


class UnknownSizeVectorSpace(VectorSpaceVariable):
    fields = VectorSpaceVariable.fields | {"name"}
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
    
    def __str__(self) -> str:
        return self.name


class VectorSpace(VectorSpaceVariable, abc.ABC):
    fields = VectorSpaceVariable.fields | {"dim"}
    
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, **kwargs)
    
    @property
    @abc.abstractmethod
    def symbol(self) -> str:
        pass
    
    def __str__(self) -> str:
        return f"{self.symbol}{self.dim}"


class RealVectorSpace(VectorSpace):
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, **kwargs)
    
    @property
    def symbol(self) -> str:
        return "R"


class TensorSpace(Node):
    fields = Node.fields | {"spaces"}
    
    def __init__(self, spaces: Iterable[VectorSpaceVariable], **kwargs):
        symbol = None
        for space in spaces:
            if isinstance(space, VectorSpace):
                if symbol == None:
                    symbol = space.symbol
                if symbol != space.symbol:
                    raise ValueError
        super().__init__(spaces=tuple(spaces), **kwargs)
    
    def __iter__(self):
        return iter(self.spaces)
    
    def __str__(self) -> str:
        return "x".join(str(space) for space in self.spaces)
    
    @property
    def shape(self):
        return tuple(space.dim for space in self.spaces)
    
    def __mul__(self, other):
        if isinstance(other, VectorSpaceVariable):
            return TensorSpace((list(self.spaces) + [other]))
        else:
            return NotImplemented
    
    @property
    def operands(self) -> Iterable[Node]:
        return self.spaces
    
    def with_operands(self, operands: List):
        return self.copy(spaces=operands)
    
    def new(self, name: str):
        return TensorVariable(self, name)


class TensorExpr(Node, abc.ABC):
    @property
    @abc.abstractmethod
    def space(self):
        """The `TensorSpace` of the expression."""
    
    def __getitem__(self, indices: Iterable[Index]):
        if not isinstance(indices, collections.abc.Iterable):
            indices = (indices,)
        indices = tuple(indices)
        return IndexedTensor(self, indices)


class ScalarExpr(Node, abc.ABC):
    @property
    @abc.abstractmethod
    def indices(self) -> Iterable[Index]:
        """Return all the indices used in this scalar_expression
        """
        pass
    
    @property
    @abc.abstractmethod
    def free_indices(self) -> Iterable[Index]:
        """Return all the indices used in this scalar_expression
         that have not been explicitly contracted
        """
        pass
    
    def forall(self, *indices: Index) -> "deIndex":
        return deIndex(self, indices)
    
    @property
    @abc.abstractmethod
    def index_spaces(self):
        """Return a dict containing the expression indices
        coupled with their spaces.
        """
        pass
    
    def __add__(self, other: "ScalarExpr") -> "ScalarExpr":
        return AddBinOp(self, other)
    
    def __mul__(self, other: "ScalarExpr") -> "ScalarExpr":
        return MulBinOp(self, other)


class Literal(ScalarExpr, Terminal):
    fields = ScalarExpr.fields | {"f"}
    
    def __init__(self, f: float, **kwargs):
        super().__init__(f=f, **kwargs)
    
    indices = ()
    free_indices = ()
    
    @property
    def index_spaces(self):
        return {}
    
    def __str__(self) -> str:
        return str(self.f)


class IndexedTensor(ScalarExpr):
    fields = ScalarExpr.fields | {"tensor_expr", "tensor_indices"}
    
    def __init__(self, tensor_expr: TensorExpr, tensor_indices: Iterable[Index], **kwargs):
        super().__init__(tensor_expr=tensor_expr, tensor_indices=tuple(tensor_indices), **kwargs)
    
    @property
    def indices(self):
        return self.tensor_indices
    
    @property
    def free_indices(self) -> Iterable[Index]:
        return self.indices
    
    @property
    def index_spaces(self):
        return dict(zip(self.tensor_indices, self.tensor_expr.space))
    
    @property
    def operands(self) -> Iterable[Node]:
        return self.tensor_expr, *self.tensor_indices
    
    def with_operands(self, operands: List):
        return self.copy(tensor_expr=operands[0], tensor_indices=tuple(operands[1:]))
    
    def postOrder(self, fn: typing.Callable):
        return fn(self.copy(tensor_expr=self.tensor_expr.postOrder(fn)),
                  tensor_indices=self.tensor_indices.postOrder(fn))
    
    def __str__(self) -> str:
        return f"{self.tensor_expr}[{','.join(map(str, self.indices))}]"


class BinOp(ScalarExpr, abc.ABC):
    fields = ScalarExpr.fields | {"lhs", "rhs"}
    
    def __init__(self, lhs: ScalarExpr, rhs: ScalarExpr, **kwargs):
        # check that common indices share the same space
        if any(
            lhs.index_spaces[idx] != rhs.index_spaces[idx]
            for idx in set(lhs.indices) & set(rhs.indices)
        ):
            raise ValueError(
                "Indices common across subexpressions must act over the same space"
            )
        super().__init__(lhs=lhs, rhs=rhs, **kwargs)
    
    @property
    def indices(self) -> FrozenSet[Index]:
        # binops store their indices in a set rather than a tuple because
        # the ordering of the indices is only determined by a surrounding
        # unindex node.
        return frozenset(self.lhs.indices) | frozenset(self.rhs.indices)

    @property
    def free_indices(self) -> Iterable[Index]:
        return self.indices
    
    @property
    def index_spaces(self):
        return {**self.lhs.index_spaces, **self.rhs.index_spaces}
    
    @property
    def operands(self) -> Iterable[Node]:
        return self.lhs, self.rhs
    
    def with_operands(self, operands: List):
        return self.copy(lhs=operands[0], rhs=operands[1])
    
    def postOrder(self, fn: typing.Callable):
        return fn(self.copy(lhs=self.lhs.postOrder(fn)), rhs=self.rhs.postOrder(fn))
    
    def __str__(self) -> str:
        return f"{self.lhs} {self.symbol} {self.rhs}"


class MulBinOp(BinOp):
    symbol = "*"


class AddBinOp(BinOp):
    symbol = "+"


class UnaryOp(ScalarExpr, abc.ABC):
    fields = ScalarExpr.fields | {"scalar_expr"}
    
    def __init__(self, scalar_expr: IndexedTensor, **kwargs):
        super().__init__(scalar_expr=scalar_expr, **kwargs)
    
    @property
    def operands(self):
        return self.tensor,
    
    def with_operands(self, operands: List):
        return self.copy(tensor=operands[0])
    
    def postOrder(self, fn: typing.Callable):
        return fn(self.copy(tensor=self.tensor.postOrder(fn)))
    
    @property
    def indices(self) -> Iterable[Index]:
        return self.tensor.indices

    @property
    def free_indices(self) -> Iterable[Index]:
        return self.indices
    
    @property
    def index_spaces(self):
        return self.tensor.index_spaces
    
    def __str__(self) -> str:
        return f"{self.name}({self.tensor})"


class Abs(UnaryOp):
    name = "abs"


class IndexSum(ScalarExpr):
    fields = ScalarExpr.fields | {"scalar_expr", "sum_indices"}
    
    def __init__(self, scalar_expr: ScalarExpr, sum_indices: Iterable[Index], **kwargs):
        if any(idx not in scalar_expr.indices for idx in sum_indices):
            raise ValueError(
                "Indices summed over must refer to Indices in subexpression"
            )
        super().__init__(scalar_expr=scalar_expr, sum_indices=tuple(sum_indices), **kwargs)
    
    @property
    def indices(self) -> Iterable[Index]:
        return self.scalar_expr.indices

    @property
    def free_indices(self) -> Iterable[Index]:
        return frozenset(self.indices) - frozenset(self.sum_indices)
    
    @property
    def index_spaces(self):
        return self.scalar_expr.index_spaces
    
    @property
    def operands(self):
        return self.scalar_expr, *self.sum_indices
    
    def with_operands(self, operands: List):
        return self.copy(scalar_expr=operands[0], sum_indices=tuple(operands[1:]))
    
    def postOrder(self, fn: typing.Callable):
        new_sub = self.scalar_expr.postOrder(fn)
        new_indices = tuple(op.postOrder(fn) for op in self.sum_indices)
        return fn(self.copy(scalar_expr=new_sub, sum_indices=new_indices))
    
    def __str__(self) -> str:
        return f"Sum[{','.join(map(str, self.sum_indices))}]({self.scalar_expr})"


class TensorVariable(TensorExpr):
    fields = TensorExpr.fields | {"tensor_space", "name"}
    
    def __init__(self, tensor_space: Union[VectorSpace, TensorSpace], name: str, **kwargs):
        if isinstance(tensor_space, VectorSpace):
            tensor_space = TensorSpace((tensor_space,))
        
        super().__init__(tensor_space=tensor_space, name=name, **kwargs)
    
    @property
    def space(self):
        return self.tensor_space
    
    @property
    def operands(self):
        return self.tensor_space,
    
    def with_operands(self, operands: List):
        return self.copy(tensor_space=operands[0])
    
    def postOrder(self, fn: typing.Callable):
        return fn(self.copy(tensor_space=self.space.postOrder(fn)))
    
    # @property
    # def _key(self):
    #     return self.space, self.name
    #
    # def __hash__(self):
    #     return hash(self._key)
    #
    # def __eq__(self, other):
    #     if isinstance(other, TensorVariable):
    #         return self._key == other._key
    #     else:
    #         return NotImplemented
    
    def __str__(self) -> str:
        return f"{self.name}{{{str(self.space)}}}"


class deIndex(TensorExpr):
    fields = TensorExpr.fields | {"scalar_expr", "indices"}
    
    def __init__(self, scalar_expr: ScalarExpr, indices: Iterable[Index], **kwargs):
        if set(indices) != set(scalar_expr.free_indices):
            scalar_expr = IndexSum(scalar_expr, set(scalar_expr.free_indices) - set(indices))
        super().__init__(scalar_expr=scalar_expr, indices=tuple(indices), **kwargs)
    
    @property
    def space(self):
        return TensorSpace(self.index_spaces.values())
    
    @property
    def index_spaces(self):
        return {idx: self.scalar_expr.index_spaces[idx] for idx in self.indices}
    
    @property
    def operands(self) -> Iterable[Node]:
        return self.scalar_expr, *self.indices
    
    def with_operands(self, operands: List):
        return self.copy(scalar_expr=operands[0], indices=tuple(operands[1:]))
    
    def postOrder(self, fn: typing.Callable):
        new_scalar_expr = self.scalar_expr.postOrder(fn)
        new_indices = tuple(op.postOrder(fn) for op in self.indices)
        return fn(self.copy(scalar_expr=new_scalar_expr, indices=new_indices))
    
    def __str__(self) -> str:
        return f"({self.scalar_expr})|{','.join(map(str, self.indices))}|"


class Lambda(Node):
    fields = Node.fields | {"vars", "tensor_expr"}
    
    def __init__(self, vars: List[TensorVariable], tensor_expr: TensorExpr, **kwargs):
        super().__init__(vars=tuple(vars), tensor_expr=tensor_expr, **kwargs)
    
    @property
    def operands(self) -> Iterable[Node]:
        return *self.vars, self.tensor_expr
    
    def with_operands(self, operands: List):
        return self.copy(vars=tuple(operands[:-1]), tensor_expr=operands[-1])
    
    def postOrder(self, fn: typing.Callable):
        new_vars = tuple(op.postOrder(fn) for op in self.vars)
        new_sub = self.tensor_expr.postOrder(fn)
        return fn(self.copy(vars=new_vars, tensor_expr=new_sub))
    
    def __str__(self) -> str:
        return f"λ{','.join([str(v) for v in self.vars])}.{self.tensor_expr}"

# class LambdaApplication(TensorExpr):
#     lambda_node: Lambda
#     args: List[TensorExpr]
#
#     def __init__(self, lambda_node: Lambda, args: List[TensorExpr], **kwargs):
#         super().__init__(**kwargs)
#         for la, ra in zip(lambda_node.vars, args):
#             if la.space != ra.space:
#                 raise ValueError(
#                     f"Lambda Expression applied with mismatched spaces: Lambda expected {la.space}, got {ra.space}"
#                 )
#
#         self.lambda_node = lambda_node
#         self.args = args
#
#     def __str__(self) -> str:
#         return f"{str(self.lambda_node)}({','.join([str(a) for a in self.args])})"
#
#     @property
#     def space(self):
#         return self.lambda_node.sub.space
#
#     @property
#     def operands(self):
#         pass
#
#     @functools.singledispatch
#     def swapArgs(node: Node):
#         return copy()
#
#     @swapArgs.TensorVariable
#     def swapArgs_TensorVariable(node: TensorVariable):
#
