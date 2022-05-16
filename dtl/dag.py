import abc
import collections
import collections.abc
import typing
from dataclasses import dataclass
import numbers
from typing import List, Iterable, FrozenSet, Union
from inspect import currentframe, getframeinfo


class Node(abc.ABC):
    def __init__(self, **kwargs):
        self._attributes = {}
        self._attributes.update(kwargs)
        
        frame = currentframe()
        here = getframeinfo(currentframe())
        while(getframeinfo(frame).filename is here.filename):
            frame = frame.f_back
        frame_info = getframeinfo(frame)
        self._attributes["frame_info"] = frame_info
    
    @property
    @abc.abstractmethod
    def operands(self):
        pass

    @property
    def attributes(self) -> typing.Dict[str, typing.Any]:
        return self._attributes
    

class Terminal(Node):
    operands = ()


@dataclass()
class Index(Terminal):
    _name: str
    
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self._name = name
        
    def __str__(self) -> str:
        return self._name

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o)

    def __hash__(self) -> int:
        return super().__hash__()


def indices(*names: str):
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


@dataclass
class VectorSpaceVariable(Terminal, abc.ABC):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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


@dataclass
class UnknownSizeVectorSpace(VectorSpaceVariable):
    name: str
    
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
    
    def __str__(self) -> str:
        return self.name


@dataclass
class VectorSpace(VectorSpaceVariable):
    dim: int
    
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
    
    @property
    @abc.abstractmethod
    def symbol(self) -> str:
        pass
    
    def __str__(self) -> str:
        return f"{self.symbol}{self.dim}"


class RealVectorSpace(VectorSpace):
    
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim, **kwargs)
        
    @property
    def symbol(self) -> str:
        return "R"


class TensorSpace(Node):
    
    def __init__(self, spaces: Iterable[VectorSpaceVariable], **kwargs):
        super().__init__(**kwargs)
        symbol = None
        for space in spaces:
            if isinstance(space, VectorSpace):
                if symbol == None:
                    symbol = space.symbol
                if symbol != space.symbol:
                    raise ValueError
        self.spaces = tuple(spaces)
    
    def __iter__(self):
        return iter(self.spaces)
    
    def __hash__(self):
        return hash(self._hashkey)
    
    def __eq__(self, other):
        if isinstance(other, TensorSpace):
            return self._hashkey == other._hashkey
        else:
            return NotImplemented
    
    def __str__(self) -> str:
        return "x".join(str(space) for space in self.spaces)
    
    @property
    def shape(self):
        return tuple(space.dim for space in self.spaces)
    
    @property
    def _hashkey(self):
        return (self.spaces,)
    
    def __mul__(self, other):
        if isinstance(other, VectorSpaceVariable):
            return TensorSpace((list(self.spaces) + [other]))
        else:
            return NotImplemented
        
    @property
    def operands(self) -> Iterable[Node]:
        return self.spaces


class TensorExpr(Node, abc.ABC):
    @property
    @abc.abstractmethod
    def space(self):
        """The `TensorSpace` of the expression."""

    def __getitem__(self, indices: Iterable[Index]):
        if not isinstance(indices, collections.abc.Iterable):
            indices = (indices,)
        indices = [i for i in indices]
        return IndexedTensor(self, indices)


class ScalarExpr(Node, abc.ABC):
    @property
    @abc.abstractmethod
    def indices(self) -> Iterable[Index]:
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


@dataclass
class Literal(ScalarExpr, Terminal):
    f: float
    
    def __init__(self, f: float, **kwargs):
        super().__init__(**kwargs)
        self.f = f

    indices = ()

    @property
    def index_spaces(self):
        return {}

    def __str__(self) -> str:
        return str(self.f)


@dataclass
class IndexedTensor(ScalarExpr):
    tensor_expr: TensorExpr
    _indices: typing.Sequence[Index]

    def __init__(self, tensor_expr: TensorExpr, _indices: typing.Sequence[Index], **kwargs):
        super().__init__(**kwargs)
        self.tensor_expr = tensor_expr
        self._indices = _indices

    @property
    def indices(self):
        return self._indices

    @property
    def index_spaces(self):
        return dict(zip(self._indices, self.tensor_expr.space))

    @property
    def operands(self) -> Iterable[Node]:
        return self.tensor_expr, *self.indices

    def __str__(self) -> str:
        return f"{self.tensor_expr}[{','.join(map(str, self.indices))}]"


@dataclass
class BinOp(ScalarExpr, abc.ABC):
    def __init__(self, lhs: IndexedTensor, rhs: IndexedTensor, **kwargs):
        super().__init__(**kwargs)
        # check that common indices share the same space
        if any(
            lhs.index_spaces[idx] != rhs.index_spaces[idx]
            for idx in set(lhs.indices) & set(rhs.indices)
        ):
            raise ValueError(
                "Indices common across subexpressions must act over the same space"
            )

        self.lhs = lhs
        self.rhs = rhs

    @property
    def indices(self) -> FrozenSet[Index]:
        # binops store their indices in a set rather than a tuple because
        # the ordering of the indices is only determined by a surrounding
        # unindex node.
        return frozenset(self.lhs.indices) | frozenset(self.rhs.indices)

    @property
    def index_spaces(self):
        return {**self.lhs.index_spaces, **self.rhs.index_spaces}

    @property
    def operands(self) -> Iterable[Node]:
        return self.lhs, self.rhs

    def __str__(self) -> str:
        return f"{self.lhs} {self.symbol} {self.rhs}"


class MulBinOp(BinOp):
    symbol = "*"


class AddBinOp(BinOp):
    symbol = "+"


@dataclass
class UnaryOp(ScalarExpr, abc.ABC):
    tensor: ScalarExpr

    def __init__(self, scalar_expr: IndexedTensor, **kwargs):
        super().__init__(**kwargs)
        self.tensor = scalar_expr
        
    @property
    def operands(self):
        return (self.tensor,)

    @property
    def indices(self) -> Iterable[Index]:
        return self.tensor.indices

    @property
    def index_spaces(self):
        return self.tensor.index_spaces

    def __str__(self) -> str:
        return f"{self.name}({self.tensor})"


class Abs(UnaryOp):
    name = "abs"


@dataclass
class IndexSum(ScalarExpr):
    sub: ScalarExpr
    _indices: Iterable[Index]

    def __init__(self, sub: ScalarExpr, _indices: Iterable[Index], **kwargs):
        super().__init__(**kwargs)
        
        if any(idx not in sub.indices for idx in _indices):
            raise ValueError(
                "Indices summed over must refer to Indices in subexpression"
            )
        
        self.sub = sub
        self._indices = _indices

    @property
    def indices(self) -> Iterable[Index]:
        return self.sub.indices
        # return frozenset(self._indices) | frozenset(self.sub.indices)

    @property
    def index_spaces(self):
        return self.sub.index_spaces

    @property
    def operands(self):
        return self.sub, *self._indices
    
    def __str__(self) -> str:
        return f"Sum[{','.join(map(str, self._indices))}]({self.sub})"


class TensorVariable(TensorExpr):
    name: str
    _space: TensorSpace

    def __init__(self, space: Union[VectorSpace, TensorSpace], name: str, **kwargs):
        super(TensorVariable, self).__init__(**kwargs)
        if isinstance(space, VectorSpace):
            space = TensorSpace((space,))
        
        self.name = name
        self._space = space
        self._key = space, name

    @property
    def space(self):
        return self._space

    @property
    def operands(self):
        return (self.space,)

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        if isinstance(other, TensorVariable):
            return self._key == other._key
        else:
            return NotImplemented

    def __str__(self) -> str:
        return f"{self.name}{{{str(self.space)}}}"


@dataclass
class deIndex(TensorExpr):
    scalar_expr: ScalarExpr
    indices: List[Index]

    def __init__(self, scalar_expr: ScalarExpr, indices: List[Index], **kwargs):
        super().__init__(**kwargs)
        
        if set(indices) != set(scalar_expr.indices):
            scalar_expr = IndexSum(scalar_expr, set(scalar_expr.indices)-set(indices))
        
        self.scalar_expr = scalar_expr
        self.indices = indices

    @property
    def space(self):
        return TensorSpace(self.index_spaces.values())

    @property
    def index_spaces(self):
        return {idx: self.scalar_expr.index_spaces[idx] for idx in self.indices}

    @property
    def operands(self) -> Iterable[Node]:
        return self.scalar_expr, *self.indices

    def __str__(self) -> str:
        return f"({self.scalar_expr})|{','.join(map(str, self.indices))}|"


@dataclass
class Lambda(Node):
    vars: List[TensorVariable]
    sub: TensorExpr

    def __init__(self, vars: List[TensorVariable], sub: TensorExpr, **kwargs):
        super().__init__(**kwargs)
        self.vars = vars
        self.sub = sub

    @property
    def operands(self) -> Iterable[Node]:
        return *self.vars, self.sub

    def __str__(self) -> str:
        return f"λ{','.join([str(v) for v in self.vars])}.{self.sub}"
