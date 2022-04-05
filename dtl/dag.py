import abc
import collections
import collections.abc
from dataclasses import dataclass
import numbers
from typing import List, Iterable, FrozenSet, Union


@dataclass(frozen=True)
class VectorSpace(abc.ABC):
    dim: int

    @property
    @abc.abstractmethod
    def symbol(self) -> str:
        pass

    def __str__(self) -> str:
        return f"{self.symbol}{self.dim}"

    def __mul__(self, other):
        if isinstance(other, VectorSpace):
            return TensorSpace(self, other)
        else:
            return NotImplemented

    def __pow__(self, other):
        if isinstance(other, numbers.Integral):
            return TensorSpace([self] * other)
        else:
            return NotImplemented


@dataclass(frozen=True)
class TensorSpace:
    spaces: Iterable[VectorSpace]

    def __iter__(self):
        return iter(self.spaces)

    @property
    def shape(self):
        return tuple(space.dim for space in self.spaces)


class Node(abc.ABC):
    @property
    @abc.abstractmethod
    def operands(self):
        pass


class Terminal(Node):
    operands = ()


@dataclass(frozen=True)
class Index(Terminal):
    name: str

    def __str__(self) -> str:
        return self.name


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
class Literal(ScalarExpr):
    f: float

    indices = ()

    @property
    def index_spaces(self):
        return {}

    def __str__(self) -> str:
        return str(self.f)


@dataclass
class IndexedTensor(ScalarExpr):
    tensor_expr: TensorExpr
    _indices: collections.abc.Sequence[Index]

    @property
    def indices(self):
        return self._indices

    @property
    def index_spaces(self):
        return dict(zip(self._indices, self.tensor_expr.space))

    @property
    def operands(self) -> Iterable[Node]:
        return self.tensor, *self.indices

    def __str__(self) -> str:
        return f"{self.tensor}[{','.join(map(str, self.indices))}]"


@dataclass
class BinOp(ScalarExpr, abc.ABC):
    def __init__(self, lhs: IndexedTensor, rhs: IndexedTensor):
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

    def operands(self) -> Iterable[Node]:
        return self.lhs, self.rhs

    def __str__(self) -> str:
        return f"{self.lhs} {self.symbol} {self.rhs}"


class MulBinOp(BinOp):
    symbol = "*"


class AddBinOp(BinOp):
    symbol = "+"


@dataclass
class UnaryOp(ScalarExpr):
    tensor: IndexedTensor

    @property
    def operands(self):
        return (self.tensor,)

    def __str__(self) -> str:
        return f"{self.name}({self.tensor})"


class Abs(UnaryOp):
    name = "abs"


@dataclass
class IndexSum(ScalarExpr):
    sub: ScalarExpr
    indices: Iterable[Index]

    def __str__(self) -> str:
        return f"Sum{self.indices}){self.sub})"


@dataclass
class TensorVariable(TensorExpr, Terminal):
    def __init__(self, space: Union[VectorSpace, TensorSpace], name: str):
        if isinstance(space, VectorSpace):
            space = TensorSpace([space])

        self._space = space
        self.name = name

    @property
    def space(self):
        return self._space

    def __str__(self) -> str:
        return self.name


@dataclass
class deIndex(TensorExpr):
    scalar_expr: ScalarExpr
    indices: List[Index]

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

    @property
    def operands(self) -> Iterable[Node]:
        return self.vars, self.sub

    def __str__(self) -> str:
        return f"λ{','.join([str(v) for v in self.vars])}.{self.sub}"
