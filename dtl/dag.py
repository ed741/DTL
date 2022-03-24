import abc
import collections
import collections.abc
from dataclasses import dataclass
from typing import List, Iterable, FrozenSet


@dataclass(frozen=True)
class VectorSpace(abc.ABC):
    dim: int

    @property
    @abc.abstractmethod
    def symbol(self) -> str:
        pass

    def __str__(self) -> str:
        return f"{self.symbol}{self.dim}"


class RealVectorSpace(VectorSpace):
    symbol = "R"


@dataclass
class TensorSpace:
    spaces: Iterable[VectorSpace]


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

    # FIXME An index should just be a label and the vector space associated with
    # it should be derived from the tensor expression rather than what we do here
    # and directly associate the index with the space.
    space: VectorSpace

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class VarIndex(Index):
    var: str

    def __str__(self) -> str:
        return self.var


@dataclass(frozen=True)
class IntIndex(Index):
    i: int

    def __str__(self) -> str:
        return str(self.i)


class TensorExpr(Node, abc.ABC):
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

    def __add__(self, other: "ScalarExpr") -> "ScalarExpr":
        return AddBinOp(self, other)

    def __mul__(self, other: "ScalarExpr") -> "ScalarExpr":
        return MulBinOp(self, other)


@dataclass
class Literal(ScalarExpr):
    f: float

    indices = ()

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
    def operands(self) -> Iterable[Node]:
        return self.tensor, *self.indices

    def __str__(self) -> str:
        return f"{self.tensor}[{','.join(map(str, self.indices))}]"


@dataclass
class BinOp(ScalarExpr, abc.ABC):
    lhs: IndexedTensor
    rhs: IndexedTensor

    @property
    def indices(self) -> FrozenSet[Index]:
        # binops store their indices in a set rather than a tuple because
        # the ordering of the indices is only determined by a surrounding
        # unindex node.
        return frozenset(self.lhs.indices) | frozenset(self.rhs.indices)

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
    name: str
    space: TensorSpace

    def __str__(self) -> str:
        return self.name


@dataclass
class deIndex(TensorExpr):
    scalar_expr: ScalarExpr
    indices: List[Index]

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
