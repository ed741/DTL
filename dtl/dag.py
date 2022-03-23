import abc
import collections
from dataclasses import dataclass
from typing import List, Iterable


@dataclass
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


@dataclass
class Index(Terminal):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class VarIndex(Index):
    var: str

    def __str__(self) -> str:
        return self.var


@dataclass
class IntIndex(Index):
    i: int

    def __str__(self) -> str:
        return str(self.i)


class TensorExpr(Node, abc.ABC):

    def __getitem__(self, indices: Iterable[Index]):
        if not isinstance(indices, collections.abc.Iterable):
            indices = (indices,)
        return IndexedTensor(self, indices)


class ScalarExpr(Node, abc.ABC):
    def forall(self, *indices: Index) -> "deIndex":
        return deIndex(self, indices)

    def __str__(self) -> str:
        return "(!!Scalar Expr NODE TYPE!!)"

    def __mul__(self, other: "ScalarExpr") -> "ScalarExpr":
        return MulBinOp(self, other)


@dataclass
class Literal(ScalarExpr):
    f: float

    def __str__(self) -> str:
        return str(self.f)


@dataclass
class IndexedTensor(ScalarExpr):
    tensor: TensorExpr
    indices: Iterable[Index]

    @property
    def operands(self) -> Iterable[Node]:
        return self.tensor, *self.indices

    def __str__(self) -> str:
        return f"{self.tensor}[{','.join(map(str, self.indices))}]"


@dataclass
class BinOp(ScalarExpr, abc.ABC):
    lhs: IndexedTensor
    rhs: IndexedTensor

    def operands(self) -> Iterable[Node]:
        return self.lhs, self.rhs

    def __str__(self) -> str:
        return f"{self.lhs} {self.symbol} {self.rhs}"


@dataclass
class MulBinOp(BinOp):
    symbol = "*"


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
    tensor: ScalarExpr
    indices: List[Index]

    @property
    def operands(self) -> Iterable[Node]:
        return self.tensor, *self.indices

    def __str__(self) -> str:
        return f"({self.tensor})|{','.join(map(str, self.indices))}|"


@dataclass
class Lambda(Node):
    vars: List[TensorVariable]
    sub: Node

    @property
    def operands(self) -> Iterable[Node]:
        return self.vars, self.sub

    def __str__(self) -> str:
        return f"λ{','.join([str(v) for v in self.vars])}.{self.sub}"


if __name__ == "__main__":
    i = Index("i")
    j = Index("j")
    k = Index("k")
    T1 = Lambda([A := TensorVariable("A")], deIndex(A[j, i], [i, j]))
    # T2 = Lambda([A := TensorVariable("A"),B := TensorVariable("B")], (A[j, i]*Abs(B[j,i])|[i, j])[k]|[k])
    A = TensorVariable("A")
    B = TensorVariable("B")
    T2 = Lambda([A, B], A[k] | [k])
    print(str([j, i]))
    # print(str(T1))
    print(str(T2))
