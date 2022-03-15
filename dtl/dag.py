import abc
import collections
from dataclasses import dataclass
from typing import List, Iterable


class VSpace:
    def __init__(self):
        raise TypeError

    def __str__(self) -> str:
        return "(!!VSPACE!!)"


@dataclass
class RVSpace:
    dim: int

    def __str__(self) -> str:
        return f"R{self.dim}"


class astNode(abc.ABC):
    @property
    @abc.abstractmethod
    def operands(self):
        pass

    def __str__(self) -> str:
        return "(!!BASE NODE TYPE!!)"


class Terminal(astNode):

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


class TensorExpr(astNode):
    def __str__(self) -> str:
        return "(!!TENSOR EXPR NODE TYPE!!)"

    def __getitem__(self, indices: List[Index]):
        if not isinstance(indices, collections.abc.Iterable):
            indices = (indices,)
        return IndexedTensor(self, indices)


class ScalarExpr(astNode):
    def __str__(self) -> str:
        return "(!!Scalar Expr NODE TYPE!!)"

    def __or__(self, indices: List[Index]) -> "deIndex":
        return deIndex(self, indices)

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
    indices: List[Index]

    @property
    def operands(self) -> Iterable[astNode]:
        return self.tensor, *self.indices

    def __str__(self) -> str:
        return f"{self.tensor}[{','.join(map(str, self.indices))}]"


@dataclass
class BinOp(ScalarExpr, abc.ABC):
    lhs: IndexedTensor
    rhs: IndexedTensor

    def operands(self) -> Iterable[astNode]:
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


@dataclass
class Abs(UnaryOp):
    name = "abs"


@dataclass
class indexSum(ScalarExpr):
    sub: ScalarExpr
    indices: Iterable[Index]

    def __str__(self) -> str:
        return f"Sum{self.indices}){self.sub})"

@dataclass
class TensorVariable(TensorExpr, Terminal):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class deIndex(TensorExpr):
    tensor: ScalarExpr
    indices: List[Index]

    @property
    def operands(self) -> Iterable[astNode]:
        return self.tensor, *self.indices

    def __str__(self) -> str:
        return f"({self.tensor})|{','.join(map(str, self.indices))}|"


@dataclass
class Lambda(astNode):
    vars: List[TensorVariable]
    sub: astNode

    @property
    def operands(self) -> Iterable[astNode]:
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
