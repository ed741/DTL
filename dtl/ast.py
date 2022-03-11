import collections
from dataclasses import dataclass
from typing import List


class VSpace():
    def __init__(self):
        raise TypeError
    def __str__(self) -> str:
        return "(!!VSPACE!!)"

@dataclass
class RVSpace():
    dim : int
    def __str__(self) -> str:
        return f"R{self.dim}"

@dataclass
class varIndexOBJ():
    vspace : VSpace
    def __str__(self) -> str:
        return f"indexOBJ({self.vspace})"

class astNode():
    def __init__(self):
        raise TypeError
    def __str__(self) -> str:
        return "(!!BASE NODE TYPE!!)"

@dataclass
class Index(astNode):
    name : str
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
            indices = indices,
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
    f : float
    def __str__(self) -> str:
        return str(self.f)

@dataclass
class IndexedTensor(ScalarExpr):
    tensor: TensorExpr
    indices: List[Index]
    def __str__(self) -> str:
        return f"{self.tensor}[{','.join(map(str, self.indices))}]"

@dataclass
class MulBinOp(ScalarExpr):
    lhs : IndexedTensor
    rhs : IndexedTensor
    def __str__(self) -> str:
        return f"{self.lhs} * {self.rhs}"

@dataclass
class Abs(ScalarExpr):
    sub : ScalarExpr
    def __str__(self) -> str:
        return f"abs({self.sub})"

@dataclass
class indexSum(ScalarExpr):
    sub : ScalarExpr
    indices : List[Index]
    def __str__(self) -> str:
        return f"Sum{self.indices}){self.sub})"

@dataclass
class TensorVariable(TensorExpr):
    name : str
    def __str__(self) -> str:
        return self.name

@dataclass
class deIndex(TensorExpr):
    tensor : ScalarExpr
    indices : List[Index]
    def __str__(self) -> str:
        return f"({self.tensor})|{','.join(map(str, self.indices))}|"

@dataclass()
class Lambda(astNode):
    vars : List[TensorVariable]
    sub : astNode
    def __str__(self) -> str:
        return f"{type(self).__name__}({','.join(map(str, self.vars))}):{self.sub}"

if __name__ == '__main__':
    i = Index("i")
    j = Index("j")
    k = Index("k")
    T1 = Lambda([A := TensorVariable("A")], deIndex(A[j, i], [i, j]))
    # T2 = Lambda([A := TensorVariable("A"),B := TensorVariable("B")], (A[j, i]*Abs(B[j,i])|[i, j])[k]|[k])
    A = TensorVariable("A")
    B = TensorVariable("B")
    T2 = Lambda([A,B], A[k] | [k])
    print(str([j, i]))
    # print(str(T1))
    print(str(T2))
