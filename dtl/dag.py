import abc
import typing
import numbers
from typing import List, Iterable, Union, Dict, Any, Sequence
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
                if frame.f_back is None:
                    break
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
    def with_operands(self, operands: Dict):
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
    
    def makes_scope(self, index):
        return False
    
    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def shortStr(self) -> str:
        """ Short version for making plots etc, As a general rule include:
        - something identifying the Type of the node
        - All information that is in __str__ but not operands
        - Unless those operands are Terminals, when it can help readability
        Use terminalShortStr as a helper to get '.' if not a terminal and the shortStr
        of that terminal if it is."""
        pass
    
    def terminalShortStr(self) -> str:
        return "."


class Terminal(Node, abc.ABC):
    operands = ()
    
    def with_operands(self, operands: List):
        return self.copy()
    
    def shortStr(self) -> str:
        return str(self)
    
    def terminalShortStr(self) -> str:
        return self.shortStr()


class ConstInt(Terminal):
    fields = Terminal.fields | {"value"}
    
    def __init__(self, val: int, **kwargs):
        super().__init__(value=val, **kwargs)
    
    def __str__(self):
        return str(self.value)


class Index(Terminal):
    fields = Terminal.fields | {"name"}
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
    
    def __str__(self) -> str:
        return self.name
    
class _NoneIndex(Terminal):
    def __init__(self):
        super().__init__()

    def __str__(self) -> str:
        return "dtlNone"
    
    def with_operands(self, operands: List):
        return self

NoneIndex = _NoneIndex()


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
        elif isinstance(other, TensorSpace):
            return TensorSpace([self] + (list(other.spaces)))
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
    
    def __init__(self, spaces: Iterable[Union[VectorSpaceVariable, 'TensorSpace']], **kwargs):
        n_spaces = []
        for space in spaces:
            if isinstance(space, VectorSpaceVariable):
                n_spaces.append(space)
            elif isinstance(space, TensorSpace):
                for s in space.spaces:
                    n_spaces.append(s)
            else:
                raise ValueError(
                    f"TensorSpace accepts only an Iterable consisting of VectorSpaceVariable and TensorSpace. {str(space)} does not confirm.")
        spaces = n_spaces
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
        return '×'.join(str(space) for space in self.spaces)
    
    def shortStr(self) -> str:
        return str(self)
    
    @property
    def shape(self):
        return tuple(space.dim for space in self.spaces)
    
    def __mul__(self, other):
        if isinstance(other, VectorSpaceVariable):
            return TensorSpace((list(self.spaces) + [other]))
        elif isinstance(other, TensorSpace):
            return TensorSpace((list(self.spaces) + (list(other.spaces))))
        else:
            return NotImplemented
    
    @property
    def operands(self) -> Dict:
        return {"spaces": list(self.spaces)}
    
    def with_operands(self, operands: Dict):
        return self.copy(spaces=operands["spaces"])
    
    def new(self, name: str):
        return TensorVariable(self, name)


# class ArgType():
#     def __init__(self, i: Index,  v: VectorSpaceVariable):
#         self.i = i
#         self.v = v
#
#     def __str__(self):
#         if self.i is not None:
#             return f"{str(self.i)}:{str(self.v)}"
#         else:
#             return f"{str(self.v)}"

class ResultType(abc.ABC):
    @abc.abstractmethod
    def withExtraDims(self, new_dims: Iterable[VectorSpaceVariable]):
        pass
    
    @abc.abstractmethod
    def __str__(self):
        pass
    
    @property
    @abc.abstractmethod
    def count(self):
        pass
    
    @property
    @abc.abstractmethod
    def isScalar(self) -> bool:
        pass
    
    @property
    @abc.abstractmethod
    def isSingular(self) -> bool:
        pass
    
    @abc.abstractmethod
    def matchShape(self, match: typing.Union[tuple, VectorSpaceVariable, str]) -> Dict[str,VectorSpaceVariable]:
        pass


class ShapeType(ResultType):
    def __init__(self, dims: List[VectorSpaceVariable]):
        self.dims = tuple(dims)
    
    def withExtraDims(self, new_dims: Iterable[VectorSpaceVariable]):
        d = list(self.dims)
        d.extend(new_dims)
        return ShapeType(d)
    
    def __str__(self):
        return f"<{','.join([str(d) for d in self.dims])}>"
    
    @property
    def count(self):
        return 1
    
    @property
    def isScalar(self) -> bool:
        return len(self.dims) == 0
    
    @property
    def isSingular(self) -> bool:
        return True
    
    @property
    def nDims(self) -> int:
        return len(self.dims)
    
    def matchShape(self, match: typing.Union[tuple, VectorSpaceVariable, str]) -> dict[Any, VectorSpaceVariable] | None:
        if not isinstance(match, tuple) or (len(match) != self.nDims):
            return None
        ret = {}
        for m,d in zip(match, self.dims):
            if isinstance(m, VectorSpaceVariable):
                if m != d:
                    return None
            else:
                if m in ret and ret[m] != d:
                    return None
                else:
                    ret[m] = d
        return ret


class ResultTupleType(ResultType):
    def __init__(self, results: Sequence[ResultType]):
        self.results = tuple(results)
    
    def withExtraDims(self, new_dims: Iterable[VectorSpaceVariable]):
        return ResultTupleType([r.withExtraDims(new_dims) for r in self.results])
    
    def __str__(self):
        return '(' + ','.join([str(r) for r in self.results]) + ')'
    
    @property
    def count(self):
        return sum([r.count for r in self.results])
    
    @property
    def isScalar(self) -> bool:
        return False
    
    @property
    def isSingular(self) -> bool:
        return False
    
    @property
    def tupleSize(self) -> int:
        return len(self.results)
    
    def get(self, i: int) -> ResultType:
        return self.results[i]
    
    def matchShape(self, match: typing.Union[tuple, VectorSpaceVariable, str]) -> dict[Any, VectorSpaceVariable] | None:
        if not isinstance(match, tuple) or (len(match) != self.tupleSize):
            return None
        ret = {}
        for m,d in zip(match, self.results):
            child_ret = d.matchShape(m)
            if child_ret is None:
                return None
            for k,v in child_ret.items():
                if k in ret and ret[k] != v:
                    return None
                ret[k] = v
        return ret


class DTLType():
    def __init__(self, args: Dict[Index, VectorSpaceVariable], result: ResultType):
        self.args = args
        self.result = result
    
    def withArgs(self, args: Dict[Index, VectorSpaceVariable]):
        return DTLType(args, self.result)
    
    def withResult(self, result: ResultType):
        return DTLType(self.args, result)
    
    def withoutArgs(self, indices):
        return DTLType({i: s for i, s in self.args.items() if i not in indices}, self.result)
    
    def __str__(self):
        return f"{','.join([str(i) + ':' + str(s) for i, s in self.args.items()])} -> {str(self.result)}"
    
    def spaceOf(self, i: Index):
        if not i in self.args:
            return None
        return self.args[i]
    
    @property
    def nResults(self) -> int:
        return self.result.count
    
    @property
    def indices(self):
        return frozenset(self.args.keys())
    
    @staticmethod
    def checkCommonIndices(types: Sequence["DTLType"], names: Sequence[str] = None,
                           message: str = "Common indices must act over the same space"):
        types = list(types)
        if names is not None:
            names = list(names)
        else:
            names = [str(i) for i in range(len(types))]
        if len(names) != len(types):
            raise ValueError("names for checkCommonIndices must be same length as types provided")
        if any(
            len({t.spaceOf(idx) for t in types} - {None}) > 1
            for idx in {idx for type in types for idx in type.indices}
        ):
            tl = "Name".ljust(8) + " " + (' '.join([name.ljust(8) for name in names])) + "\n"
            l = [f"{str(idx).ljust(8)} " + (
                ' '.join([(str(t.spaceOf(idx)) if idx in t.indices else '_').ljust(8) for t in types])) + (
                     "<-**" if len({t.spaceOf(idx) for t in types} - {None}) > 1 else "")
                 for idx in {idx for type in types for idx in type.indices}]
            nl = '\n'
            out = tl + nl.join(l)
            raise ValueError(
                f"{message}\n{out}"
            )


class Expr(Node, abc.ABC):
    
    @property
    @abc.abstractmethod
    def type(self) -> DTLType:
        pass
    
    @staticmethod
    def __getIndicesFromIndexingPattern(indices: Union[Sequence, Index, ConstInt, int],
                                        shapes: Union[ResultType, VectorSpaceVariable]) -> Dict[
        Index, VectorSpaceVariable]:
        if isinstance(shapes, VectorSpaceVariable):
            space = shapes
            # Base case for individual Dims
            if indices is None:
                return {}  # They can not index a VectorSpaceVariable if they like
            if isinstance(indices, Index):
                index = indices
                return {index: space}
            if isinstance(indices, int):
                return {}
            if isinstance(indices, ConstInt):
                return {}
            raise ValueError(f"Cannot index {str(space)} with {str(indices)}, index must be Index or ConstInt (or int)")
        if isinstance(shapes, ShapeType):
            shape = shapes
            if isinstance(indices, Sequence):
                if len(indices) != shape.nDims:
                    raise ValueError(
                        f"Cannot index space {str(shape)} with {','.join([str(i) for i in indices])}, number of dimensions must match")
                return {index: space for idx, dims in zip(indices, shape.dims) for index, space in
                        Expr.__getIndicesFromIndexingPattern(idx, dims).items()}
            raise ValueError(f"Cannot index {str(shape)} with {str(indices)}, indices must be a Sequence (list/tuple)")
        if isinstance(shapes, ResultTupleType):
            if isinstance(indices, Sequence):
                if len(indices) != shapes.tupleSize:
                    raise ValueError(
                        f"Cannot index space {str(shapes)} with {str(indices)}, number of dimensions must match")
                return {index: space for idx, dims in zip(indices, shapes.results) for index, space in
                        Expr.__getIndicesFromIndexingPattern(idx, dims).items()}
            raise ValueError(f"Cannot index {str(shapes)} with {str(indices)}, indices must be a Sequence (list/tuple)")
        raise ValueError("Shapes must be of type ResultType")
    
    def __getitem__(self, indices: Union[Sequence, Index, slice]) -> "Expr":
        if isinstance(indices, Sequence) or isinstance(indices, Index) or isinstance(indices, int) or isinstance(indices, ConstInt):
            if isinstance(indices, Sequence) and len(indices) > 0 and isinstance(indices[0], slice):
                if any(s.step != None for s in indices): raise ValueError(
                    "When binding Indices to vector spaces with expr[i:space] notation you must not use the step feature of slices: expr[i:space:_]")
                idxs = [s.start for s in indices]
                pSpaces = [s.stop for s in indices]
                spaces = []
                for idx, v in zip(idxs, pSpaces):
                    if isinstance(v, VectorSpaceVariable):
                        spaces.append(v)
                    elif isinstance(v, Index):
                        nv = self.type.spaceOf(v)
                        if nv is None:
                            raise ValueError(f"Cannot bind {str(idx)} to the same space as {str(v)} when {str(v)} is not defined in {str(self.type)} in expression {str(self)}")
                        spaces.append(nv)
                    else:
                        raise ValueError(
                            f"Cannot bind {str(idx)} to {str(v)} when {str(v)} is not defined a VectorSpace Variable, or an Index already in {str(self.type)} in expression {str(self)}")
                        
                spaces = [s.stop if isinstance(s.stop, VectorSpaceVariable) else self.type.spaceOf(s.stop) for s in indices]
                return IndexBinding(self, idxs, spaces)
            if isinstance(indices, Index) or isinstance(indices, int) or isinstance(indices, ConstInt):
                indices = [indices]
            expr = self
            type = expr.type
            indexDict = Expr.__getIndicesFromIndexingPattern(indices, type.result)
            unboundIndices = [i for i in indexDict if type.spaceOf(i) is None]
            if len(unboundIndices) > 0:
                spaces = [indexDict[i] for i in unboundIndices]
                expr = IndexBinding(expr, unboundIndices, spaces, attrs={"AutoGenerated": True})
            return IndexExpr(expr, indices)
        elif isinstance(indices, Dict):
            idxs, spaces = zip(*indices.items())
            return IndexBinding(self, idxs, spaces)
        elif isinstance(indices, slice):
            if indices.step != None: raise ValueError(
                "When binding Indices to vector spaces with expr[i:space] notation you must not use the step feature of slices: expr[i:space:_]")
            idx = indices.start
            if isinstance(indices.stop, VectorSpaceVariable):
                space = indices.stop
            elif isinstance(indices.stop, Index):
                space = self.type.spaceOf(indices.stop)
                if space is None:
                    raise ValueError(
                        f"Cannot bind {str(idx)} to the same space as {str(indices.stop)} when {str(indices.stop)} is not defined in {str(self.type)} in expression {str(self)}")
            else:
                raise ValueError(
                    f"Cannot bind {str(idx)} to {str(indices.stop)} when {str(indices.stop)} is not defined a VectorSpace Variable, or an Index already in {str(self.type)} in expression {str(self)}")
            return IndexBinding(self, [idx], [space])
        else:
            raise ValueError("Unable to Parse __getitem__ argument: "+ str(indices))
    
    def tuple(self) -> tuple:
        if isinstance(self.type.result, ResultTupleType):
            return tuple([IndexedExprTuple(self, i) for i in range(self.type.result.tupleSize)])
        else:
            raise ValueError("Cannot make tuple from expr with type " + str(self.type))
    
    def forall(self, *indices: Index) -> "DeindexExpr":
        return DeindexExpr(self, indices)
    
    def sum(self, *indices: Index) -> "IndexSum":
        return IndexSum(self, indices)
    
    def __add__(self, other: "Expr") -> "Expr":
        return AddBinOp(self, other)
    
    def __sub__(self, other: "Expr") -> "Expr":
        return SubBinOp(self, other)
    
    def __mul__(self, other: "Expr") -> "Expr":
        return MulBinOp(self, other)
    
    @staticmethod
    def exprInputConversion(expr: "ExprTypeHint"):
        if isinstance(expr, Expr):
            return expr
        elif isinstance(expr, int) or isinstance(expr, float):
            return Literal(expr)
        elif isinstance(expr, tuple):
                return ExprTuple(tuple([Expr.exprInputConversion(e) for e in expr]))
        else:
            raise ValueError(f"Cannot convert {expr} to ExprTuple. Expected type a = Expr|tuple[a]" )

ExprTypeHint = typing.Union[Expr, typing.Tuple["ExprTypeHint",...], int, float]

class TensorVariable(Expr):  # -> <...>
    fields = Expr.fields | {"tensor_space", "name"}
    
    def __init__(self, tensor_space: Union[VectorSpace, TensorSpace], name: str, **kwargs):
        if isinstance(tensor_space, VectorSpace):
            tensor_space = TensorSpace((tensor_space,))
        
        super().__init__(tensor_space=tensor_space, name=name, **kwargs)
    
    @property
    def operands(self):
        return {"tensor_space": self.tensor_space}
    
    def with_operands(self, operands: Dict):
        return self.copy(tensor_space=operands["tensor_space"])
    
    @property
    def type(self) -> DTLType:
        return DTLType({}, ShapeType(self.tensor_space.spaces))
    
    def __str__(self) -> str:
        return f"{self.name}<{str(self.tensor_space)}>"
    
    def shortStr(self) -> str:
        return f"{self.name}<>"


class IndexBinding(Expr):  # [...] -> <...>... , {b:B} => [...b:B] -> <...>...
    fields = Expr.fields | {"expr", "indices", "spaces"}
    
    def __init__(self, expr: ExprTypeHint, indices: Sequence[Index], spaces: Sequence[VectorSpaceVariable], **kwargs):
        expr = Expr.exprInputConversion(expr)
        if len(indices) != len(spaces):
            raise ValueError(f"IndexBinding, every index must map to a space - indices Sequence and spaces Sequence have different lengths!:\n"
                             f"  indices:[{','.join([str(i) for i in indices])}]\n"
                             f"   spaces:[{','.join([str(s) for s in spaces])}]\n")
        exprType = expr.type
        for index, space in zip(indices, spaces):
            if exprType.spaceOf(index) != None:
                raise ValueError(
                    f"IndexBinding cannot redeclare index {str(index)}:{str(space)} already found in expr with type {str(exprType)}")
            if not isinstance(space, VectorSpaceVariable):
                raise ValueError(
                    f"IndexBinding cannot declare index {str(index)}:{str(space)} where the space is not a vector space variable (use Expr.__getItem__ for syntactic sugar that allows this input)")
        super().__init__(expr=expr, indices=tuple(indices), spaces=tuple(spaces), **kwargs)
    
    @property
    def type(self) -> DTLType:
        exprType = self.expr.type
        args = exprType.args
        for i, s in zip(self.indices, self.spaces):
            args[i] = s
        return exprType.withArgs(args)
    
    @property
    def operands(self):
        return {"expr": self.expr, "indices": self.indices, "spaces": self.spaces}
    
    def with_operands(self, operands: Dict):
        return self.copy(expr=operands["expr"], indices=operands["indices"], spaces=operands["spaces"])
        # return self.copy(expr=operands["expr"], indices=operands["indices"], spaces=operands["spaces"])
    
    def __str__(self):
        return f"{str(self.expr)}{{{','.join([str(i) + ':' + str(s) for i, s, in zip(self.indices, self.spaces)])}}}"
    
    def shortStr(self) -> str:
        return f"{self.expr.terminalShortStr()}{{{','.join([str(i) + ':' + str(s) for i, s, in zip(self.indices, self.spaces)])}}}"


class IndexExpr(Expr):  # [a:A...] -> <A...>... , [[a..],...] => [a:A...] -> <...>...
    fields = Expr.fields | {"expr", "indices"}
    
    @staticmethod
    def checkIndexingPattern(exprType: DTLType,
                             indices: Union[Sequence, Index, ConstInt, int, _NoneIndex],
                             shapes: Union[ResultType, VectorSpaceVariable]):
        if isinstance(shapes, VectorSpaceVariable):
            space = shapes
            # Base case for individual Dims
            if indices is None or isinstance(indices, _NoneIndex):
                return NoneIndex  # They can not index a VectorSpaceVariable if they like
            if isinstance(indices, Index):
                index = indices
                # If they have used an Index check it's space
                idxSpace = exprType.spaceOf(index)
                if idxSpace is None:
                    raise ValueError(f"IndexExpr index ({str(index)}) must be an argument in Expr type {exprType}")
                elif idxSpace != space:
                    raise ValueError(
                        f"IndexExpr index ({str(index)}) has space {str(idxSpace)} but is indexing {str(space)} in the expr with type {exprType}")
                else:
                    return index
            if isinstance(indices, int):
                indices = ConstInt(indices, attrs={"AutoGenerated": True})
            if isinstance(indices, ConstInt):
                i = indices
                # if they have used a ConstInt check it's in bounds (if possible to check)
                if isinstance(space, VectorSpace) and space.dim <= i.value:
                    raise ValueError(f"Cannot index space {str(space)} with int {str(i)}")
                else:
                    return i
            raise ValueError(f"Cannot index {str(space)} with {str(indices)}, index must be Index or ConstInt (or int)")
        if isinstance(shapes, ShapeType):
            shape = shapes
            if isinstance(indices, Sequence):
                if len(indices) != shape.nDims:
                    raise ValueError(
                        f"Cannot index space {str(shape)} with {str(indices)}, number of dimensions must match")
                return tuple(
                    [IndexExpr.checkIndexingPattern(exprType, idx, dims) for idx, dims in zip(indices, shape.dims)])
            raise ValueError(f"Cannot index {str(shape)} with {str(indices)}, indices must be a Sequence (list/tuple)")
        if isinstance(shapes, ResultTupleType):
            if isinstance(indices, Sequence):
                if len(indices) != shapes.tupleSize:
                    raise ValueError(
                        f"Cannot index space {str(shapes)} with {str(indices)}, number of dimensions must match")
                return tuple(
                    [IndexExpr.checkIndexingPattern(exprType, idx, dims) for idx, dims in zip(indices, shapes.results)])
            raise ValueError(f"Cannot index {str(shapes)} with {str(indices)}, indices must be a Sequence (list/tuple)")
        raise ValueError("Shapes must be of type ResultType")
    
    def __init__(self, expr: ExprTypeHint, indices: Union[Index, ConstInt, int, Sequence], **kwargs):
        expr = Expr.exprInputConversion(expr)
        exprType = expr.type
        indices = IndexExpr.checkIndexingPattern(exprType, indices, exprType.result)
        super().__init__(expr=expr, indices=indices, **kwargs)
    
    # We assume a well-formed indices argument here as it should have already been checked by checkIndexingPattern
    @staticmethod
    def _generateResultType(indices: Union[Sequence, Index, ConstInt],
                            shapes: ResultType) -> ResultType:
        if isinstance(shapes, ShapeType):
            shape = shapes
            return ShapeType([dim for idx, dim in zip(indices, shape.dims) if idx == NoneIndex])
        if isinstance(shapes, ResultTupleType):
            return ResultTupleType(
                [IndexExpr._generateResultType(idx, dims) for idx, dims in zip(indices, shapes.results)])
    
    @property
    def type(self) -> DTLType:
        exprType = self.expr.type
        newResult = IndexExpr._generateResultType(self.indices, exprType.result)
        return exprType.withResult(newResult)
    
    @property
    def operands(self):
        return {"expr": self.expr, "indices": self.indices}
    
    def with_operands(self, operands: Dict):
        return self.copy(expr=operands["expr"], indices=operands["indices"])
    
    @staticmethod
    def _str_indices(indices) -> str:
        if isinstance(indices, Sequence):
            return '[' + ','.join([IndexExpr._str_indices(i) for i in indices]) + ']'
        else:
            return str(indices)
    
    def __str__(self):
        return f"{str(self.expr)}{IndexExpr._str_indices(self.indices)}"
    
    def shortStr(self) -> str:
        return f"{self.expr.terminalShortStr()}{IndexExpr._str_indices(self.indices)}"
    


class Literal(Expr, Terminal):  # [] -> <>
    fields = Expr.fields | Terminal.fields | {"f"}
    
    def __init__(self, f: float, **kwargs):
        super().__init__(f=float(f), **kwargs)
    
    @property
    def type(self) -> DTLType:
        return DTLType({}, ShapeType([]))
    
    def __str__(self) -> str:
        return str(self.f)


class ScalarBinOp(Expr, abc.ABC):  # [a] -> <>, [b] -> <> => [a|b] -> <>
    fields = Expr.fields | {"lhs", "rhs"}
    
    def __init__(self, lhs: ExprTypeHint, rhs: ExprTypeHint, **kwargs):
        lhs = Expr.exprInputConversion(lhs)
        rhs = Expr.exprInputConversion(rhs)
        
        lhsType = lhs.type
        rhsType = rhs.type
        
        if not lhsType.result.isScalar:
            raise ValueError(
                f"lhs must be a single result scalar ( [...] -> <> ) but {str(lhs)} has type {str(lhsType)}")
        if not rhsType.result.isScalar:
            raise ValueError(
                f"rhs must be a single result scalar ( [...] -> <> ) but {str(rhs)} has type {str(rhsType)}")
        
        # check that common indices share the same space
        DTLType.checkCommonIndices([lhsType, rhsType], names=["lhs", "rhs"],
                                   message="Indices common across subexpressions must act over the same space")
        if any(
            lhsType.spaceOf(idx) != rhsType.spaceOf(idx)
            for idx in lhsType.indices & rhsType.indices
        ):
            l = [f"{str(idx).ljust(8)} " \
                 f"{(str(lhsType.spaceOf(idx)) if idx in lhsType.indices else '_').ljust(8)} " \
                 f"{(str(rhsType.spaceOf(idx)) if idx in rhsType.indices else '_').ljust(8)} "
                 for idx in set(lhsType.indices) & set(rhsType.indices)]
            nl = '\n'
            raise ValueError(
                f"Indices common across subexpressions must act over the same space\n"
                f"Name     lhs      rhs\n"
                f"{nl.join(l)}"
            )
        super().__init__(lhs=lhs, rhs=rhs, **kwargs)
    
    @property
    def type(self) -> DTLType:
        lhsType = self.lhs.type
        rhsType = self.rhs.type
        args = {}
        for idx in lhsType.indices:
            args[idx] = lhsType.spaceOf(idx)
        for idx in rhsType.indices:
            args[idx] = rhsType.spaceOf(idx)
        return DTLType(args, ShapeType([]))
    
    @property
    def operands(self):
        return {"lhs": self.lhs, "rhs": self.rhs}
    
    def with_operands(self, operands):
        return self.copy(lhs=operands["lhs"], rhs=operands["rhs"])
    
    @property
    @abc.abstractmethod
    def symbol(self) -> str:
        pass
    
    def __str__(self) -> str:
        return f"{self.lhs} {self.symbol} {self.rhs}"
    
    def shortStr(self) -> str:
        return f"{self.lhs.terminalShortStr()} {self.symbol} {self.rhs.terminalShortStr()}"
    
    

class MulBinOp(ScalarBinOp):
    symbol = "*"


class AddBinOp(ScalarBinOp):
    symbol = "+"


class SubBinOp(ScalarBinOp):
    symbol = "-"


class ScalarUnaryOp(Expr, abc.ABC):  # [...] -> <> => [...] -> <>
    fields = Expr.fields | {"expr"}
    
    def __init__(self, expr: ExprTypeHint, **kwargs):
        expr = Expr.exprInputConversion(expr)
        exprType = expr.type
        if not exprType.result.isScalar:
            raise ValueError(
                f" must be a single result scalar ( [...] -> <> ) but {str(expr)} has type {str(exprType)}")
        super().__init__(expr=expr, **kwargs)
    
    def type(self) -> DTLType:
        return self.expr.type
    
    @property
    def operands(self):
        return {"expr": self.expr}
    
    def with_operands(self, operands):
        return self.copy(expr=operands["expr"])
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass
    
    def __str__(self) -> str:
        return f"{self.name}({self.expr})"

    def shortStr(self) -> str:
        return f"{self.name}({self.expr.terminalShortStr()})"


class Abs(ScalarUnaryOp):
    name = "abs"


class IndexSum(Expr):  # [a:A...] -> <...>... , [a..] => [...] -> <...>...
    fields = Expr.fields | {"expr", "sum_indices"}
    
    def __init__(self, expr: ExprTypeHint, sum_indices: Sequence[Index], **kwargs):
        expr = Expr.exprInputConversion(expr)
        exprType = expr.type
        if any(idx not in exprType.indices for idx in sum_indices):
            raise ValueError(
                "Indices summed over must refer to Indices in subexpression"
            )
        super().__init__(expr=expr, sum_indices=tuple(sum_indices), **kwargs)
    
    @property
    def type(self) -> DTLType:
        return self.expr.type.withoutArgs(self.sum_indices)
    
    @property
    def operands(self):
        return {"expr": self.expr, "sum_indices": self.sum_indices}
    
    def with_operands(self, operands):
        return self.copy(expr=operands["expr"], sum_indices=operands["sum_indices"])
    
    def __str__(self) -> str:
        return f"Sum[{','.join(map(str, self.sum_indices))}]({self.expr})"

    def shortStr(self) -> str:
        return f"Σ[{','.join(map(str, self.sum_indices))}]({self.expr.terminalShortStr()})"
    
    def makes_scope(self, index):
        return index in self.sum_indices


# DeindexFormatTypeHint = typing.Union[typing.Tuple["DeindexFormatTypeHint",...], Index, VectorSpaceVariable]

class DeindexExpr(Expr):
    fields = Expr.fields | {"expr", "indices"}
    
    def __init__(self, expr: ExprTypeHint, indices: Sequence[Index], **kwargs):
        expr = Expr.exprInputConversion(expr)
        exprType = expr.type
        unfreeIndices = set(indices).difference(exprType.indices)
        if len(unfreeIndices) > 0:
            raise ValueError(
                f"Cannot Deindex along indices [{','.join(str(i) for i in unfreeIndices)}] that are not free in the expression of type: {str(exprType)} in expression {expr}")
        # if set(indices) != set(exprType.indices):
        #     expr = IndexSum(expr, tuple(set(exprType.indices) - set(indices)), attrs={"AutoGenerated": True})
        super().__init__(expr=expr, indices=tuple(indices), **kwargs)
    
    @property
    def type(self) -> DTLType:
        exprType = self.expr.type
        args = {i: exprType.spaceOf(i) for i in exprType.indices if i not in self.indices}
        dims = [exprType.spaceOf(i) for i in self.indices]
        result = exprType.result.withExtraDims(dims)
        return DTLType(args, result)
    
    @property
    def operands(self):
        return {"expr": self.expr, "indices": self.indices}
    
    def with_operands(self, operands):
        return self.copy(expr=operands["expr"], indices=operands["indices"])
    
    def __str__(self) -> str:
        return f"({self.expr})|{','.join(map(str, self.indices))}|"

    def shortStr(self) -> str:
        return f"({self.expr.terminalShortStr()})|{','.join(map(str, self.indices))}|"
    
    def makes_scope(self, index):
        return index in self.indices


class ExprTuple(Expr):
    fields = Expr.fields | {"exprs"}
    
    def __init__(self, exprs: Sequence[ExprTypeHint], **kwargs):
        exprs = [Expr.exprInputConversion(expr) for expr in exprs]
        exprTypes = [expr.type for expr in exprs]
        DTLType.checkCommonIndices(exprTypes)
        super().__init__(exprs=tuple(exprs), **kwargs)
    
    @property
    def type(self) -> DTLType:
        exprTypes = [expr.type for expr in self.exprs]
        args = {i: exprType.spaceOf(i) for exprType in exprTypes for i in exprType.indices}
        result = ResultTupleType([exprType.result for exprType in exprTypes])
        return DTLType(args, result)
    
    @property
    def operands(self):
        return {"exprs": self.exprs}
    
    def with_operands(self, operands: Dict):
        return self.copy(exprs=operands["exprs"])
    
    def __str__(self) -> str:
        return f"({','.join([str(e) for e in self.exprs])})"
    
    def shortStr(self) -> str:
        return f"({','.join([e.terminalShortStr() for e in self.exprs])})"


class IndexedExprTuple(Expr):
    fields = Expr.fields | {"expr", "n"}
    
    def __init__(self, expr: ExprTypeHint, n: int, **kwargs):
        expr = Expr.exprInputConversion(expr)
        exprType = expr.type
        if not isinstance(exprType.result, ResultTupleType):
            raise ValueError("Cannot extract Expr from Tuple that does not have tuple result type: " + str(exprType))
        if n >= exprType.result.tupleSize or n < -exprType.result.tupleSize:
            raise ValueError(
                f"Cannot extract tuple element {n} from tuple of length {exprType.result.tupleSize} in type: {exprType}")
        super().__init__(expr=expr, n=n, **kwargs)
    
    @property
    def type(self) -> DTLType:
        exprType = self.expr.type
        return exprType.withResult(exprType.result.get(self.n))
    
    @property
    def operands(self):
        return {"expr": self.expr}
    
    def with_operands(self, operands: Dict):
        return self.copy(expr=operands["expr"])
    
    def __str__(self) -> str:
        return f"{str(self.expr)}[@{self.n}]"
    
    def shortStr(self) -> str:
        return f"@{self.n}"
