import dtlutils
from dtl import *
from dtl.dag_ext_non_scalar import *
from dtlpp.backends.native import PythonDtlNode, KernelBuilder, CodeComment, SeqNode, ExprConst, AssignTemp, CodeNode, \
    ExpressionNode
from dtlutils import visualise, ndVisualise

dag_ext_non_scalar_init()


class MarginaliseFactor(PythonDtlNode):
    fields = ScalarExpr.fields | {"eta_expr", "lambda_expr", "slot_index",
                                  "index_eta_slot", "index_eta",
                                  "index_lambda1_slot", "index_lambda1",
                                  "index_lambda2_slot", "index_lambda2"}
    
    def __init__(self, eta_expr: NdExpr, lambda_expr: NdExpr, slot_index: Index, **kwargs):
        if len(eta_expr.nd_space.spaces) != 2 or len(lambda_expr.nd_space.spaces) != 4 :
            raise ValueError("Marginalisation requires a eta to have 2 exposed dimension (connectivity * vector)"
                             "and lambda to have 4: (connectivity * connectivity * vector * vector) ")
        if not (lambda_expr.nd_space.spaces[0] == lambda_expr.nd_space.spaces[1] == eta_expr.nd_space.spaces[0]):
            raise ValueError("Marginalisation requires a eta and lambda to have the same spaces for their"
                             "factor's connectivity")
        if not (lambda_expr.nd_space.spaces[2] == lambda_expr.nd_space.spaces[3] == eta_expr.nd_space.spaces[1]):
            raise ValueError("Marginalisation requires a eta and lambda to have the same spaces for the"
                             "variable vector spaces")
        if slot_index not in eta_expr.nd_free_indices:
            raise ValueError(f"Marginalisation requires a eta to have {slot_index.name} in its nd_free_indices")
        if slot_index not in lambda_expr.nd_free_indices:
            raise ValueError(f"Marginalisation requires a lambda to have {slot_index.name} in its nd_free_indices")
        if any(
            eta_expr.index_spaces[idx] != lambda_expr.index_spaces[idx]
            for idx in set(eta_expr.indices) & set(lambda_expr.indices)
        ):
            l = [ f"{str(idx).ljust(8)} " \
                  f"{(str(eta_expr.index_spaces[idx]) if idx in eta_expr.index_spaces else '_').ljust(8)} " \
                  f"{(str(lambda_expr.index_spaces[idx]) if idx in lambda_expr.index_spaces else '_').ljust(8)} "
                  for idx in set(eta_expr.indices) & set(lambda_expr.indices)]
            nl = '\n'
            raise ValueError(
                f"Indices common across subexpressions must act over the same space\n"
                f"Name     lhs      rhs\n"
                f"{nl.join(l)}"
            )
        _index_eta = eta_expr.nd_space_indices[1]
        if 'index_eta' in kwargs:
            if kwargs['index_eta'] != _index_eta:
                raise ValueError(f"Index for eta ({kwargs['index_eta']}) must match that used in eta_expr argument: {_index_eta}")
            _index_eta = kwargs['index_eta']
            kwargs.pop('index_eta')
        _index_eta_slot = eta_expr.nd_space_indices[0]
        if 'index_eta_slot' in kwargs:
            if kwargs['index_eta_slot'] != _index_eta_slot:
                raise ValueError(
                    f"Index for eta connectivity ({kwargs['index_eta_slot']}) must match that used in eta_expr argument: {_index_eta_slot}")
            _index_eta_slot = kwargs['index_eta_slot']
            kwargs.pop('index_eta_slot')
        _index_lambda1 = lambda_expr.nd_space_indices[2]
        if 'index_lambda1' in kwargs:
            if kwargs['index_lambda1'] != _index_lambda1:
                raise ValueError(
                    f"Index for internal (1) lambda ({kwargs['index_lambda1']}) must match that used in lambda_expr argument: {_index_lambda1}")
            _index_lambda1 = kwargs['index_lambda1']
            kwargs.pop('index_lambda1')
        _index_lambda1_slot = lambda_expr.nd_space_indices[0]
        if 'index_lambda1_slot' in kwargs:
            if kwargs['index_lambda1_slot'] != _index_lambda1_slot:
                raise ValueError(
                    f"Index for lambda1 connectivty ({kwargs['index_lambda1_slot']}) must match that used in lambda_expr argument: {_index_lambda1_slot}")
            _index_lambda1_slot = kwargs['index_lambda1_slot']
            kwargs.pop('index_lambda1_slot')
        _index_lambda2 = lambda_expr.nd_space_indices[3]
        if 'index_lambda2' in kwargs:
            if kwargs['index_lambda2'] != _index_lambda2:
                raise ValueError(
                    f"Index for internal (2) lambda ({kwargs['index_lambda2']}) must match that used in lambda_expr argument: {_index_lambda2}")
            _index_lambda2 = kwargs['index_lambda2']
            kwargs.pop('index_lambda2')
        _index_lambda2_slot = lambda_expr.nd_space_indices[1]
        if 'index_lambda2_slot' in kwargs:
            if kwargs['index_lambda2_slot'] != _index_lambda2_slot:
                raise ValueError(
                    f"Index for lambda2 connectivity ({kwargs['index_lambda2_slot']}) must match that used in lambda_expr argument: {_index_lambda2_slot}")
            _index_lambda2_slot = kwargs['index_lambda2_slot']
            kwargs.pop('index_lambda2_slot')
        super().__init__(eta_expr=eta_expr, lambda_expr=lambda_expr, slot_index=slot_index,
                         index_eta_slot = _index_eta_slot, index_eta=_index_eta,
                         index_lambda1_slot = _index_lambda1_slot, index_lambda1=_index_lambda1,
                         index_lambda2_slot = _index_lambda2_slot, index_lambda2=_index_lambda2, **kwargs)

    @property
    def operands(self):
        return [self.eta_expr, self.lambda_expr, self.slot_index,
                self.index_eta_slot, self.index_eta,
                self.index_lambda1_slot, self.index_lambda1,
                self.index_lambda2_slot, self.index_lambda2]

    def with_operands(self, operands: List):
        return self.copy(eta_expr=operands[0], lambda_expr=operands[1], slot_index=operands[2],
                         index_eta_slot=operands[3], index_eta=operands[4],
                         index_lambda1_slot=operands[5], index_lambda1=operands[6],
                         index_lambda2_slot=operands[7], index_lambda2=operands[8])

    def __str__(self) -> str:
        return f"Marginalised({str(self.eta_expr)},{str(self.lambda_expr)}, {str(self.slot_index)})"
    
    def Eta(self):
        return GetEta(self)
    
    def Lambda(self):
        return GetLambda(self)

    def makes_scope(self, index):
        return index in [self.index_eta_slot, self.index_eta,
                         self.index_lambda1_slot, self.index_lambda1,
                         self.index_lambda2_slot, self.index_lambda2]
        
    def get_expression(self, kernelBuilder):
        inst_eta, exp_eta = kernelBuilder._get_expression(self.eta_expr)
        inst_lambda, exp_lambda = kernelBuilder._get_expression(self.lambda_expr)
        
        print(inst_eta, exp_eta)
        print(inst_lambda, exp_lambda)
        print("ahhhhh")

        class NPMarginaliseCode(CodeNode):
            def __init__(cself, eta_expr: ExpressionNode, lambda_expr: ExpressionNode, slot_var: ExprConst):
                cself.eta_expr = eta_expr
                cself.lambda_expr = lambda_expr
                cself.slot_var = slot_var
    
            def code(cself, indent: int) -> str:
                return f"{'    ' * indent}#{self.comment}"
    
            def do(cself, args: typing.Dict[str, typing.Any]):
                eta_np = cself.eta_expr.do(args)
                lambda_np = cself.lambda_expr.do(args)
                # eta_np.reorder
                
        
        return SeqNode([CodeComment("Do ETA:"), inst_eta,
                        CodeComment("Do Lambda:"), inst_lambda,
                        CodeComment(f"Do Marginalisation here for {self.slot_index.name}"),
                        CodeComment(f"Eta:{exp_eta.code()}"),
                        CodeComment(f"Lambda:{exp_lambda.code()}"),]), ExprConst("marginalised Value")


class GetEta(NdExpr, PythonDtlNode):
    fields = NdExpr.fields | {"marginalised", "connectivity_index", "index"}

    def __init__(self, marginalised: MarginaliseFactor, **kwargs):
        _index = marginalised.index_eta
        if 'index' in kwargs:
            # This should not be a problem here as GetEta should be able ot use its own index for the nd_space as
            # marginalisation scopes the previous eta nd_space index
            # if kwargs['index'] != _index:
            #     raise ValueError(
            #         f"Index for internal eta ({kwargs['index_eta']}) must match that used in eta_expr argument: {_index}")
            _index= kwargs['index']
            kwargs.pop('index')
        _connectivity_index = marginalised.index_eta_slot
        if 'connectivity_index' in kwargs:
            _connectivity_index = kwargs.pop('connectivity_index')
            
            
        super().__init__(marginalised=marginalised, connectivity_index = _connectivity_index, index=_index, **kwargs)

    @property
    def indices(self) -> Iterable[Index]:
        return frozenset(self.marginalised.eta_expr.indices) \
            |  frozenset(self.marginalised.lambda_expr.indices) \
            |  frozenset([self.connectivity_index, self.index])
    
    @property
    def nd_free_indices(self) -> Iterable[Index]:
        return frozenset(self.marginalised.eta_expr.nd_free_indices)
    @property
    def index_spaces(self):
        return {**self.marginalised.eta_expr.index_spaces, self.connectivity_index: self.nd_space.spaces[0], self.index:self.nd_space.spaces[1]}
    
    @property
    def nd_space_indices(self):
        return frozenset([self.connectivity_index, self.index])
    
    @property
    def nd_space(self):
        return self.marginalised.eta_expr.nd_space
    
    @property
    def operands(self):
        return [self.marginalised, self.connectivity_index, self.index]
    
    def with_operands(self, operands: List):
        return self.copy(marginalised=operands[0], connectivity_index=operands[1], index=operands[2])
    
    # def makes_scope(self, index):
    #     return index in self.nd_expr.nd_space_indices
    
    def __str__(self) -> str:
        return f"({str(self.marginalised)}.Eta)"

    def get_expression(self, kernelBuilder):
        inst, exp = kernelBuilder._get_expression(self.marginalised)
    
        print(inst, exp)
        print("ETA ahhhhh")
        return SeqNode([inst, CodeComment("Eta Instructions"), AssignTemp("Eta_expr", exp)]), ExprConst("Eta_Expr")



class GetLambda(NdExpr, PythonDtlNode):
    fields = NdExpr.fields | {"marginalised", "connectivity_index1", "connectivity_index2", "index1", "index2"}
    
    def __init__(self, marginalised: MarginaliseFactor, **kwargs):
        _connectivity_index2 = marginalised.index_lambda2_slot
        _connectivity_index1 = marginalised.index_lambda1_slot
        _index1 = marginalised.index_lambda1
        _index2 = marginalised.index_lambda2
        if 'index1' in kwargs:
            _index1= kwargs.pop('index1')
        if 'index2' in kwargs:
            _index2= kwargs.pop('index2')
        if 'connectivity_index1' in kwargs:
            _connectivity_index1 = kwargs.pop('connectivity_index1')
        if 'connectivity_index2' in kwargs:
            _connectivity_index2 = kwargs.pop('connectivity_index2')
        super().__init__(marginalised=marginalised,
                         connectivity_index1=_connectivity_index1, connectivity_index2=_connectivity_index2,
                         index1=_index1, index2=_index2, **kwargs)
        
    @property
    def indices(self) -> Iterable[Index]:
        return frozenset(self.marginalised.eta_expr.indices) \
            | frozenset(self.marginalised.lambda_expr.indices) \
            | frozenset([self.connectivity_index1, self.connectivity_index2, self.index1, self.index2])
    
    @property
    def nd_free_indices(self) -> Iterable[Index]:
        return frozenset(self.marginalised.lambda_expr.nd_free_indices)
    
    @property
    def index_spaces(self):
        return {**self.marginalised.eta_expr.index_spaces,
                self.connectivity_index1: self.nd_space.spaces[0],
                self.connectivity_index2: self.nd_space.spaces[1],
                self.index1: self.nd_space.spaces[2],
                self.index2: self.nd_space.spaces[3]}
    
    @property
    def nd_space_indices(self):
        return frozenset([self.connectivity_index1, self.connectivity_index2, self.index1, self.index2])
    
    @property
    def nd_space(self):
        return self.marginalised.lambda_expr.nd_space
    
    @property
    def operands(self):
        return [self.marginalised, self.connectivity_index1, self.connectivity_index2, self.index1, self.index2]
    
    def with_operands(self, operands: List):
        return self.copy(marginalised=operands[0],
                         connectivity_index1=operands[1], connectivity_index2=operands[2],
                         index1=operands[3], index2=operands[4])
    
    # def makes_scope(self, index):
    #     return index in self.nd_expr.nd_space_indices
    
    def __str__(self) -> str:
        return f"({str(self.marginalised)}.Lambda)"

    def get_expression(self, kernelBuilder):
        inst, exp = kernelBuilder._get_expression(self.marginalised)
    
        print(inst, exp)
        print("Lambda ahhhhh")
        return SeqNode([inst, CodeComment("Lambda Instructions"), AssignTemp("Lambda_expr", exp)]), ExprConst("Lambda_Expr")
        
# i = Index("i")
# j = Index("j")
# k = Index("k")
# l = Index("l")
# vsR1 = RealVectorSpace(10)
# vsR2 = RealVectorSpace(2)
# vsR3 = RealVectorSpace(3)



"""

F0 -- P0 -- F1 -- P1 -- F2 -- P2 -- F3 -- P3

"""
N_poses = RealVectorSpace(5)
n = Index("n")
PoseVec = RealVectorSpace(3)
e = Index("e")
l1 = Index("l1")
l2 = Index("l2")
Factor_Connectivity = RealVectorSpace(2)
c = Index("c")
s = Index("s")
sp = Index("sp")
M_Measurements = RealVectorSpace(4)
m = Index("m")
# InternalMeasurement_Size = RealVectorSpace(PoseVec.dim * Factor_Connectivity.dim)
# ie = Index("ie")
# il1 = Index("il1")
# il2 = Index("il2")

etaTS = PoseVec ** 1 #TensorSpace([PoseVec])
lambdaTS = PoseVec ** 2 #TensorSpace([PoseVec, PoseVec])
etaiTS = Factor_Connectivity * etaTS #TensorSpace([Factor_Connectivity, etaTS])
lambdaiTS = Factor_Connectivity * Factor_Connectivity * lambdaTS #TensorSpace([Factor_Connectivity, lambdaTS])

AdjacencyMatrix = TensorVariable(M_Measurements*Factor_Connectivity*N_poses, "A")
FactorBaseEtas = TensorVariable(M_Measurements*etaiTS, "FBE")
FactorBaseLambdas = TensorVariable(M_Measurements*lambdaiTS, "FBL")
# SlotToEta = TensorVariable(Factor_Connectivity*etaTS*Factor_Connectivity*etaiTS, "SlotToEta")
# SlotToLambda = TensorVariable(Factor_Connectivity*lambdaTS*Factor_Connectivity*lambdaiTS, "SlotToLambda")
Connectivity_Ones= TensorVariable(Factor_Connectivity, "1")

FactorToVariableEtaMessages = TensorVariable(M_Measurements*Factor_Connectivity*etaTS, "FVEM")
FactorToVariableLambdaMessages = TensorVariable(M_Measurements*Factor_Connectivity*lambdaTS, "FVLM")

VariablesTotalEtas = IndexSum(AdjacencyMatrix[m,c,n] * FactorToVariableEtaMessages[m,c,e], [m,c]) # [n,e]
VariablesTotalLambdas = IndexSum(AdjacencyMatrix[m,c,n] * FactorToVariableLambdaMessages[m,c,l1,l2], [m,c]) # [n,l1,l2]

VariableToFactorEtaMessages = IndexSum(VariablesTotalEtas * AdjacencyMatrix[m,c,n], [n]) - FactorToVariableEtaMessages[m,c,e] # [m,c,e]
VariableToFactorLambdaMessages = IndexSum(VariablesTotalLambdas * AdjacencyMatrix[m,c,n], [n]) - FactorToVariableLambdaMessages[m,c,l1,l2] # [m,c,l1,l2]


FactorEtas = (VariableToFactorEtaMessages * Connectivity_Ones[s]) + FactorBaseEtas[m,s,e] # [m,c,s,e]
FactorLambdas = (VariableToFactorLambdaMessages * Connectivity_Ones[s] * Connectivity_Ones[sp]) + FactorBaseLambdas[m,s,sp,l1,l2]# [m,c,s,sp,l1,l2]

marginalised = MarginaliseFactor(FactorEtas.expose([s,e]), FactorLambdas.expose([s,sp,l1,l2]), c)
etas = marginalised.Eta().as_scalar().forall(m,c,e)
lambdas = marginalised.Lambda().as_scalar().forall(m,c,l1,l2)

# ex2 = dtlutils.names.make_Index_names_unique(etas)
visualise.plot_dag([etas, lambdas], view=True, coalesce_duplicates=True, label_edges=True)
print([str(i) for i in etas.scalar_expr.free_indices])
print([str(x) for x in etas.indices])
print("done")

builder = KernelBuilder([etas, lambdas])
kernel = builder.build()
print("================================")
print(builder.codeNode.code())