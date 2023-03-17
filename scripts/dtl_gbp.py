import typing

import numpy as np

import dtl
from dtl import Expr, Index, RealVectorSpace, TensorVariable, IndexSum, ExprTuple, DTLType
from dtlpp.backends import native
from dtlpp.backends.native import PythonDtlNode, KernelBuilder, CodeComment, SeqNode, ExprConst, CodeNode, \
    ExpressionNode
from dtlutils import visualise, traversal, optimise


class MarginaliseFactor(dtl.Expr, PythonDtlNode):
    fields = Expr.fields | {"expr", "slot_index"}
    
    def __init__(self, expr: dtl.ExprTypeHint, slot_index: Index, **kwargs):
        expr = Expr.exprInputConversion(expr)
        exprType = expr.type
        typeCheck = exprType.result.matchShape((("v","t"),("v","v","t","t")))
        if typeCheck is None:
            raise ValueError("Marginalisation requires a eta to have 2 exposed dimension (connectivity * vector) and lambda to have 4: (connectivity * connectivity * vector * vector) ")
        if slot_index not in exprType.indices:
            raise ValueError(f"Marginalisation requires a the input expr to have {slot_index.name} in its type's argument indices")
        super().__init__(expr=expr, slot_index=slot_index, **kwargs)
        
    @property
    def type(self) -> DTLType:
        return self.expr.type
    
    @property
    def operands(self):
        return {"expr":self.expr, "slot_index":self.slot_index}

    def with_operands(self, operands):
        return self.copy(expr=operands["expr"], slot_index=operands["slot_index"])

    def __str__(self) -> str:
        return f"Marginalised({str(self.expr)}, {str(self.slot_index)})"

    def shortStr(self) -> str:
        return f"Marg({self.expr.terminalShortStr()}, {str(self.slot_index)})"

    def get_expression(self, kernelBuilder: native.KernelBuilder, indexMap: typing.Dict[dtl.Index, str], path: typing.Tuple[str,...]):
        inst, expressions = kernelBuilder._get_expression(traversal.operandLabelled(self, ["expr"]), indexMap, path)
        exp_eta, exp_lambda = expressions
        class NPMarginaliseCode(CodeNode):
            def __init__(cself, eta_expr: ExpressionNode, lambda_expr: ExpressionNode, slot_var: ExprConst,
                         temp_eta_name:str, temp_lambda_name:str):
                cself.eta_expr = eta_expr
                cself.lambda_expr = lambda_expr
                cself.slot_var = slot_var
                cself.temp_eta_name = temp_eta_name
                cself.temp_lambda_name = temp_lambda_name
    
            def code(cself, indent: int) -> str:
                id = '    ' * indent
                return f"{id}#MAGINALISE STUFF\n" \
                       f"{id}eta_np = {cself.eta_expr.code()}\n" \
                       f"{id}lambda_np = {cself.lambda_expr.code()}\n" \
                       f"{id}slot_idx = {cself.slot_var.code()}\n" \
                       f"{id}eta_np[[0, slot_idx],:] = eta_np[[slot_idx, 0],:]\n" \
                       f"{id}lambda_np[:, [0, slot_idx], :, :] = lambda_np[:, [slot_idx, 0], :, :]\n" \
                       f"{id}lambda_np[[0, slot_idx], :, :, :] = lambda_np[[slot_idx, 0], :, :, :]\n" \
                       f"{id}ea = eta_np[0,:]\n" \
                       f"{id}eb = eta_np[1:,:].flatten()\n" \
                       f"{id}aa = lambda_np[0,0,:,:]\n" \
                       f"{id}ab = lambda_np[0,1:,:,:].transpose([1,0,2]).reshape(lambda_np.shape[-1],-1)\n" \
                       f"{id}ba = lambda_np[1:, 0, :, :].reshape(-1,lambda_np.shape[-1])\n" \
                       f"{id}bb = lambda_np[1:, 1:, :, :].transpose([0,2,1,3]).reshape(-1,lambda_np.shape[-1]*(lambda_np.shape[0]-1))\n" \
                       f"{id}bbInv = np.linalg.inv(bb)\n" \
                       f"{id}abbbInv = np.matmul(ab, bbInv)\n" \
                       f"{id}{cself.temp_eta_name} = ea - np.matmul(abbbInv, eb)\n" \
                       f"{id}{cself.temp_lambda_name} = aa - np.matmul(abbbInv, ba)"

            def do(cself, args: typing.Dict[str, typing.Any]):
                eta_np = cself.eta_expr.do(args)
                lambda_np = cself.lambda_expr.do(args)
                slot_idx = cself.slot_var.do(args)
                eta_np[[0, slot_idx],:] = eta_np[[slot_idx, 0],:]
                lambda_np[:, [0, slot_idx], :, :] = lambda_np[:, [slot_idx, 0], :, :]
                lambda_np[[0, slot_idx], :, :, :] = lambda_np[[slot_idx, 0], :, :, :]
                ea = eta_np[0,:]
                eb = eta_np[1:,:].flatten()
                aa = lambda_np[0,0,:,:]
                ab = lambda_np[0,1:,:,:].transpose([1,0,2]).reshape(lambda_np.shape[-1],-1)
                ba = lambda_np[1:, 0, :, :].reshape(-1,lambda_np.shape[-1])
                bb = lambda_np[1:, 1:, :, :].transpose([0,2,1,3]).reshape(-1,lambda_np.shape[-1]*(lambda_np.shape[0]-1))
                bbInv = np.linalg.inv(bb)
                abbbInv = np.matmul(ab, bbInv)
                emarg = ea - np.matmul(abbbInv, eb)
                lmarg = aa - np.matmul(abbbInv, ba)
                args[cself.temp_eta_name] = emarg
                args[cself.temp_lambda_name] = lmarg
                
        temp_eta_name = "marginalisedEtaValue"
        temp_lambda_name = "marginalisedLambdaValue"
        return SeqNode([CodeComment("Do ETA and Lambda:"), inst,
                        CodeComment(f"Do Marginalisation here for {self.slot_index.name} named: {indexMap[self.slot_index]}"),
                        CodeComment(f"Eta:{exp_eta.code()}"),
                        CodeComment(f"Lambda:{exp_lambda.code()}"),
                        NPMarginaliseCode(exp_eta, exp_lambda, ExprConst(indexMap[self.slot_index]),
                                          temp_eta_name, temp_lambda_name),
                        ]),\
            (ExprConst("marginalisedEtaValue"),ExprConst("marginalisedLambdaValue"))



"""
Code for our GBP Problem:

F0 -- P0 -- F1 -- P1 -- F2 -- P2 -- F3 -- P3

"""
# N_poses is the number of poses (variable nodes in the Factorgraph)
N_poses = RealVectorSpace(5)
# For clarity, we prefer using specific index names for different dimensions though this is not needed and only for ease
# of reading
n = Index("n")
# PoseVec is the internal size of each variable in the factorGraph - for example (x,y,z)
PoseVec = RealVectorSpace(3)
e = Index("e")
l1 = Index("l1")
l2 = Index("l2")
# Factor_connectivity is the maximum number of variables each measurement (factor node) is connected to.
Factor_Connectivity = RealVectorSpace(2)
c = Index("c")
s = Index("s")
sp = Index("sp")
# M_Measurements is the number of measurements in the graph (factor nodes in the factor graph)
M_Measurements = RealVectorSpace(4)
m = Index("m")

# These define some helpful tensorSpaces that can be combined later with more vector spaces
etaTS = PoseVec ** 1 #TensorSpace([PoseVec])
lambdaTS = PoseVec ** 2 #TensorSpace([PoseVec, PoseVec])
etaiTS = Factor_Connectivity * etaTS #TensorSpace([Factor_Connectivity, etaTS])
lambdaiTS = Factor_Connectivity * Factor_Connectivity * lambdaTS #TensorSpace([Factor_Connectivity, lambdaTS])

# Each measurement has 'slots' equal to Factor_connectivity. This allows us to describe which connection from the
# measurement node each pose is - this is needed as the relationships are not the same, unlike with pose nodes where it
# doesn't matter which connected measurement node is which

# Adjacency matrix elements are either 0 for no connection, and 1 where this is an edge in the graph. Each Measurement
# Node slot may have up to 1 non-zero value as each
AdjacencyMatrix = TensorVariable(M_Measurements*Factor_Connectivity*N_poses, "A")
FactorBaseEtas = TensorVariable(M_Measurements*etaiTS, "FBE")
FactorBaseLambdas = TensorVariable(M_Measurements*lambdaiTS, "FBL")
# SlotToEta = TensorVariable(Factor_Connectivity*etaTS*Factor_Connectivity*etaiTS, "SlotToEta")
# SlotToLambda = TensorVariable(Factor_Connectivity*lambdaTS*Factor_Connectivity*lambdaiTS, "SlotToLambda")
Connectivity_Ones= TensorVariable(Factor_Connectivity, "One")

FactorToVariableEtaMessages = TensorVariable(M_Measurements*Factor_Connectivity*etaTS, "FVEM")
FactorToVariableLambdaMessages = TensorVariable(M_Measurements*Factor_Connectivity*lambdaTS, "FVLM")

VariablesTotalEtas = IndexSum(AdjacencyMatrix[m,c,n] * FactorToVariableEtaMessages[m,c,e], [m,c]) # [n,e]
VariablesTotalLambdas = IndexSum(AdjacencyMatrix[m,c,n] * FactorToVariableLambdaMessages[m,c,l1,l2], [m,c]) # [n,l1,l2]

VariableToFactorEtaMessages = native.InstantiationExprNode(IndexSum(VariablesTotalEtas * AdjacencyMatrix[m,c,n], [n]) - FactorToVariableEtaMessages[m,c,e]) # [m,c,e]
VariableToFactorLambdaMessages = native.InstantiationExprNode(IndexSum(VariablesTotalLambdas * AdjacencyMatrix[m,c,n], [n]) - FactorToVariableLambdaMessages[m,c,l1,l2]) # [m,c,l1,l2]


# FactorEtas_t = native.InstantiationExprNode(((VariableToFactorEtaMessages[s:c]) + FactorBaseEtas[m,s,e]).forall(m,c,s,e))
# FactorEtas = FactorEtas_t[m,c,s,e] # [m,c,s,e]
# FactorEtas = ((VariableToFactorEtaMessages[s:c]) + FactorBaseEtas[m,s,e])
FactorEtas = native.SequenceExprNode(VariableToFactorEtaMessages, ((VariableToFactorEtaMessages[s:c]) + FactorBaseEtas[m,s,e]).forall(s))
# FactorLambdas_t = native.InstantiationExprNode(((VariableToFactorLambdaMessages[s:c, sp:c]) + FactorBaseLambdas[m,s,sp,l1,l2]).forall(m,c,s,sp,l1,l2))
# FactorLambdas = FactorLambdas_t[m,c,s,sp,l1,l2]# [m,c,s,sp,l1,l2]
# FactorLambdas = ((VariableToFactorLambdaMessages[s:c, sp:c]) + FactorBaseLambdas[m,s,sp,l1,l2])
FactorLambdas = native.SequenceExprNode(VariableToFactorLambdaMessages, ((VariableToFactorLambdaMessages[s:c, sp:c]) + FactorBaseLambdas[m,s,sp,l1,l2]).forall(s,sp))



# preMarg = native.InstantiationExprNode((FactorEtas.forall(s,e), FactorLambdas.forall(s,sp,l1,l2)))
# preMarg = (FactorEtas.forall(s,e), FactorLambdas.forall(s,sp,l1,l2))
preMarg = (FactorEtas.forall(e), FactorLambdas.forall(l1,l2))
marginalised = MarginaliseFactor(preMarg, c).forall(m,c)

# expr = native.SequenceExprNode((FactorEtas_t, FactorLambdas_t), marginalised)
expr = marginalised

expr = optimise.use_common_subexpressions(expr, ["AutoGenerated", "frame_info"])

# ex2 = dtlutils.names.make_Index_names_unique(etas)
visualise.plot_dag(expr, view=True, coalesce_duplicates=True, label_edges=True, short_strs=True, show_types=True, skip_terminals=True)
# print([str(i) for i in etas.scalar_expr.free_indices])
# print([str(x) for x in etas.indices])
print("done")

builder = KernelBuilder(expr)
kernel = builder.build()
print("================================")
print(builder.codeNode.code())
out = kernel(A=np.zeros([4,6,5]), FVEM=np.zeros([4,6,3]), FVLM=np.zeros([4,6,3,3]), One=np.ones([6]), FBE=np.random.rand(5,6,3), FBL=np.random.rand(5,6,6,3,3))
print(out)
e,l = out
print(e.shape)
print(l.shape)