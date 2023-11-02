import typing

import numpy as np

import dtl
from dtl import Expr, Index, RealVectorSpace, TensorVariable, IndexSum, DTLType
from dtl.dtlMatrix import Invert
from dtlpp.backends import native
from dtlpp.backends.native import PythonDtlNode, KernelBuilder, CodeComment, SeqNode, ExprConst, CodeNode, \
    ExpressionNode
from dtl.dtlutils import visualise, traversal
from dtl.passes import optimise


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
        exprType = self.expr.type
        typeCheck = exprType.result.matchShape((("v", "t"), ("v", "v", "t", "t")))
        results = dtl.ResultTupleType([dtl.ShapeType([typeCheck["t"]]),dtl.ShapeType([typeCheck["t"],typeCheck["t"]])])
        return exprType.withResult(results)
    
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
                       f"{id}if bb[0,0] !=0.0:\n" \
                       f"{id}   bbInv = np.linalg.inv(bb)\n" \
                       f"{id}else:\n" \
                       f"{id}   bbInv = bb\n" \
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
                if bb[0,0] !=0.0:
                    bbInv = np.linalg.inv(bb)
                else:
                    bbInv = bb
                abbbInv = np.matmul(ab, bbInv)
                emarg = ea - np.matmul(abbbInv, eb)
                lmarg = aa - np.matmul(abbbInv, ba)
                # print(emarg)
                # print(lmarg)
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
Truth:
X:            0           2           3           4
Y:            0           2           3           4
Z:            0           2           3           4
"""

HMatrixArray = np.zeros((4,3,2,3))
LambdasArray = np.zeros((4,3,3))
JacobiansArray = np.zeros((4,3,2,3))
ZArray = np.zeros((4,3))
AdjacencyArray = np.zeros((4,2,4))
def make_middle_factor(f,va,vb, x,y,z, var):
    AdjacencyArray[f, 0, va] = 1
    AdjacencyArray[f, 1, vb] = 1
    ZArray[f,:] = np.array([x,y,z])
    LambdasArray[f,:,:] = var*np.identity(3)
    JacobiansArray[f,:,0,:] = -1*np.identity(3)
    JacobiansArray[f,:,1,:] = np.identity(3)
    HMatrixArray[f,:,0,:] = -1*np.identity(3)
    HMatrixArray[f,:,1,:] = np.identity(3)
    
make_middle_factor(1,0,1,100,10,10,1) # v0-f1-v1
make_middle_factor(2,1,2,10,100,10,1) # v1-f2-v2
make_middle_factor(3,2,3,10,10,100,1) # v2-f3-v3

def make_anchor_factor(f,v,x,y,z, var):
    AdjacencyArray[f, 0, v] = 1
    ZArray[f, :] = np.array([x, y, z])
    LambdasArray[f, :, :] = var * np.identity(3)
    JacobiansArray[f, :, 0, :] = np.identity(3)
    HMatrixArray[f, :, 0, :] = np.identity(3)
    
make_anchor_factor(0,0,0,0,0,100) # f0-v0
# make_anchor_factor(1,1,10,10,-5,1) # f0-v0
# make_anchor_factor(2,2,20,20,-5,10) # f0-v0
# make_anchor_factor(3,3,30,30,-5,100) # f0-v0

print("Jacobians")
print(JacobiansArray[1])



# N_variables is the number of poses (variable nodes in the Factorgraph)
N_variables = RealVectorSpace(4)
# For clarity, we prefer using specific index names for different dimensions though this is not needed and only for ease
# of reading
n = Index("n")
# VarVec is the internal size of each variable in the factorGraph - for example (x,y,z)
VarVec = RealVectorSpace(3)
e = Index("e")
l1 = Index("l1")
l2 = Index("l2")
ep = Index("ep")
lp1 = Index("lp1")
lp2 = Index("lp2")
# Factor_connectivity is the maximum number of variables each measurement (factor node) is connected to.
Factor_Connectivity = RealVectorSpace(2)
c = Index("c")
cp = Index("cp")
s = Index("s")
sp = Index("sp")
# M_Measurements is the number of measurements in the graph (factor nodes in the factor graph)
M_Measurements = RealVectorSpace(4)
m = Index("m")

# These define some helpful tensorSpaces that can be combined later with more vector spaces
etaTS = VarVec ** 1 #TensorSpace([VarVec])
lambdaTS = VarVec ** 2 #TensorSpace([VarVec, VarVec])
etaiTS = Factor_Connectivity * etaTS #TensorSpace([Factor_Connectivity, etaTS])
lambdaiTS = Factor_Connectivity * Factor_Connectivity * lambdaTS #TensorSpace([Factor_Connectivity, lambdaTS])

# Each measurement has 'slots' equal to Factor_connectivity. This allows us to describe which connection from the
# measurement node each pose is - this is needed as the relationships are not the same, unlike with pose nodes where it
# doesn't matter which connected measurement node is which

# Adjacency matrix elements are either 0 for no connection, and 1 where this is an edge in the graph. Each Measurement
# Node slot may have up to 1 non-zero value as each
AdjacencyMatrix = TensorVariable(M_Measurements * Factor_Connectivity * N_variables, "A")

Jacobians = TensorVariable(M_Measurements * VarVec * Factor_Connectivity * VarVec, "J")
HMatrix = TensorVariable(M_Measurements * VarVec * Factor_Connectivity * VarVec, "HM")
FactorInputZs = TensorVariable(M_Measurements * VarVec, "Z")
FactorInputLambdas = TensorVariable(M_Measurements * lambdaTS, "L")

FactorToVariableEtaMessages = TensorVariable(M_Measurements*Factor_Connectivity*etaTS, "FVEM")
FactorToVariableLambdaMessages = TensorVariable(M_Measurements*Factor_Connectivity*lambdaTS, "FVLM")

VariablesTotalEtasT = native.InstantiationExprNode((AdjacencyMatrix[m,c,n] * FactorToVariableEtaMessages[m,c,e]).sum(m,c).forall(n,e))
VariablesTotalEtas = VariablesTotalEtasT[n,e]# [n,e]
VariablesTotalLambdasT = native.InstantiationExprNode((AdjacencyMatrix[m,c,n] * FactorToVariableLambdaMessages[m,c,l1,l2]).sum(m,c).forall(n,l1,l2))
VariablesTotalLambdas = VariablesTotalLambdasT[n,l1,l2]# [n,l1,l2]

VariablesTotalCovariances = Invert(VariablesTotalLambdas.forall(l1,l2), skip_zero=True) #[n | 3x3]
VariablesMusT = native.InstantiationExprNode((VariablesTotalCovariances[e,l2] * VariablesTotalEtas).sum(l2).forall(n,e))
VariablesMus = VariablesMusT[n,e]# [n, e]

VariableToFactorEtaMessages = native.InstantiationExprNode(IndexSum(VariablesTotalEtas * AdjacencyMatrix[m,s,n], [n]) - FactorToVariableEtaMessages[m,s,e]) # [m,s,e]
VariableToFactorLambdaMessages = native.InstantiationExprNode(IndexSum(VariablesTotalLambdas * AdjacencyMatrix[m,cp,n], [n]) - FactorToVariableLambdaMessages[m,cp,l1,l2]) # [m,cp,l1,l2]
VariableToFactorMuMessages = native.InstantiationExprNode(IndexSum(VariablesMus * AdjacencyMatrix[m,c,n], [n])) # [m,c,e]

# HLocal = native.IndexRebinding((HMatrix[m,ep,c,e]*VariableToFactorMuMessages).sum(c,e), [ep], [e]) # [m,e]
HLocalT = native.InstantiationExprNode((HMatrix[m,ep,c,e]*VariableToFactorMuMessages).sum(c,e).forall(m,ep))
HLocal = HLocalT[m,e] # [m,e]


FactorBaseLambdas = (Jacobians[m,lp1,s,l1] * FactorInputLambdas[m,lp1,lp2] * Jacobians[m,lp2,sp,l2]).sum(lp1,lp2) # m,s,l1,sp,l2
Temp_eta_base_rhs = native.IndexRebinding((Jacobians[m,e,c,ep]*VariableToFactorMuMessages).sum(c,ep) + FactorInputZs[m,e] - HLocal, [e], [lp2]) # m,lp2
FactorBaseEtas = native.IndexRebinding(((Jacobians[m,lp1,s,l1] * FactorInputLambdas[m,lp1,lp2]).sum(lp1) * Temp_eta_base_rhs).sum(lp2),[l1],[e]) # m,s,e
# FactorEtas_t = native.InstantiationExprNode(((VariableToFactorEtaMessages[s:c]) + FactorBaseEtas[m,s,e]).forall(m,c,s,e))
# FactorEtas = FactorEtas_t[m,c,s,e] # [m,c,s,e]
# FactorEtas = ((VariableToFactorEtaMessages[s:c]) + FactorBaseEtas[m,s,e])

ID = native.InstantiationExprNode(dtl.Literal(1)[sp:Factor_Connectivity].forall(sp,sp))
D = native.InstantiationExprNode((dtl.Literal(1)[s:Factor_Connectivity,c:Factor_Connectivity] - ID[s,c]).forall(s,c))
D = native.SequenceExprNode(ID, D)

FactorEtas = ((VariableToFactorEtaMessages*D[c,s]) + FactorBaseEtas).forall(s,e) # m,c -> [s,e]
# FactorLambdas_t = native.InstantiationExprNode(((VariableToFactorLambdaMessages[s:c, sp:c]) + FactorBaseLambdas[m,s,sp,l1,l2]).forall(m,c,s,sp,l1,l2))
# FactorLambdas = FactorLambdas_t[m,c,s,sp,l1,l2]# [m,c,s,sp,l1,l2]
# FactorLambdas = ((VariableToFactorLambdaMessages[s:c, sp:c]) + FactorBaseLambdas[m,s,sp,l1,l2])
FactorLambdas = ((VariableToFactorLambdaMessages* ID[sp,s]*ID[cp,s]*D[c,cp]).sum(cp) + FactorBaseLambdas).forall(s,sp, l1, l2) # m,c -> [s,sp,l1,l2]



# preMarg = native.InstantiationExprNode((FactorEtas.forall(s,e), FactorLambdas.forall(s,sp,l1,l2)))
# preMarg = (FactorEtas.forall(s,e), FactorLambdas.forall(s,sp,l1,l2))
preMargT = native.InstantiationExprNode(dtl.ExprTuple([FactorEtas, FactorLambdas]).forall(m,c))
preMarg = preMargT[(dtl.NoneIndex, dtl.NoneIndex, m,c),(dtl.NoneIndex, dtl.NoneIndex,dtl.NoneIndex, dtl.NoneIndex, m,c)]
marg = MarginaliseFactor(preMarg, c)
marginalised = dtl.DeindexExpr(marg, ((m,c,VarVec),(m,c,VarVec,VarVec)))
expr = marginalised
# expr = native.SequenceExprNode((FactorEtas_t, FactorLambdas_t), marginalised)
expr = native.SequenceExprNode(preMargT, expr)
expr = native.SequenceExprNode(HLocalT, expr)
expr = native.SequenceExprNode(VariablesMusT, expr)
expr = native.SequenceExprNode(VariablesTotalLambdasT, expr)
expr = native.SequenceExprNode(VariablesTotalEtasT, expr)
expr = native.SequenceExprNode(D, expr)


expr = optimise.use_common_subexpressions(expr, ["AutoGenerated", "frame_info"])

# ex2 = dtlutils.names.make_Index_names_unique(etas)
visualise.plot_dag(expr, view=True, coalesce_duplicates=True, label_edges=True, short_strs=True, show_types=True, skip_terminals=True)
# print([str(i) for i in etas.scalar_expr.free_indices])
# print([str(x) for x in etas.indices])
print("done")

builder = KernelBuilder(expr, debug_comments=0)
kernel = builder.build()
print("================================")
print(builder.codeNode.code())

builderMu = KernelBuilder(VariablesMus.forall(n,e))
MuFunc = builderMu.build()

fvem = np.zeros([4,2,3])
fvlm = np.zeros([4,2,3,3])
for step in range(5):
    print(f"Step:{step}")
    out = kernel(A=AdjacencyArray, FVEM=fvem, FVLM=fvlm, J=JacobiansArray, HM=HMatrixArray, L=LambdasArray, Z=ZArray)
    # print(out)
    out_eta,out_lambda = out
    # print(out_eta.shape)
    # print(out_lambda.shape)
    factors = out_eta.shape[0]
    # for f in range(factors):
    #     v = np.argmax(AdjacencyArray[f,0,:])
    #     print(f"f->v :: f:{f}, va:{v}")
    #     print(out_eta[f,0,:])
    #     print(out_lambda[f, 0, :,:])
    #     v = np.argmax(AdjacencyArray[f, 1, :])
    #     print(f"f->v :: f:{f}, vb:{v}")
    #     print(out_eta[f, 1, :])
    #     print(out_lambda[f, 1, :, :])
    fvem = out_eta
    fvlm = out_lambda
    
    mus = MuFunc(FVEM=fvem, FVLM=fvlm, A=AdjacencyArray)
    print("mu:")
    print(mus)
    print(mus.shape)


