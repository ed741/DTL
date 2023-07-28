import dtl
import xdsl.dialects.arith
from dtl import *
from dtl.dag import RealVectorSpace, Index
import dtlpp.backends.xdsl as xdtl
import numpy as np

from dtl.dtlutils import visualise
from xdsl import printer, ir
from xdsl.dialects import memref
from xdsl.dialects.builtin import Float64Type, f32, ModuleOp
from xdsl.dialects.func import FuncOp, Return
from xdsl.ir import SSAValue, Region
from xdsl.irdl import SingleBlockRegionDef

i = Index('i')
j = Index('j')
k = Index('k')
v5 = RealVectorSpace(5)
v6 = RealVectorSpace(6)
vu = UnknownSizeVectorSpace("VU")
A = TensorVariable(v5*v5*vu, "A")
expr = ((A[i,j,k]+Literal(2))*Literal(3)-Literal(2)).sum(j).forall(k)
# expr, _ = ExprTuple.tupleOf(expr, Literal(10)[i:v5]).tuple()
print("expr: "+str(expr))
print("expr type: "+str(expr.type))

type = xdtl.DTLType_to_xdtl(expr.type)
print("type")
print(type)

# module = ModuleOp([])
# block = module.body.block
block = ir.Block()
a = block.insert_arg(xdsl.dialects.builtin.TensorType.from_type_and_list(f32, [5,5]), 0)
vu_len = block.insert_arg(xdsl.dialects.builtin.i32, 1)
i_val = block.insert_arg(xdsl.dialects.builtin.i32, 2)
mem = block.insert_arg(memref.MemRefType.from_element_type_and_shape(f32, (5,)), 3)

# lines, output = xdtl.get_xdsl_dtl_version(expr, tensorVariables={A:a})
lines, output = xdtl.get_xdsl_dtl_exec_version(expr, space_map={vu:vu_len}, arg_map={i:i_val}, tensor_variables={A:a}, output=mem)

# block.add_ops(lines)

ret = Return()
block.add_ops(lines+[ret])

region = Region([block])
argTypes = [arg.type for arg in block.args]
# retTypes = [r.type for r in ret.operands]
retTypes = []
func = FuncOp.from_region("main", argTypes, retTypes, region)
module = ModuleOp([func])

print("lines:")
p = printer.Printer()
p.print(block)
block.verify()

print("Moduel???")
print(module)
print("args")
print(func.args)
print("results")
print(func.get_return_op())
print("func type")
print(func.function_type)
