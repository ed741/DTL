import dtl
import xdsl.dialects.arith
from dtl import *
from dtl.dag import RealVectorSpace, Index
import dtlpp.backends.xdsl as xdtl
import numpy as np

from dtl.dtlutils import visualise
from xdsl import printer, ir
from xdsl.dialects.builtin import Float64Type, f32
from xdsl.ir import SSAValue
from xdsl.irdl import SingleBlockRegionDef

i = Index('i')
j = Index('j')
v5 = RealVectorSpace(5)
v6 = RealVectorSpace(6)
A = TensorVariable(v5*v5*v6, "A")
expr = A[i,j,None].sum(j).forall(i)
type = xdtl.DTLType_to_xdtl(expr.type)
print("type")
print(type)

block = ir.Block()
a = block.insert_arg(xdsl.dialects.builtin.TensorType.from_type_and_list(f32, [5,5]), 0)
lines, output = xdtl.get_xdsl_dtl_version(expr, tensorVariables={A:a})
block.add_ops(lines)
print("lines:")
p = printer.Printer()
p.print(block)
block.verify()