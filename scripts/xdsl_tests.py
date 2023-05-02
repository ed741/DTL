import dtl
from dtl import *
from dtl.dag import RealVectorSpace, Index
import dtlpp.backends.xdsl as xdtl
import numpy as np

from dtl.dtlutils import visualise
from xdsl import printer, ir
from xdsl.ir import SSAValue

i = Index('i')
j = Index('j')
A = TensorVariable(RealVectorSpace(5)*RealVectorSpace(5), "A")
expr = A[i:UnknownSizeVectorSpace("ha")]
type = xdtl.DTLType_to_xdtl(expr.type)
print("type")
print(type)

lines, output = xdtl.get_xdsl_dtl_version(expr, tensorVariables={A:ir.BlockArgument(xdtl.DTLType_to_xdtl(A.type))})
print("lines:")
p = printer.Printer()
p.print(*lines)