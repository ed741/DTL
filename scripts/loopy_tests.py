from dtl import *
from dtl.dag import RealVectorSpace, Index, Lambda
from dtlpp.backends.loopy import KernelBuilder
import loopy as lp
import numpy as np


vs = RealVectorSpace(5)
i = Index("i")
j = Index("j")
print("loopy_test.1")
# expr = Lambda((B:= (vs*vs).new("B"),), B[i,j].forall(i))
B = (vs*vs).new("B")
expr = B[i,j].forall(i)
print("loopy_test.2")
builder = KernelBuilder(expr)
print("loopy_test.3")
kernel = builder.build()
print("loopy_test.4")
print(kernel)
print("loopy_test.5")
print(lp.generate_code_v2(kernel).device_code())  # the C code
print("loopy_test.6")
input_tensor = np.ones((5,5))
output_tensor = np.zeros(5)
kernel(mytemporarytemporary=output_tensor, B=input_tensor)

print(output_tensor)