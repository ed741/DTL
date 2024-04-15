from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
from xdsl.ir import Block, Use
from xdsl.printer import Printer

a = arith.Constant(IntegerAttr(10, IndexType()))
b = arith.Constant(IntegerAttr(11, IndexType()))

cond = arith.Constant(IntegerAttr(1, IntegerType(1)))

if_op = scf.If(cond, [IndexType()], [a, a_y := scf.Yield(a)], [b, b_y := scf.Yield(b)])


block = Block()
block.add_op(cond)
block.add_op(if_op)
print(block)
Printer().print(block)
from xdsl.traits import UseDefChainTrait

print("get defs of a.result's uses:")
print([UseDefChainTrait.get_defs_following_from_operand(use) for use in a.result.uses])
print("get defs of b.result's uses:")
print([UseDefChainTrait.get_defs_following_from_operand(use) for use in b.result.uses])
print("get defs of cond.result's uses:")
print([UseDefChainTrait.get_defs_following_from_operand(use) for use in b.result.uses])


print("get operands leading to if_op.output[0]:")
print(UseDefChainTrait.get_operands_leading_to_op_result(if_op.output[0]))
print("Done")