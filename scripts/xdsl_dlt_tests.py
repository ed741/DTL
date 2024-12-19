from dtl.visualise import LayoutPlotter, PtrGraphPlotter
from xdsl.dialects import arith, builtin, func
from xdsl.dialects.builtin import ArrayAttr, FunctionType, IndexType, IntegerType, ModuleOp, f32, i32
from xdsl.dialects.experimental import dlt as dlt
from xdsl.dialects.experimental.dlt import SetAttr
from xdsl.ir import Block, Region
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker
from xdsl.transforms.experimental.dlt.generate_dlt_identities import DLTGenerateIterateOpIdentitiesRewriter, \
    DLTGeneratePtrIdentitiesRewriter, \
    DLTSimplifyPtrIdentitiesRewriter
from xdsl.transforms.experimental.dlt.iteration_map import IterationMap
from xdsl.transforms.experimental.dlt.layout_graph import DLTPtrConsistencyError, LayoutGraph
from xdsl.transforms.experimental.dlt.layout_manipulation import InConsistentLayoutException
from xdsl.utils.exceptions import VerifyException

block = Block(arg_types=[IndexType(), IndexType(), IntegerType(1)])
i, j, b = block.args
layout_part = dlt.AbstractLayoutAttr([
    ([],[dlt.DimensionAttr("k", 3)],dlt.PrimitiveLayoutAttr(f32))
])
layout_a = dlt.AbstractLayoutAttr([
    ([dlt.MemberAttr("r","fs"), dlt.MemberAttr("f","l")],[dlt.DimensionAttr("i", 100),dlt.DimensionAttr("j", 3)], layout_part),
    ([dlt.MemberAttr("r","fs"), dlt.MemberAttr("f","e")],[dlt.DimensionAttr("i", 100)], layout_part),
    ([dlt.MemberAttr("r","fs"), dlt.MemberAttr("f","lk")],[dlt.DimensionAttr("i", 100), dlt.DimensionAttr("m", 2)],dlt.PrimitiveLayoutAttr(i32))])
# layout_a = dlt.AbstractLayoutAttr([
#     ([dlt.MemberAttr("r","fs"), dlt.MemberAttr("f","l")],[dlt.DimensionAttr("i", 100),dlt.DimensionAttr("j", 3),dlt.DimensionAttr("k", 3)], dlt.PrimitiveLayoutAttr(f32)),
#     ([dlt.MemberAttr("r","fs"), dlt.MemberAttr("f","e")],[dlt.DimensionAttr("i", 100),dlt.DimensionAttr("k", 3)], dlt.PrimitiveLayoutAttr(f32)),
#     ([dlt.MemberAttr("r","fs"), dlt.MemberAttr("f","lk")],[dlt.DimensionAttr("i", 100), dlt.DimensionAttr("m", 2)],dlt.PrimitiveLayoutAttr(i32))])
ptr_a = dlt.PtrType(layout_a.contents_type, layout_a, SetAttr([]), ArrayAttr([]), ArrayAttr([]), base=True, identity="p1")
alloc_op = dlt.AllocOp(ptr_a, {}, [])
block.add_op(alloc_op)
sel_l_op = dlt.SelectOp(alloc_op.res, [dlt.MemberAttr("r","fs"), dlt.MemberAttr("f","l")], [], [])
block.add_op(sel_l_op)
sel_ij_op = dlt.SelectOp(sel_l_op.res, [], [dlt.DimensionAttr("i", 100), dlt.DimensionAttr("j", 3)], [i,j])
block.add_op(sel_ij_op)

sel_e_op = dlt.SelectOp(alloc_op.res, [dlt.MemberAttr("r","fs"), dlt.MemberAttr("f","e")], [dlt.DimensionAttr("i", 100)], [i])
block.add_op(sel_e_op)

arith_sel_op = arith.Select(b, sel_ij_op.res, sel_e_op.res)
block.add_op(arith_sel_op)

block.add_op(func.Return(arith_sel_op.result))

funcType = FunctionType.from_lists([a.type for a in block.args], [arith_sel_op.result.type])
func = func.FuncOp("foo", funcType, Region([block]))

scope_op = dlt.LayoutScopeOp([], [func])
module = ModuleOp([scope_op])
print(module)

try:
    module.verify()
except Exception as e:
    print(e)

first_identity_gen = DLTGeneratePtrIdentitiesRewriter()
dlt_ident_applier = PatternRewriteWalker(
    first_identity_gen, walk_regions_first=False
)
dlt_ident_applier.rewrite_module(module)

first_layout_graph = first_identity_gen.layouts.get(scope_op, LayoutGraph({}, [], []))
PtrGraphPlotter.plot_graph(first_layout_graph, ptr_types=None, marked_idents=None,
                           name="ptr_graph_first", view=True)
LayoutPlotter.plot_layout(first_layout_graph.get_type_map(), name="layouts_first", entry_points=set(first_layout_graph.get_entry_layouts()), view=True)


simplify_graph = DLTSimplifyPtrIdentitiesRewriter(
    first_identity_gen.layouts, first_identity_gen.initial_identities
)
dlt_simplify_applier = PatternRewriteWalker(
    simplify_graph, walk_regions_first=False
)
dlt_simplify_applier.rewrite_module(module)

layout_graph_generator = DLTGeneratePtrIdentitiesRewriter()
iteration_map_generator = DLTGenerateIterateOpIdentitiesRewriter()
dlt_graph_gen_applier = PatternRewriteWalker(
    GreedyRewritePatternApplier(
        [layout_graph_generator, iteration_map_generator]
    ),
    walk_regions_first=False,
)
dlt_graph_gen_applier.rewrite_module(module)

layout_graph = layout_graph_generator.layouts.get(scope_op, LayoutGraph({}, [], []))
iteration_map = iteration_map_generator.iteration_maps.get(scope_op, IterationMap({}))

print(f" layout graph consistency: {layout_graph.is_consistent()}")
print(f" iter map consistency: {iteration_map.is_consistent()}")

PtrGraphPlotter.plot_graph(layout_graph, ptr_types=None, marked_idents=None,
                           name="ptr_graph_pre", view=True)
LayoutPlotter.plot_layout(layout_graph.get_type_map(), name="layouts_pre", entry_points=set(layout_graph.get_entry_layouts()), view=True)

psi = layout_graph.get_type_map()
keys = set(psi.keys())
for key in keys:
    try:
        print(f"propagate: {key.data} ::", end="")
        new_psi = dict(psi)
        layout_graph.propagate_type(key, new_psi, set())
        layout_graph.backpropagate_type(key, new_psi, set())
        psi = new_psi

        # PtrGraphPlotter.plot_graph(layout_graph, ptr_types=psi, marked_idents=None,
        #                            name="ptr_graph_post", view=True)
        # LayoutPlotter.plot_layout(psi, name="layouts_post", entry_points=set(layout_graph.get_entry_layouts()),
        #                           view=True)
        print("success")
    except VerifyException as e:
        print("failed - VerifyException")
        pass
    except DLTPtrConsistencyError as e:
        print("failed - DLTPtrConsistencyError")
        pass
    except InConsistentLayoutException as e:
        print("failed - InConsistentLayoutException")
        pass


PtrGraphPlotter.plot_graph(layout_graph, ptr_types=psi, marked_idents=None,
                           name="ptr_graph_post", view=True)
LayoutPlotter.plot_layout(psi, name="layouts_post", entry_points=set(layout_graph.get_entry_layouts()), view=True)

print(f" layout graph consistency: {layout_graph.is_consistent(psi)}")
print(f" iter map consistency: {iteration_map.is_consistent()}")

rewriter = layout_graph.use_types(psi)
rewriter.rewrite_op(module)
module.verify()

print(module)