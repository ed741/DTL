import ctypes
import functools
import tempfile
import typing
from ctypes import cdll
from dataclasses import dataclass

from dtl import VectorSpaceVariable, UnknownSizeVectorSpace, VectorSpace, TensorVariable, Expr, Index
from dtlpp.backends import xdsl as xdtl
from xdsl import ir
from xdsl.dialects import builtin, arith, printf, llvm, func
from xdsl.dialects.builtin import f32, IndexType, ModuleOp, i64
from xdsl.dialects.experimental import dlt
from xdsl.dialects.func import Return, FuncOp
from xdsl.ir import Region, MLContext, TypeAttribute
from xdsl.pattern_rewriter import PatternRewriteWalker, GreedyRewritePatternApplier
from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations
from xdsl.transforms.experimental.dlt import lower_dlt_to_
from xdsl.transforms.experimental.convert_return_to_pointer_arg import PassByPointerRewriter
from xdsl.transforms.experimental.generate_dlt_layouts import DLTLayoutRewriter, _make_dense_layouts, _try_apply_sparse
from xdsl.transforms.experimental.generate_dlt_ptr_identities import DLTGeneratePtrIdentitiesRewriter, \
    DLTSimplifyPtrIdentitiesRewriter
from xdsl.transforms.experimental.lower_dtl_to_dlt import DTLRewriter
from xdsl.transforms.printf_to_llvm import PrintfToLLVM
from xdsl.transforms.printf_to_putchar import PrintfToPutcharPass
from xdsl.transforms.reconcile_unrealized_casts import reconcile_unrealized_casts
from xdslDTL import compilec

_T = typing.TypeVar("_T")
TupleStruct: typing.TypeAlias = typing.Union[tuple["TupleStruct[_T]", ...], _T]

def _flatten(ts: TupleStruct[_T]) -> list[_T]:
    if isinstance(ts, tuple):
        return [e for es in ts for e in _flatten(es)]
    else:
        return [ts]

@dataclass
class FuncTypeDescriptor:
    params: list
    return_types: list
    structure: TupleStruct


class LibBuilder:
    def __init__(self):
        self.vs_namer = {}
        self.vs_name_idx = 0
        self.tensor_var_details: dict[TupleStruct[TensorVariable], dlt.PtrType] = {}
        self.tensor_var_elems: dict[TupleStruct[TensorVariable], TupleStruct[dlt.ElementAttr]] = {}
        # self.tensor_var_ptrs: dict[TupleStruct[TensorVariable], dlt.PtrType] = {}
        self.tensor_var_dims: dict[TensorVariable, list[dlt.DimensionAttr]] = {}
        self.tensor_var_namer_idx = 0

        self.funcs = []
        self.func_map: dict[str, FuncTypeDescriptor] = {}

    def _get_vs_name(self, vs: VectorSpaceVariable, new: bool=False):
        if new:
            name = "dim_"+ str(self.vs_name_idx)
            self.vs_name_idx += 1
            return name
        if vs not in self.vs_namer:
            raise ValueError()
            name = "dim_" + str(self.vs_name_idx)
            self.vs_name_idx += 1
            self.vs_namer[vs] = name
        return self.vs_namer[vs]

    def _get_vs_extent(self, vs: VectorSpaceVariable):
        if isinstance(vs, UnknownSizeVectorSpace):
            return dlt.InitDefinedExtentAttr(vs.name)
        if isinstance(vs, VectorSpace):
            return dlt.StaticExtentAttr(vs.dim)

    def _add_member_in_tupleStruct_layer_elements(self, elements: TupleStruct[dlt.ElementAttr], add: dlt.MemberAttr = None) -> TupleStruct[dlt.ElementAttr]:
        if isinstance(elements, tuple):
            return tuple([self._add_member_in_tupleStruct_layer_elements(e) for e in elements])
        elif isinstance(elements, dlt.ElementAttr):
            element = typing.cast(dlt.ElementAttr, elements)
            members = list(element.member_specifiers)
            if add is not None:
                members.append(add)
            return dlt.ElementAttr(members, element.dimensions, element.base_type)

    # def _increment_tupleStruct_layer_members(self, members: list[dlt.MemberAttr]) -> list[dlt.MemberAttr]:
    #     new_members = []
    #     for member in members:
    #         assert isinstance(member, dlt.MemberAttr)
    #         member = typing.cast(dlt.MemberAttr, member)
    #         if member.structName.data.startswith("T_") and member.structName.data[2:].isdecimal():
    #             layer = int(member.structName.data[2:])
    #             new_member = dlt.MemberAttr(f"T_{layer}", member.memberName)
    #             new_members.append(new_member)
    #         else:
    #             raise ValueError()
    #             # new_members.append(member)
    #     return new_members

    def _set_dlt_type_elems_for_tensor_var(self, tensor_vars: TupleStruct[TensorVariable], level=0) -> TupleStruct[dlt.ElementAttr]:
        assert tensor_vars not in self.tensor_var_elems
        if isinstance(tensor_vars, tuple):
            elements: TupleStruct[dlt.ElementAttr] = tuple()
            for i, tensor_var in enumerate(tensor_vars):
                child_elements = self._set_dlt_type_elems_for_tensor_var(tensor_var, level=level + 1)
                new_member = dlt.MemberAttr(f"T_{level}", f"{i}")
                child_elements = self._add_member_in_tupleStruct_layer_elements(child_elements, add=new_member)
                elements = tuple([*elements, child_elements])
            self.tensor_var_elems[tensor_vars] = elements
            return elements
        else:
            tensor_var = typing.cast(TensorVariable, tensor_vars)
            vector_space_vars = tensor_var.tensor_space.spaces
            vector_space_names = [self._get_vs_name(vs, new=True) for vs in vector_space_vars]
            vector_space_extents = [self._get_vs_extent(vs) for vs in vector_space_vars]
            dimensions = [dlt.DimensionAttr(vs_name, vs_extent) for vs_name, vs_extent in
                          zip(vector_space_names, vector_space_extents)]
            element = dlt.ElementAttr([], dimensions, builtin.f32)
            self.tensor_var_elems[tensor_var] = element
            self.tensor_var_dims[tensor_var] = dimensions
            return element

    def _set_dlt_type_for_tensor_var(self, tensor_vars: TupleStruct[TensorVariable], ptr_type: dlt.PtrType = None, members: set[dlt.MemberAttr] = None, level=0):
        assert tensor_vars not in self.tensor_var_details
        if members is None:
            members = set()
        if isinstance(tensor_vars, tuple):
            elements = _flatten(self.tensor_var_elems[tensor_vars])
            if ptr_type is None:

                dlt_TypeType = dlt.TypeType(elements)
                all_dims = [dim for elem in elements for dim in elem.dimensions]
                init_extents = list({base_extent for dim in all_dims for base_extent in dim.extent.base_extents() if
                                     isinstance(base_extent, dlt.InitDefinedExtentAttr)})
                ptr_id = str(self.tensor_var_namer_idx)
                self.tensor_var_namer_idx += 1
                ptr_type = dlt.PtrType(dlt_TypeType, members=dlt.SetAttr(members), extents=init_extents, identity=ptr_id)
            else:
                ptr_id = str(self.tensor_var_namer_idx)
                self.tensor_var_namer_idx += 1
                ptr_type = dlt.PtrType(ptr_type.contents_type.select_members(members), ptr_type.layout,
                                       ptr_type.filled_members.add(members),
                                       ptr_type.filled_dimensions,
                                       ptr_type.filled_extents,
                                       ptr_type.is_base,
                                       ptr_id)
            self.tensor_var_details[tensor_vars] = ptr_type
            for i, child in enumerate(tensor_vars):
                self._set_dlt_type_for_tensor_var(child, ptr_type,
                                                  members.union({dlt.MemberAttr(f"T_{level}", f"{i}")}),
                                                  level=level + 1)
        else:
            tensor_var = typing.cast(TensorVariable, tensor_vars)
            element = self.tensor_var_elems[tensor_var]
            if ptr_type is None:
                dlt_TypeType = dlt.TypeType([element])
                all_dims = [dim for dim in element.dimensions]
                init_extents = list({base_extent for dim in all_dims for base_extent in dim.extent.base_extents() if
                                     isinstance(base_extent, dlt.InitDefinedExtentAttr)})
                ptr_id = str(self.tensor_var_namer_idx)
                self.tensor_var_namer_idx += 1
                ptr_type = dlt.PtrType(dlt_TypeType, members=dlt.SetAttr(members), extents=init_extents, identity=ptr_id)
            else:
                ptr_id = str(self.tensor_var_namer_idx)
                self.tensor_var_namer_idx += 1
                ptr_type = dlt.PtrType(ptr_type.contents_type.select_members(members), ptr_type.layout,
                                       ptr_type.filled_members.add(members),
                                       ptr_type.filled_dimensions,
                                       ptr_type.filled_extents,
                                       ptr_type.is_base,
                                       ptr_id)
            self.tensor_var_details[tensor_var] = ptr_type

    def _get_tensor_var_info(self, tensor_vars: TupleStruct[TensorVariable]) -> tuple[dlt.PtrType, TupleStruct[dlt.ElementAttr]]:
        if tensor_vars not in self.tensor_var_details:
            self._set_dlt_type_elems_for_tensor_var(tensor_vars)
            self._set_dlt_type_for_tensor_var(tensor_vars)
        return self.tensor_var_details[tensor_vars], self.tensor_var_elems[tensor_vars]

    def _get_select_ops(self, ssa_in: ir.SSAValue, elements: TupleStruct[dlt.ElementAttr], tensor_vars: TupleStruct[TensorVariable]) -> tuple[list[ir.Operation], list[ir.SSAValue]]:
        if isinstance(elements, tuple):
            ops = []
            results = []
            for element, tensor_var in zip(elements, tensor_vars):
                child_ops, child_results = self._get_select_ops(ssa_in, element, tensor_var)
                ops.extend(child_ops)
                results.extend(child_results)
            return ops, results
        elif isinstance(elements, dlt.ElementAttr):
            element = typing.cast(dlt.ElementAttr, elements)
            assert isinstance(tensor_vars, TensorVariable)
            tensor_var = typing.cast(TensorVariable, tensor_vars)
            result_type = self.tensor_var_details[tensor_var]
            return [op := dlt.SelectOp(ssa_in, element.member_specifiers, [], [], result_type=result_type)], [op.res]

    def make_init(self,
                  name: str,
                  tensor_vars: TupleStruct[TensorVariable],
                  extents: list[UnknownSizeVectorSpace],
                  ):
        ptr_type, elements = self._get_tensor_var_info(tensor_vars)
        base_ptr_type = ptr_type.as_base()
        base_ptr_type = ptr_type.as_base().with_identification("R_"+base_ptr_type.identification.data)
        block = ir.Block()
        arg_idx = 0
        extents_map = {}
        for e in extents:
            arg = block.insert_arg(IndexType(), arg_idx)
            debug = [c := builtin.UnrealizedConversionCastOp.get([arg], [i64]), trunc_op := arith.TruncIOp(c.outputs[0], builtin.i32), printf.PrintIntOp(trunc_op.result)] + printf.PrintCharOp.from_constant_char_ops("\n")
            block.add_ops(debug)
            arg_idx += 1
            if e is not None:
                assert isinstance(e, UnknownSizeVectorSpace)
                extents_map[self._get_vs_extent(e)] = arg

        alloc_op = dlt.AllocOp(base_ptr_type, extents_map)
        block.add_op(alloc_op)
        select_elem_ops, select_elem_ssa = self._get_select_ops(alloc_op.res, elements, tensor_vars)
        block.add_ops(select_elem_ops)
        ret = Return(alloc_op.res, *select_elem_ssa)
        block.add_op(ret)

        region = Region([block])
        argTypes = [arg.type for arg in block.args]
        retTypes = [r.type for r in ret.operands]
        func = FuncOp.from_region(name, argTypes, retTypes, region)
        self.funcs.append(func)
        func_type = func.function_type
        self.func_map[name] = FuncTypeDescriptor(list(func_type.inputs), list(func_type.outputs), (1, tensor_vars))
        return func

    def make_dummy(self, name, ret_val):
        block = ir.Block()
        block.add_op(const := arith.Constant(builtin.IntegerAttr(ret_val, IndexType())))
        ret = Return(const.result)
        block.add_op(ret)

        region = Region([block])
        argTypes = [arg.type for arg in block.args]
        retTypes = [r.type for r in ret.operands]
        func = FuncOp.from_region(name, argTypes, retTypes, region)
        self.funcs.append(func)
        func_type = func.function_type
        self.func_map[name] = FuncTypeDescriptor(list(func_type.inputs), list(func_type.outputs), ret_val)
        return func

    def make_setter(self, name, tensor_var: TensorVariable, members: typing.Iterable[dlt.MemberAttr], args: list[int]):
        block = ir.Block()
        ptr_type, element = self._get_tensor_var_info(tensor_var)
        dims = self.tensor_var_dims[tensor_var]

        dlt_tensor = block.insert_arg(ptr_type, 0)

        assert len(args) == len(tensor_var.tensor_space.spaces)
        index_args = []
        for i in range(max(args) + 1):
            index_args.append(block.insert_arg(IndexType(), len(block.args)))

        dim_values = []
        for arg_idx in args:
            dim_values.append(index_args[arg_idx])

        select = dlt.SelectOp(dlt_tensor, members, dims, dim_values)
        block.add_op(select)
        value_arg = block.insert_arg(element.base_type, len(block.args))
        store = dlt.SetOp(select.res, element.base_type, value_arg)
        block.add_op(store)
        ret = Return()
        block.add_op(ret)

        region = Region([block])
        argTypes = [arg.type for arg in block.args]
        retTypes = [r.type for r in ret.operands]
        func = FuncOp.from_region(name, argTypes, retTypes, region)
        self.funcs.append(func)
        func_type = func.function_type
        self.func_map[name] = FuncTypeDescriptor(list(func_type.inputs), list(func_type.outputs), None)
        return func

    def make_print_tensorVar(self, name: str, tensor_var, extents: list[UnknownSizeVectorSpace]):
        block = ir.Block()
        ptr_type, elements = self._get_tensor_var_info(tensor_var)
        dims = self.tensor_var_dims[tensor_var]
        tensor_var_arg = block.insert_arg(ptr_type, 0)
        arg_idx = 1
        dynamic_extents = {}
        for extent in extents:
            arg = block.insert_arg(IndexType(), arg_idx)
            arg_idx += 1
            dynamic_extents[self._get_vs_extent(extent)] = arg
        extents = [dim.extent for dim in dims]
        extent_args = []
        for e in extents:
            if e.is_dynamic():
                extent_args.append(dynamic_extents[e])
            if e.is_init_time():
                block.add_op(ex := dlt.ExtractExtentOp(tensor_var_arg, e))
                extent_args.append(ex.res)
        block.add_op(iter_op := dlt.IterateOp(extents, extent_args, [[[d] for d in dims]], [tensor_var_arg], [], builtin.StringAttr("nested")))
        inner_idxs = iter_op.body.block.args[0:len(extents)]
        inner_ptr = iter_op.get_block_arg_for_tensor_arg_idx(0)
        ret = iter_op.body.block.last_op
        iter_op.body.block.insert_op_before(get_op := dlt.GetOp(inner_ptr, get_type=builtin.f32), ret)
        iter_op.body.block.insert_op_before(printf.PrintFormatOp(f"{name}:: "+(",".join([f"{dim.dimensionName} ({dim.extent}): {{}}" for dim in dims])) + " -> {}", *inner_idxs, get_op.res), ret)
        block.add_op(Return())

        region = Region([block])
        argTypes = [arg.type for arg in block.args]
        retTypes = []
        func = FuncOp.from_region(name, argTypes, retTypes, region)
        self.funcs.append(func)
        func_type = func.function_type
        self.func_map[name] = FuncTypeDescriptor(list(func_type.inputs), list(func_type.outputs), None)
        return func

    def make_function(self,
                      name: str,
                      expr: Expr,
                      results: list[TensorVariable],
                      tensor_vars: list[TensorVariable],
                      extents: list[UnknownSizeVectorSpace],
                      index_args: list[Index],
                      # extent_map: dict[UnknownSizeVectorSpace, int] = {},
                      ):
        block = ir.Block()
        # space_map = {}
        # for vs, length in extent_map.items():
        #     block.add_op(const_op := arith.Constant(IntegerAttr(length, IndexType())))
        #     space_map[vs] = const_op.result
        arg_idx = 0
        outputs = []
        for result in results:
            ptr_type, elements = self._get_tensor_var_info(result)
            dims = self.tensor_var_dims[result]
            arg = block.insert_arg(ptr_type, arg_idx)
            arg_idx += 1
            dim_names = [dim.dimensionName.data for dim in dims]
            outputs.append((arg, dim_names))
        tensor_variables = {}
        for tensor_var in tensor_vars:
            ptr_type, elements = self._get_tensor_var_info(tensor_var)
            dims = self.tensor_var_dims[tensor_var]
            arg = block.insert_arg(ptr_type, arg_idx)
            arg_idx += 1
            dim_names = [dim.dimensionName.data for dim in dims]
            tensor_variables[tensor_var] = (arg, dim_names)
        dynamic_extents = {}
        for extent in extents:
            arg = block.insert_arg(IndexType(), arg_idx)
            arg_idx += 1
            dynamic_extents[extent] = arg
        arg_map = {}
        for index_arg in index_args:
            arg = block.insert_arg(IndexType(), arg_idx)
            arg_idx += 1
            arg_map[index_arg] = arg

        exec, output = xdtl.get_xdsl_dtl_exec_version(expr, space_map=dynamic_extents,
                                                      arg_map=arg_map,
                                                      tensor_variables=tensor_variables,
                                                      outputs=outputs)
        block.add_ops(exec)
        block.add_op(output)
        block.add_op(Return())

        region = Region([block])
        argTypes = [arg.type for arg in block.args]
        retTypes = []
        func = FuncOp.from_region(name, argTypes, retTypes, region)
        self.funcs.append(func)
        func_type = func.function_type
        self.func_map[name] = FuncTypeDescriptor(list(func_type.inputs), list(func_type.outputs), tuple(results))
        return func

    def build(self):
        malloc_func = llvm.FuncOp("malloc", llvm.LLVMFunctionType([builtin.i64], llvm.LLVMPointerType.opaque()),
                                  linkage=llvm.LinkageAttr("external"))
        free_func = llvm.FuncOp("free", llvm.LLVMFunctionType([llvm.LLVMPointerType.opaque()], None),
                                  linkage=llvm.LinkageAttr("external"))
        memcpy_func = llvm.FuncOp("memcpy", llvm.LLVMFunctionType([llvm.LLVMPointerType.opaque(), llvm.LLVMPointerType.opaque(), builtin.i64], None),
                                  linkage=llvm.LinkageAttr("external"))
        abort_func = llvm.FuncOp("abort", llvm.LLVMFunctionType([]),
                                 linkage=llvm.LinkageAttr("external"))

        scope_op = dlt.LayoutScopeOp([], self.funcs)
        module = ModuleOp([malloc_func, free_func, memcpy_func, abort_func, scope_op])
        module.verify()

        print(module)

        # DTL -> DLT
        dtl_to_dlt_applier = PatternRewriteWalker(DTLRewriter(),
                                                  walk_regions_first=False)
        dtl_to_dlt_applier.rewrite_module(module)
        module.verify()

        first_identity_gen = DLTGeneratePtrIdentitiesRewriter()
        dlt_ident_applier = PatternRewriteWalker(first_identity_gen,
                                                  walk_regions_first=False)
        dlt_ident_applier.rewrite_module(module)
        simplify_graph = DLTSimplifyPtrIdentitiesRewriter(first_identity_gen.layouts, first_identity_gen.initial_identities)
        dlt_simplify_applier = PatternRewriteWalker(simplify_graph,
                                                  walk_regions_first=False)
        dlt_simplify_applier.rewrite_module(module)

        layout_graph_generator = DLTGeneratePtrIdentitiesRewriter()
        dlt_graph_gen_applier = PatternRewriteWalker(layout_graph_generator,
                                                 walk_regions_first=False)
        dlt_graph_gen_applier.rewrite_module(module)

        module.verify()

        print(module)

        layout_graph = layout_graph_generator.layouts[scope_op]
        layout_graph.consistent_check()
        check = layout_graph.is_consistent()
        if not check:
            raise ValueError("Layout Graph has been check and found to be inconsistent")
        original_type_map = layout_graph.get_type_map()
        new_type_map = original_type_map.copy()


        while any(ptr.layout.is_abstract for ptr in new_type_map.values()):
            old_type_map = new_type_map.copy()
            idents = {ident for ident in layout_graph.get_entry_layouts().keys() if new_type_map[ident].layout.is_abstract}
            if len(idents) == 0:
                idents = {ident for ident in new_type_map.keys() if
                          new_type_map[ident].layout.is_abstract}
            if len(idents) == 0:
                raise ValueError("some layout is abstract but cannot find an abstract layout")
            ident = idents.pop()
            print(f"Reifying {ident.data}")
            ptr = new_type_map.pop(ident)
            new_layout = ptr.layout
            new_layout = _try_apply_sparse(new_layout)
            new_layout = _make_dense_layouts(new_layout, {})
            # new_layout = dlt.StructLayoutAttr([new_layout])
            new_ptr = ptr.with_new_layout(new_layout, preserve_ident=True)
            new_type_map[ident] = new_ptr
            layout_graph.propagate_type(ident, new_type_map)
            changed = {i for (i, ptr) in new_type_map.items() if old_type_map[i] != ptr}
            print("Propagated changes to: " + ",".join([i.data for i in changed]))

        # idents, ptr_types = zip(*type_map.items())
        # layouts = [p.layout for p in ptr_types]
        # name_map = {}
        # new_layouts = _make_dense_layouts(layouts, name_map)
        # new_ptr_types = [p.with_new_layout(l, preserve_ident=True, remove_bloat=True) for p, l in zip(ptr_types, new_layouts)]
        # new_type_map = {i:p for i, p in zip(idents, new_ptr_types)}
        check = layout_graph.is_consistent(new_type_map)
        if not check:
            raise ValueError("Layout Graph has been check and found to be inconsistent")

        print("making type updating pass")
        rewriter = layout_graph.use_types(new_type_map)
        print("rewriting types")
        rewriter.rewrite_op(scope_op)
        print("verifying module")
        module.verify()

        print(module)

        # # generate-layouts-> DLT
        # dlt_to_dlt_applier = PatternRewriteWalker(DLTLayoutRewriter(),
        #                                           walk_regions_first=False)
        # dlt_to_dlt_applier.rewrite_module(module)
        # module.verify()
        #
        # identifiers = {}
        # for op in module.walk():
        #     for ptr in ([typing.cast(dlt.PtrType, r.type) for r in op.results if isinstance(r.type, dlt.PtrType)] +
        #                 [typing.cast(dlt.PtrType, arg.type) for region in op.regions for block in region.blocks for arg in block.args if isinstance(arg.type, dlt.PtrType)]):
        #         if ptr.has_identity:
        #             if ptr.identification in identifiers:
        #
        #                 if identifiers[ptr.identification] != ptr:
        #                     print(op)
        #                     print(ptr)
        #                     print(identifiers[ptr.identification])
        #             else:
        #                 identifiers[ptr.identification] = ptr
        #         else:
        #             print(op)



        print(module)

        # DLT -> MLIR builtin
        dlt_to_llvm_applier = PatternRewriteWalker(GreedyRewritePatternApplier(
            [RemoveUnusedOperations(),
             lower_dlt_to_.DLTSelectRewriter(),
             lower_dlt_to_.DLTGetRewriter(),
             lower_dlt_to_.DLTSetRewriter(),
             lower_dlt_to_.DLTAllocRewriter(),
             lower_dlt_to_.DLTIterateRewriter(),
             lower_dlt_to_.DLTCopyRewriter(),
             lower_dlt_to_.DLTExtractExtentRewriter(),
             ]),
            walk_regions_first=False)

        print(module)
        module.verify()

        dlt_to_llvm_applier.rewrite_module(module)

        print(module)
        module.verify()

        rem_scope = PatternRewriteWalker(GreedyRewritePatternApplier(
            [lower_dlt_to_.DLTScopeRewriter(),
             lower_dlt_to_.DLTPtrTypeRewriter(recursive=True),
             lower_dlt_to_.DLTIndexTypeRewriter(recursive=True),
             lower_dlt_to_.DLTIndexRangeTypeRewriter(recursive=True),
             ])
        )
        rem_scope.rewrite_module(module)
        PrintfToPutcharPass().apply(MLContext(True), module)
        PrintfToLLVM().apply(MLContext(True), module)

        function_types = {}
        func_mod_applier = PatternRewriteWalker(GreedyRewritePatternApplier([PassByPointerRewriter(
            [func.sym_name for func in self.funcs], originals=function_types
        )]))
        func_mod_applier.rewrite_module(module)

        reconcile_unrealized_casts(module)
        module.verify()

        print(module)

        lib_fd, lib_name = tempfile.mkstemp(suffix=".so")
        print(lib_name)
        compilec.compile(module, lib_name)
        function_types = {name.data:v for name,v in function_types.items()}

        return DTLCLib(lib_name, self.func_map, function_types)

class FunctionCaller:
    def __init__(self, func, param_types, ret_types, return_structure = None):
        self.func = func
        self.param_types = param_types
        self.ret_types = ret_types
        if return_structure is None:
            return_structure = tuple(ret_types)
        self.return_structure: TupleStruct = return_structure

    def __call__(self, *args, **kwargs):
        rets = []
        func_args = []
        for ret_type in self.ret_types:
            rets.append(ret_type())
            func_args.append(ctypes.pointer(rets[-1]))
        for param_type, arg in zip(self.param_types, args):
            if not isinstance(arg, param_type):
                print(type(arg))
                print(param_type)
                print(type(arg) == param_type)
                arg = param_type(arg) # try to make init the right type?
            assert isinstance(arg, param_type)
            if isinstance(arg, StructType):
                arg = ctypes.pointer(arg)
            func_args.append(arg)
        self.func(*func_args)

        return_structure = self.tupleStruct_from(rets, self.return_structure)
        assert len(rets) == 0
        if isinstance(return_structure, tuple) and len(return_structure) == 0:
            return
        return return_structure

    def tupleStruct_from(self, rets: list, structure: TupleStruct) -> TupleStruct:
        if isinstance(structure, tuple):
            return tuple([self.tupleStruct_from(rets, s) for s in structure])
        else:
            return rets.pop(0)


class DTLCLib:
    def __init__(self, library_path, funcs, function_types: dict[str, builtin.FunctionType]):
        self._library_path = library_path
        self._funcs: dict[str, FuncTypeDescriptor] = funcs # describe funcs as original (with return vals) with dlt types
        self._callers: dict[str, FunctionCaller] = {}
        self._lib = cdll.LoadLibrary(self._library_path)
        self._func_types = function_types # describe funcs as original (with return vals) with llvm types
        self._ctype_classes: dict = {}


    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        if name.startswith('_'):
            raise AttributeError(name)
        func = self.__getitem__(name)
        setattr(self, name, func)
        return func

    def __getitem__(self, name) -> FunctionCaller:
        if name not in self._funcs:
            raise KeyError(name)
        if name in self._callers:
            return self._callers[name]
        func_descriptor = self._funcs[name]
        func_type = self._func_types[name]
        assert len(func_type.inputs) == len(func_descriptor.params)
        assert len(func_type.outputs) == len(func_descriptor.return_types)
        ret_types = list(func_type.outputs.data)
        param_types = list(func_type.inputs.data)
        ctype_ret_types = [self.convert_llvm_type_to_ctypes(t, dlt_type=dlt_t) for t, dlt_t in zip(ret_types, func_descriptor.return_types)]
        ctype_ret_pointer_types = [ctypes.POINTER(t) for t in ctype_ret_types]
        ctype_param_types = [self.convert_llvm_type_to_ctypes(t, dlt_type=dlt_t) for t, dlt_t in zip(param_types, func_descriptor.params)]
        self._lib[name].argtypes = ctype_ret_pointer_types + ctype_param_types
        caller = FunctionCaller(self._lib[name], ctype_param_types, ctype_ret_types)
        self._callers[name] = caller
        return caller

    def convert_llvm_type_to_ctypes(self, llvm_type: TypeAttribute, dlt_type: TypeAttribute = None):
        if dlt_type is not None:
            if dlt_type not in self._ctype_classes:
                type = self._convert_llvm_type_to_ctypes(llvm_type, dlt_type)
                self._ctype_classes[dlt_type] = type
            return self._ctype_classes[dlt_type]
        else:
            return self._convert_llvm_type_to_ctypes(llvm_type)

    @functools.singledispatchmethod
    def _convert_llvm_type_to_ctypes(self, llvm_type: TypeAttribute, dlt_type: TypeAttribute = None):
        raise NotImplementedError(f"Cannot convert type: {llvm_type} to ctypes")

    @_convert_llvm_type_to_ctypes.register
    def _(self, llvm_type: builtin.IntegerType, dlt_type: TypeAttribute = None):
        if llvm_type == builtin.i64:
            return ctypes.c_uint64
        else:
            raise NotImplementedError(f"Cannot convert Integer type: {llvm_type} to ctypes")

    @_convert_llvm_type_to_ctypes.register
    def _(self, llvm_type: llvm.LLVMPointerType, dlt_type: TypeAttribute = None):
        return ctypes.c_void_p

    @_convert_llvm_type_to_ctypes.register
    def _(self, llvm_type: llvm.LLVMStructType, dlt_type: TypeAttribute = None):
        member_types = [self.convert_llvm_type_to_ctypes(member) for member in llvm_type.types]
        if dlt_type is not None:
            if isinstance(dlt_type, dlt.PtrType):
                dlt_type = typing.cast(dlt.PtrType, dlt_type)
                field_names = ["ptr"]
                for e in dlt_type.filled_extents:
                    field_names.append(e.get_id().data)
                for d in dlt_type.filled_dimensions:
                    field_names.append(d.dimensionName.data)
                assert len(field_names) == len(member_types)
                # return StructType(list(zip(field_names, member_types)))
                class PtrStructType(StructType):
                    _fields_ = list(zip(field_names, member_types))
                return PtrStructType
        # return StructType([(str(i), member) for i, member in enumerate(member_types)])
        class OtherStructType(StructType):
            _fields_ = [(str(i), member) for i, member in enumerate(member_types)]
        return OtherStructType

class StructType(ctypes.Structure):
    pass



'''
(!dlt.ptr<
    !dlt.type<
        ({T_0:0},dim_0:#dlt.InitDefinedExtent<"Q">->dim_1:#dlt.StaticExtent<10 : index>->f32),
        ({T_0:1},dim_2:#dlt.StaticExtent<10 : index>->dim_3:#dlt.InitDefinedExtent<"S">->f32)
    >,
    #dlt.layout.struct<[
        #dlt.layout.member<
            #dlt.layout.dense<
                #dlt.layout.dense<
                    #dlt.layout.primitive<f32>,
                #dlt.dimension<dim_3:#dlt.InitDefinedExtent<"S">>
                >,
            #dlt.dimension<dim_2:#dlt.StaticExtent<10 : index>>
            >,
        #dlt.member<T_0:1>
        >,
        #dlt.layout.member<
            #dlt.layout.dense<
                #dlt.layout.dense<
                    #dlt.layout.primitive<f32>,
                #dlt.dimension<dim_1:#dlt.StaticExtent<10 : index>>
                >,
            #dlt.dimension<dim_0:#dlt.InitDefinedExtent<"Q">>
            >,
        #dlt.member<T_0:0>>
    ]>,
    #dlt.set{},
    [],
    [#dlt.InitDefinedExtent<"S">, #dlt.InitDefinedExtent<"Q">],
    "Y",
    "">,
!dlt.ptr<
    !dlt.type<
        ({},dim_0:#dlt.InitDefinedExtent<"Q">->dim_1:#dlt.StaticExtent<10 : index>->f32)
    >,
    #dlt.layout.struct<
        [#dlt.layout.member<#dlt.layout.dense<#dlt.layout.dense<#dlt.layout.primitive<f32>, #dlt.dimension<dim_3:#dlt.InitDefinedExtent<"S">>>, #dlt.dimension<dim_2:#dlt.StaticExtent<10 : index>>>, #dlt.member<T_0:1>>, #dlt.layout.member<#dlt.layout.dense<#dlt.layout.dense<#dlt.layout.primitive<f32>, #dlt.dimension<dim_1:#dlt.StaticExtent<10 : index>>>, #dlt.dimension<dim_0:#dlt.InitDefinedExtent<"Q">>>, #dlt.member<T_0:0>>]>, #dlt.set{#dlt.member<T_0:0>}, [], [#dlt.InitDefinedExtent<"S">, #dlt.InitDefinedExtent<"Q">], "Y", "">, !dlt.ptr<!dlt.type<({},dim_2:#dlt.StaticExtent<10 : index>->dim_3:#dlt.InitDefinedExtent<"S">->f32)>, #dlt.layout.struct<[#dlt.layout.member<#dlt.layout.dense<#dlt.layout.dense<#dlt.layout.primitive<f32>, #dlt.dimension<dim_3:#dlt.InitDefinedExtent<"S">>>, #dlt.dimension<dim_2:#dlt.StaticExtent<10 : index>>>, #dlt.member<T_0:1>>, #dlt.layout.member<#dlt.layout.dense<#dlt.layout.dense<#dlt.layout.primitive<f32>, #dlt.dimension<dim_1:#dlt.StaticExtent<10 : index>>>, #dlt.dimension<dim_0:#dlt.InitDefinedExtent<"Q">>>, #dlt.member<T_0:0>>]>, #dlt.set{#dlt.member<T_0:1>}, [], [#dlt.InitDefinedExtent<"S">, #dlt.InitDefinedExtent<"Q">], "Y", "">) {
      

'''