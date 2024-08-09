import ctypes
import functools
import os
import tempfile
import typing
from ctypes import cdll
from dataclasses import dataclass

import dtl
from dtl import (
    VectorSpaceVariable,
    UnknownSizeVectorSpace,
    VectorSpace,
    TensorVariable,
    Expr,
    Index,
)
from dtlpp.backends import xdsl as xdtl
from xdsl import ir
from xdsl.dialects import builtin, arith, printf, llvm, func
from xdsl.dialects.builtin import IndexType, ModuleOp, i64
from xdsl.dialects.experimental import dlt
from xdsl.dialects.func import Return, FuncOp
from xdsl.ir import Attribute, Region, MLContext, TypeAttribute
from xdsl.pattern_rewriter import PatternRewriteWalker, GreedyRewritePatternApplier
from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations
from xdsl.transforms.experimental.dlt import lower_dlt_to_
from xdsl.transforms.experimental.convert_return_to_pointer_arg import (
    PassByPointerRewriter,
)
from xdsl.transforms.experimental.dlt.generate_dlt_iteration_orders import (
    IterationGenerator,
    _make_nested_order,
)
from xdsl.transforms.experimental.dlt.iteration_map import IterationMap
from xdsl.transforms.experimental.dlt.layout_graph import LayoutGraph
from xdsl.transforms.experimental.dlt.generate_dlt_layouts import (
    LayoutGenerator,
    _make_dense_layouts,
    _try_apply_index_replace,
    _try_apply_sparse,
)
from xdsl.transforms.experimental.dlt.generate_dlt_identities import (
    DLTGenerateIterateOpIdentitiesRewriter,
    DLTGeneratePtrIdentitiesRewriter,
    DLTSimplifyPtrIdentitiesRewriter,
)
from xdsl.transforms.experimental.lower_dtl_to_dlt import DTLRewriter
from xdsl.transforms.printf_to_llvm import PrintfToLLVM
from xdsl.transforms.printf_to_putchar import PrintfToPutcharPass
from xdsl.transforms.reconcile_unrealized_casts import reconcile_unrealized_casts

from dtl import compilec

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


class FunctionCaller:
    def __init__(self, func, param_types, ret_types, return_structure=None):
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
                # print(type(arg))
                # print(param_type)
                # print(type(arg) == param_type)
                arg = param_type(arg)  # try to make init the right type?
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
    def __init__(
        self, library_path, funcs, function_types: dict[str, builtin.FunctionType]
    ):
        self._library_path = library_path
        self._funcs: dict[str, FuncTypeDescriptor] = (
            funcs  # describe funcs as original (with return vals) with dlt types
        )
        self._callers: dict[str, FunctionCaller] = {}
        self._lib = cdll.LoadLibrary(self._library_path)
        self._func_types = function_types  # describe funcs as original (with return vals) with llvm types
        self._ctype_classes: dict = {}

        self._dlclose_func = ctypes.cdll.LoadLibrary('').dlclose
        self._dlclose_func.argtypes = [ctypes.c_void_p]
        self._handle = self._lib._handle

    def _close(self, delete=False):
        del self._lib
        self._dlclose_func(self._handle)
        lib_path = self._library_path
        os.remove(lib_path)


    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name.startswith("_"):
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
        if len(func_type.inputs) != len(func_descriptor.params):
            print(name)
            print(func_type.inputs)
            print(func_descriptor.params)
        assert len(func_type.outputs) == len(func_descriptor.return_types)
        ret_types = list(func_type.outputs.data)
        param_types = list(func_type.inputs.data)
        ctype_ret_types = [
            self.convert_llvm_type_to_ctypes(t, dlt_type=dlt_t)
            for t, dlt_t in zip(ret_types, func_descriptor.return_types)
        ]
        ctype_ret_pointer_types = [ctypes.POINTER(t) for t in ctype_ret_types]
        ctype_param_types = [
            self.convert_llvm_type_to_ctypes(t, dlt_type=dlt_t)
            for t, dlt_t in zip(param_types, func_descriptor.params)
        ]
        self._lib[name].argtypes = ctype_ret_pointer_types + ctype_param_types
        caller = FunctionCaller(
            self._lib[name],
            ctype_param_types,
            ctype_ret_types,
            return_structure=func_descriptor.structure,
        )
        self._callers[name] = caller
        return caller

    def convert_llvm_type_to_ctypes(
        self, llvm_type: Attribute, dlt_type: TypeAttribute = None
    ):
        assert isinstance(llvm_type, TypeAttribute)
        if dlt_type is not None:
            if dlt_type not in self._ctype_classes:
                type = self._convert_llvm_type_to_ctypes(llvm_type, dlt_type)
                self._ctype_classes[dlt_type] = type
            return self._ctype_classes[dlt_type]
        else:
            return self._convert_llvm_type_to_ctypes(llvm_type)

    @functools.singledispatchmethod
    def _convert_llvm_type_to_ctypes(
        self, llvm_type: TypeAttribute, dlt_type: TypeAttribute = None
    ):
        raise NotImplementedError(f"Cannot convert type: {llvm_type} to ctypes")

    @_convert_llvm_type_to_ctypes.register
    def _(self, llvm_type: builtin.IntegerType, dlt_type: TypeAttribute = None):
        if llvm_type == builtin.i64:
            return ctypes.c_uint64
        else:
            raise NotImplementedError(
                f"Cannot convert Integer type: {llvm_type} to ctypes"
            )

    @_convert_llvm_type_to_ctypes.register
    def _(self, llvm_type: builtin.Float32Type, dlt_type: TypeAttribute = None):
        if llvm_type == builtin.f32:
            return ctypes.c_float
        else:
            raise NotImplementedError(
                f"Cannot convert Floating type: {llvm_type} to ctypes"
            )

    @_convert_llvm_type_to_ctypes.register
    def _(self, llvm_type: llvm.LLVMPointerType, dlt_type: TypeAttribute = None):
        return ctypes.c_void_p

    @_convert_llvm_type_to_ctypes.register
    def _(self, llvm_type: llvm.LLVMStructType, dlt_type: TypeAttribute = None):
        member_types = [
            self.convert_llvm_type_to_ctypes(member) for member in llvm_type.types
        ]
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


class LibBuilder:
    def __init__(self, scope_extents: dict[dtl.UnknownSizeVectorSpace, int] = None):
        if scope_extents is None:
            scope_extents = {}
        self._scope_vector_spaces = scope_extents

        self.vs_namer = {}
        self.vs_name_idx = 0
        self.tensor_var_details: dict[TupleStruct[TensorVariable], dlt.PtrType] = {}
        self.tensor_var_elems: dict[
            TupleStruct[TensorVariable], TupleStruct[dlt.ElementAttr]
        ] = {}
        self.tensor_elem_tuple_name_idx = 0
        # self.tensor_var_ptrs: dict[TupleStruct[TensorVariable], dlt.PtrType] = {}
        self.tensor_var_dims: dict[TensorVariable, list[dlt.DimensionAttr]] = {}
        self.tensor_var_namer_idx = 0

        self.funcs = []
        self.func_map: dict[str, FuncTypeDescriptor] = {}

    def _get_vs_name(self, vs: VectorSpaceVariable, new: bool = False):
        if new:
            name = "dim_" + str(self.vs_name_idx)
            self.vs_name_idx += 1
            return name
        if vs not in self.vs_namer:
            raise ValueError()
            # name = "dim_" + str(self.vs_name_idx)
            # self.vs_name_idx += 1
            # self.vs_namer[vs] = name
        return self.vs_namer[vs]

    def _get_vs_extent(self, vs: VectorSpaceVariable):
        if isinstance(vs, UnknownSizeVectorSpace):
            if vs in self._scope_vector_spaces:
                return dlt.ScopeDefinedExtentAttr(vs.name)
            else:
                return dlt.InitDefinedExtentAttr(vs.name)
        if isinstance(vs, VectorSpace):
            return dlt.StaticExtentAttr(vs.dim)

    def _add_member_in_tupleStruct_layer_elements(
        self, elements: TupleStruct[dlt.ElementAttr], add: dlt.MemberAttr = None
    ) -> TupleStruct[dlt.ElementAttr]:
        if isinstance(elements, tuple):
            return tuple(
                [self._add_member_in_tupleStruct_layer_elements(e) for e in elements]
            )
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

    def _set_dlt_type_elems_for_tensor_var(
        self, tensor_vars: TupleStruct[TensorVariable]
    ) -> TupleStruct[dlt.ElementAttr]:
        if tensor_vars in self.tensor_var_elems:
            assert tensor_vars not in self.tensor_var_elems
        if isinstance(tensor_vars, tuple):
            elements: TupleStruct[dlt.ElementAttr] = tuple()
            level = str(self.tensor_elem_tuple_name_idx)
            self.tensor_elem_tuple_name_idx += 1

            for i, tensor_var in enumerate(tensor_vars):
                child_elements = self._set_dlt_type_elems_for_tensor_var(
                    tensor_var
                )
                new_member = dlt.MemberAttr(f"T_{level}", f"{i}")
                child_elements = self._add_member_in_tupleStruct_layer_elements(
                    child_elements, add=new_member
                )
                elements = tuple([*elements, child_elements])
            self.tensor_var_elems[tensor_vars] = elements
            return elements
        else:
            tensor_var = typing.cast(TensorVariable, tensor_vars)
            vector_space_vars = tensor_var.tensor_space.spaces
            vector_space_names = [
                self._get_vs_name(vs, new=True) for vs in vector_space_vars
            ]
            vector_space_extents = [self._get_vs_extent(vs) for vs in vector_space_vars]
            dimensions = [
                dlt.DimensionAttr(vs_name, vs_extent)
                for vs_name, vs_extent in zip(vector_space_names, vector_space_extents)
            ]
            element = dlt.ElementAttr([], dimensions, builtin.f32)
            self.tensor_var_elems[tensor_var] = element
            self.tensor_var_dims[tensor_var] = dimensions
            return element

    def _set_dlt_type_for_tensor_var(
        self,
        tensor_vars: TupleStruct[TensorVariable],
        ptr_type: dlt.PtrType = None,
        members: set[dlt.MemberAttr] = None,
    ):
        assert tensor_vars not in self.tensor_var_details
        if members is None:
            members = set()
        if isinstance(tensor_vars, tuple):
            elements = _flatten(self.tensor_var_elems[tensor_vars])
            if ptr_type is None:

                dlt_TypeType = dlt.TypeType(elements)
                all_dims = [dim for elem in elements for dim in elem.dimensions]
                init_extents = list(
                    {
                        base_extent
                        for dim in all_dims
                        for base_extent in dim.extent.base_extents()
                        if isinstance(base_extent, dlt.InitDefinedExtentAttr)
                    }
                )
                ptr_id = str(self.tensor_var_namer_idx)
                self.tensor_var_namer_idx += 1
                ptr_type = dlt.PtrType(
                    dlt_TypeType,
                    members=dlt.SetAttr(members),
                    extents=init_extents,
                    identity=ptr_id,
                )
            else:
                ptr_id = str(self.tensor_var_namer_idx)
                self.tensor_var_namer_idx += 1
                ptr_type = dlt.PtrType(
                    ptr_type.contents_type.select_members(members),
                    ptr_type.layout,
                    ptr_type.filled_members.add(members),
                    ptr_type.filled_dimensions,
                    ptr_type.filled_extents,
                    ptr_type.is_base,
                    ptr_id,
                )
            self.tensor_var_details[tensor_vars] = ptr_type
            common_member_name = set.intersection(*[{m.structName.data for m in elem.member_specifiers} for elem in ptr_type.contents_type.elements])
            assert len(common_member_name) == 1
            common_member_name = common_member_name.pop()
            for i, child in enumerate(tensor_vars):
                self._set_dlt_type_for_tensor_var(
                    child,
                    ptr_type,
                    members.union({dlt.MemberAttr(common_member_name, f"{i}")}),
                )
        else:
            tensor_var = typing.cast(TensorVariable, tensor_vars)
            element = self.tensor_var_elems[tensor_var]
            if ptr_type is None:
                dlt_TypeType = dlt.TypeType([element])
                all_dims = [dim for dim in element.dimensions]
                init_extents = list(
                    {
                        base_extent
                        for dim in all_dims
                        for base_extent in dim.extent.base_extents()
                        if isinstance(base_extent, dlt.InitDefinedExtentAttr)
                    }
                )
                ptr_id = str(self.tensor_var_namer_idx)
                self.tensor_var_namer_idx += 1
                ptr_type = dlt.PtrType(
                    dlt_TypeType,
                    members=dlt.SetAttr(members),
                    extents=init_extents,
                    identity=ptr_id,
                )
            else:
                ptr_id = str(self.tensor_var_namer_idx)
                self.tensor_var_namer_idx += 1
                ptr_type = dlt.PtrType(
                    ptr_type.contents_type.select_members(members),
                    ptr_type.layout,
                    ptr_type.filled_members.add(members),
                    ptr_type.filled_dimensions,
                    ptr_type.filled_extents,
                    ptr_type.is_base,
                    ptr_id,
                )
            self.tensor_var_details[tensor_var] = ptr_type

    def _get_tensor_var_info(
        self, tensor_vars: TupleStruct[TensorVariable]
    ) -> tuple[dlt.PtrType, TupleStruct[dlt.ElementAttr]]:
        if tensor_vars not in self.tensor_var_details:
            self._set_dlt_type_elems_for_tensor_var(tensor_vars)
            self._set_dlt_type_for_tensor_var(tensor_vars)
        return self.tensor_var_details[tensor_vars], self.tensor_var_elems[tensor_vars]

    def _get_select_ops(
        self,
        ssa_in: ir.SSAValue,
        elements: TupleStruct[dlt.ElementAttr],
        tensor_vars: TupleStruct[TensorVariable],
    ) -> tuple[list[ir.Operation], list[ir.SSAValue]]:
        if isinstance(elements, tuple):
            ops = []
            results = []
            for element, tensor_var in zip(elements, tensor_vars):
                child_ops, child_results = self._get_select_ops(
                    ssa_in, element, tensor_var
                )
                ops.extend(child_ops)
                results.extend(child_results)
            return ops, results
        elif isinstance(elements, dlt.ElementAttr):
            element = typing.cast(dlt.ElementAttr, elements)
            assert isinstance(tensor_vars, TensorVariable)
            tensor_var = typing.cast(TensorVariable, tensor_vars)
            result_type = self.tensor_var_details[tensor_var]
            return [
                op := dlt.SelectOp(
                    ssa_in, element.member_specifiers, [], [], result_type=result_type
                )
            ], [op.res]

    def make_init(
        self,
        name: str,
        tensor_vars: TupleStruct[TensorVariable],
        extents: list[UnknownSizeVectorSpace],
        free_name: str = None,
    ):
        ptr_type, elements = self._get_tensor_var_info(tensor_vars)
        base_ptr_type = ptr_type.as_base()
        base_ptr_type = ptr_type.as_base().with_identification(
            "R_" + base_ptr_type.identification.data
        )
        block = ir.Block()
        arg_idx = 0
        extents_map = {}
        for e in extents:
            arg = block.insert_arg(IndexType(), arg_idx)
            debug = [
                c := builtin.UnrealizedConversionCastOp.get([arg], [i64]),
                trunc_op := arith.TruncIOp(c.outputs[0], builtin.i32),
                printf.PrintIntOp(trunc_op.result),
            ] + printf.PrintCharOp.from_constant_char_ops("\n")
            block.add_ops(debug)
            arg_idx += 1
            if e is not None:
                assert isinstance(e, UnknownSizeVectorSpace)
                extents_map[self._get_vs_extent(e)] = arg

        alloc_op = dlt.AllocOp(base_ptr_type, extents_map)
        block.add_op(alloc_op)
        select_elem_ops, select_elem_ssa = self._get_select_ops(
            alloc_op.res, elements, tensor_vars
        )
        block.add_ops(select_elem_ops)
        ret = Return(alloc_op.res, *select_elem_ssa)
        block.add_op(ret)

        region = Region([block])
        argTypes = [arg.type for arg in block.args]
        retTypes = [r.type for r in ret.operands]
        func = FuncOp.from_region(name, argTypes, retTypes, region)
        self.funcs.append(func)
        func_type = func.function_type
        self.func_map[name] = FuncTypeDescriptor(
            list(func_type.inputs), list(func_type.outputs), (1, tensor_vars)
        )

        if free_name != None:
            free_block = ir.Block()
            free_block.add_op(dlt.DeallocOp(free_block.insert_arg(base_ptr_type, 0)))
            free_block.add_op(ret := Return())
            free_region = Region([free_block])
            free_argTypes = [arg.type for arg in free_block.args]
            free_retTypes = [r.type for r in ret.operands]
            free_func = FuncOp.from_region(
                free_name, free_argTypes, free_retTypes, free_region
            )
            self.funcs.append(free_func)
            free_func_type = free_func.function_type
            self.func_map[free_name] = FuncTypeDescriptor(
                list(free_func_type.inputs), list(free_func_type.outputs), None
            )

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
        self.func_map[name] = FuncTypeDescriptor(
            list(func_type.inputs), list(func_type.outputs), ret_val
        )
        return func

    def make_setter(
        self,
        name,
        tensor_var: TensorVariable,
        members: typing.Iterable[dlt.MemberAttr],
        args: list[int],
    ):
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
        self.func_map[name] = FuncTypeDescriptor(
            list(func_type.inputs), list(func_type.outputs), None
        )
        return func

    def make_getter(
        self,
        name,
        tensor_var: TensorVariable,
        members: typing.Iterable[dlt.MemberAttr],
        args: list[int],
    ):
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

        load = dlt.GetOp(select.res, element.base_type)
        block.add_op(load)
        ret = Return(load.res)
        block.add_op(ret)

        region = Region([block])
        argTypes = [arg.type for arg in block.args]
        retTypes = [r.type for r in ret.operands]
        func = FuncOp.from_region(name, argTypes, retTypes, region)
        self.funcs.append(func)
        func_type = func.function_type
        self.func_map[name] = FuncTypeDescriptor(
            list(func_type.inputs), list(func_type.outputs), 1
        )
        return func

    def make_print_tensorVar(
        self, name: str, tensor_var, dynamic_vector_spaces: list[UnknownSizeVectorSpace]
    ):
        block = ir.Block()
        ptr_type, elements = self._get_tensor_var_info(tensor_var)
        dims = self.tensor_var_dims[tensor_var]
        tensor_var_arg = block.insert_arg(ptr_type, 0)

        dynamic_extents = {}
        for vs in dynamic_vector_spaces:
            arg = block.insert_arg(IndexType(), len(block.args))
            dynamic_extents[self._get_vs_extent(vs)] = arg

        extents = [dim.extent for dim in dims]
        extent_args = []
        for e in extents:
            if e.is_dynamic():
                extent_args.append(dynamic_extents[e])
            if e.is_init_time():
                block.add_op(ex := dlt.ExtractExtentOp(tensor_var_arg, e))
                extent_args.append(ex.res)
        block.add_op(
            iter_op := dlt.IterateOp(
                extents, extent_args, [[[d] for d in dims]], [tensor_var_arg], []
            )
        )
        inner_idxs = iter_op.body.block.args[0 : len(extents)]
        inner_ptr = iter_op.get_block_arg_for_tensor_arg_idx(0)
        ret = iter_op.body.block.last_op
        iter_op.body.block.insert_op_before(
            get_op := dlt.GetOp(inner_ptr, get_type=builtin.f32), ret
        )
        iter_op.body.block.insert_op_before(
            printf.PrintFormatOp(
                f"{name}:: "
                + (
                    ",".join(
                        [f"{dim.dimensionName} ({dim.extent}): {{}}" for dim in dims]
                    )
                )
                + " -> {}",
                *inner_idxs,
                get_op.res,
            ),
            ret,
        )
        block.add_op(Return())

        region = Region([block])
        argTypes = [arg.type for arg in block.args]
        retTypes = []
        func = FuncOp.from_region(name, argTypes, retTypes, region)
        self.funcs.append(func)
        func_type = func.function_type
        self.func_map[name] = FuncTypeDescriptor(
            list(func_type.inputs), list(func_type.outputs), None
        )
        return func

    def make_function(
        self,
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

        exec, output = xdtl.get_xdsl_dtl_exec_version(
            expr,
            dynamic_space_map=dynamic_extents,
            scope_spaces=set(self._scope_vector_spaces.keys()),
            arg_map=arg_map,
            tensor_variables=tensor_variables,
            outputs=outputs,
        )
        block.add_ops(exec)
        block.add_op(output)
        block.add_op(Return())

        region = Region([block])
        argTypes = [arg.type for arg in block.args]
        retTypes = []
        func = FuncOp.from_region(name, argTypes, retTypes, region)
        self.funcs.append(func)
        func_type = func.function_type
        self.func_map[name] = FuncTypeDescriptor(
            list(func_type.inputs), list(func_type.outputs), None
        )
        return func

    def prepare(self, verbose=2) -> tuple[ModuleOp, LayoutGraph, IterationMap]:
        if verbose > 0:
            print(f"Preparing module")
        malloc_func = llvm.FuncOp(
            "malloc",
            llvm.LLVMFunctionType([builtin.i64], llvm.LLVMPointerType.opaque()),
            linkage=llvm.LinkageAttr("external"),
        )
        free_func = llvm.FuncOp(
            "free",
            llvm.LLVMFunctionType([llvm.LLVMPointerType.opaque()], None),
            linkage=llvm.LinkageAttr("external"),
        )
        memcpy_func = llvm.FuncOp(
            "memcpy",
            llvm.LLVMFunctionType(
                [
                    llvm.LLVMPointerType.opaque(),
                    llvm.LLVMPointerType.opaque(),
                    builtin.i64,
                ],
                None,
            ),
            linkage=llvm.LinkageAttr("external"),
        )
        abort_func = llvm.FuncOp(
            "abort", llvm.LLVMFunctionType([]), linkage=llvm.LinkageAttr("external")
        )

        scope_op = dlt.LayoutScopeOp([(dlt.ScopeDefinedExtentAttr(vs.name), i) for vs, i in self._scope_vector_spaces.items()], self.funcs)
        module = ModuleOp([malloc_func, free_func, memcpy_func, abort_func, scope_op])
        module.verify()

        if verbose > 1:
            print(module)

        # DTL -> DLT
        dtl_to_dlt_applier = PatternRewriteWalker(
            DTLRewriter(), walk_regions_first=False
        )
        dtl_to_dlt_applier.rewrite_module(module)
        if verbose > 1:
            print(module)
        module.verify()

        first_identity_gen = DLTGeneratePtrIdentitiesRewriter()
        dlt_ident_applier = PatternRewriteWalker(
            first_identity_gen, walk_regions_first=False
        )
        dlt_ident_applier.rewrite_module(module)
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

        module.verify()

        if verbose > 1:
            print(module)

        layout_graph = layout_graph_generator.layouts[scope_op]
        iteration_map = iteration_map_generator.iteration_maps[scope_op]

        layout_graph.check_consistency()
        iteration_map.check_consistency()

        check = layout_graph.is_consistent()
        if not check:
            raise ValueError(
                "Layout Graph has been checked and found to be inconsistent"
            )
        check = iteration_map.is_consistent()
        if not check:
            raise ValueError(
                "Iteration Map has been checked and found to be inconsistent"
            )
        return module, layout_graph, iteration_map

    def lower(
        self,
        module: ModuleOp,
        layout_graph: LayoutGraph,
        new_type_map: dict[builtin.StringAttr, dlt.PtrType],
        iteration_map: IterationMap,
        new_order_map: dict[builtin.StringAttr, dlt.IterationOrder],
        verbose=2,
    ):
        if verbose > 0:
            print(f"Compiling module")
        check = layout_graph.is_consistent(new_type_map)
        if not check:
            raise ValueError("Layout Graph has been check and found to be inconsistent")
        if verbose > 1:
            print("making dlt.ptr type updating pass")
        rewriter = layout_graph.use_types(new_type_map)
        if verbose > 1:
            print("rewriting types")
        rewriter.rewrite_op(module)
        if verbose > 1:
            print("verifying module")
        module.verify()

        if verbose > 1:
            print("making iteration order updating pass")
        rewriter = iteration_map.use_orders(new_order_map)
        if verbose > 1:
            print("rewriting iterate ops")
        rewriter.rewrite_op(module)
        if verbose > 1:
            print("verifying module")
        module.verify()

        if verbose > 1:
            print(module)

        # DLT -> MLIR builtin
        dlt_to_llvm_applier = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RemoveUnusedOperations(),
                    lower_dlt_to_.DLTSelectRewriter(),
                    lower_dlt_to_.DLTGetRewriter(),
                    lower_dlt_to_.DLTSetRewriter(),
                    lower_dlt_to_.DLTAllocRewriter(),
                    lower_dlt_to_.DLTDeallocRewriter(),
                    lower_dlt_to_.DLTIterateRewriter(),
                    lower_dlt_to_.DLTCopyRewriter(),
                    lower_dlt_to_.DLTExtractExtentRewriter(),
                ]
            ),
            walk_regions_first=False,
        )

        # print(module)
        # module.verify()

        dlt_to_llvm_applier.rewrite_module(module)

        if verbose > 1:
            print(module)
        module.verify()

        rem_scope = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    lower_dlt_to_.DLTScopeRewriter(),
                    lower_dlt_to_.DLTPtrTypeRewriter(recursive=True),
                    lower_dlt_to_.DLTIndexTypeRewriter(recursive=True),
                    lower_dlt_to_.DLTIndexRangeTypeRewriter(recursive=True),
                ]
            )
        )
        rem_scope.rewrite_module(module)
        PrintfToPutcharPass().apply(MLContext(True), module)
        PrintfToLLVM().apply(MLContext(True), module)

        function_types = {}
        func_mod_applier = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    PassByPointerRewriter(
                        [func.sym_name for func in self.funcs], originals=function_types
                    )
                ]
            )
        )
        func_mod_applier.rewrite_module(module)

        reconcile_unrealized_casts(module)
        module.verify()

        if verbose > 1:
            print(module)
        return function_types

    def compile(
        self,
        module: ModuleOp,
        function_types: dict[builtin.StringAttr, func.FunctionType],
        llvm_out: str = None,
        llvm_only: bool = False,
        verbose=2,
    ) -> DTLCLib | None:

        lib_fd, lib_name = tempfile.mkstemp(suffix=".so")
        os.close(lib_fd)
        if verbose > 0:
            print(f"library name: {lib_name}")
        compilec.mlir_compile(module, lib_name, llvm_out=llvm_out, llvm_only=llvm_only, verbose=verbose)
        if llvm_only:
            return None
        function_types = {name.data: v for name, v in function_types.items()}

        return DTLCLib(lib_name, self.func_map, function_types)

    def compile_from(self, llvm_path: str, function_types: dict[builtin.StringAttr, func.FunctionType], verbose = 2) -> DTLCLib:
        lib_fd, lib_name = tempfile.mkstemp(suffix=".so")
        os.close(lib_fd)
        if verbose > 0:
            print(f"library name: {lib_name}")
        compilec.clang_compile(llvm_path, lib_name, verbose=verbose)
        function_types = {name.data: v for name, v in function_types.items()}
        return DTLCLib(lib_name, self.func_map, function_types)

    def build(self, verbose: int = 0):

        module, layout_graph, iteration_map = self.prepare(verbose)

        iteration_generator = IterationGenerator(iteration_map)
        # new_iter_maps = iteration_generator.generate_mappings()
        layout_generator = LayoutGenerator(layout_graph)
        # new_layout_maps = layout_generator.generate_mappings()

        original_type_map = layout_graph.get_type_map()
        new_type_map = original_type_map.copy()

        while any(ptr.layout.is_abstract for ptr in new_type_map.values()):
            old_type_map = new_type_map.copy()
            idents = {
                ident
                for ident in layout_graph.get_entry_layouts().keys()
                if new_type_map[ident].layout.is_abstract
            }
            if len(idents) == 0:
                idents = {
                    ident
                    for ident in new_type_map.keys()
                    if new_type_map[ident].layout.is_abstract
                }
            if len(idents) == 0:
                raise ValueError(
                    "some layout is abstract but cannot find an abstract layout"
                )
            ident = idents.pop()
            print(f"Reifying {ident.data}")
            ptr = new_type_map.pop(ident)
            new_layout = ptr.layout
            # new_layout = _try_apply_index_replace(new_layout)
            new_layout = _try_apply_sparse(new_layout)
            new_layout = _make_dense_layouts(new_layout, {})
            # new_layout = dlt.StructLayoutAttr([new_layout])
            new_ptr = ptr.with_new_layout(new_layout, preserve_ident=True)
            new_type_map[ident] = new_ptr
            layout_graph.propagate_type(ident, new_type_map)
            changed = {i for (i, ptr) in new_type_map.items() if old_type_map[i] != ptr}
            print("Propagated changes to: " + ",".join([i.data for i in changed]))

        original_order_map = iteration_map.get_map()
        new_order_map = original_order_map.copy()
        new_orders = {
            id: _make_nested_order(order) for id, order in new_order_map.items()
        }

        function_types = self.lower(module, layout_graph, new_type_map, iteration_map, new_orders, verbose=verbose)
        return self.compile(module, function_types, verbose=verbose)


"""
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
      

"""
