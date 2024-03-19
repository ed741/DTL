from dtl import VectorSpaceVariable, UnknownSizeVectorSpace, VectorSpace, TensorVariable, Expr, Index
from dtlpp.backends import xdsl as xdtl
from xdsl import ir
from xdsl.dialects import builtin, arith, printf
from xdsl.dialects.builtin import f32, IndexType
from xdsl.dialects.experimental import dlt
from xdsl.dialects.func import Return, FuncOp
from xdsl.ir import Region


class LibBuilder:
    def __init__(self):
        self.vs_namer = {}
        self.vs_name_idx = 0
        self.tensor_var_details = {}
        self.tensor_var_namer_idx = 0

        self.funcs = []

    def _get_vs_name(self, vs: VectorSpaceVariable, new: bool=False):
        if new:
            name = str(self.vs_name_idx)
            self.vs_name_idx += 1
            return name
        if vs not in self.vs_namer:
            name = str(self.vs_name_idx)
            self.vs_name_idx += 1
            self.vs_namer[vs] = name
        return self.vs_namer[vs]

    def _get_vs_extent(self, vs: VectorSpaceVariable):
        if isinstance(vs, UnknownSizeVectorSpace):
            return dlt.InitDefinedExtentAttr(vs.name)
        if isinstance(vs, VectorSpace):
            return dlt.StaticExtentAttr(vs.dim)

    def _get_tensor_var_info(self, tensor_var: TensorVariable, base=False):
        if tensor_var not in self.tensor_var_details:
            vector_space_vars = tensor_var.tensor_space.spaces
            vector_space_names = [self._get_vs_name(vs, new=True) for vs in vector_space_vars]
            vector_space_extents = [self._get_vs_extent(vs) for vs in vector_space_vars]
            dimensions = [dlt.DimensionAttr(vs_name, vs_extent) for vs_name, vs_extent in zip(vector_space_names, vector_space_extents)]
            init_extents = list({base_extent for dim in dimensions for base_extent in dim.extent.base_extents() if isinstance(base_extent, dlt.InitDefinedExtentAttr)})
            ptr_id = str(self.tensor_var_namer_idx)
            self.tensor_var_namer_idx +=1
            ptr_type = dlt.PtrType(dlt.TypeType([([],dimensions,f32)]),extents=init_extents, identity=ptr_id)
            if base:
                ptr_type = ptr_type.as_base()
            self.tensor_var_details[tensor_var] = (ptr_type, dimensions)
        return self.tensor_var_details[tensor_var]

    def make_init(self,
                  name: str,
                  tensor_var: TensorVariable,
                  extents: list[UnknownSizeVectorSpace],
                  ):
        ptr_type, dims = self._get_tensor_var_info(tensor_var, base=True)
        block = ir.Block()
        arg_idx = 0
        dlt_extents = []
        alloc_args = []
        for e in extents:
            arg = block.insert_arg(IndexType(), arg_idx)
            debug = [trunc_op := arith.IndexCastOp(arg, builtin.i32), printf.PrintIntOp(trunc_op.result)] + printf.PrintCharOp.from_constant_char_ops("\n")
            block.add_ops(debug)
            arg_idx += 1
            if e is not None:
                assert isinstance(e, UnknownSizeVectorSpace)
                alloc_args.append(arg)
                dlt_extents.append(self._get_vs_extent(e))

        alloc_op = dlt.AllocOp(operands=[[], alloc_args],
                               attributes={"init_extents": builtin.ArrayAttr(dlt_extents)},
                               result_types=[ptr_type])
        block.add_op(alloc_op)
        ret = Return(alloc_op.res)
        block.add_op(ret)

        region = Region([block])
        argTypes = [arg.type for arg in block.args]
        retTypes = [r.type for r in ret.operands]
        func = FuncOp.from_region(name, argTypes, retTypes, region)
        self.funcs.append(func)
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
        return func

    def make_funcion(self,
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
            ptr_type, dims = self._get_tensor_var_info(result)
            arg = block.insert_arg(ptr_type, arg_idx)
            arg_idx += 1
            dim_names = [dim.dimensionName for dim in dims]
            outputs.append((arg, dim_names))
        tensor_variables = {}
        for tensor_var in tensor_vars:
            ptr_type, dims = self._get_tensor_var_info(tensor_var)
            arg = block.insert_arg(ptr_type, arg_idx)
            arg_idx += 1
            dim_names = [dim.dimensionName for dim in dims]
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
        return func
