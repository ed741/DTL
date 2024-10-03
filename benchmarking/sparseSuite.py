import abc
import sys
import timeit
from random import Random

import numpy as np
import scipy

from benchmark import Benchmark
from dtl import Index, RealVectorSpace, TensorVariable
from dtl.libBuilder import LibBuilder, NpArrayCtype, TupleStruct
from xdsl.dialects import arith, builtin, func, llvm, printf, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType, StringAttr
from xdsl.dialects.experimental import dlt
from xdsl.ir import Block
from xdsl.transforms.experimental.dlt.generate_dlt_iteration_orders import IterationMapping
from xdsl.transforms.experimental.dlt.generate_dlt_layouts import PtrMapping, ReifyConfig

_Epsilon = 0.00001

class SparseSuite(Benchmark, abc.ABC):
    def __init__(
        self,
        matrix_path: str,
        vector_path: str,
        seed: int,
        base_dir: str,
        name: str,
        runs: int,
        repeats: int,
        opt_num: int,
        epsilon: float,
        np_suffix: str = "",
        **kwargs,
    ):
        new_base_dir = f"{base_dir}/sparseSuite"
        results_base_dir = (
            f"{new_base_dir}/{name}_O{opt_num}_{seed}_{runs}"
        )
        self.seed = seed
        super().__init__(
            results_base_dir,
            f"{new_base_dir}/layouts",
            f"{new_base_dir}/orders",
            runs,
            repeats,
            opt_num,
            epsilon,
            **kwargs,
        )

        # print("setting random values in np a & b")
        r = Random(seed)
        ref_a = scipy.io.mmread(matrix_path)
        assert isinstance(ref_a, scipy.sparse.coo_matrix)
        ref_b = scipy.io.mmread(vector_path)

        self.i, self.j = ref_a.shape
        self.nnz = ref_a.nnz

        ref_c_l = []
        def make_c():
            ref_c_l.append(ref_a.astype(np.float32) * ref_b.astype(np.float32))
        result = timeit.timeit(make_c, number=100)
        print(f"Time to make ref_c with scipy: {result}s")

        ref_c = ref_c_l[0]


        self.ref_a = ref_a
        self.np_a_row = ref_a.row.astype(np.int32)#[0:1000]
        self.np_a_col = ref_a.col.astype(np.int32)#[0:1000]
        self.np_a_val = ref_a.data.astype(np.float32)#[0:1000]
        self.np_b = ref_b.astype(np.float32).reshape((-1,))
        self.np_c = ref_c.astype(np.float32).reshape((-1,))

        #self.nnz = 1000


        self.tensor_variables = None

        self.handle_reference_array(self.np_a_row, "np_a_row"+np_suffix, True, False, "np_a_row", binary=True, dtype=np.int32)
        self.handle_reference_array(self.np_a_col, "np_a_col" + np_suffix, True, False, "np_a_col", binary=True, dtype=np.int32)
        self.handle_reference_array(self.np_a_val, "np_a_val" + np_suffix, True, False, "np_a_val", binary=True)
        self.handle_reference_array(self.np_b, "np_b"+np_suffix, True, False, "np_b", binary=True)
        self.handle_reference_array(self.np_c, "np_c"+np_suffix, False, True, "np_c", binary=True)


    @abc.abstractmethod
    def construct_lib_builder(
        self,
        lib_builder: LibBuilder,
        a: TensorVariable,
        b: TensorVariable,
        c: TensorVariable,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def get_configs_for_tensors(
        self, a: TensorVariable, b: TensorVariable, c: TensorVariable
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        raise NotImplementedError

    def get_configs_for_DTL_tensors(
        self,
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        assert self.tensor_variables is not None
        return self.get_configs_for_tensors(*self.tensor_variables)

    def define_lib_builder(self) -> LibBuilder:
        vi = RealVectorSpace(self.i)
        vj = RealVectorSpace(self.j)

        A = TensorVariable(vi * vj, "A")
        B = TensorVariable(vj, "B")
        C = TensorVariable(vi, "C")

        self.tensor_variables = (A, B, C)

        _i = Index("i")
        _j = Index("j")
        spmvmul = (A[_i, _j] * B[_j]).sum(_j).forall(_i)

        lib_builder = LibBuilder({})
        self.construct_lib_builder(lib_builder, A, B, C)

        lib_builder.make_setter("set_A", (A), {}, [0, 1])
        lib_builder.make_setter("set_B", (B), {}, [0])

        lib_builder.make_getter("get_C", (C), {}, [0])
        lib_builder.make_function("spmvmul", spmvmul, [C], [A, B], [], [])

        self._make_setup_func_A(lib_builder, "setup_A", A, self.nnz)
        self._make_setup_func_B(lib_builder, "setup_B", B, self.j)
        self._make_check_func_C(lib_builder, "check_C", C, self.i, self.epsilon)
        return lib_builder
    #
    @staticmethod
    def _make_setup_func_A(lib_builder: LibBuilder, name: str, t_var: TensorVariable, nnz: int):
        setup_a_block = Block(arg_types=[lib_builder.tensor_var_details[t_var], llvm.LLVMPointerType.opaque(), llvm.LLVMPointerType.opaque(), llvm.LLVMPointerType.opaque()])
        arg_a, arg_np_ptr_row, arg_np_ptr_col, arg_np_ptr_val = setup_a_block.args
        arg_a_dims = lib_builder.tensor_var_dims[t_var]

        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        setup_a_block.add_ops([zero_op, one_op])

        nnz_ub_op = arith.Constant(IntegerAttr(nnz, IndexType()))
        nnz_ub_cast_op = builtin.UnrealizedConversionCastOp.get([nnz_ub_op.result], [builtin.i64])
        nnz_ub_i64 = nnz_ub_cast_op.outputs[0]
        setup_a_block.add_ops([nnz_ub_op, nnz_ub_cast_op])

        lb = zero_op.result
        step = one_op.result
        ub = nnz_ub_op.result

        loop_block = Block(arg_types=[IndexType()])
        nnz_index_cast_op = builtin.UnrealizedConversionCastOp.get([loop_block.args[0]], [builtin.i64])
        loop_block.add_op(nnz_index_cast_op)
        nnz_idx_i64 = nnz_index_cast_op.outputs[0]

        row_ptr_arith_op = llvm.GEPOp(arg_np_ptr_row, [0, llvm.GEP_USE_SSA_VAL], [nnz_idx_i64], pointee_type=llvm.LLVMArrayType.from_size_and_type(nnz, builtin.i32))
        row_load_op = llvm.LoadOp(row_ptr_arith_op.result, builtin.i32)
        col_ptr_arith_op = llvm.GEPOp(arg_np_ptr_col, [0, llvm.GEP_USE_SSA_VAL], [nnz_idx_i64],
                                   pointee_type=llvm.LLVMArrayType.from_size_and_type(nnz, builtin.i32))
        col_load_op = llvm.LoadOp(col_ptr_arith_op.result, builtin.i32)
        val_ptr_arith_op = llvm.GEPOp(arg_np_ptr_val, [0, llvm.GEP_USE_SSA_VAL], [nnz_idx_i64],
                                      pointee_type=llvm.LLVMArrayType.from_size_and_type(nnz, builtin.f32))
        val_load_op = llvm.LoadOp(val_ptr_arith_op.result, builtin.f32)
        loop_block.add_ops([row_ptr_arith_op, row_load_op, col_ptr_arith_op, col_load_op, val_ptr_arith_op, val_load_op])

        row_i64_op = arith.ExtUIOp(row_load_op.dereferenced_value, builtin.i64)
        row_index_op = builtin.UnrealizedConversionCastOp.get([row_i64_op.result], [IndexType()])
        row_index = row_index_op.outputs[0]
        col_i64_op = arith.ExtUIOp(col_load_op.dereferenced_value, builtin.i64)
        col_index_op = builtin.UnrealizedConversionCastOp.get([col_i64_op.result], [IndexType()])
        col_index = col_index_op.outputs[0]
        loop_block.add_ops([row_i64_op, row_index_op, col_i64_op, col_index_op])

        select_op = dlt.SelectOp(arg_a, [], arg_a_dims, [row_index, col_index])
        set_op = dlt.SetOp(select_op.res, builtin.f32, val_load_op.dereferenced_value)
        loop_block.add_ops([select_op, set_op])

        # print_op = printf.PrintFormatOp("Set {} {} to {} ({}/{})", row_i64_op.result, col_i64_op.result, val_load_op.dereferenced_value, nnz_idx_i64, nnz_ub_i64)
        # loop_block.add_op(print_op)

        loop_block.add_op(scf.Yield())

        loop_op = scf.For(lb, ub, step, [], loop_block)
        setup_a_block.add_ops([loop_op, func.Return()])


        np_ptr_type_val = NpArrayCtype((nnz,))
        np_ptr_type_dim = NpArrayCtype((nnz,), np.int32)
        lib_builder.make_custom_function(name, setup_a_block, [t_var, np_ptr_type_dim, np_ptr_type_dim, np_ptr_type_val])

    @staticmethod
    def _make_setup_func_B(lib_builder: LibBuilder, name: str, t_var: TensorVariable, dim: int):
        setup_a_block = Block(arg_types=[lib_builder.tensor_var_details[t_var], llvm.LLVMPointerType.opaque()])
        arg_a, arg_np_ptr_val = setup_a_block.args
        arg_a_dims = lib_builder.tensor_var_dims[t_var]

        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        setup_a_block.add_ops([zero_op, one_op])

        dim_ub_op = arith.Constant(IntegerAttr(dim, IndexType()))
        setup_a_block.add_ops([dim_ub_op])

        lb = zero_op.result
        step = one_op.result
        ub = dim_ub_op.result

        loop_block = Block(arg_types=[IndexType()])
        dim_index = loop_block.args[0]
        dim_index_cast_op = builtin.UnrealizedConversionCastOp.get([dim_index], [builtin.i64])
        loop_block.add_op(dim_index_cast_op)
        dim_i64 = dim_index_cast_op.outputs[0]

        val_ptr_arith_op = llvm.GEPOp(arg_np_ptr_val, [0, llvm.GEP_USE_SSA_VAL], [dim_i64],
                                      pointee_type=llvm.LLVMArrayType.from_size_and_type(dim, builtin.f32))
        val_load_op = llvm.LoadOp(val_ptr_arith_op.result, builtin.f32)
        loop_block.add_ops(
            [val_ptr_arith_op, val_load_op])

        select_op = dlt.SelectOp(arg_a, [], arg_a_dims, [dim_index])
        set_op = dlt.SetOp(select_op.res, builtin.f32, val_load_op.dereferenced_value)
        loop_block.add_ops([select_op, set_op, scf.Yield()])

        loop_op = scf.For(lb, ub, step, [], loop_block)
        setup_a_block.add_ops([loop_op, func.Return()])

        np_ptr_type = NpArrayCtype((dim,))
        lib_builder.make_custom_function(name, setup_a_block,
                                         [t_var, np_ptr_type])


    @staticmethod
    def _make_check_func_C(lib_builder: LibBuilder, name: str, t_var: TensorVariable, dim: int, epsilon: float):
        setup_a_block = Block(arg_types=[lib_builder.tensor_var_details[t_var], llvm.LLVMPointerType.opaque(), lib_builder.tensor_var_details[t_var]])
        arg_a, arg_np_ptr, first_arg_a = setup_a_block.args
        arg_a_dims = lib_builder.tensor_var_dims[t_var]

        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        f_zero_op = arith.Constant(builtin.FloatAttr(0.0, builtin.f32))
        f_epsilon_op = arith.Constant(builtin.FloatAttr(epsilon, builtin.f32))
        setup_a_block.add_ops([zero_op, one_op, false_op, true_op, f_zero_op, f_epsilon_op])

        ub_op = arith.Constant(IntegerAttr(dim, IndexType()))
        setup_a_block.add_ops([ub_op])

        epsilon_correct = true_op.result
        total_error = f_zero_op.result
        bit_wise_consistent = true_op.result

        lb = zero_op.result
        step = one_op.result
        ub = ub_op.result

        loop_block = Block(arg_types=[IndexType(), IntegerType(1), builtin.f32, IntegerType(1)])
        dim_index = loop_block.args[0]
        select_op = dlt.SelectOp(arg_a, [], arg_a_dims, [dim_index])
        get_op = dlt.GetOp(select_op.res, builtin.f32)
        loop_block.add_ops([select_op, get_op])

        dim_index_cast_op = builtin.UnrealizedConversionCastOp.get([dim_index], [builtin.i64])
        dim_i64 = dim_index_cast_op.outputs[0]

        ptr_arith_op = llvm.GEPOp(arg_np_ptr,
                                  [0, llvm.GEP_USE_SSA_VAL],
                                  [dim_i64],
                                  pointee_type=llvm.LLVMArrayType.from_size_and_type(dim, builtin.f32))
        np_load_op = llvm.LoadOp(ptr_arith_op, builtin.f32)
        loop_block.add_ops([dim_index_cast_op, ptr_arith_op, np_load_op])

        error_op = arith.Subf(get_op.res, np_load_op.dereferenced_value)
        neg_error_op = arith.Negf(error_op.result)
        abs_error_op = arith.Maximumf(error_op.result, neg_error_op.result)
        norm_signed_epsilon_op = arith.Mulf(np_load_op.dereferenced_value, f_epsilon_op.result)
        norm_neg_signed_epsilon_op = arith.Negf(norm_signed_epsilon_op.result)
        norm_epsilon_op = arith.Maximumf(norm_signed_epsilon_op.result, norm_neg_signed_epsilon_op.result)
        new_total_error_op = arith.Addf(loop_block.args[2], abs_error_op.result)
        cmp_error_op = arith.Cmpf(abs_error_op.result, norm_epsilon_op.result, "ogt")
        new_epsilon_correct_op = arith.Select(cmp_error_op, false_op.result, loop_block.args[1])
        if_error_op = scf.If(cmp_error_op, [], [
            printf.PrintFormatOp(
                "Result miss match at {} : reference = {}, result = {}, error = {} > {}",
                dim_i64,
                np_load_op.dereferenced_value,
                get_op.res,
                abs_error_op.result,
                norm_epsilon_op.result,
            ), scf.Yield()])
        loop_block.add_ops([error_op, neg_error_op, abs_error_op, norm_signed_epsilon_op, norm_neg_signed_epsilon_op, norm_epsilon_op, new_total_error_op, cmp_error_op, new_epsilon_correct_op, if_error_op])

        select_first_op = dlt.SelectOp(first_arg_a, [], arg_a_dims, [dim_index])
        get_first_op = dlt.GetOp(select_first_op.res, builtin.f32)
        cmp_first_op = arith.Cmpf(get_op.res, get_first_op.res, "one")
        new_bit_wise_consistent_op = arith.Select(cmp_first_op, false_op.result, loop_block.args[3])
        if_bit_wise_op = scf.If(cmp_first_op, [], [
            printf.PrintFormatOp(
                "Result does not match previous result at {}: first result = {}, this result = {}",
                dim_i64,
                get_first_op.res,
                get_op.res,
            ), scf.Yield()
        ])
        loop_block.add_ops([select_first_op, get_first_op, cmp_first_op, new_bit_wise_consistent_op, if_bit_wise_op])

        loop_block.add_op(scf.Yield(new_epsilon_correct_op.result, new_total_error_op.result, new_bit_wise_consistent_op.result))

        loop_op = scf.For(lb, ub, step, [epsilon_correct, total_error, bit_wise_consistent], loop_block)
        output_epsilon_check_op = arith.ExtUIOp(loop_op.res[0], builtin.i64)
        output_total_error = loop_op.res[1]
        output_bit_consistent_op = arith.ExtUIOp(loop_op.res[2], builtin.i64)
        setup_a_block.add_ops([loop_op,
                               output_epsilon_check_op,
                               output_bit_consistent_op,
                               func.Return(
                                   output_epsilon_check_op,
                                   output_total_error,
                                   output_bit_consistent_op,
                               )])

        np_ptr_type = NpArrayCtype((dim,))
        lib_builder.make_custom_function(name, setup_a_block, [t_var, np_ptr_type, t_var])

    @abc.abstractmethod
    def init_layouts(self) -> str:
        # must define 'a', 'b', 'c'
        raise NotImplementedError

    def get_setup(self) -> str:
        code = self.init_layouts()
        code += "lib.setup_A(a, np_a_row, np_a_col, np_a_val)\n"
        code += "lib.setup_B(b, np_b)\n"
        return code

    def get_benchmark(self) -> str:
        return "lib.spmvmul(c, a, b)"

    def get_test(self) -> str:
        # must define 'correct', 'total_error', 'consistent' in the scope
        code =  "results = lib.check_C(c, np_c, f_c)\n"
        code += "correct = bool(results[0].value)\n"
        code += "total_error = float(results[1].value)\n"
        code += "consistent = bool(results[2].value)\n"
        return code


class SCircuit(SparseSuite):

    def __init__(
        self,
        seed: int,
        base_dir: str,
        name: str,
        runs: int,
        repeats: int,
        opt_level: int,
        epsilon: float,
    ):
        super().__init__(
            "sparse_suite/scircuit/scircuit.mtx",
            "sparse_suite/scircuit/scircuit_b.mtx",
            seed,
            base_dir,
            "scircuit_"+name,
            runs,
            repeats,
            opt_level,
            epsilon,
            benchmark_child_process=True,
            benchmark_timeout=300,
            waste_of_time_threshold= 20,
        )


    def construct_lib_builder(
        self,
        lib_builder: LibBuilder,
        a: TensorVariable,
        b: TensorVariable,
        c: TensorVariable,
    ):
        lib_builder.make_init("init_A", (a), [], free_name="dealloc_A")
        lib_builder.make_init("init_B", (b), [], free_name="dealloc_B")
        lib_builder.make_init("init_C", (c), [], free_name="dealloc_C")

    def get_configs_for_tensors(
        self, a: TensorVariable, b: TensorVariable, c: TensorVariable
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        return {
            a: ReifyConfig(coo_buffer_options=frozenset([0]), dense=True, coo_minimum_dims=1),
            b: ReifyConfig(coo_buffer_options=frozenset([0]), dense=True, coo_minimum_dims=1),
            c: ReifyConfig(dense=True, coo_minimum_dims=1),
        }

    def skip_layout(self, l: PtrMapping) -> bool:
        a_layout = l.make_ptr_dict()[StringAttr("R_0")]
        if isinstance(a_layout.layout, dlt.DenseLayoutAttr):
            if isinstance(a_layout.layout.child, dlt.DenseLayoutAttr):
                return True
        return False

    def skip_order(self, o: IterationMapping) -> bool:
         return all(isinstance(i, dlt.NestedIterationOrderAttr) for i in o.values)

    def init_layouts(self) -> str:
        code =  "r_a, (a) = lib.init_A()\n"
        code += "r_b, (b) = lib.init_B()\n"
        code += "r_c, (c) = lib.init_C()\n"
        return code

    def get_clean(self) -> str:
        code =  "lib.dealloc_A(r_a)\n"
        code += "lib.dealloc_B(r_b)\n"
        code += "lib.dealloc_C(r_c)\n"
        return code



if __name__ == "__main__":

    repeats = 1
    runs = 1
    benchmarks = []

    print(f"Args: {sys.argv}")

    # if len(sys.argv) == 1 or "1" in sys.argv:
    #     benchmarks.append(StaticTriple(128,128,128,True, 0, "./results", repeats=repeats, runs=runs, opt_level=3, epsilon=_Epsilon))
    if len(sys.argv) == 1 or "scircuit" in sys.argv:
        benchmarks.append(
            SCircuit(
                0,
                "./results",
                "",
                repeats=repeats,
                runs=runs,
                opt_level=3,
                epsilon=_Epsilon,
            )
        )

    benchmark_options = [a for a in sys.argv if a.startswith("-")]
    for benchmark in benchmarks:
        benchmark.run(benchmark_options)

    if "run" in sys.argv:
        for benchmark in benchmarks:
            benchmark.run()