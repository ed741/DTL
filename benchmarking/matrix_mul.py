import sys
from random import Random

import numpy as np

from benchmarking.benchmark import Benchmark
from dtl import *
from dtl.dag import RealVectorSpace, Index

from dtl.libBuilder import LibBuilder, NpArrayCtype, StructType, TupleStruct
from xdsl.dialects import arith, builtin, func, llvm, printf, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
from xdsl.dialects.experimental import dlt
from xdsl.ir import Block
from xdsl.transforms.experimental.dlt.generate_dlt_layouts import ReifyConfig

_Epsilon = 0.00001

_Args = tuple[Any, StructType, StructType, StructType]


class MatMul(Benchmark, abc.ABC):
    def __init__(
        self,
        i: int,
        j: int,
        k: int,
        use_scope_vars: bool,
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
        var_name = "scope" if use_scope_vars else "static"
        new_base_dir = f"{base_dir}/matmul"
        results_base_dir = (
            f"{new_base_dir}/{name}_{var_name}_O{opt_num}_{i}.{j}.{k}_{seed}_{runs}"
        )

        self.i, self.j, self.k = i, j, k
        self.seed = seed
        self.use_scope_vars = use_scope_vars
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

        np_a, np_b = self.make_a_b(r)

        # print("generating np c")
        np_c = np.matmul(np_a, np_b, dtype=np.float32, casting="no")
        # np_d = np.zeros_like(np_c)
        # for i_i in range(self.i):
        #     for i_k in range(self.k):
        #         for i_j in range(self.j):
        #             _a = np_a[i_i, i_j]
        #             _b = np_b[i_j, i_k]
        #             prod = _a * _b
        #             _d = np_d[i_i, i_k]
        #             sum = _d + prod
        #             np_d[i_i, i_k] = sum
        # self.np_d = np_d

        self.np_a = np_a
        self.np_b = np_b
        self.np_c = np_c

        self.tensor_variables = None

        self.handle_reference_array(np_a, "np_a"+np_suffix, True, False, "np_a")
        self.handle_reference_array(np_b, "np_b"+np_suffix, True, False, "np_b")
        self.handle_reference_array(np_c, "np_c"+np_suffix, False, True, "np_c")

    def make_a_b(self, r: Random) -> tuple[np.ndarray, np.ndarray]:
        np_a = np.zeros((self.i, self.j), dtype=np.float32)
        np_b = np.zeros((self.j, self.k), dtype=np.float32)
        for i_i in range(self.i):
            for i_j in range(self.j):
                num = r.random()
                np_a[i_i, i_j] = num
        for i_j in range(self.j):
            for i_k in range(self.k):
                num = r.random()
                np_b[i_j, i_k] = num
        return np_a, np_b

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
        if self.use_scope_vars:
            vi = UnknownSizeVectorSpace("vi")
            vj = UnknownSizeVectorSpace("vj")
            vk = UnknownSizeVectorSpace("vk")
            scope_var_map = {vi: self.i, vj: self.j, vk: self.k}
        else:
            vi = RealVectorSpace(self.i)
            vj = RealVectorSpace(self.j)
            vk = RealVectorSpace(self.k)
            scope_var_map = {}
        A = TensorVariable(vi * vj, "A")
        B = TensorVariable(vj * vk, "B")
        C = TensorVariable(vi * vk, "C")

        self.tensor_variables = (A, B, C)

        _i = Index("i")
        _j = Index("j")
        _k = Index("k")
        matmul = (A[_i, _j] * B[_j, _k]).sum(_j).forall(_i, _k)

        lib_builder = LibBuilder(scope_var_map)
        self.construct_lib_builder(lib_builder, A, B, C)

        lib_builder.make_setter("set_A", (A), {}, [0, 1])
        lib_builder.make_setter("set_B", (B), {}, [0, 1])

        lib_builder.make_getter("get_C", (C), {}, [0, 1])
        lib_builder.make_function("matmul", matmul, [C], [A, B], [], [])

        self._make_setup_func(lib_builder, "setup_A", A, self.i, self.j)
        self._make_setup_func(lib_builder, "setup_B", B, self.j, self.k)
        self._make_check_func(lib_builder, "check_C", C, self.i, self.k, self.epsilon)
        return lib_builder

    @staticmethod
    def _make_setup_func(lib_builder: LibBuilder, name: str, t_var: TensorVariable, dim_1: int, dim_2: int):
        setup_a_block = Block(arg_types=[lib_builder.tensor_var_details[t_var], llvm.LLVMPointerType.opaque()])
        arg_a, arg_np_ptr = setup_a_block.args
        arg_a_dims = lib_builder.tensor_var_dims[t_var]

        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        setup_a_block.add_ops([zero_op, one_op])

        i_ub_op = arith.Constant(IntegerAttr(dim_1, IndexType()))
        j_ub_op = arith.Constant(IntegerAttr(dim_2, IndexType()))
        setup_a_block.add_ops([i_ub_op, j_ub_op])

        lb = zero_op.result
        step = one_op.result
        ub_i = i_ub_op.result
        ub_j = j_ub_op.result

        i_loop_block = Block(arg_types=[IndexType()])
        j_loop_block = Block(arg_types=[IndexType()])
        select_op = dlt.SelectOp(arg_a, [], arg_a_dims, [i_loop_block.args[0], j_loop_block.args[0]])
        i_index_cast_op = builtin.UnrealizedConversionCastOp.get([i_loop_block.args[0]], [builtin.i64])
        j_index_cast_op = builtin.UnrealizedConversionCastOp.get([j_loop_block.args[0]], [builtin.i64])
        ptr_arith_op = llvm.GEPOp(arg_np_ptr,
                                  [0, llvm.GEP_USE_SSA_VAL, llvm.GEP_USE_SSA_VAL],
                                  [i_index_cast_op.outputs[0], j_index_cast_op.outputs[0]],
                                  pointee_type=llvm.LLVMArrayType.from_size_and_type(dim_1,
                                                                                     llvm.LLVMArrayType.from_size_and_type(
                                                                                         dim_2, builtin.f32)))
        load_op = llvm.LoadOp(ptr_arith_op, builtin.f32)
        set_op = dlt.SetOp(select_op.res, builtin.f32, load_op.dereferenced_value)
        j_loop_block.add_ops([select_op, i_index_cast_op, j_index_cast_op, ptr_arith_op, load_op, set_op, scf.Yield()])
        j_loop_op = scf.For(lb, ub_j, step, [], j_loop_block)
        i_loop_block.add_ops([j_loop_op, scf.Yield()])
        i_loop_op = scf.For(lb, ub_i, step, [], i_loop_block)
        setup_a_block.add_ops([i_loop_op, func.Return()])

        np_ptr_type = NpArrayCtype((dim_1, dim_2))
        lib_builder.make_custom_function(name, setup_a_block, [t_var, np_ptr_type])

    @staticmethod
    def _make_check_func(lib_builder: LibBuilder, name: str, t_var: TensorVariable, dim_1: int, dim_2: int, epsilon: float):
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

        i_ub_op = arith.Constant(IntegerAttr(dim_1, IndexType()))
        j_ub_op = arith.Constant(IntegerAttr(dim_2, IndexType()))
        setup_a_block.add_ops([i_ub_op, j_ub_op])

        epsilon_correct = true_op.result
        total_error = f_zero_op.result
        bit_wise_consistent = true_op.result

        lb = zero_op.result
        step = one_op.result
        ub_i = i_ub_op.result
        ub_j = j_ub_op.result

        i_loop_block = Block(arg_types=[IndexType(), IntegerType(1), builtin.f32, IntegerType(1)])
        j_loop_block = Block(arg_types=[IndexType(), IntegerType(1), builtin.f32, IntegerType(1)])
        select_op = dlt.SelectOp(arg_a, [], arg_a_dims, [i_loop_block.args[0], j_loop_block.args[0]])
        get_op = dlt.GetOp(select_op.res, builtin.f32)
        j_loop_block.add_ops([select_op, get_op])

        i_index_cast_op = builtin.UnrealizedConversionCastOp.get([i_loop_block.args[0]], [builtin.i64])
        j_index_cast_op = builtin.UnrealizedConversionCastOp.get([j_loop_block.args[0]], [builtin.i64])
        ptr_arith_op = llvm.GEPOp(arg_np_ptr,
                                  [0, llvm.GEP_USE_SSA_VAL, llvm.GEP_USE_SSA_VAL],
                                  [i_index_cast_op.outputs[0], j_index_cast_op.outputs[0]],
                                  pointee_type=llvm.LLVMArrayType.from_size_and_type(dim_1,
                                                                                     llvm.LLVMArrayType.from_size_and_type(
                                                                                         dim_2, builtin.f32)))
        np_load_op = llvm.LoadOp(ptr_arith_op, builtin.f32)
        j_loop_block.add_ops([i_index_cast_op, j_index_cast_op, ptr_arith_op, np_load_op])

        error_op = arith.Subf(get_op.res, np_load_op.dereferenced_value)
        neg_error_op = arith.Negf(error_op.result)
        abs_error_op = arith.Maximumf(error_op.result, neg_error_op.result)
        norm_epsilon_op = arith.Mulf(np_load_op.dereferenced_value, f_epsilon_op.result)
        new_total_error_op = arith.Addf(j_loop_block.args[2], abs_error_op.result)
        cmp_error_op = arith.Cmpf(abs_error_op.result, norm_epsilon_op.result, "ogt")
        new_epsilon_correct_op = arith.Select(cmp_error_op, false_op.result, j_loop_block.args[1])
        if_error_op = scf.If(cmp_error_op, [], [
            printf.PrintFormatOp(
                "Result miss match at {}, {}: reference = {}, result = {}, error = {} >= {}",
                i_index_cast_op.outputs[0],
                j_index_cast_op.outputs[0],
                np_load_op.dereferenced_value,
                get_op.res,
                abs_error_op.result,
                norm_epsilon_op.result,
            ), scf.Yield()])
        j_loop_block.add_ops([error_op, neg_error_op, abs_error_op, norm_epsilon_op, new_total_error_op, cmp_error_op, new_epsilon_correct_op, if_error_op])

        select_first_op = dlt.SelectOp(first_arg_a, [], arg_a_dims, [i_loop_block.args[0], j_loop_block.args[0]])
        get_first_op = dlt.GetOp(select_first_op.res, builtin.f32)
        cmp_first_op = arith.Cmpf(get_op.res, get_first_op.res, "one")
        new_bit_wise_consistent_op = arith.Select(cmp_first_op, false_op.result, j_loop_block.args[3])
        if_bit_wise_op = scf.If(cmp_first_op, [], [
            printf.PrintFormatOp(
                "Result does not match previous result at {}, {}: first result = {}, this result = {}",
                i_index_cast_op.outputs[0],
                j_index_cast_op.outputs[0],
                get_first_op.res,
                get_op.res,
            ), scf.Yield()
        ])
        j_loop_block.add_ops([select_first_op, get_first_op, cmp_first_op, new_bit_wise_consistent_op, if_bit_wise_op])

        j_loop_block.add_op(scf.Yield(new_epsilon_correct_op.result, new_total_error_op.result, new_bit_wise_consistent_op.result))

        j_loop_op = scf.For(lb, ub_j, step, i_loop_block.args[1:], j_loop_block)
        i_loop_block.add_ops([j_loop_op, scf.Yield(*j_loop_op.res)])
        i_loop_op = scf.For(lb, ub_i, step, [epsilon_correct, total_error, bit_wise_consistent], i_loop_block)
        output_epsilon_check_op = arith.ExtUIOp(i_loop_op.res[0], builtin.i64)
        output_total_error = i_loop_op.res[1]
        output_bit_consistent_op = arith.ExtUIOp(i_loop_op.res[2], builtin.i64)
        setup_a_block.add_ops([i_loop_op,
                               output_epsilon_check_op,
                               output_bit_consistent_op,
                               func.Return(
                                   output_epsilon_check_op,
                                   output_total_error,
                                   output_bit_consistent_op,
                               )])

        np_ptr_type = NpArrayCtype((dim_1, dim_2))
        lib_builder.make_custom_function(name, setup_a_block, [t_var, np_ptr_type, t_var])

    @abc.abstractmethod
    def init_layouts(self) -> str:
        # must define 'a', 'b', 'c'
        raise NotImplementedError

    def get_setup(self) -> str:
        code = self.init_layouts()
        code += "lib.setup_A(a, np_a)\n"
        code += "lib.setup_B(b, np_b)\n"
        return code

    def get_benchmark(self) -> str:
        return "lib.matmul(c, a, b)"

    def get_test(self) -> str:
        # must define 'correct', 'total_error', 'consistent' in the scope
        code =  "results = lib.check_C(c, np_c, f_c)\n"
        code += "correct = bool(results[0].value)\n"
        code += "total_error = float(results[1].value)\n"
        code += "consistent = bool(results[2].value)\n"
        return code


class StaticTriple(MatMul):

    def __init__(
        self,
        i: int,
        j: int,
        k: int,
        use_scope_vars: bool,
        seed: int,
        base_dir: str,
        name: str,
        runs: int,
        repeats: int,
        opt_level: int,
        epsilon: float,
    ):
        name = f"triple_{name}"
        super().__init__(
            i,
            j,
            k,
            use_scope_vars,
            seed,
            base_dir,
            name,
            runs,
            repeats,
            opt_level,
            epsilon,
        )

    def construct_lib_builder(
        self,
        lib_builder: LibBuilder,
        a: TensorVariable,
        b: TensorVariable,
        c: TensorVariable,
    ):
        lib_builder.make_init("init", (a, b, c), [], free_name="dealloc")

    def get_configs_for_tensors(
        self, a: TensorVariable, b: TensorVariable, c: TensorVariable
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        return {(a, b, c): ReifyConfig()}

    def init_layouts(self) -> str:
        return "root, (a, b, c) = lib.init()"

    def get_clean(self) -> str:
        return "lib.dealloc(root)"


class StaticPair(MatMul):

    def __init__(
        self,
        i: int,
        j: int,
        k: int,
        use_scope_vars: bool,
        seed: int,
        base_dir: str,
        name: str,
        runs: int,
        repeats: int,
        opt_level: int,
        epsilon: float,
    ):
        name = f"pair_{name}"
        super().__init__(
            i,
            j,
            k,
            use_scope_vars,
            seed,
            base_dir,
            name,
            runs,
            repeats,
            opt_level,
            epsilon,
        )

    def construct_lib_builder(
        self,
        lib_builder: LibBuilder,
        a: TensorVariable,
        b: TensorVariable,
        c: TensorVariable,
    ):
        lib_builder.make_init("init_AB", (a, b), [], free_name="dealloc_AB")
        lib_builder.make_init("init_C", (c), [], free_name="dealloc_C")

    def get_configs_for_tensors(
        self, a: TensorVariable, b: TensorVariable, c: TensorVariable
    ) -> dict[TupleStruct[TensorVariable], ReifyConfig]:
        return {
            (a, b): ReifyConfig(coo_buffer_options=frozenset([0])),
            c: ReifyConfig(),
        }

    def init_layouts(self) -> str:
        code =  "r_ab, (a, b) = lib.init_AB()\n"
        code += "r_c, (c) = lib.init_C()\n"
        return code

    def get_clean(self) -> str:
        code =  "lib.dealloc_AB(r_ab)\n"
        code += "lib.dealloc_C(r_c)\n"
        return code


class StaticSingles(MatMul):

    def __init__(
        self,
        i: int,
        j: int,
        k: int,
        use_scope_vars: bool,
        seed: int,
        base_dir: str,
        name: str,
        runs: int,
        repeats: int,
        opt_level: int,
        epsilon: float,
        **kwargs,
    ):
        name = f"singles_{name}"
        super().__init__(
            i,
            j,
            k,
            use_scope_vars,
            seed,
            base_dir,
            name,
            runs,
            repeats,
            opt_level,
            epsilon,
            **kwargs,
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
            a: ReifyConfig(coo_buffer_options=frozenset([0])),
            b: ReifyConfig(coo_buffer_options=frozenset([0])),
            c: ReifyConfig(),
        }

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


class RandomSparseSingles(StaticSingles):
    def __init__(
        self,
        i: int,
        j: int,
        k: int,
        use_scope_vars: bool,
        seed: int,
        rate_a: float,
        rate_b: float,
        base_dir: str,
        name: str,
        runs: int,
        repeats: int,
        opt_level: int,
        epsilon: float,
        **kwargs,
    ):
        name = f"randomSparse_{name}"
        self.rate_a = rate_a
        self.rate_b = rate_b
        super().__init__(
            i,
            j,
            k,
            use_scope_vars,
            seed,
            base_dir,
            name,
            runs,
            repeats,
            opt_level,
            epsilon,
            np_suffix = f"_{rate_a}_{rate_b}_",
            **kwargs,
        )

    def make_a_b(self, r: Random) -> tuple[np.ndarray, np.ndarray]:
        np_a = np.zeros((self.i, self.j), dtype=np.float32)
        np_b = np.zeros((self.j, self.k), dtype=np.float32)
        for i_i in range(self.i):
            for i_j in range(self.j):
                if r.random() < self.rate_a:
                    num = r.random()
                    np_a[i_i, i_j] = num
        for i_j in range(self.j):
            for i_k in range(self.k):
                if r.random() < self.rate_b:
                    num = r.random()
                    np_b[i_j, i_k] = num
        return np_a, np_b

    def get_test_id_from_row(self, row) -> tuple[tuple, int, tuple[int, float]]:
        layout_num = int(row[0])
        order_num = int(row[1])
        rate_a = float(row[2])
        rate_b = float(row[3])
        rep = int(row[4])
        runs = int(row[5])
        time = float(row[6])
        correct = row[7] == "True"
        mean_error = float(row[8])
        consistent = row[9] == "True"
        waiting_time = float(row[10])
        finished = row[11] == "True"
        return (layout_num, order_num, rate_a, rate_b), rep, (runs, time)

    def get_test_id(self, layout_num, order_num) -> tuple:
        return (layout_num, order_num, self.rate_a, self.rate_b)

    def get_results_header(self):
        return [
            "layout_mapping",
            "iter_mapping",
            "rate_a",
            "rate_b",
            "rep",
            "runs",
            "time",
            "correct",
            "mean_error",
            "consistent",
            "waiting_time",
            "finished"
        ]


class RowSparseSingles(StaticSingles):
    def __init__(
        self,
        i: int,
        j: int,
        k: int,
        use_scope_vars: bool,
        seed: int,
        rate_a: float,
        rate_b: float,
        base_dir: str,
        name: str,
        runs: int,
        repeats: int,
        opt_level: int,
        epsilon: float,
    ):
        name = f"randomSparse_{rate_a}_{rate_b}_{name}"
        self.rate_a = rate_a
        self.rate_b = rate_b
        super().__init__(
            i,
            j,
            k,
            use_scope_vars,
            seed,
            base_dir,
            name,
            runs,
            repeats,
            opt_level,
            epsilon,
        )

    def sparsify(self, np_a, np_b, r: Random) -> tuple[np.ndarray, np.ndarray]:
        for i_i in range(np_a.shape[0]):
            if r.random() < self.rate_a:
                for i_j in range(np_a.shape[1]):
                    np_a[i_i, i_j] = 0
        for i_j in range(np_b.shape[0]):
            if r.random() < self.rate_b:
                for i_k in range(np_b.shape[1]):
                    np_b[i_j, i_k] = 0
        return np_a, np_b


if __name__ == "__main__":

    repeats = 5
    runs = 10
    benchmarks = []

    print(f"Args: {sys.argv}")

    # if len(sys.argv) == 1 or "1" in sys.argv:
    #     benchmarks.append(StaticTriple(128,128,128,True, 0, "./results", repeats=repeats, runs=runs, opt_level=3, epsilon=_Epsilon))
    if len(sys.argv) == 1 or "2" in sys.argv:
        benchmarks.append(
            StaticPair(
                128,
                128,
                128,
                True,
                0,
                "./results",
                "",
                repeats=repeats,
                runs=runs,
                opt_level=3,
                epsilon=_Epsilon,
            )
        )
    if len(sys.argv) == 1 or "3" in sys.argv:
        benchmarks.append(
            StaticSingles(
                128,
                128,
                128,
                True,
                0,
                "./results",
                "",
                repeats=repeats,
                runs=runs,
                opt_level=3,
                epsilon=_Epsilon,
            )
        )
    # if len(sys.argv) == 1 or "4" in sys.argv:
    #     benchmarks.append(StaticTriple(8, 8, 8, True, 0, "./results", repeats=repeats, runs=runs, opt_level=3, epsilon=_Epsilon))
    if len(sys.argv) == 1 or "5" in sys.argv:
        benchmarks.append(
            StaticPair(
                8,
                8,
                8,
                True,
                0,
                "./results",
                "",
                repeats=repeats,
                runs=runs,
                opt_level=3,
                epsilon=_Epsilon,
            )
        )
    if len(sys.argv) == 1 or "6" in sys.argv:
        benchmarks.append(
            StaticSingles(
                8,
                8,
                8,
                True,
                0,
                "./results",
                "",
                repeats=repeats,
                runs=runs,
                opt_level=3,
                epsilon=_Epsilon,
            )
        )

    # if len(sys.argv) == 1 or "7" in sys.argv:
    #     benchmarks.append(StaticTriple(128, 128, 128, False, 0, "./results", repeats=repeats, runs=runs, opt_level=3, epsilon=_Epsilon))
    if len(sys.argv) == 1 or "8" in sys.argv:
        benchmarks.append(
            StaticPair(
                128,
                128,
                128,
                False,
                0,
                "./results",
                "",
                repeats=repeats,
                runs=runs,
                opt_level=3,
                epsilon=_Epsilon,
            )
        )
    if len(sys.argv) == 1 or "9" in sys.argv:
        benchmarks.append(
            StaticSingles(
                128,
                128,
                128,
                False,
                0,
                "./results",
                "",
                repeats=repeats,
                runs=runs,
                opt_level=3,
                epsilon=_Epsilon,
            )
        )
    # if len(sys.argv) == 1 or "10" in sys.argv:
    #     benchmarks.append(StaticTriple(8, 8, 8, False, 0, "./results", repeats=repeats, runs=runs, opt_level=3, epsilon=_Epsilon))
    if len(sys.argv) == 1 or "11" in sys.argv:
        benchmarks.append(
            StaticPair(
                8,
                8,
                8,
                False,
                0,
                "./results",
                "",
                repeats=repeats,
                runs=runs,
                opt_level=3,
                epsilon=_Epsilon,
            )
        )
    if len(sys.argv) == 1 or "12" in sys.argv:
        benchmarks.append(
            StaticSingles(
                8,
                8,
                8,
                False,
                0,
                "./results",
                "",
                repeats=repeats,
                runs=runs,
                opt_level=3,
                epsilon=_Epsilon,
            )
        )

    for rate in ["0.1", "0.01", "0.001", "0.0001", "0.00001"]:
        if len(sys.argv) == 1 or f"13-{rate}" in sys.argv:
            benchmarks.append(
                RandomSparseSingles(
                    1024,
                    1024,
                    1024,
                    False,
                    0,
                    float(rate),
                    float(rate),
                    "./results",
                    "",
                    repeats=repeats,
                    runs=runs,
                    opt_level=3,
                    epsilon=_Epsilon,
                    waste_of_time_threshold = 2.0,
                    test_too_short_threshold = 0.01,
                    long_run_multiplier = 10,
                    benchmark_timeout = 5.0,
                )
            )

    # benchmarks.append(
    #     RandomSparseSingles(1024, 1024, 1024, False, 0, 1, 1,"./results", "", repeats=repeats, runs=runs, opt_level=3, epsilon=_Epsilon))

    for benchmark in benchmarks:
        benchmark.skip_testing = "skip_testing" in sys.argv
        benchmark.only_compile_to_llvm = "only_to_llvm" in sys.argv
        benchmark.do_not_compile_mlir = "no_mlir" in sys.argv
        benchmark.do_not_lower = "do_not_lower" in sys.argv
        # benchmark.take_first_layouts = 5
        # benchmark.take_first_orders = 5
        benchmark.run()
        benchmark.skip_testing = False
        benchmark.only_compile_to_llvm = False
        benchmark.do_not_compile_mlir = False
        benchmark.do_not_lower = False

    if len(sys.argv) == 1 or "run" in sys.argv:
        for benchmark in benchmarks:
            benchmark.run()
