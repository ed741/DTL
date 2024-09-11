import sys
from random import Random

import numpy as np

from benchmarking.benchmark import Benchmark
from dtl import *
from dtl.dag import RealVectorSpace, Index

from dtl.libBuilder import DTLCLib, LibBuilder, StructType, TupleStruct
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

        self.handle_reference_array(np_a, "np_a")
        self.handle_reference_array(np_b, "np_b")
        self.handle_reference_array(np_c, "np_c")

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
        lib_builder.make_function("copy_A", A, [A], [A], [], [])
        lib_builder.make_function("copy_B", B, [B], [B], [], [])

        return lib_builder

    @abc.abstractmethod
    def init_layouts(
        self, lib: DTLCLib
    ) -> tuple[Any, StructType, StructType, StructType]:
        raise NotImplementedError

    def setup(self, lib: DTLCLib, first_args: _Args = None) -> _Args:
        root, a, b, c = self.init_layouts(lib)

        if first_args is None:
        # print("set a & b")
            for i_i in range(self.np_a.shape[0]):
                for i_j in range(self.np_a.shape[1]):
                    if self.np_a[i_i, i_j] != 0:
                        lib.set_A(a, i_i, i_j, self.np_a[i_i, i_j])
            for i_j in range(self.np_b.shape[0]):
                for i_k in range(self.np_b.shape[1]):
                    if self.np_b[i_j, i_k] != 0:
                        lib.set_B(b, i_j, i_k, self.np_b[i_j, i_k])

            matmul_func = (
                lib.matmul
            )  # force the runtime to load and prepare the function (so it's cached)
        else:
            f_root, f_a, f_b, f_c = first_args
            lib.copy_A(a, f_a)
            lib.copy_B(b, f_b)
        return root, a, b, c

    def get_benchmark(self, lib: DTLCLib) -> typing.Callable[[_Args], None]:
        def benchmark(args: _Args) -> None:
            root, a, b, c = args
            lib.matmul(c, a, b)

        return benchmark

    def test(
        self, lib: DTLCLib, args: _Args, first_args: _Args
    ) -> tuple[bool, float, bool]:  # (within_epsilon, mean_error, bit_wise_reapeatable)
        root, a, b, c = args
        f_root, f_a, f_b, f_c = first_args
        within_epsilon = True
        bitwise_consistency = True

        total_error = 0
        for i_i in range(self.np_c.shape[0]):
            for i_k in range(self.np_c.shape[1]):
                np_num = self.np_c[i_i, i_k]
                res = lib.get_C(c, i_i, i_k).value
                error = abs(res - np_num)
                total_error += error
                normalised_epsilon = np_num * self.epsilon
                if error > normalised_epsilon:
                    print(
                        f"Result miss match! at i: {i_i}, k: {i_k}, np_c = {np_num}, c = {res}, error = {res - np_num}, epsilon(abs) = {self.epsilon}, epsilon(norm) = {normalised_epsilon}"
                    )
                    within_epsilon = False
                first_res = lib.get_C(f_c, i_i, i_k).value
                if res != first_res:
                    print(
                        f"Result does not match previous result at i: {i_i}, k: {i_k}. first result: {first_res}, this result: {res}"
                    )
                    bitwise_consistency = False

        return within_epsilon, total_error, bitwise_consistency


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

    def init_layouts(
        self, lib: DTLCLib
    ) -> tuple[Any, StructType, StructType, StructType]:
        root, (a, b, c) = lib.init()
        return root, a, b, c

    def teardown(self, lib: DTLCLib, args: _Args) -> None:
        root, a, b, c = args
        lib.dealloc(root)


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

    def init_layouts(
        self, lib: DTLCLib
    ) -> tuple[Any, StructType, StructType, StructType]:
        root_ab, (a, b) = lib.init_AB()
        root_c, (c) = lib.init_C()
        return (root_ab, root_c), a, b, c

    def teardown(self, lib: DTLCLib, args: _Args) -> None:
        (root_ab, root_c), a, b, c = args
        lib.dealloc_AB(root_ab)
        lib.dealloc_C(root_c)


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

    def init_layouts(
        self, lib: DTLCLib
    ) -> tuple[Any, StructType, StructType, StructType]:
        root_a, (a) = lib.init_A()
        root_b, (b) = lib.init_B()
        root_c, (c) = lib.init_C()
        return (root_a, root_b, root_c), a, b, c

    def teardown(self, lib: DTLCLib, args: _Args) -> None:
        (root_a, root_b, root_c), a, b, c = args
        lib.dealloc_A(root_a)
        lib.dealloc_B(root_b)
        lib.dealloc_C(root_c)


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
        within_epsilon = row[7] == "True"
        per_run_error = float(row[8])
        bit_repeatable = row[9] == "True"
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
            "within_epsilon",
            "per_run_error",
            "bit_repeatable",
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
                    waste_of_time_threshold = 5.0,
                    test_too_short_threshold = 0.01,
                    long_run_multiplier = 10,
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
