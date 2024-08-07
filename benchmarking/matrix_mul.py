# import xdsl.dialects.arith
import csv
import ctypes
import os
import pickle
import time
import timeit
from io import StringIO
from random import Random

import numpy as np

from benchmarking.benchmark import Benchmark
from dtl import *
from dtl.dag import RealVectorSpace, Index
import dtlpp.backends.xdsl as xdtl

from dtl.libBuilder import DTLCLib, LibBuilder, StructType
from dtlpp.backends import native
from xdsl.dialects import arith, scf, func
from xdsl.dialects.builtin import ModuleOp, f32, IntegerAttr, IntegerType
from xdsl.dialects.experimental import dlt
from xdsl.ir import Block, Region
from xdsl.printer import Printer
from xdsl.transforms.experimental.dlt.generate_dlt_iteration_orders import IterationGenerator
from xdsl.transforms.experimental.dlt.generate_dlt_layouts import LayoutGenerator
from xdsl.transforms.experimental.dlt.iteration_map import IterationMap
from xdsl.transforms.experimental.dlt.layout_graph import LayoutGraph

_Epsilon = 0.00001

_Args = tuple[Any, StructType, StructType, StructType]


class MatMul(Benchmark, abc.ABC):
    def __init__(self, i: int, j: int, k: int, seed: int, base_dir: str, name: str, runs: int, repeats: int, epsilon: float):
        base_dir = f"{base_dir}/matmul_{name}_{i}.{j}.{k}_{seed}_{runs}"
        self.i, self.j, self.k = i, j, k
        self.seed = seed
        super().__init__(base_dir, runs, repeats, epsilon)

        # print("initing np a & b")
        np_a = np.zeros((i, j), dtype=np.float32)
        np_b = np.zeros((j, k), dtype=np.float32)

        # print("setting random values in np a & b")
        r = Random(seed)
        for i_i in range(i):
            for i_j in range(j):
                num = r.random()
                np_a[i_i, i_j] = num
        for i_j in range(j):
            for i_k in range(k):
                num = r.random()
                np_b[i_j, i_k] = num

        # print("generating np c")
        np_c = np.matmul(np_a, np_b)

        self.np_a = np_a
        self.np_b = np_b
        self.np_c = np_c


    @abc.abstractmethod
    def construct_lib_builder(self, lib_builder: LibBuilder, a: TensorVariable, b: TensorVariable, c: TensorVariable):
        raise NotImplementedError

    def define_lib_builder(self) -> LibBuilder:
        vi = UnknownSizeVectorSpace("vi")
        vj = UnknownSizeVectorSpace("vj")
        vk = UnknownSizeVectorSpace("vk")
        A = TensorVariable(vi * vj, "A")
        B = TensorVariable(vj * vk, "B")
        C = TensorVariable(vi * vk, "C")

        _i = Index('i')
        _j = Index('j')
        _k = Index('k')
        matmul = (A[_i, _j] * B[_j, _k]).sum(_j).forall(_i, _k)

        lib_builder = LibBuilder({vi: self.i, vj: self.j, vk: self.k})
        lib_builder.make_setter("set_A", (A), {}, [0, 1])
        lib_builder.make_setter("set_B", (B), {}, [0, 1])

        lib_builder.make_getter("get_C", (C), {}, [0, 1])
        lib_builder.make_function("matmul", matmul, [C], [A, B], [], [])
        self.construct_lib_builder(lib_builder, A, B, C)

        return lib_builder


    @abc.abstractmethod
    def init_layouts(self, lib: DTLCLib) -> tuple[Any, StructType, StructType, StructType]:
        raise NotImplementedError

    def setup(self, lib: DTLCLib) -> _Args:
        root, a, b, c = self.init_layouts(lib)

        # print("set a & b")
        for i_i in range(self.np_a.shape[0]):
            for i_j in range(self.np_a.shape[1]):
                lib.set_A(a, i_i, i_j, self.np_a[i_i, i_j])
        for i_j in range(self.np_b.shape[0]):
            for i_k in range(self.np_b.shape[1]):
                lib.set_B(b, i_j, i_k, self.np_b[i_j, i_k])
        return root, a, b, c

    def get_benchmark(self, lib: DTLCLib) -> typing.Callable[[_Args], None]:
        def benchmark(args: _Args) -> None:
            root, a, b, c = args
            lib.matmul(c, a, b)

        return benchmark

    def test(self, lib: DTLCLib, args: _Args, first_args: _Args) -> tuple[bool, float, bool]: # (within_epsilon, mean_error, bit_wise_reapeatable)
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
                if error > self.epsilon:
                    print(
                        f"Result miss match! at i: {i_i}, k: {i_k}, np_c = {np_num}, c = {res}, error = {res - np_num}")
                    within_epsilon = False
                first_res = lib.get_C(f_c, i_i, i_k).value
                if res != first_res:
                    print(
                        f"Result does not match previous result at i: {i_i}, k: {i_k}. first result: {first_res}, this result: {res}")
                    bitwise_consistency = False

        return within_epsilon, total_error, bitwise_consistency


class StaticTriple(MatMul):

    def __init__(self, i: int, j: int, k: int, seed: int, base_dir: str, runs: int, repeats: int, epsilon: float):
        name = "static_triple"
        super().__init__(i, j, k, seed, base_dir, name, runs, repeats, epsilon)

    def construct_lib_builder(self, lib_builder: LibBuilder, a: TensorVariable, b: TensorVariable, c: TensorVariable):
        lib_builder.make_init("init", (a, b, c), [], free_name="dealloc")


    def init_layouts(self, lib: DTLCLib) -> tuple[Any, StructType, StructType, StructType]:
        root, (a, b, c) = lib.init()
        return root, a, b, c

    def teardown(self, lib: DTLCLib, args: _Args) -> None:
        root, a, b, c = args
        lib.dealloc(root)


class StaticPair(MatMul):

    def __init__(self, i: int, j: int, k: int, seed: int, base_dir: str, runs: int, repeats: int, epsilon: float):
        name = "static_pair"
        super().__init__(i, j, k, seed, base_dir, name, runs, repeats, epsilon)

    def construct_lib_builder(self, lib_builder: LibBuilder, a: TensorVariable, b: TensorVariable, c: TensorVariable):
        lib_builder.make_init("init_AB", (a, b), [], free_name="dealloc_AB")
        lib_builder.make_init("init_C", (c), [], free_name="dealloc_C")

    def init_layouts(self, lib: DTLCLib) -> tuple[Any, StructType, StructType, StructType]:
        root_ab, (a, b) = lib.init_AB()
        root_c, (c) = lib.init_C()
        return (root_ab, root_c), a, b, c

    def teardown(self, lib: DTLCLib, args: _Args) -> None:
        (root_ab, root_c), a, b, c = args
        lib.dealloc_AB(root_ab)
        lib.dealloc_C(root_c)


class StaticSingles(MatMul):

    def __init__(self, i: int, j: int, k: int, seed: int, base_dir: str, runs: int, repeats: int, epsilon: float):
        name = "static_singles"
        super().__init__(i, j, k, seed, base_dir, name, runs, repeats, epsilon)


    def construct_lib_builder(self, lib_builder: LibBuilder, a: TensorVariable, b: TensorVariable, c: TensorVariable):
        lib_builder.make_init("init_A", (a), [], free_name="dealloc_A")
        lib_builder.make_init("init_B", (b), [], free_name="dealloc_B")
        lib_builder.make_init("init_C", (c), [], free_name="dealloc_C")

    def init_layouts(self, lib: DTLCLib) -> tuple[Any, StructType, StructType, StructType]:
        root_a, (a) = lib.init_A()
        root_b, (b) = lib.init_B()
        root_c, (c) = lib.init_C()
        return (root_a, root_b, root_c), a, b, c

    def teardown(self, lib: DTLCLib, args: _Args) -> None:
        (root_a, root_b, root_c), a, b, c = args
        lib.dealloc_A(root_a)
        lib.dealloc_B(root_b)
        lib.dealloc_C(root_c)


if __name__ == '__main__':
    static_benchmark = StaticTriple(128,128,128, 0, "./results", repeats=10, runs=10, epsilon=_Epsilon)
    static_benchmark.run()
    static_benchmark = StaticPair(128,128,128, 0, "./results", repeats=10, runs=10, epsilon=_Epsilon)
    static_benchmark.run()
    static_benchmark = StaticSingles(128, 128, 128, 0, "./results", repeats=10, runs=10, epsilon=_Epsilon)
    static_benchmark.run()

    static_benchmark = StaticTriple(8, 8, 8, 0, "./results", repeats=10, runs=10, epsilon=_Epsilon)
    static_benchmark.run()
    static_benchmark = StaticPair(8, 8, 8, 0, "./results", repeats=10, runs=10, epsilon=_Epsilon)
    static_benchmark.run()
    static_benchmark = StaticSingles(8, 8, 8, 0, "./results", repeats=10, runs=10, epsilon=_Epsilon)
    static_benchmark.run()
