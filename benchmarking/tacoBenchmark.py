import abc
import ctypes
import itertools
import os
import pickle
import subprocess
from collections.abc import Callable
from typing import Any, TypeAlias, TypeVar, Generic

from benchmarking.benchmark import (
    Benchmark,
    BenchmarkSettings,
    ID_Tuple,
    Options,
    PythonCode,
    Test,
    TestCode,
)
from dtl.libBuilder import DTLCLib, FuncTypeDescriptor, TupleStruct
from xdsl.dialects.builtin import FunctionType, StringAttr

TypeMap: TypeAlias = tuple[int, dict[str, tuple[tuple[str, ...], tuple[int, ...]]]]
# CppCode: TypeAlias = str


class TacoTest(Test, abc.ABC):

    def __init__(
        self, code: TestCode, taco_path: str, cpp_path: str, type_mapping: TypeMap
    ):
        super().__init__(code)
        self.taco_path = taco_path
        self.cpp_path = cpp_path
        self.type_mapping = type_mapping

    def get_lib_name(self) -> str:
        return f"lib_taco_benchmark.so"

    def get_func_types_name(self) -> str:
        return f"taco_func_types.ft"

    def get_path_str(self) -> str:
        return "_".join([str(p) for p in self.get_id()])

    def get_test_path(self, tests_path: str) -> str:
        return f"{tests_path}/{self.get_path_str()}"

    def get_load(self, tests_path: str, rep: int) -> PythonCode:
        test_path = self.get_test_path(tests_path)
        code = (
            """
import pickle
import ctypes
from dtl.libBuilder import DTLCLib, FuncTypeDescriptor
from xdsl.dialects.builtin import FunctionType, StringAttr
with open(##func_types_path##, "rb") as f:
    function_types_tuple = pickle.load(f)
    function_types, dlt_func_types = function_types_tuple
    assert isinstance(function_types, dict)
    assert isinstance(dlt_func_types, dict)
    for k, v in function_types.items():
        assert isinstance(k, StringAttr)
        assert isinstance(v, FunctionType)
    for k, v in dlt_func_types.items():
        assert isinstance(k, str)
        assert isinstance(v, FuncTypeDescriptor)
function_types_str = {k.data: v for k, v in function_types.items()}
taco_lib_path = ##taco_path##
taco_lib = ctypes.cdll.LoadLibrary(taco_lib_path)
lib = DTLCLib(##lib_path##, dlt_func_types, function_types_str)
        """.replace(
                "##func_types_path##", f"\"{test_path}/{self.get_func_types_name()}\""
            )
            .replace("##lib_path##", f"\"{test_path}/{self.get_lib_name()}\"")
            .replace("##taco_path##", f"\"{self.taco_path}\"")
        )
        return code

class BasicTacoTest(TacoTest):

    @classmethod
    def get_id_headings(cls) -> list[tuple[str, type[str] | type[int] | type[bool]]]:
        return [
            ("taco_layout", int),
        ]

    @classmethod
    def get_result_headings(
        cls,
    ) -> list[tuple[str, type[int] | type[bool] | type[float]]]:
        return [("correct", bool), ("total_error", float), ("consistent", bool)]

    def get_id(self) -> ID_Tuple:
        return (self.type_mapping[0],)



T_Taco = TypeVar("T_Taco", bound=TacoTest)


class TacoBenchmark(Benchmark[T_Taco, DTLCLib], abc.ABC, Generic[T_Taco]):

    def __init__(
        self,
        base_dir: str,
        taco_include_path: str,
        taco_lib_path: str,
        settings: BenchmarkSettings,
        opt_num: int,
        skip_func: Callable[[TypeMap], bool] = None,
    ):
        super().__init__(base_dir, settings)
        self.taco_include_path = taco_include_path
        self.taco_lib_path = taco_lib_path
        self.taco_path = f"{taco_lib_path}/libtaco.so"
        self.taco_lib = ctypes.cdll.LoadLibrary(self.taco_path)
        self.opt_num = opt_num
        self.skip_func = skip_func if skip_func is not None else lambda t: False

    @abc.abstractmethod
    def get_format_options(self) -> dict[str, tuple[list[str], ...]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_code_injection(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    @abc.abstractmethod
    def make_tests_for(self, type_map: TypeMap) -> list[T_Taco]:
        raise NotImplementedError

    @abc.abstractmethod
    def make_func_types(
        self,
    ) -> tuple[dict[StringAttr, FunctionType], dict[str, FuncTypeDescriptor]]:
        raise NotImplementedError

    def enumerate_tests(self, type_maps: list[TypeMap]) -> list[T_Taco]:
        return [t for l in type_maps for t in self.make_tests_for(l)]

    def parse_options(self, benchmark_options: list[str] = None) -> dict[str, Any]:
        options = super().parse_options(benchmark_options)
        return options

    def get_extra_clang_args(self) -> list[str]:
        match self.opt_num:
            case 0:
                return []
            case 1:
                return ["-O1"]
            case 2:
                return ["-O2"]
            case 3:
                return ["-O3"]
            case 4:
                return ["-O3", "-march=native"]

    def initialise_benchmarks(self, options: Options) -> list[T_Taco]:
        layout_options = self.get_format_options()

        layouts = self.get_layouts(layout_options)
        layouts = [l for l in layouts if not self.skip_func(l)]
        tests = self.enumerate_tests(layouts)
        return tests

    def get_layouts(
        self, layout_options: dict[str, tuple[list[str], ...]]
    ) -> list[TypeMap]:
        formats_per_tensor = []
        names = []
        for name, options in layout_options.items():
            format_tuples = []
            for format in itertools.product(*options):
                for order in itertools.permutations(tuple(range(len(format)))):
                    format_tuples.append((format, order))
            formats_per_tensor.append(format_tuples)
            names.append(name)

        layouts = []
        for i, format_tuples in enumerate(itertools.product(*formats_per_tensor)):
            type_map: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {}
            for name, format_tuple in zip(names, format_tuples):
                type_map[name] = format_tuple
            layouts.append((i, type_map))
        return layouts

    def unload_lib(self, lib: DTLCLib):
        lib._close(delete=False)

    def load_lib(
        self, test: T_Taco, test_path: str, options: Options, load: bool = True
    ) -> DTLCLib | None:
        # test_path = test.get_test_path(tests_path)
        return self.get_compiled_lib(
            test_path,
            test.get_lib_name(),
            test.get_func_types_name(),
            test.type_mapping,
            self.get_code_injection(),
            test.cpp_path,
            options,
            load = load,
        )

    def get_compiled_lib(
        self,
        test_path: str,
        lib_name: str,
        func_types_name: str,
        type_mapping: TypeMap,
        code_injection: list[tuple[str, str]],
        cpp_path: str,
        options: Options,
        load: bool = True,
    ) -> DTLCLib | None:
        type_map_number, type_map = type_mapping
        count = self.start_inline_log(f"Getting lib for mapping: {type_map_number} :: ")

        os.makedirs(test_path, exist_ok=True)
        lib_path = f"{test_path}/{lib_name}"
        test_cpp_path = f"{lib_path}.cpp"
        func_types_path = f"{test_path}/{func_types_name}"

        func_types_exists = os.path.exists(func_types_path)
        count = self.inline_log(count, "func_types: ", append=True)
        if func_types_exists:
            with open(func_types_path, "rb") as f:
                function_types_tuple = pickle.load(f)
                function_types, dlt_func_types = function_types_tuple
            count = self.inline_log(count, "loaded, ", append=True)
        else:
            function_types, dlt_func_types = self.make_func_types()
            with open(func_types_path, "wb") as f:
                f.write(pickle.dumps((function_types, dlt_func_types)))
            count = self.inline_log(count, "made, ", append=True)

        lib_exists = os.path.exists(lib_path)
        self.inline_log(0, "lib: ")
        if lib_exists:
            if load:
                function_types_str = {k.data: v for k, v in function_types.items()}
                lib = DTLCLib(lib_path, dlt_func_types, function_types_str)
            else:
                lib = None
            self.end_inline_log(count, "found.", append=True)
            return lib
        else:
            count = self.inline_log(count, "not found, ", append=True)

        count = self.inline_log(count, "cpp: ", append=True)
        with open(cpp_path, "r") as f:
            cpp_code = f.read()

        for old, replacement in code_injection:
            cpp_code = cpp_code.replace(old, replacement)

        for name, (format, order) in type_map.items():
            format_str = (
                f"{{{','.join(format)}}}, {{{','.join([str(o) for o in order])}}}"
            )
            cpp_code = cpp_code.replace(name, format_str)

        with open(test_cpp_path, "w") as f:
            f.write(cpp_code)

        count = self.inline_log(count, "written, ", append=True)
        count = self.inline_log(count, "lib: ", append=True)
        clang_args = ["clang++"]
        clang_args.extend(["-o", lib_path])
        clang_args.append("-shared")
        clang_args.append("-fPIC")
        clang_args.append("-Wno-override-module")
        clang_args.extend(self.get_extra_clang_args())
        clang_args.extend(["-I", self.taco_include_path])
        clang_args.extend(["-L", self.taco_lib_path])
        clang_args.append("-ltaco")
        clang_args.append(test_cpp_path)
        # print(" ".join(clang_args))

        process_clang = subprocess.Popen(
            clang_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        out, err = process_clang.communicate()
        if (out is not None and len(out)>0) or (err is not None and len(err)>0):
            self.log("stdout:", error=True)
            print(out.decode("utf8") if out is not None else None)
            self.log("stderr:", error=True)
            print(err.decode("utf8") if err is not None else None)
        process_clang.wait()

        count = self.inline_log(count, f"compiled", append=True)

        if load:
            function_types_str = {k.data: v for k, v in function_types.items()}
            lib = DTLCLib(lib_path, dlt_func_types, function_types_str)
            self.end_inline_log(count, f" and loaded", append=True)
        else:
            lib = None
        return lib
