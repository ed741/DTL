import abc
import csv
import datetime
import os
import pickle
import subprocess
import time
import typing
from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, TypeAlias, TypeVar, overload

import nptyping
import numpy as np
import psutil

from benchmarking import benchmarkRunner

K = TypeVar("K")
Options: TypeAlias = dict[str, Any]
PythonCode: TypeAlias = str

ID_Tuple: TypeAlias = tuple[str|int|bool, ...]
U_Res_Tuple: TypeAlias = tuple[int, float, float, bool]
Res_Tuple: TypeAlias = tuple[int|bool|float, ...]
Res_Dict: TypeAlias = dict[tuple[ID_Tuple, int], tuple[U_Res_Tuple, Res_Tuple]]

@dataclass(frozen=True)
class TestCode():
    setup: PythonCode
    benchmark: PythonCode
    test: PythonCode # must define the variables named by 'get_result_headings()' e.g. 'correct', 'total_error', 'consistent' in the scope
    clean: PythonCode

class Test(abc.ABC):
    def __init__(self, code: TestCode):
        self.code = code

    @classmethod
    @abc.abstractmethod
    def get_id_headings(cls) -> list[tuple[str,type[str]|type[int]|type[bool]]]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_result_headings(cls) -> list[tuple[str, type[int]|type[bool]|type[float]]]:
        raise NotImplementedError # return [("correct", bool),("total_error", float),("consistent", bool)]

    @abc.abstractmethod
    def get_id(self) -> ID_Tuple:
        raise NotImplementedError

    def get_id_str(self) -> str:
        return "-".join([str(p) for p in self.get_id()])

    @abc.abstractmethod
    def get_load(self, tests_path: str) -> PythonCode:
        # must define 'lib'
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_path(self, tests_path: str) -> str:
        raise NotImplementedError

T = TypeVar('T', bound=Test)
L = TypeVar('L')

@dataclass(frozen=True)
class BenchmarkSettings:
    runs: int = 10
    repeats: int = 3
    waste_of_time_threshold: float = 0.1
    test_too_short_threshold: float = 0.001
    long_run_multiplier: int = 100
    setup_timeout: float = 3
    benchmark_timeout: float = 3
    testing_timeout: float = 3
    tear_down_timeout: float = 3
    benchmark_trial_child_process: bool = True
    benchmark_in_child_process: bool = False
    ram_use_cap: float = 0.75

class Benchmark(abc.ABC, Generic[T, L]):

    def __init__(
        self,
        base_dir: str,
        settings: BenchmarkSettings,
    ):
        self.base_dir = base_dir
        self.settings = settings

        self.np_arg_paths: dict[str, tuple[type, str]] = {}
        self.np_args: dict[str, np.ndarray] = {}
        self.np_res_paths: dict[str, tuple[type, str]] = {}
        self.np_ress: dict[str, np.ndarray] = {}

        self.lib: L | None = None
        self.lib_id: ID_Tuple | None = None


        os.makedirs(self.base_dir, exist_ok=True)

    @property
    def test_class(self) -> type[T]:
        return typing.get_args(self.__orig_bases__[0])[0]

    @abc.abstractmethod
    def initialise_benchmarks(self, options: Options) -> list[T]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_lib(self, test: T, test_path: str, options: Options, load: bool = True) -> L:
        raise NotImplementedError

    @abc.abstractmethod
    def unload_lib(self, lib: L):
        raise NotImplementedError


    def parse_options(self, benchmark_options: list[str] = None) -> Options:
        if benchmark_options is None:
            return {}
        return {"skip-testing":"--skip-testing" in benchmark_options,
                   "valgrind":"--valgrind" in benchmark_options,
                "no-timeout":"--no-timeout" in benchmark_options,}

    def run(self, benchmark_options: list[str] = None):
        options = self.parse_options(benchmark_options)
        self.log(f"Running benchmark {type(self)} in {self.base_dir}")
        self.run_benchmarking(options)

    @staticmethod
    def get_universal_result_parts() -> tuple[tuple[str, type[int]], tuple[str, type[float]], tuple[str, type[float]], tuple[str, type[bool]]]:
        return ("runs", int), ("time", float), ("waiting_time", float), ("finished", bool)

    @overload
    def get_test_result_from_list(self, row, only_res: Literal[False] = False) -> tuple[ID_Tuple, int, U_Res_Tuple, Res_Tuple]: ...

    @overload
    def get_test_result_from_list(self, row, only_res: Literal[True] = False) -> Res_Tuple: ...

    def get_test_result_from_list(self, row, only_res = False) -> tuple[ID_Tuple, int, U_Res_Tuple, Res_Tuple] | Res_Tuple:
        id_parts = self.test_class.get_id_headings()
        universal_parts = self.get_universal_result_parts()
        result_parts = self.test_class.get_result_headings()
        row_idx = 0
        test_id: tuple[()] | tuple[str|int|bool, ...] = ()
        if not only_res:
            for name, t in id_parts:
                if t == bool:
                    if row[row_idx] == "True":
                        test_id = *test_id, True
                    elif row[row_idx] == "False":
                        test_id = *test_id, False
                    else:
                        raise ValueError(f"Unexpected value for bool '{name}': '{row[row_idx]}' in row: {row} at {row_idx}")
                else:
                    test_id = *test_id, t(row[row_idx])
                row_idx += 1
            assert len(test_id) > 0
            o_id = typing.cast(tuple[str|int|bool, ...], test_id)

            repeat = int(row[row_idx])
            row_idx += 1
        else:
            o_id = None
            repeat = None

        if not only_res:
            u_res: tuple[()] | tuple[float|bool, ...] = ()
            for name, t in universal_parts:
                if t == bool:
                    if row[row_idx] == "True":
                        u_res = *u_res, True
                    elif row[row_idx] == "False":
                        u_res = *u_res, False
                    else:
                        raise ValueError(f"Unexpected value for bool '{name}': '{row[row_idx]}' in row: {row} at {row_idx}")
                else:
                    u_res = *u_res, t(row[row_idx])
                row_idx += 1
            assert len(u_res) > 0
            o_u_res = typing.cast(U_Res_Tuple, u_res)
        else:
            o_u_res = None

        res: tuple[()] | tuple[int|bool|float, ...] = ()
        for name, t in result_parts:
            if t == bool:
                if row[row_idx] == "True":
                    res = *res, True
                elif row[row_idx] == "False":
                    res = *res, False
                else:
                    raise ValueError(f"Unexpected value for bool '{name}': '{row[row_idx]}' in row: {row} at {row_idx}")
            else:
                res = *res, t(row[row_idx])
            row_idx += 1
        assert len(res) > 0
        o_res = typing.cast(tuple[int|bool|float, ...], res)

        if only_res:
            return o_res
        else:
            return o_id, repeat, o_u_res, o_res

    def get_results_path(self)-> str:
        return f"{self.base_dir}/results.csv"

    def load_existing_results(self) -> Res_Dict:
        count = self.start_inline_log("Loading from existing results, ","...")
        results_done = {}
        results_file = self.get_results_path()
        if os.path.exists(results_file):
            with open(results_file, "r", newline="") as f:
                r = csv.reader(f)
                next(r, None)  # skip header
                for row in r:
                    test_id, rep, u_res, result = self.get_test_result_from_list(row)
                    results_done[(test_id, rep)] = (u_res, result)
        else:
            with open(results_file, "a", newline="") as csv_results:
                result_writer = csv.writer(csv_results)
                header = self._get_results_header()
                result_writer.writerow(header)

        self.end_inline_log(count, f"Found {len(results_done)} results")
        return results_done

    def get_runs_for_time(self, trial_time: float) -> int:
        if trial_time < 0:
            return 0
        if trial_time > self.settings.waste_of_time_threshold:
            return 0
        if trial_time < self.settings.test_too_short_threshold:
            return self.settings.runs * self.settings.long_run_multiplier
        return self.settings.runs

    def remove_skipable_tests(self, tests: list[T], results: Res_Dict) -> list[T]:
        new_tests = []
        for t in tests:
            if (t.get_id(), -1) in results:
                (runs, t_time, w_time, fin), res = results[(t.get_id(), -1)]
                trial_time = t_time/runs
                runs_to_do = self.get_runs_for_time(trial_time)
                if runs_to_do > 0:
                    for rep in range(self.settings.repeats):
                        if (t.get_id(), rep) not in results:
                            new_tests.append(t)
                            break
            else:
                new_tests.append(t)
        self.log(f"Able to Skip {len(tests) - len(new_tests)} / {len(tests)} of the all the tests")
        return new_tests

    def _stringify(self, p: str|int|bool|float) -> str:
        if isinstance(p, str):
            return p
        elif isinstance(p, int):
            return str(p)
        elif isinstance(p, bool):
            return "True" if p else "False"
        elif isinstance(p, float):
            s = str(p)
            new_p = float(s)
            if p != new_p:
                self.log(f"Turning float '{repr(p)}' into a str '{s}' produced an inconsistency.", error=True)
            return s

    def _get_results_header(self) -> list[str]:
        return [s for (s,t) in self.test_class.get_id_headings() + [("repeats", int)] + list(self.get_universal_result_parts()) + self.test_class.get_result_headings()]

    def write_result(self, test: T, repeat: int, u_res: U_Res_Tuple, results: Res_Tuple) -> None:
        row = []
        for p in test.get_id():
            row.append(self._stringify(p))
        row.append(self._stringify(repeat))
        for p in u_res:
            row.append(self._stringify(p))
        for p in results:
            row.append(self._stringify(p))

        results_file = self.get_results_path()
        exists = os.path.exists(results_file)
        with open(results_file, "a", newline="") as f:
            result_writer = csv.writer(f)
            if not exists:
                header = self._get_results_header()
                result_writer.writerow(header)
            result_writer.writerow(row)
            f.flush()

    def _non_result_for[K](self, heading: list[tuple[str, type[K]]]) -> tuple[K, ...]:
        new_result = []
        for n, t in heading:
            if t == str:
                new_result.append("")
            elif t == int:
                new_result.append(0)
            elif t == bool:
                new_result.append(True)
            elif t == float:
                new_result.append(0.0)
            else:
                raise NotImplementedError(f"{t} not implemented")
        return tuple(new_result)


    def run_benchmarking(self, options: Options) -> None:
        tests = self.initialise_benchmarks(options)

        loaded_results = self.load_existing_results()

        tests = self.remove_skipable_tests(tests, loaded_results)

        count = 0
        tests_to_run = len(tests)

        for test in tests:
            count += 1
            self.log(f"\tRunning Benchmarks for {test.get_id_str()}. ({count}/{tests_to_run}) : {test.get_test_path(self.get_tests_path())}")

            runs_to_do = self.get_runs_to_do(test, loaded_results, options)
            if runs_to_do < 1:
                self.log(f"Skipping test id: ({test.get_id_str()}), as test has {runs_to_do} runs.")
            else:
                for rep in range(self.settings.repeats):
                    if (test.get_id(), rep) in loaded_results:
                        self.log(f"Skipping test id: {test.get_id_str()}, rep: {rep} as it is already in the results file")
                        continue
                    char_count = self.start_inline_log(f"Running benchmark repeat :: test id: {test.get_id_str()}, rep: {rep}, runs: {runs_to_do}, ")
                    if options["skip-testing"]:
                        self.end_inline_log(char_count, f"Skipping testing")
                        continue

                    universal_results, results = (
                        self.run_test(test, runs_to_do, rep, options)
                    )
                    self.write_result(test, rep, universal_results, results)
                    runs_done, test_time, wait_time, finished = universal_results
                    assert runs_done == runs_to_do
                    universal_result_string = f"time: {test_time}, wait: {wait_time}, finished: {finished}"
                    custom_result_string = ", ".join([f"{n}: {r}" for ((n, t), r) in zip(test.get_result_headings(), results)])
                    self.end_inline_log(char_count, universal_result_string + " :: " + custom_result_string)
            self.close_lib()
        print(f"finished")

    def get_runs_to_do(self, test: T, loaded_results: Res_Dict, options: Options) -> int:
        if (test.get_id(), -1) in loaded_results:
            (test_runs, test_time, test_w_time, test_fin), res = loaded_results[(test.get_id(), -1)]
            count = self.start_inline_log(f"Trail test result found in loaded results: ", "...")
        else:
            lib = self.get_lib(test, options)
            if lib is None:
                self.log(f"Cannot trail test because lib is None")
                return 0
            if self.settings.benchmark_trial_child_process:
                count = self.start_inline_log(f"Trail test starting in new process: ")
                u_res, res = self.run_external_test(test, 1, -1, options)
                self.write_result(test, -1, u_res, res)
                test_runs, test_time, test_w_time, test_fin = u_res
            else:
                count = self.start_inline_log(f"Trail test: ", "...")
                u_res, res = self.run_internal_test(test, lib, 1)
                self.write_result(test, -1, u_res, res)
                test_runs, test_time, test_w_time, test_fin = u_res
        # self.end_inline_log(count, f"runs: {test_runs}, time: {test_time}, wait time: {test_w_time}, finished: {test_fin}")
        universal_result_string = f"runs: {test_runs}, time: {test_time}, wait time: {test_w_time}, finished: {test_fin}"
        custom_result_string = ", ".join([f"{n}: {r}" for ((n, t), r) in zip(test.get_result_headings(), res)])
        self.end_inline_log(count, universal_result_string + " :: " + custom_result_string)
        trial_time = test_time / test_runs
        return self.get_runs_for_time(trial_time)

    def run_test(self, test: T, runs: int, rep: int, options: Options) -> tuple[U_Res_Tuple, Res_Tuple]:
        if self.settings.benchmark_in_child_process:
            lib = self.get_lib(test, options, load=False)
            return self.run_external_test(test, runs, rep, options)
        else:
            lib = self.get_lib(test, options, load=True)
            return self.run_internal_test(test, lib, runs)

    def run_internal_test(self, test: T, lib: L, runs: int) -> tuple[U_Res_Tuple, Res_Tuple]:
        start_time = time.time()
        test_time, res = benchmarkRunner.run_benchmark(lib,
                                                       runs,
                                                       self.np_args,
                                                       self.np_ress,
                                                       test.code.setup,
                                                       test.code.benchmark,
                                                       test.code.test,
                                                       test.code.clean,
                                                       test.get_result_headings(),
                                                       print_updates=True,
                                                       inline_updates=True,
                                                       )
        waiting_time = time.time() - start_time
        return (runs, test_time, waiting_time, True), res

    def run_external_test(self,  test: T, runs: int, rep: int, options: Options) -> tuple[U_Res_Tuple, Res_Tuple]:

        dump_dir = f"{test.get_test_path(self.get_dump_path())}/{test.get_id_str()}/{rep}"
        dump_path = f"{dump_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
        os.makedirs(dump_path, exist_ok=False)


        count = self.inline_log(0, "setting up")
        args = []
        if options["valgrind"]:
            args.extend(["valgrind", "--leak-check=full", "--track-origins=yes"])
        args.extend(["python", "benchmarking/benchmarkRunner.py"])
        args.append(f"{runs}")
        np_type_map = {np.float32: ":f32", np.float64: ":f64", np.int32: ":i32", np.int64: ":i64"}
        for np_arg_name, (np_arg_ty, np_arg_path) in self.np_arg_paths.items():
            args.extend([f"-a{np_type_map[np_arg_ty]}={np_arg_name}", np_arg_path])
        for np_res_name, (np_res_ty, np_res_path) in self.np_res_paths.items():
            args.extend([f"-r{np_type_map[np_res_ty]}={np_res_name}", np_res_path])

        for name, type in test.get_result_headings():
            t = {int: "i", bool: "b", float: "f"}[type]
            args.append(f"-o:{t}={name}")
        args.extend(["--load", test.get_load(self.get_tests_path())])
        args.extend(["--setup", test.code.setup])
        args.extend(["--benchmark", test.code.benchmark])
        args.extend(["--test", test.code.test])
        args.extend(["--clean", test.code.clean])
        args.extend(["-ptg"])

        current_env = os.environ.copy()

        def abort_test(oc, process, msg:str = ""):
            c = self.inline_log(0, "timed out")
            process.kill()
            c = self.inline_log(c, f"{msg}killed, ")
            return c + oc


        with open(f"{dump_path}/stdout", "w") as write_stdout, open(f"{dump_path}/stderr", "w") as write_stderr, open(f"{dump_path}/stdout", "r", 1) as read_stdout:
            count = self.inline_log(count, "starting")
            start_time = time.time()
            process_benchmark = subprocess.Popen(
                args, env=current_env, stdin=None, stderr=write_stderr,
                stdout=write_stdout, text=True,
            )
            process_live = True
            exit_code = None
            count = self.inline_log(count, "waiting for benchmark process")

            busy_wait_time = 1.0
            out = []
            info_strings = []

            max_ram_usage = 0

            # setup_done = False
            # section_start = time.time()
            # wait_time = self.settings.setup_timeout*runs
            # if process_live:
            #     count = self.inline_log(count, "benchmark process: setup, ")
            #     loop_count = self.inline_log(0,"")
            #     while process_live and not setup_done:
            #         loop_count = self.inline_log(loop_count, ".", append=True)
            #         ram_usage = psutil.virtual_memory()[2]/100.0
            #         max_ram_usage = max(max_ram_usage, ram_usage)
            #         if ram_usage > self.settings.ram_use_cap:
            #             loop_count = self.inline_log(loop_count, "")
            #             count = abort_test(count, process_benchmark, msg=f"Ram%: {int(ram_usage*100.0)}, ")
            #             info_strings.append(f"Aborting due to Ram usage reaching {ram_usage*100.0}%")
            #         try:
            #             exit_code = process_benchmark.wait(min(busy_wait_time, wait_time))
            #             process_live = False
            #         except subprocess.TimeoutExpired as _:
            #             pass
            #         current_output = read_stdout.read()
            #         current_lines = current_output.splitlines()
            #         out.extend([l for l in current_lines if not l.strip().startswith("#")])
            #         setup_done = "~setup-done~" in out
            #         if process_live and not setup_done and time.time() - section_start > wait_time:
            #             loop_count = self.inline_log(loop_count, "")
            #             count = abort_test(count, process_benchmark)
            #             info_strings.append(f"Aborting due to time")
            #             process_live = False
            #     loop_count = self.inline_log(loop_count, "")
            # info_strings.append(f"setup took {time.time()-section_start}s")
            #
            # benchmark_done = False
            # section_start = time.time()
            # wait_time = self.settings.benchmark_timeout * runs
            # if process_live:
            #     count = self.inline_log(count, "benchmark process: benchmarking, ")
            #     loop_count = self.inline_log(0,"")
            #     ram_usage = psutil.virtual_memory()[2] / 100.0
            #     max_ram_usage = max(max_ram_usage, ram_usage)
            #     if ram_usage > self.settings.ram_use_cap:
            #         loop_count = self.inline_log(loop_count, "")
            #         count = abort_test(count, process_benchmark, msg=f"Ram%: {int(ram_usage * 100.0)}, ")
            #         info_strings.append(f"Aborting due to Ram usage reaching {ram_usage * 100.0}%")
            #     while process_live and not benchmark_done:
            #         loop_count = self.inline_log(loop_count, ".", append=True)
            #         try:
            #             exit_code = process_benchmark.wait(min(busy_wait_time, wait_time))
            #             process_live = False
            #         except subprocess.TimeoutExpired as _:
            #             pass
            #         current_output = read_stdout.read()
            #         current_lines = current_output.splitlines()
            #         out.extend([l for l in current_lines if not l.strip().startswith("#")])
            #         benchmark_done = "~benchmark-done~" in out
            #         if process_live and not benchmark_done and time.time() - section_start > wait_time:
            #             loop_count = self.inline_log(loop_count, "")
            #             count = abort_test(count, process_benchmark)
            #             info_strings.append(f"Aborting due to time")
            #     loop_count = self.inline_log(loop_count, "")
            # info_strings.append(f"benchmarking took {time.time() - section_start}s")
            #
            # testing_done = False
            # section_start = time.time()
            # wait_time = self.settings.testing_timeout * runs
            # if process_live:
            #     count = self.inline_log(count, "benchmark process: testing, ")
            #     loop_count = self.inline_log(0,"")
            #     ram_usage = psutil.virtual_memory()[2] / 100.0
            #     max_ram_usage = max(max_ram_usage, ram_usage)
            #     if ram_usage > self.settings.ram_use_cap:
            #         loop_count = self.inline_log(loop_count, "")
            #         count = abort_test(count, process_benchmark, msg=f"Ram%: {int(ram_usage * 100.0)}, ")
            #         info_strings.append(f"Aborting due to Ram usage reaching {ram_usage * 100.0}%")
            #     while process_live and not testing_done:
            #         loop_count = self.inline_log(loop_count, ".", append=True)
            #         try:
            #             exit_code = process_benchmark.wait(min(busy_wait_time, wait_time))
            #             process_live = False
            #         except subprocess.TimeoutExpired as _:
            #             pass
            #         current_output = read_stdout.read()
            #         current_lines = current_output.splitlines()
            #         out.extend([l for l in current_lines if not l.strip().startswith("#")])
            #         testing_done = "~testing-done~" in out
            #         if process_live and not testing_done and time.time() - section_start > wait_time:
            #             loop_count = self.inline_log(loop_count, "")
            #             count = abort_test(count, process_benchmark)
            #             info_strings.append(f"Aborting due to time")
            #     loop_count = self.inline_log(loop_count, "")
            # info_strings.append(f"testing took {time.time() - section_start}s")

            testing_sections = [("setup", "~setup-done~", self.settings.setup_timeout),
                                ("benchmark", "~benchmark-done~", self.settings.benchmark_timeout),
                                ("testing", "~testing-done~", self.settings.testing_timeout),
                                ("tear down", "~tear-down-done~", self.settings.tear_down_timeout)]
            for section_name, section_tag, section_timeout in testing_sections:
                if not process_live:
                    break
                section_done = False
                section_start = time.time()
                wait_time = section_timeout * runs
                count = self.inline_log(count, f"benchmark process: {section_name}, ")
                loop_count = self.inline_log(0,"")
                ram_usage = psutil.virtual_memory()[2] / 100.0
                max_ram_usage = max(max_ram_usage, ram_usage)
                if ram_usage > self.settings.ram_use_cap:
                    loop_count = self.inline_log(loop_count, "")
                    count = abort_test(count, process_benchmark, msg=f"Ram%: {int(ram_usage * 100.0)}, ")
                    info_strings.append(f"Aborting due to Ram usage reaching {ram_usage * 100.0}%")
                while process_live and not section_done:
                    loop_count = self.inline_log(loop_count, ".", append=True)
                    try:
                        exit_code = process_benchmark.wait(min(busy_wait_time, wait_time))
                        process_live = False
                    except subprocess.TimeoutExpired as _:
                        pass
                    current_output = read_stdout.read()
                    current_lines = current_output.splitlines()
                    out.extend([l for l in current_lines if not l.strip().startswith("#")])
                    section_done = section_tag in out
                    if process_live and not section_done and time.time() - section_start > wait_time:
                        loop_count = self.inline_log(loop_count, "")
                        if options["no-timeout"]:
                            count = self.inline_log(count, "Not Aborting, but we should have")
                            info_strings.append(f"Not Aborting due to time (though time reached)")
                        else:
                            count = abort_test(count, process_benchmark)
                            info_strings.append(f"Aborting due to time")
                loop_count = self.inline_log(loop_count, "")
                info_strings.append(f"{section_name} took {time.time() - section_start}s")

            process_benchmark.communicate()
        waiting_time = time.time() - start_time
        finished = all(status in out for status in ["~setup-done~", "~benchmark-done~", "~testing-done~", "~tear-down-done~", "~results-done~"])

        split_out = [s for s in out if not s.startswith("~")]

        info_strings.append(f"Ram usage (measured) peaked at {max_ram_usage}")
        info_strings.append(f"Process return code: {process_benchmark.returncode}")
        with open(f"{dump_path}/info", "w") as f:
            for line in info_strings:
                f.write(f"{line}\n")

        if len(split_out) != 1 + len(test.get_result_headings()):
            if finished:
                self.end_inline_log(count, " ERROR", append=True)
                self.log(f"{len(split_out)} != 1 + {len(test.get_result_headings())}")
                self.log(f"Process return code: {process_benchmark.returncode}", error=True)
                self.log(f" Test finished but output is: {split_out} ", error=True)
                self.log(f" outputs from process stored in {dump_path}", error=True)
                return (runs, -1.0, waiting_time, True), self._non_result_for(test.get_result_headings())
            else:
                return (runs, -1.0, waiting_time, False), self._non_result_for(test.get_result_headings())

        try:
            res_part = split_out[1:]
            res = self.get_test_result_from_list(res_part, only_res=True)
        except ValueError as e:
            self.end_inline_log(count, " ERROR", append=True)
            self.log(f"Process exit code: {process_benchmark.returncode}", error=True)
            self.log(f" Test finished but output is: {split_out} ", error=True)
            self.log(f" outputs from process stored in {dump_path}", error=True)
            self.log(str(e), error=True)
            return (runs, -2.0, waiting_time, finished), self._non_result_for(test.get_result_headings())

        if exit_code == 0:
            self.inline_log(count, "")
        u_res = runs, float(split_out[0]), waiting_time, finished
        return u_res, res

    def get_tests_path(self) -> str:
        return f"{self.base_dir}/tests"

    def get_dump_path(self) -> str:
        return f"{self.base_dir}/dump"

    def get_lib(self, test: T, options: Options, load: bool = True) -> L | None:
        if self.lib_id != test.get_id():
            self.close_lib()
            tests_path = self.get_tests_path()
            test_path = test.get_test_path(tests_path)
            lib = self.load_lib(test, test_path, options, load=load)
            self.lib = lib
            self.lib_id = test.get_id()
        return self.lib

    def close_lib(self):
        if self.lib is not None:
            self.unload_lib(self.lib)
            self.lib = None
            assert self.lib_id is not None
        self.lib_id = None


    @overload
    def search_store(self, store_path: str, key: K, match_function: Callable[[K, K], bool] = None,
                     lock: Literal[True] = True) -> tuple[Literal[True], int, Any] | tuple[Literal[False], int, str]:
        ...
    @overload
    def search_store(self, store_path: str, key: K, match_function: Callable[[K, K], bool] = None,
                     lock: Literal[False] = True) -> tuple[Literal[True], int, Any] | tuple[Literal[False], Literal[-1], None]:
        ...
    @overload
    def search_store(self, store_path: str, key: K, match_function: Callable[[K, K], bool] = None,
                     lock: bool = True) -> tuple[Literal[True], int, Any] | tuple[Literal[False], int, str] | tuple[Literal[False], Literal[-1], None]:
        ...
    def search_store(self, store_path: str, key: K, match_function: Callable[[K, K], bool] = None, lock: bool = True) -> tuple[Literal[True], int, Any] | tuple[Literal[False], int, str] | tuple[Literal[False], Literal[-1], None]:
        if match_function is None:
            match_function = lambda x, y: x == y
        store_keys_path = f"{store_path}/keys"
        store_values_path = f"{store_path}/values"
        store_gen_path = f"{store_path}/gen"
        os.makedirs(store_keys_path, exist_ok=True)
        os.makedirs(store_values_path, exist_ok=True)
        os.makedirs(store_gen_path, exist_ok=True)

        count = self.start_inline_log(f"Searching store: {store_path}, ", "...")
        checked_keys = set()
        files = os.listdir(store_keys_path)
        for file in files:
            count = self.inline_log(count, ".", append=True)
            key_file_name = os.fsdecode(file)
            is_lock = key_file_name.endswith(".lock")
            key_int = int(key_file_name.removesuffix(".lock"))
            checked_keys.add(key_int)
            is_locked = is_lock or os.path.exists(f"{store_keys_path}/{key_int}.lock")

            if not is_locked:
                with open(f"{store_keys_path}/{key_int}", "rb") as f:
                    loaded_key = pickle.load(f)
                if match_function(key, loaded_key):
                    count = self.inline_log(count, f"Found matching key: {key_int}, ")
                    with open(f"{store_values_path}/{key_int}", "rb") as f:
                        loaded = pickle.load(f)
                    with open(f"{self.base_dir}/store_use", "a") as f:
                        f.write(f"{datetime.datetime.now()} Loaded {key_int} from store: {store_path}\n")
                    self.end_inline_log(count, "Loaded value.", append=True)
                    return True, key_int, loaded

        self.end_inline_log(count, f"{len(checked_keys)} keys checked, No match found.")
        if lock:
            checked_keys.add(-1)
            new_key = max(checked_keys) + 1
            new_key_str = f"{new_key}?"
            count = self.start_inline_log(" Attempting to lock key: ", f"{new_key_str}")
            file_found = False
            pid = os.getpid()
            while not file_found:
                try:
                    with open(f"{store_keys_path}/{new_key}.lock", "x") as f:
                        f.write(f"{int(pid)}")
                    file_found = True
                except FileExistsError as _:
                    new_key = new_key + 1
                    new_key_str = f"{new_key}?"
                    count = self.inline_log(count, f"{new_key_str}")
            with open(f"{self.base_dir}/store_use", "a") as f:
                f.write(f"{datetime.datetime.now()} Locked {new_key} by {int(pid)} from store: {store_path}\n")
            self.end_inline_log(count, f"{new_key}")
            new_store_gen_path = f"{store_gen_path}/{new_key}"
            os.makedirs(new_store_gen_path, exist_ok=True)
            return False, new_key, new_store_gen_path
        return False, -1, None

    def write_store(self, store_path: str, key: K, value: Any, key_int: int) -> bool:
        if key_int < 0:
            return False

        store_keys_path = f"{store_path}/keys"
        store_values_path = f"{store_path}/values"
        store_gen_path = f"{store_path}/gen"
        os.makedirs(store_keys_path, exist_ok=True)
        os.makedirs(store_values_path, exist_ok=True)
        os.makedirs(store_gen_path, exist_ok=True)

        pid = os.getpid()
        with open(f"{store_keys_path}/{key_int}.lock", "r") as f:
            lock_pid = int(f.read())
            if lock_pid != pid:
                self.log(f"lock pid: {lock_pid} does not match our pid {pid} - something has tampered with the store directory {store_path}", error=True)
                assert False, f"lock pid: {lock_pid} does not match out pid {pid} - something has tampered with the store directory {store_path}"

        with open(f"{store_values_path}/{key_int}", "wb") as f:
            f.write(pickle.dumps(value))
        with open(f"{store_keys_path}/{key_int}", "wb") as f:
            f.write(pickle.dumps(key))
        os.remove(f"{store_keys_path}/{key_int}.lock")
        with open(f"{self.base_dir}/store_use", "a") as f:
            f.write(f"{datetime.datetime.now()} Unlocked and generated record {key_int} by {int(pid)} from store: {store_path}\n")
        self.log(f"New key: {key_int}, Added to store: {store_path}")
        return True


    def handle_reference_array(self, array: nptyping.NDArray, array_path: str, is_arg: bool, is_res: bool, scope_name: str, binary: bool = True, dtype: type = np.float32):
        if scope_name is None:
            scope_name = array_path
        if binary:
            array_path += ".npy"
        path = f"{self.base_dir}/{array_path}"
        if os.path.exists(path):
            if binary:
                loaded_array = np.load(path)
            else:
                loaded_array = np.loadtxt(path)
            if (array.shape != loaded_array.shape) or (not np.equal(array, loaded_array).all()):
                self.log(f"ERROR! loaded array does not match newly calculated reference array: {path}", error=True)
            else:
                # print(f"Reference array consistent with {path}")
                pass
        else:
            dir = os.path.dirname(path)
            os.makedirs(dir, exist_ok=True)
            if binary:
                np.save(path, array)
                np.savetxt(path+".txt", array)
            else:
                np.savetxt(path, array)
            self.log(f"Reference array saved to {path}")
        if is_arg:
            assert scope_name not in self.np_arg_paths
            self.np_arg_paths[scope_name] = dtype, path
            assert scope_name not in self.np_args
            self.np_args[scope_name] = array

        if is_res:
            assert scope_name not in self.np_res_paths
            self.np_res_paths[scope_name] = dtype, path
            assert scope_name not in self.np_ress
            self.np_ress[scope_name] = array


    def log(self, string: str, error:bool = False):
        print(f"{datetime.datetime.now()}: {"ERROR: " if error else ""}{string}")

    def start_inline_log(self, string: str, temp: str = "") -> int:
        print(f"{datetime.datetime.now()}: {string}{temp}", end="")
        return len(temp)

    def inline_log(self, chars: int, string: str, append: bool = False) -> int:
        if not append:
            print("\b"*chars, end="")
            chars = 0
        print(string, end="")
        return chars+len(string)

    def end_inline_log(self, chars: int, string: str, append: bool = False):
        if not append:
            print("\b" * chars, end="")
        print(string)
