import csv
import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import evaluation.evaluateTools as et
from xdsl.dialects.experimental import dlt


@dataclass(frozen=True)
class Stat:
    test_key: tuple
    program_key: tuple
    time: dict
    idx_by: dict
    sparse: dict
    test_map: dict
    runs: int
    entries: list[dict]

def generate_data(
    experiment_name: str,
    test_key: list[tuple[str, Callable[[str], Any]]] = None,
    program_keys: list[str] = None,
    layout_targets: dict[str, str] = None,
    layout_gen: Callable[[dict[str, dlt.PtrType]], dict[str, str]] = None,
    layout_export: list[str] = None,
    instances: tuple[int, int] = None,
    min_wait_time: int = None,
    time_options: list[str] = None,
    idx_time_options: list[str] = None,
    stat_gen: Callable[[Stat], dict[str, str]] = None,
    stat_export: list[str] = None,
    explore_dump_codes: bool = False,
    tests_gen: Callable[[Stat, str], dict[str, str]] = None,
    tests_export: list[str] = None,
):

    if test_key is None:
        test_key = [("layout", int), ("order", int)]
    if program_keys is None:
        program_keys = ["layout", "order"]
    assert set(program_keys).issubset(set([k for k, fn in test_key]))

    if layout_export is None:
        layout_export = []
    if stat_gen is None:
        stat_gen = lambda s: {}
    if stat_export is None:
        stat_export = []
    if tests_export is None:
        tests_export = []

    experiment_dir = f"{et.RESULTS_DIR_BASE}/{experiment_name}"
    data_dir = f"{et.OUTPUT_DATA_DIR}/{experiment_name}"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Experiment name: {experiment_name}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Data directory: {data_dir}")

    experiment_info_map = {}

    if layout_targets is not None:
        layout_graph, layouts = et.load_dlt_layouts(experiment_dir)
        experiment_info_map["AllLayouts"] = len(layouts)
    else:
        layouts = []

    results = et.load_results_file(
        f"{experiment_dir}/results.csv",
        columns=test_key
        + [
            ("repeats", int),
            ("runs", int),
            ("time", float),
            ("waiting_time", float),
            ("finished", et.parse_bool),
            ("correct", et.parse_bool),
        ],
    )

    total_tests = len({tuple([row[k] for k, f in test_key]) for row in results})
    print("total_tests:", total_tests)
    experiment_info_map["TotalTests"] = total_tests

    total_programs = len({tuple([row[k] for k in program_keys]) for row in results})
    print("total_programs:", total_programs)
    experiment_info_map["TotalPrograms"] = total_programs

    for k, fu in test_key:
        total_k = len({row[k] for row in results})
        print(f"total_{k}:", total_k)
        experiment_info_map[f"Total{k.capitalize()}"] = total_k

    finished = [r for r in results if r["finished"]]
    finished_trails = [r for r in finished if r["repeats"] == -1]
    total_tests_trail_finished = len(
        {tuple([row[k] for k, f in test_key]) for row in finished_trails}
    )
    print("total_tests_trail_finished:", total_tests_trail_finished)
    experiment_info_map["TotalTestsTrialFinished"] = total_tests_trail_finished

    total_incorrect_finished_trials = len(
        {
            tuple([row[k] for k, f in test_key])
            for row in finished_trails
            if not row["correct"]
        }
    )
    print("total_incorrect_finished_trials:", total_incorrect_finished_trials)
    experiment_info_map["TotalIncorrectTrialFinished"] = total_incorrect_finished_trials

    if min_wait_time is not None:
        total_defo_crashed_trails = len(
            {
                tuple([row[k] for k, f in test_key])
                for row in results
                if row["repeats"] == -1
                and not row["finished"]
                and row["waiting_time"] < min_wait_time
            }
        )
        print("total_defo_crashed_trails:", total_defo_crashed_trails)
        experiment_info_map["TotalCrashedTrial"] = total_defo_crashed_trails

    full_tests = [r for r in finished if r["repeats"] >= 0]

    total_full_tests = len({tuple([row[k] for k, f in test_key]) for row in full_tests})
    print("total_full_tests:", total_full_tests)
    experiment_info_map["TotalFullTests"] = total_full_tests

    total_full_programs = len(
        {tuple([row[k] for k in program_keys]) for row in full_tests}
    )
    print("total_full_programs:", total_full_programs)
    experiment_info_map["TotalFullPrograms"] = total_full_programs

    tests = {}
    for r in results:
        tests.setdefault(tuple([r[k] for k, fu in test_key]), []).append(r)

    stats = []

    for t_key, rs in tests.items():
        ts = [r["time"] / r["runs"] for r in rs if r["repeats"] >= 0]
        if len(ts) > 0:
            t_mean = np.mean(ts)
            t_min = np.min(ts)
            t_max = np.max(ts)
            t_std = np.std(ts)
            t_median = np.median(ts)
        else:
            t_mean = float("nan")
            t_min = float("nan")
            t_max = float("nan")
            t_std = float("nan")
            t_median = float("nan")
        trial_r = [r for r in rs if r["repeats"] == -1][0]
        t_trail = trial_r["time"] / trial_r["runs"]

        runss = [r["runs"] for r in rs if r["repeats"] >= 0]
        assert len(set(runss)) <= 1
        runs = runss[0] if len(runss) > 0 else 0

        if layout_gen is not None:
            l_idx = [k for k, fn in test_key].index("layout")
            l = t_key[l_idx]
            ls = {lo for lo in layouts if lo.number == l}
            if len(ls) != 1:
                print(ls)
            assert len(ls) == 1
            # print(l)
            l_map = {id.data: p for id, p in ls.pop().make_ptr_dict().items()}

            ptr_map = {name: l_map[target] for name, target in layout_targets.items()}
            sparses = layout_gen(ptr_map)

        else:
            sparses = {}

        program_key = []
        for k in program_keys:
            idx = [tk for tk, fn in test_key].index(k)
            program_key.append(t_key[idx])
        program_key = tuple(program_key)

        stats.append(
            Stat(
                t_key,
                program_key,
                {
                    "mean": float(t_mean),
                    "min": t_min,
                    "max": t_max,
                    "std": float(t_std),
                    "median": float(t_median),
                    "trial": t_trail,
                },
                {},
                sparses,
                {},
                runs,
                rs,
            )
        )

    all_time_options = ["mean", "min", "max", "std", "median", "trial"]
    if time_options is not None:
        assert set(time_options).issubset(set(all_time_options))
    else:
        time_options = all_time_options
    if idx_time_options is not None:
        assert set(idx_time_options).issubset(set(all_time_options))
    else:
        idx_time_options = all_time_options

    for val in all_time_options:
        sorted_stats = list(stats)
        sorted_stats.sort(key=lambda s: float('inf') if math.isnan(t := s.time[val]) else t)
        for i, stat in enumerate(sorted_stats):
            stat.idx_by[val] = i
        post_sort = sorted(stats, key=lambda s: float('inf') if math.isnan(t := s.time[val]) else t)
        assert sorted_stats == post_sort

    assert sorted(stats, key=lambda s: float('inf') if math.isnan(t := s.time["median"]) else t) == sorted(
        stats, key=lambda stat: stat.idx_by["median"]
    )

    fastest_median_test = min(
            [s for s in stats if all(r["correct"] for r in s.entries)], key=lambda stat: stat.idx_by["median"]
        )
    print(fastest_median_test.idx_by)
    for i, (k, fu) in enumerate(test_key):
        fastest_median_program_k = fastest_median_test.test_key[i]
        print(f"fastest_median_test_{k}: {fastest_median_program_k}")
        experiment_info_map[f"FastestMedianTest{k.capitalize()}"] = (
            fastest_median_program_k
        )

    fastest_median_program_time = fastest_median_test.time["median"]
    print(f"fastest_median_Test_time: {fastest_median_program_time}")
    experiment_info_map["FastestMedianTestTime"] = fastest_median_program_time

    if instances is not None:
        experiment_info_map["InstancesNormal"] = instances[0]
        experiment_info_map["InstancesExtra"] = instances[1]
        experiment_info_map["InstancesNormalTests"] = str(
            len({t.test_key for t in stats if t.runs == instances[0]})
        )
        experiment_info_map["InstancesExtraTests"] = str(
            len({t.test_key for t in stats if t.runs == instances[1]})
        )

    if tests_gen is not None:
        for stat in stats:
            test_path = et.get_test_path(experiment_dir, stat.program_key)
            m = tests_gen(stat, test_path)
            stat.test_map.update(m)
            pass




    if explore_dump_codes:
        segfaulted = 0
        time_aborted = 0

        for stat in stats:
            dump_path = et.get_dump_path(experiment_dir, stat.program_key, stat.test_key, -1)
            with open(f"{dump_path}/info", "r") as f:
                exit_code = None
                for line in f:
                    if line.startswith("Process return code: "):
                        assert exit_code is None
                        exit_code = int(line.removeprefix("Process return code: "))
            if exit_code != 0:
                print(f"{stat.test_key} :: exit code: {exit_code}")
                if exit_code == -9:
                    print("info:")
                    with open(f"{dump_path}/info", "r") as f:
                        for line in f:
                            print(line)
                    print("stdout:")
                    with open(f"{dump_path}/stdout", "r") as f:
                        for line in f:
                            if not line.startswith("# result incorrect!"):
                                print(line)
                            else:
                                print("result was incorrect!\n...")
                                break
                    print("stderr:")
                    with open(f"{dump_path}/stderr", "r") as f:
                        for line in f:
                            print(line)


    table_path = f"{data_dir}/table.csv"
    with open(table_path, "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(
            [
                *[k.replace("_", "") for k, fn in test_key],
                "runs",
                *[f"time{s.capitalize()}" for s in time_options],
                *[f"idx{s.capitalize()}" for s in idx_time_options],
                *[s for s in layout_export],
                *[s for s in stat_export],
                *[s for s in tests_export]
            ]
        )
        for stat in stats:
            export_stats = stat_gen(stat)

            csv_writer.writerow(
                [
                    *stat.test_key,
                    stat.runs,
                    *[stat.time[s] for s in time_options],
                    *[stat.idx_by[s] for s in idx_time_options],
                    *[str(stat.sparse[s]) for s in layout_export],
                    *[str(export_stats[s]) for s in stat_export],
                    *[str(stat.test_map[s]) for s in tests_export]
                ]
            )
    print(f"Written table to {table_path}")

    info_path = f"{data_dir}/info.tex"
    with open(info_path, "w") as f:
        for i, v in experiment_info_map.items():
            f.write(r"\newcommand{\dataInfo" + i.replace("_", "") + r"}{" + str(v) + "}\n")
    print(f"Written info to {info_path}")

    for key, val in experiment_info_map.items():
        print(f"{key}: {val}")
