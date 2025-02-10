import csv
import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import evaluation.evaluateTools as et
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.experimental import dlt


@dataclass(frozen=True)
class Stat:
    test_key: tuple
    program_key: tuple
    time: dict
    idx_by: dict
    info_map: dict
    runs: int
    entries: list[dict]


def make_general_info(
        test_key: list[str] = None,
        program_keys: list[str] = None,
        results: list[dict[str, Any]] = None,
        min_wait_time: float = None,
) -> dict[str, Any]:
    experiment_info_map = {}

    total_tests = len({tuple([row[k] for k in test_key]) for row in results})
    print("total_tests:", total_tests)
    experiment_info_map["TotalTests"] = total_tests

    total_programs = len({tuple([row[k] for k in program_keys]) for row in results})
    print("total_programs:", total_programs)
    experiment_info_map["TotalPrograms"] = total_programs

    for k in test_key:
        total_k = len({row[k] for row in results})
        print(f"total_{k}:", total_k)
        experiment_info_map[f"Total{k.capitalize()}"] = total_k

    finished = [r for r in results if r["finished"]]
    finished_trails = [r for r in finished if r["repeats"] == -1]
    total_tests_trail_finished = len(
        {tuple([row[k] for k in test_key]) for row in finished_trails}
    )
    print("total_tests_trail_finished:", total_tests_trail_finished)
    experiment_info_map["TotalTestsTrialFinished"] = total_tests_trail_finished

    total_incorrect_finished_trials = len(
        {
            tuple([row[k] for k in test_key])
            for row in finished_trails
            if not row["correct"]
        }
    )
    print("total_incorrect_finished_trials:", total_incorrect_finished_trials)
    experiment_info_map["TotalIncorrectTrialFinished"] = total_incorrect_finished_trials

    if min_wait_time is not None:
        total_defo_crashed_trails = len(
            {
                tuple([row[k] for k in test_key])
                for row in results
                if row["repeats"] == -1
                and not row["finished"]
                and row["waiting_time"] < min_wait_time
            }
        )
        print("total_defo_crashed_trails:", total_defo_crashed_trails)
        experiment_info_map["TotalCrashedTrial"] = total_defo_crashed_trails

    full_tests = [r for r in finished if r["repeats"] >= 0]

    total_full_tests = len({tuple([row[k] for k in test_key]) for row in full_tests})
    print("total_full_tests:", total_full_tests)
    experiment_info_map["TotalFullTests"] = total_full_tests

    for k in test_key:
        total_k = len({row[k] for row in full_tests})
        print(f"total_full_tests_{k}:", total_k)
        experiment_info_map[f"TotalFullTests{k.capitalize()}"] = total_k

    total_full_programs = len(
        {tuple([row[k] for k in program_keys]) for row in full_tests}
    )
    print("total_full_programs:", total_full_programs)
    experiment_info_map["TotalFullPrograms"] = total_full_programs

    return experiment_info_map

def make_stats(
        test_key: list[str] = None,
        program_keys: list[str] = None,
        results : list[dict[str, Any]] = None,
        time_options: list[str] = None,
        idx_time_options: list[str] = None,
) -> list[Stat]:
    all_time_options = ["mean", "min", "max", "std", "median", "trial"]
    if time_options is not None:
        assert set(time_options).issubset(set(all_time_options))
    else:
        time_options = all_time_options
    if idx_time_options is not None:
        assert set(idx_time_options).issubset(set(time_options))
    else:
        idx_time_options = time_options

    tests = {}
    for r in results:
        tests.setdefault(tuple([r[k] for k in test_key]), []).append(r)

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



        program_key = []
        for k in program_keys:
            idx = [tk for tk in test_key].index(k)
            program_key.append(t_key[idx])
        program_key = tuple(program_key)

        stats.append(
            Stat(
                t_key,
                program_key,
                {s:{
                    "mean": float(t_mean),
                    "min": t_min,
                    "max": t_max,
                    "std": float(t_std),
                    "median": float(t_median),
                    "trial": t_trail,
                }[s] for s in time_options},
                {},
                {},
                runs,
                rs,
            )
        )

    for val in idx_time_options:
        sorted_stats = list(stats)
        sorted_stats.sort(key=lambda s: float('inf') if math.isnan(t := s.time[val]) else t)
        for i, stat in enumerate(sorted_stats):
            stat.idx_by[val] = i
        post_sort = sorted(stats, key=lambda s: float('inf') if math.isnan(t := s.time[val]) else t)
        assert sorted_stats == post_sort

    return stats

def add_layout_info(
        stats: list[Stat],
        experiment_dir: str,
        test_key: list[str],
        layout_targets: dict[str, str] = None,
        layout_gen: Callable[[dict[str, dlt.PtrType]], dict[str, str]] = None,
) -> dict[str, Any]:
    experiment_info_map = {}

    layout_graph, layouts = et.load_dlt_layouts(experiment_dir)
    experiment_info_map["AllLayouts"] = len(layouts)
    if layout_targets is None or layout_gen is None:
        return experiment_info_map

    l_idx = test_key.index("layout")
    layouts_map = {l.number:l for l in layouts}
    et.log_progress_start(len(stats))
    for s in stats:
        et.log_progress_tick()
        t_key = s.test_key
        l_number = t_key[l_idx]
        ptr_layout_map = layouts_map[l_number]
        ls = {lo for lo in layouts if lo.number == l_number}
        if len(ls) != 1:
            print(ls)
        assert len(ls) == 1
        # print(l)
        l_map = ptr_layout_map.make_ptr_dict()
        ptr_map = {name: l_map[StringAttr(target)] for name, target in layout_targets.items()}
        s.info_map.update(layout_gen(ptr_map))
    et.log_progress_end()
    return experiment_info_map

def min_stat_by_info(
        stats: list[Stat],
        test_keys: list[str],
        key: Callable[[Stat], Any],
        name: str,
        min_name: str = None,
        extras: list[tuple[str, Callable[[Stat], Any]]] = None,
        take_max: bool = False,
) -> dict[str, Any]:
    experiment_info_map = {}

    valid_stats = [s for s in stats if s.runs>0 and all(r["correct"] for r in s.entries)]
    if len(valid_stats):
        if take_max:
            min_test = max(valid_stats, key=key)
        else:
            min_test = min(valid_stats, key=key)
    else:
        min_test = None

    for i, k in enumerate(test_keys):
        if min_test is not None:
            min_test_key = min_test.test_key[i]
        else:
            min_test_key = float("nan")
        print(f"{name}_{k}: {min_test_key}")
        experiment_info_map[f"{name}{k.capitalize()}"] = (
            min_test_key
        )

    if min_name is not None:
        if min_test is not None:
            min_val = key(min_test)
        else:
            min_val = float("nan")
        print(f"{name}{min_name}: {min_val}")
        experiment_info_map[f"{name}{min_name}"] = min_val

    if extras is not None:
        for n, fn in extras:
            if min_test is not None:
                val = fn(min_test)
            else:
                val = float("nan")
                
            print(f"{n}: {val}")
            experiment_info_map[f"{name}{n}"] = val

    return experiment_info_map

def instances_info(
        stats: list[Stat],
        instances: list[int] = None
) -> dict[str, str]:
    experiment_info_map = {}
    if instances is not None:
        experiment_info_map["InstancesNormal"] = instances[0]
        experiment_info_map["InstancesExtra"] = instances[1]
        experiment_info_map["InstancesNormalTests"] = str(
            len({t.test_key for t in stats if t.runs == instances[0]})
        )
        experiment_info_map["InstancesExtraTests"] = str(
            len({t.test_key for t in stats if t.runs == instances[1]})
        )
    return experiment_info_map


def generate_data(
    experiment_name: str,
    output_table_filename: str = "table.csv",
    output_info_filename: str = "info.tex",
    test_key_def: list[tuple[str, Callable[[str], Any]]] = None,
    program_keys: list[str] = None,
    layout_targets: dict[str, str] = None,
    layout_gen: Callable[[dict[str, dlt.PtrType]], dict[str, str]] = None,
    layout_export: list[str] = None,
    instances: tuple[int, int] = None,
    min_wait_time: float = None,
    time_options: list[str] = None,
    idx_time_options: list[str] = None,
    stat_gen: Callable[[Stat], dict[str, str]] = None,
    stat_export: list[str] = None,
    explore_dump_codes: bool = False,
    tests_gen: Callable[[Stat, str], dict[str, str]] = None,
    tests_export: list[str] = None,
    experiment_info_map: dict[str, str] = None,
    grouped_stats: list[tuple[Callable[[Stat], str], list[tuple[str, Callable[[Stat], Any], str]]]] = None,
    skip_stat_func: Callable[[Stat], bool] = lambda s: False,
) -> tuple[list[Stat], dict[str, Any]]:

    if test_key_def is None:
        test_key_def = [("layout", int), ("order", int)]
    test_keys = [k for k, f in test_key_def]
    if program_keys is None:
        program_keys = test_keys
    assert set(program_keys).issubset(set(test_keys))

    if layout_export is None:
        layout_export = []
    if stat_gen is None:
        stat_gen = lambda s: {}
    if stat_export is None:
        stat_export = []
    if tests_export is None:
        tests_export = []

    if experiment_info_map is None:
        experiment_info_map = {}

    experiment_dir = et.get_experiment_dir(experiment_name)
    data_dir = et.get_data_dir(experiment_name)
    os.makedirs(data_dir, exist_ok=True)

    print(f"Experiment name: {experiment_name}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Data directory: {data_dir}")


    results = et.load_results_file(
        f"{experiment_dir}/results.csv",
        columns=test_key_def
                + [
            ("repeats", int),
            ("runs", int),
            ("time", float),
            ("waiting_time", float),
            ("finished", et.parse_bool),
            ("correct", et.parse_bool),
        ],
    )
    experiment_info_map.update(make_general_info(test_keys, program_keys, results, min_wait_time))

    stats = make_stats(test_keys, program_keys, results, time_options, idx_time_options)

    layout_info_map = add_layout_info(stats, experiment_dir, test_keys, layout_targets, layout_gen)
    experiment_info_map.update(layout_info_map)

    experiment_info_map.update(min_stat_by_info(stats, test_keys, lambda s:s.time["median"], "FastestMedianTest", "Time"))

    if grouped_stats is not None:
        for k_fn, min_options  in grouped_stats:
            sub_stats_dict = {}
            for stat in stats:
                sub_stats_dict.setdefault(k_fn(stat), []).append(stat)
            for k, sub_stats in sub_stats_dict.items():
                for min_name, v_fn, v_min_name in min_options:
                    sub_min_results = min_stat_by_info(sub_stats, test_keys, v_fn, min_name, v_min_name)
                    sub_min_results = {f"{n}For{k}":r for n, r in sub_min_results.items()}
                    experiment_info_map.update(sub_min_results)
                sub_entries = [e for stat in sub_stats for e in stat.entries]
                sub_general_results = make_general_info(test_keys, program_keys, sub_entries, min_wait_time)
                sub_general_results = {f"{n}For{k}":r for n, r in sub_general_results.items()}
                experiment_info_map.update(sub_general_results)
                sub_instances_results = instances_info(sub_stats, instances)
                sub_instances_results = {f"{n}For{k}": r for n, r in sub_instances_results.items()}
                experiment_info_map.update(sub_instances_results)


    experiment_info_map.update(instances_info(stats, instances))

    if tests_gen is not None:
        for stat in stats:
            test_path = et.get_test_path(experiment_dir, stat.program_key)
            m = tests_gen(stat, test_path)
            stat.info_map.update(m)
            pass


    if explore_dump_codes:
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


    if output_table_filename:
        table_path = f"{data_dir}/{output_table_filename}"
        with open(table_path, "w") as f:
            csv_writer = csv.writer(f, delimiter=",")
            csv_writer.writerow(
                [
                    *[k.replace("_", "") for k, fn in test_key_def],
                    "runs",
                    *[f"time{s.capitalize()}" for s in (time_options if time_options is not None else [])],
                    *[f"idx{s.capitalize()}" for s in (idx_time_options if idx_time_options is not None else [])],
                    *[s for s in layout_export],
                    *[s for s in stat_export],
                    *[s for s in tests_export]
                ]
            )
            for stat in stats:
                if skip_stat_func(stat):
                    continue
                export_stats = stat_gen(stat)

                csv_writer.writerow(
                    [
                        *stat.test_key,
                        stat.runs,
                        *[stat.time[s] for s in (time_options if time_options is not None else [])],
                        *[stat.idx_by[s] for s in (idx_time_options if idx_time_options is not None else [])],
                        *[str(stat.info_map[s]) for s in layout_export],
                        *[str(export_stats[s]) for s in stat_export],
                        *[str(stat.info_map[s]) for s in tests_export]
                    ]
                )
        print(f"Written table to {table_path}")

    if output_info_filename:
        et.write_experiment_info_map(experiment_name, experiment_info_map, output_info_filename)

    print("Experiment Info Map::")
    for key, val in experiment_info_map.items():
        print(f"\t{key}: {val}")

    return stats, experiment_info_map
