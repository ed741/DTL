import csv
import math
import sys
from typing import Any, Callable

import numpy as np

from evaluation.evaluateExpriment import Stat, generate_data, make_general_info, min_stat_by_info
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import PtrType
import evaluation.evaluateTools as et

experiment_name = "matmul/random_sparse_static_O3_1024.1024.1024_0"
target_layouts = {"A": "R_0", "B": "R_1", "C": "R_2"}


A_layouts = {}
B_layouts = {}
C_layouts = {}
def gen_func(ptr_map: dict[str, PtrType])-> dict[str,str]:
    # A_ptr = ptr_map["A"]
    # B_ptr = ptr_map["B"]
    # C_ptr = ptr_map["C"]
    #
    # CI_node = et.get_layout_node_for_dim(C_ptr.layout, "dim_4")
    # CK_node = et.get_layout_node_for_dim(C_ptr.layout, "dim_5")
    #
    # CSparseIBuffer = 0
    # if isinstance(CI_node, dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr):
    #     CSparseIBuffer = typing.cast(dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr, CI_node).buffer_scaler.data
    # CSparseKBuffer = 0
    # if isinstance(CK_node, dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr):
    #     CSparseKBuffer = typing.cast(dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr, CK_node).buffer_scaler.data
    #
    # COrder = [{"dim_4":"I", "dim_5":"K"}[d] for d in et.get_dim_order(C_ptr.layout, {"dim_4", "dim_5"})]
    #
    # if A_ptr.layout in A_layouts:
    #     A_layout_idx = A_layouts[A_ptr.layout]
    # else:
    #     A_layout_idx = len(A_layouts)
    #     A_layouts[A_ptr.layout] = A_layout_idx
    #
    # if B_ptr.layout in B_layouts:
    #     B_layout_idx = B_layouts[B_ptr.layout]
    # else:
    #     B_layout_idx = len(B_layouts)
    #     B_layouts[B_ptr.layout] = B_layout_idx
    #
    # if C_ptr.layout in C_layouts:
    #     C_layout_idx = C_layouts[C_ptr.layout]
    # else:
    #     C_layout_idx = len(C_layouts)
    #     C_layouts[C_ptr.layout] = C_layout_idx
    #
    # return {
    #         "CIsSparse": str(int(et.is_layout_sparse(C_ptr.layout))),
    #         "CCountSparse": str(int(et.count_layout_sparse(C_ptr.layout))),
    #         "CSparseI": str(int(et.is_layout_sparse_dim(C_ptr.layout, ["dim_4"]))),
    #         "CSparseK": str(int(et.is_layout_sparse_dim(C_ptr.layout, ["dim_5"]))),
    #         "CSparseIUp": str(int(isinstance(CI_node, dlt.UnpackedCOOLayoutAttr))),
    #         "CSparseIS": str(int(isinstance(CI_node, dlt.SeparatedCOOLayoutAttr))),
    #         "CSparseKUp": str(int(isinstance(CK_node, dlt.UnpackedCOOLayoutAttr))),
    #         "CSparseKS": str(int(isinstance(CK_node, dlt.SeparatedCOOLayoutAttr))),
    #         "CSparseIBuffer": str(CSparseIBuffer),
    #         "CSparseKBuffer": str(CSparseKBuffer),
    #         "COrderIK": str(int(COrder == ["I", "K"])),
    #         "COrderKI": str(int(COrder == ["K", "I"])),
    #         "AloIDX": str(A_layout_idx),
    #         "BloIDX": str(B_layout_idx),
    #         "CloIDX": str(C_layout_idx),
    #         }
    return {}


arrays = {
    0.1:"0_0.1_0.1",
    0.01:"0_0.01_0.01",
    0.001:"0_0.001_0.001",
    0.0001:"0_0.0001_0.0001",
    0.00001:"0_1e-05_1e-05",
          }
rate_variables = {0.1: "A", 0.01: "B", 0.001: "C", 0.0001: "D", 0.00001: "E", }

array_info_map = {k:{} for k in arrays}

for rate, folder in arrays.items():
    arrays_dir = f"{et.get_experiment_dir(experiment_name)}/arrays/{folder}"
    np_a = np.array(np.load(f"{arrays_dir}/np_a.npy"))
    np_b = np.array(np.load(f"{arrays_dir}/np_b.npy"))
    np_c = np.array(np.load(f"{arrays_dir}/np_c.npy"))
    print(f"loaded arrays from {arrays_dir}")
    va = np.array(np_a)
    va[va!=0] = 1
    vb = np.array(np_b)
    vb[vb!=0] = 1
    vc = np.matmul(va, vb)
    array_info_map[rate]["NnzA"] = np.count_nonzero(np_a)
    array_info_map[rate]["NnzB"] = np.count_nonzero(np_b)
    array_info_map[rate]["NnzC"] = np.count_nonzero(np_c)
    array_info_map[rate]["MultSparseAB"] = int(np.sum(np.matmul(va, vb)))
    array_info_map[rate]["MultSparseA"] = int(np.sum(np.matmul(va, np.ones_like(vb))))
    array_info_map[rate]["MultSparseB"] = int(np.sum(np.matmul(np.ones_like(va), vb)))
    array_info_map[rate]["MultDense"] = 1024**3

    print(f"NnzA         : {array_info_map[rate]['NnzA']:12.0f}")
    print(f"NnzB         : {array_info_map[rate]['NnzB']:12.0f}")
    print(f"MultSparseAB :{array_info_map[rate]["MultSparseAB"]:12.0f}")
    print(f"MultSparseA  :{array_info_map[rate]["MultSparseA"]:12.0f}")
    print(f"MultSparseB  :{array_info_map[rate]["MultSparseB"]:12.0f}")
    print(f"MultDense    :{array_info_map[rate]["MultDense"]:12.0f}")

experiment_info_map = {}
for rate, name in rate_variables.items():
    for s, v in array_info_map[rate].items():
        experiment_info_map[f"Array{s}For{name}"] = v

test_key_def = [("layout", int), ("order", int), ("rate_a", float), ("rate_b", float)]
test_keys = [n for n, fn in test_key_def]
program_keys = ["layout", "order"]
stats, experiment_info_map = generate_data(experiment_name,
              # output_table_filename="",
              output_info_filename="",
              test_key_def= test_key_def,
              program_keys = program_keys,
              # layout_targets = target_layouts,
              layout_gen=gen_func,
              layout_export=[
                  # "CIsSparse",
                  # "CCountSparse",
                  # "CSparseI",
                  # "CSparseK",
                  # "CSparseIUp",
                  # "CSparseIS",
                  # "CSparseKUp",
                  # "CSparseKS",
                  # "CSparseIBuffer",
                  # "CSparseKBuffer",
                  # "COrderIK",
                  # "COrderKI",
                  # "AloIDX",
                  # "BloIDX",
                  # "CloIDX",
              ],
              instances=(10,100),
              grouped_stats=[
                  (
                      lambda s: f"Rate{(rate_variables[s.test_key[2]])}",
                      [(
                          "FastestMedianTest",
                          lambda s: s.time['median'],
                          "Time",
                      )]
                  )
              ],
              experiment_info_map=experiment_info_map,
              )


stats_ordered = sorted(stats, key=lambda k: (k.test_key[0],-k.test_key[2],-k.test_key[3],k.test_key[1]))

fastest = {}
for stat in stats_ordered:
    layout = stat.test_key[0]
    order = stat.test_key[1]
    rate = stat.test_key[2]
    time = stat.time["median"]
    if math.isnan(time):
        continue
    if (layout, rate) not in fastest:
        fastest[(layout, rate)] = (order, time)
    else:
        f_o, f_t = fastest[(layout, rate)]
        if time < f_t:
            fastest[(layout, rate)] = (order, time)

fastest_layout_stats_ordered = [stat for stat in stats_ordered if stat.runs>0 and fastest[(stat.test_key[0], stat.test_key[2])][0]==stat.test_key[1]]

et.log_progress_start(len(fastest_layout_stats_ordered))
for i, stat in enumerate(fastest_layout_stats_ordered):
    et.log_progress_tick()
    layout = stat.test_key[0]
    order = stat.test_key[1]
    rate = stat.test_key[2]
    time = stat.time["median"]

    if stat.runs>0 and fastest[(layout, rate)][0]==order:
        setup, benchmark, check, clean = et.parse_for_memory_use(et.get_experiment_dir(experiment_name), stat.program_key, stat.test_key)
        stat.info_map["all_allocated_total"] = setup.allocated_total + benchmark.allocated_total + check.allocated_total + clean.allocated_total
        stat.info_map["setup_allocated_total"] = setup.allocated_total
        stat.info_map["benchmark_allocated_total"] = benchmark.allocated_total
        
        stat.info_map["all_mem_calls_total"] = setup.mallocs + setup.reallocs + benchmark.mallocs + benchmark.reallocs + check.mallocs + check.reallocs + clean.mallocs + clean.reallocs
        stat.info_map["setup_mem_calls_total"] = setup.mallocs + setup.reallocs
        stat.info_map["bench_mem_calls_total"] = benchmark.mallocs + benchmark.reallocs
    else:
        assert False
et.log_progress_end()



pareto_columns = {
    "layout":lambda stat: stat.test_key[0],
    "order":lambda stat: stat.test_key[1],
    "rate":lambda stat: stat.test_key[2],
    "runs":lambda stat: stat.runs,
    "median":lambda stat: stat.time["median"],
    "multSparseAB":lambda stat: array_info_map[rate]["MultSparseAB"],
    "nnzAB":lambda stat: array_info_map[rate]["NnzA"] + array_info_map[rate]["NnzB"],
    "nnzC":lambda stat: array_info_map[rate]["NnzC"],
    "memTotal":lambda stat: stat.info_map.get("all_allocated_total", float('nan')),
    "memSetup":lambda stat: stat.info_map.get("setup_allocated_total", float('nan')),
    "memBench":lambda stat: stat.info_map.get("benchmark_allocated_total", float('nan')),
    "memCallTotal":lambda stat: stat.info_map.get("all_mem_calls_total", float('nan')),
    "memCallSetup":lambda stat: stat.info_map.get("setup_mem_calls_total", float('nan')),
    "memCallBench":lambda stat: stat.info_map.get("bench_mem_calls_total", float('nan')),
}

## ParetoFront
for stat in fastest_layout_stats_ordered:
    stat.info_map["ParetoFront"] = 0
costa = lambda stat: stat.time["median"]
costb = (lambda stat: stat.info_map["all_allocated_total"] / (array_info_map[rate]["NnzA"] + array_info_map[rate]["NnzB"] + array_info_map[rate]["NnzC"]))

max_all_a = max([costa(s) for s in fastest_layout_stats_ordered])
max_all_b = max([costb(s) for s in fastest_layout_stats_ordered])

pareto_front_layouts = set()

for rate, rate_name in rate_variables.items():
    # print(rate, rate_name)
    tests = [stat for stat in fastest_layout_stats_ordered if stat.test_key[2]==rate]
    if len(tests)==0:
        continue

    min_time_func = lambda s: (
        s.time["median"],
        s.info_map["all_allocated_total"],
        s.info_map["benchmark_allocated_total"],
    )
    sub_min_results = min_stat_by_info(tests, test_keys, min_time_func, "FLFastestMedianTest", None,
                                       [
                                           ("MedianTime", lambda stat: stat.time["median"]),
                                           ("AllocAll", lambda stat: stat.info_map["all_allocated_total"]),
                                           ("AllocBench", lambda stat: stat.info_map["benchmark_allocated_total"]),
                                       ])
    experiment_info_map.update({f"{k}For{rate_name}": v for k, v in sub_min_results.items()})
    min_alloc_all_func = lambda s: (
        s.info_map["all_allocated_total"],
        s.time["median"],
        s.info_map["benchmark_allocated_total"],
    )
    sub_min_results = min_stat_by_info(tests, test_keys, min_alloc_all_func, "FLMinAllocAll", None,
                                       [
                                           ("MedianTime", lambda stat: stat.time["median"]),
                                           ("AllocAll", lambda stat: stat.info_map["all_allocated_total"]),
                                           ("AllocBench", lambda stat: stat.info_map["benchmark_allocated_total"]),
                                       ])
    experiment_info_map.update({f"{k}For{rate_name}": v for k, v in sub_min_results.items()})
    min_alloc_bench_func = lambda s: (
        s.info_map["benchmark_allocated_total"],
        s.time["median"],
        s.info_map["all_allocated_total"],
    )
    sub_min_results = min_stat_by_info(tests, test_keys, min_alloc_bench_func, "FLMinAllocBench", None,
                                       [
                                           ("MedianTime", lambda stat: stat.time["median"]),
                                           ("AllocAll", lambda stat: stat.info_map["all_allocated_total"]),
                                           ("AllocBench", lambda stat: stat.info_map["benchmark_allocated_total"]),
                                       ])
    experiment_info_map.update({f"{k}For{rate_name}": v for k, v in sub_min_results.items()})


    costs = np.array([[costa(stat), costb(stat)] for stat in tests])
    front = et.is_pareto_efficient(costs)
    # print(front)
    front_stats = [stat for i, stat in enumerate(tests) if front[i]]
    for stat in front_stats:
        stat.info_map["ParetoFront"]=1
        pareto_front_layouts.add(stat.test_key[0])
    front_stats.sort(key=costa)
    mina = np.min(costs[:,0])
    minb = np.min(costs[:,1])

    table_path = f"{et.OUTPUT_DATA_DIR}/{experiment_name}/table_pareto{rate_name}.csv"
    with open(table_path, "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(
            ["a", "b", "end"] + [k for k in pareto_columns.keys()]
        )
        csv_writer.writerow([mina, max_all_b, "1"] + [l(front_stats[0]) for k, l in pareto_columns.items()])
        for stat in front_stats:
            csv_writer.writerow([costa(stat), costb(stat), "0"] + [l(stat) for k, l in pareto_columns.items()])
        csv_writer.writerow([max_all_a, minb, "2"] + [l(front_stats[-1]) for k, l in pareto_columns.items()])
    print(f"Written table to {table_path}")

for stat in fastest_layout_stats_ordered:
    stat.info_map["ParetoFrontShare"]=1 if stat.test_key[0] in pareto_front_layouts else 0


columns = {
    "layout":lambda stat: stat.test_key[0],
    "order":lambda stat: stat.test_key[1],
    "rate":lambda stat: stat.test_key[2],
    "runs":lambda stat: stat.runs,
    "median":lambda stat: stat.time["median"],
    "multSparseAB":lambda stat: array_info_map[rate]["MultSparseAB"],
    "nnzAB":lambda stat: array_info_map[rate]["NnzA"] + array_info_map[rate]["NnzB"],
    "nnzC":lambda stat: array_info_map[rate]["NnzC"],
    "memTotal":lambda stat: stat.info_map.get("all_allocated_total", float('nan')),
    "memSetup":lambda stat: stat.info_map.get("setup_allocated_total", float('nan')),
    "memBench":lambda stat: stat.info_map.get("benchmark_allocated_total", float('nan')),
    "memCallTotal":lambda stat: stat.info_map.get("all_mem_calls_total", float('nan')),
    "memCallSetup":lambda stat: stat.info_map.get("setup_mem_calls_total", float('nan')),
    "memCallBench":lambda stat: stat.info_map.get("bench_mem_calls_total", float('nan')),
    "paretoFront":lambda stat: stat.info_map["ParetoFront"],
    "paretoFrontShare":lambda stat: stat.info_map["ParetoFrontShare"],
}
table_path = f"{et.OUTPUT_DATA_DIR}/{experiment_name}/table2.csv"
with open(table_path, "w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(
        [k for k in columns.keys()]
    )
    last_program_id = (stats[0].test_key[0])
    i = 0
    while i < len(fastest_layout_stats_ordered):
        stat = fastest_layout_stats_ordered[i]
        layout = stat.test_key[0]
        order = stat.test_key[1]
        rate = stat.test_key[2]
        time = stat.time["median"]
        if stat.runs > 0 and fastest[(layout, rate)][0]==order:
            program_id = layout
            if program_id != last_program_id:
                csv_writer.writerow([float("nan")]*8)
            last_program_id = program_id
            csv_writer.writerow([l(stat) for k, l in columns.items()])
        i += 1
print(f"Written table to {table_path}")

et.write_experiment_info_map(experiment_name, experiment_info_map)