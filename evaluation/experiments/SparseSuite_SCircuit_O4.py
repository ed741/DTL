import csv
import math
import typing

import numpy as np

from evaluation.evaluateExpriment import Stat, generate_data, make_general_info, min_stat_by_info
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import PtrType
import evaluation.evaluateTools as et

experiment_name = "sparseSuite/SCircuit_O4"
target_layouts = {"A": "R_0", "B": "R_1", "C": "R_2"}


A_layouts = {}
B_layouts = {}
C_layouts = {}
def gen_func(ptr_map: dict[str, PtrType])-> dict[str,str]:
    A_ptr = ptr_map["A"]
    B_ptr = ptr_map["B"]
    C_ptr = ptr_map["C"]

    CI_node = et.get_layout_node_for_dim(C_ptr.layout, "dim_3")

    CSparseIBuffer = -1
    if isinstance(CI_node, dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr):
        CSparseIBuffer = typing.cast(dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr, CI_node).buffer_scaler.data

    AI_node = et.get_layout_node_for_dim(A_ptr.layout, "dim_0")
    ASparseIBuffer = -1
    if isinstance(AI_node, dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr):
        ASparseIBuffer = typing.cast(dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr, AI_node).buffer_scaler.data

    AJ_node = et.get_layout_node_for_dim(A_ptr.layout, "dim_1")
    ASparseJBuffer = -1
    if isinstance(AJ_node, dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr):
        ASparseJBuffer = typing.cast(dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr, AJ_node).buffer_scaler.data

    AOrder = [{"dim_0":"I", "dim_1":"J"}[d] for d in et.get_dim_order(A_ptr.layout, {"dim_0", "dim_1"})]



    if A_ptr.layout in A_layouts:
        A_layout_idx = A_layouts[A_ptr.layout]
    else:
        A_layout_idx = len(A_layouts)
        A_layouts[A_ptr.layout] = A_layout_idx

    if B_ptr.layout in B_layouts:
        B_layout_idx = B_layouts[B_ptr.layout]
    else:
        B_layout_idx = len(B_layouts)
        B_layouts[B_ptr.layout] = B_layout_idx

    if C_ptr.layout in C_layouts:
        C_layout_idx = C_layouts[C_ptr.layout]
    else:
        C_layout_idx = len(C_layouts)
        C_layouts[C_ptr.layout] = C_layout_idx


    if isinstance(A_ptr.layout, dlt.IndexingLayoutAttr):
        A_idx = typing.cast(dlt.IndexingLayoutAttr, A_ptr.layout)
        if isinstance(A_idx.directChild, dlt.DenseLayoutAttr):
            A_dense = typing.cast(dlt.DenseLayoutAttr, A_idx.directChild)
            if A_dense.dimension.dimensionName.data == "dim_0":
                print(f"Found CSR: A_layout_idx: {A_layout_idx}")

    return {
            "CIsSparse": str(int(et.is_layout_sparse(C_ptr.layout))),
            "CCountSparse": str(int(et.count_layout_sparse(C_ptr.layout))),
            "CSparseIUp": str(int(isinstance(CI_node, dlt.UnpackedCOOLayoutAttr))),
            "CSparseIS": str(int(isinstance(CI_node, dlt.SeparatedCOOLayoutAttr))),
            "CSparseIBuffer": str(CSparseIBuffer),
            "ACountSparse": str(int(et.count_layout_sparse(A_ptr.layout))),
            "ASparseI": str(int(isinstance(AI_node, dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr))),
            "ASparseIUp": str(int(isinstance(AI_node, dlt.UnpackedCOOLayoutAttr))),
            "ASparseIS": str(int(isinstance(AI_node, dlt.SeparatedCOOLayoutAttr))),
            "ASparseIBuffer": str(ASparseIBuffer),
            "ASparseJ": str(int(isinstance(AJ_node, dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr))),
            "ASparseJUp": str(int(isinstance(AJ_node, dlt.UnpackedCOOLayoutAttr))),
            "ASparseJS": str(int(isinstance(AJ_node, dlt.SeparatedCOOLayoutAttr))),
            "ASparseJBuffer": str(ASparseJBuffer),
            "ASparseIJShare": str(int(AI_node == AJ_node)),
            "AOrderIJ": str(int(AOrder == ["I", "J"])),
            "AOrderJI": str(int(AOrder == ["J", "I"])),
            "AloIDX": str(A_layout_idx),
            "BloIDX": str(B_layout_idx),
            "CloIDX": str(C_layout_idx),
            }
    # return {}


array_info_map = {}

arrays_dir = f"{et.get_experiment_dir(experiment_name)}/arrays/SCircuit"
np_ref_a_0 = np.array(np.load(f"{arrays_dir}/np_ref_a_0.npy"))
np_ref_a_1 = np.array(np.load(f"{arrays_dir}/np_ref_a_1.npy"))
np_ref_a_val = np.array(np.load(f"{arrays_dir}/np_ref_a_val.npy"))
np_ref_b = np.array(np.load(f"{arrays_dir}/np_ref_b.npy"))
np_ref_c = np.array(np.load(f"{arrays_dir}/np_ref_c.npy"))
print(f"loaded arrays from {arrays_dir}")
# va = np.array(np_a)
# va[va!=0] = 1
a_shape = (170998, 170998)
assert np_ref_b.shape[0] == a_shape[1]
vb = np.array(np_ref_b)
vb[vb!=0] = 1
array_info_map["NnzA"] = np.count_nonzero(np_ref_a_val)
array_info_map["NnzB"] = np.count_nonzero(np_ref_b)
array_info_map["NnzC"] = np.count_nonzero(np_ref_c)
array_info_map["MultSparseAB"] = int(np.count_nonzero(vb[np_ref_a_1[np_ref_a_val!=0]]))
array_info_map["MultSparseA"] = int(np.count_nonzero(np_ref_a_val))
array_info_map["MultSparseB"] = int(np.count_nonzero(vb)*a_shape[0])
array_info_map["MultDense"] = a_shape[1] * np_ref_b.shape[0]

print(f"NnzA         : {array_info_map['NnzA']:12.0f}")
print(f"NnzB         : {array_info_map['NnzB']:12.0f}")
print(f"MultSparseAB :{array_info_map["MultSparseAB"]:12.0f}")
print(f"MultSparseA  :{array_info_map["MultSparseA"]:12.0f}")
print(f"MultSparseB  :{array_info_map["MultSparseB"]:12.0f}")
print(f"MultDense    :{array_info_map["MultDense"]:12.0f}")

experiment_info_map = {}
for s, v in array_info_map.items():
    experiment_info_map[f"Array{s}"] = str(v)

test_key_def = [("layout", int), ("order", int)]
test_keys = [n for n, fn in test_key_def]
program_keys = ["layout", "order"]

layout_export= [
    "CIsSparse",
    "CCountSparse",
    "CSparseIUp",
    "CSparseIS",
    "CSparseIBuffer",
    "ACountSparse",
    "ASparseI",
    "ASparseIUp",
    "ASparseIS",
    "ASparseIBuffer",
    "ASparseJ",
    "ASparseJUp",
    "ASparseJS",
    "ASparseJBuffer",
    "ASparseIJShare",
    "AOrderIJ",
    "AOrderJI",
    "AloIDX",
    "BloIDX",
    "CloIDX",
]
stats, experiment_info_map = generate_data(experiment_name,
              # output_table_filename="",
              output_info_filename="",
              test_key_def= test_key_def,
              program_keys = program_keys,
              layout_targets = target_layouts,
              layout_gen=gen_func,
              layout_export=layout_export,
              instances=(10,100),
              experiment_info_map=experiment_info_map,
              time_options=["median", "trial"],
              skip_stat_func=lambda stat: False,
              )


stats_ordered = sorted(stats, key=lambda k: (k.test_key[0],k.test_key[1]))

fastest = {}
for stat in stats_ordered:
    layout = stat.test_key[0]
    order = stat.test_key[1]
    time = stat.time["median"]
    if math.isnan(time):
        continue
    if (layout,) not in fastest:
        fastest[(layout,)] = (order, time)
    else:
        f_o, f_t = fastest[(layout,)]
        if time < f_t:
            fastest[(layout,)] = (order, time)

fastest_layout_stats_ordered = [stat for stat in stats_ordered if stat.runs>0 and fastest[(stat.test_key[0],)][0]==stat.test_key[1]]

print("Getting Memory use data:")
et.log_progress_start(len(fastest_layout_stats_ordered))
for i, stat in enumerate(fastest_layout_stats_ordered):
    et.log_progress_tick()
    layout = stat.test_key[0]
    order = stat.test_key[1]
    time = stat.time["median"]

    if stat.runs>0 and fastest[(layout,)][0]==order:
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
    "runs":lambda stat: stat.runs,
    "median":lambda stat: stat.time["median"],
    "multSparseAB":lambda stat: array_info_map["MultSparseAB"],
    "nnzAB":lambda stat: array_info_map["NnzA"] + array_info_map["NnzB"],
    "nnzC":lambda stat: array_info_map["NnzC"],
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
costb = (lambda stat: stat.info_map["all_allocated_total"] / (array_info_map["NnzA"] + array_info_map["NnzB"] + array_info_map["NnzC"]))

max_all_a = max([costa(s) for s in fastest_layout_stats_ordered])
max_all_b = max([costb(s) for s in fastest_layout_stats_ordered])

pareto_front_layouts = set()

tests = [stat for stat in fastest_layout_stats_ordered]

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
experiment_info_map.update(sub_min_results)
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
experiment_info_map.update(sub_min_results)
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
experiment_info_map.update(sub_min_results)

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

table_path = f"{et.OUTPUT_DATA_DIR}/{experiment_name}/table_pareto.csv"
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
    "runs":lambda stat: stat.runs,
    "median":lambda stat: stat.time["median"],
    "multSparseAB":lambda stat: array_info_map["MultSparseAB"],
    "nnzAB":lambda stat: array_info_map["NnzA"] + array_info_map["NnzB"],
    "nnzC":lambda stat: array_info_map["NnzC"],
    "memTotal":lambda stat: stat.info_map.get("all_allocated_total", float('nan')),
    # "memSetup":lambda stat: stat.info_map.get("setup_allocated_total", float('nan')),
    # "memBench":lambda stat: stat.info_map.get("benchmark_allocated_total", float('nan')),
    # "memCallTotal":lambda stat: stat.info_map.get("all_mem_calls_total", float('nan')),
    # "memCallSetup":lambda stat: stat.info_map.get("setup_mem_calls_total", float('nan')),
    # "memCallBench":lambda stat: stat.info_map.get("bench_mem_calls_total", float('nan')),
    "paretoFront":lambda stat: stat.info_map["ParetoFront"],
    # "paretoFrontShare":lambda stat: stat.info_map["ParetoFrontShare"],

} | {l: (lambda stat, ln=l: stat.info_map[str(ln)]) for l in layout_export}

table_path = f"{et.OUTPUT_DATA_DIR}/{experiment_name}/table2.csv"
with open(table_path, "w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(
        [k for k in columns.keys()]
    )
    i = 0
    while i < len(fastest_layout_stats_ordered):
        stat = fastest_layout_stats_ordered[i]
        layout = stat.test_key[0]
        order = stat.test_key[1]
        time = stat.time["median"]
        if stat.runs > 0 and fastest[(layout,)][0]==order:
            csv_writer.writerow([l(stat) for k, l in columns.items()])
        i += 1
print(f"Written table to {table_path}")

et.write_experiment_info_map(experiment_name, experiment_info_map)