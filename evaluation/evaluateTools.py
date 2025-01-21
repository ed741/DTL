import csv
import os
import pickle
import re
import typing
from collections import namedtuple
from collections.abc import Callable
from typing import Any

from scipy.optimize import direct

from xdsl.dialects.experimental import dlt
from xdsl.transforms.experimental.dlt.generate_dlt_layouts import PtrMapping
from xdsl.transforms.experimental.dlt.layout_graph import LayoutGraph

if "DTL_BASE_DIR" in os.environ:
    DTL_BASE_DIR = os.environ["DTL_BASE_DIR"]
else:
    DTL_BASE_DIR = ".."

RESULTS_DIR_BASE = f"{DTL_BASE_DIR}/results"

if "FIGURES_DIR" in os.environ:
    FIGURES_DIR = os.environ["FIGURES_DIR"]
else:
    FIGURES_DIR = "./figures"

if "OUTPUT_DATA_DIR" in os.environ:
    OUTPUT_DATA_DIR = os.environ["OUTPUT_DATA_DIR"]
else:
    OUTPUT_DATA_DIR = "./data"


def parse_bool(input: str) -> bool:
    if input in ["True", "true"]:
        return True
    elif input in ["False", "false"]:
        return False
    else:
        raise ValueError(f"Unrecognized bool input: {input}")

def load_results_file(results_file, columns: list[tuple[str, Callable[[str], Any]]]) -> list[dict]:
    with open(results_file) as f:
        csv_reader = csv.reader(f, delimiter=',')
        header = next(csv_reader)
        indices = [(c, header.index(c), fu) for c, fu in columns]
        output = []
        for row in csv_reader:
            output.append({c:fu(row[i]) for c, i, fu in indices})
    return output

def write_results_file(results_file, columns: tuple[str, ...], data: list[tuple]) -> None:
    with open(results_file, "w") as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(columns)
        for row in data:
            csv_writer.writerow(row)

def load_dlt_layouts(test_dir) -> tuple[LayoutGraph, list[PtrMapping]]:
    usage_path= f"{test_dir}/store_use"
    key_num = None
    store_dir = None
    with open(usage_path) as f:
        lines = f.readlines()
        for line in reversed(lines):
            match = re.match(r"^[\d-]+ [\d:]+\.\d+ (?:Loaded (\d+)|Unlocked and generated record (\d+) by \d+) from store: (.*layouts)$", line)
            if match is None:
                continue
            g = match.groups()
            key_num = int(g[0]) if g[0] is not None else int(g[1])
            store_dir = g[2]
            assert key_num is not None
            break
    if key_num is None:
        raise ValueError(f"Could not find key num in {usage_path}")
    if store_dir is None:
        raise ValueError(f"Could not find store_dir in {usage_path}")

    with open(f"{DTL_BASE_DIR}/{store_dir}/keys/{key_num}", "rb") as f:
        loaded_key = pickle.load(f)
    with open(f"{DTL_BASE_DIR}/{store_dir}/values/{key_num}", "rb") as f:
        loaded_ptr_mappings = pickle.load(f)
    if isinstance(loaded_key, tuple):
        layout_graph = loaded_key[0]
    else:
        layout_graph = loaded_key
        print("warning: loaded key is old format")
    if not isinstance(layout_graph, LayoutGraph):
        raise ValueError(f"layout graph is not a LayoutGraph: {layout_graph}")
    if not isinstance(loaded_ptr_mappings, list):
        raise ValueError(f"loaded ptr mappings is not a set: {loaded_ptr_mappings}")
    for ptr_mapping in loaded_ptr_mappings:
        if not isinstance(ptr_mapping, PtrMapping):
            raise ValueError(f"ptr mapping is not a PtrMapping: {ptr_mapping}")

    return layout_graph, loaded_ptr_mappings


def get_dump_path(experiment_path: str, program_id: tuple[Any,...], test_id: tuple[Any,...], repeat: int) -> str:
    dump_path = f"{experiment_path}/dump/{'_'.join([str(i) for i in program_id])}/{'-'.join([str(i) for i in test_id])}/{str(repeat)}"
    files = os.listdir(dump_path)
    assert len(files) > 0, "No dump files found"
    path = f"{dump_path}/{files[0]}"
    return path

def get_test_path(experiment_path: str, program_id: tuple[Any,...]) -> str:
    path = f"{experiment_path}/tests/{'_'.join([str(i) for i in program_id])}"
    return path


MemoryStats = namedtuple("MemoryStats", ["mallocs", "reallocs", "frees", "realloc_mallocs", "running_total", "allocated_total", "freed_total", "allocated_best_case", "allocated_worst_case", "max"])
def parse_for_memory_use(experiment_path: str, path_id: tuple[Any,...], rest_of_id: tuple[Any,...]) -> tuple[MemoryStats, MemoryStats, MemoryStats, MemoryStats]:
    dump_path = get_dump_path(experiment_path, path_id, rest_of_id, -2)
    path = f"{dump_path}/stdout"

    stages = ["~setup-done~", "~benchmark-done~", "~testing-done~", "~tear-down-done~"]
    stage_idx = 0
    stats = []

    memory_map = {}

    count_malloc = 0
    count_realloc = 0
    count_free = 0
    count_realloc_malloc = 0
    running_total = 0
    allocated_total = 0
    freed_total = 0
    allocated_best_case = 0
    allocated_worst_case = 0
    running_max = 0
    with open(path) as f:
        for line in f.readlines():
            if stage_idx >= len(stages):
                break
            if (match := re.match(f"^{stages[stage_idx]}$",line)) is not None:
                stage_idx = stage_idx + 1
                stats.append(MemoryStats(count_malloc, count_realloc, count_free, count_realloc_malloc, running_total, allocated_total, freed_total, allocated_best_case, allocated_worst_case, running_max))


                count_malloc = 0
                count_realloc = 0
                count_free = 0
                count_realloc_malloc = 0

                allocated_total = 0
                freed_total = 0
                allocated_best_case = 0
                allocated_worst_case = 0
                running_max = running_total

            elif (match := re.match(r"^# called malloc\(([0-9]+)\) -> (0x[0-9a-f]+)$", line)) is not None:
                count_malloc += 1

                size = int(match.group(1))
                ptr = match.group(2)
                assert ptr not in memory_map
                memory_map[ptr] = size

                running_total += size
                allocated_total += size
                allocated_best_case += size
                allocated_worst_case += size
            elif (match := re.match(r"^# called realloc\((0x[0-9a-f]+), ([0-9]+)\) -> (0x[0-9a-f]+)$", line)) is not None:
                count_realloc += 1

                old_ptr = match.group(1)
                size = int(match.group(2))
                new_ptr = match.group(3)
                if old_ptr != new_ptr:
                    count_realloc_malloc += 1

                assert old_ptr in memory_map
                old_size = memory_map[old_ptr]
                del memory_map[old_ptr]

                assert new_ptr not in memory_map
                memory_map[new_ptr] = size

                if old_ptr != new_ptr:
                    running_total += size
                    running_max = max(running_total, running_max)
                    running_total -= old_size
                    allocated_total += size
                    freed_total += old_size
                else:
                    running_total += size - old_size
                    allocated_total -= old_size
                    allocated_total += max(old_size, size)

                allocated_best_case -= old_size
                allocated_best_case += max(old_size, size)

                allocated_worst_case += size
            elif (match := re.match(r"^# called free\((0x[0-9a-f]+)\)$", line)) is not None:
                count_free += 1

                ptr = match.group(1)
                assert ptr in memory_map
                size = memory_map[ptr]
                del memory_map[ptr]
                freed_total += size
                running_total -= size
            running_max = max(running_total, running_max)

    assert len(memory_map) == 0
    assert len(stats) == 4
    return stats[0], stats[1], stats[2], stats[3]



def __flatten(l: dlt.Layout) -> list[dlt.Layout]:
        return [l] + [cl for c in l.get_children() for cl in __flatten(c)]


def get_layout_node_for_dim(layout: dlt.Layout, dim:str) -> dlt.Layout:
    layout_nodes = __flatten(layout)
    for node in layout_nodes:
        if isinstance(node, dlt.DenseLayoutAttr):
            if dim == typing.cast(dlt.DenseLayoutAttr, node).dimension.dimensionName.data:
                return node
        elif isinstance(node, dlt.UnpackedCOOLayoutAttr):
            if dim in [d.dimensionName.data for d in typing.cast(dlt.UnpackedCOOLayoutAttr, node).dimensions]:
                return node
        elif isinstance(node, dlt.SeparatedCOOLayoutAttr):
            if dim in [d.dimensionName.data for d in typing.cast(dlt.UnpackedCOOLayoutAttr, node).dimensions]:
                return node
        elif isinstance(node, dlt.ArithReplaceLayoutAttr):
            arith_replace_node = typing.cast(dlt.ArithReplaceLayoutAttr, node)
            if dim in [r.outer_dimension for r in arith_replace_node.replacements]:
                return get_layout_node_for_dim(arith_replace_node.child, arith_replace_node.inner_dimension().dimensionName.data)


def get_dim_order(layout: dlt.Layout, dims: set[str]) -> list[str]:
    assert len({e for e in layout.contents_type.elements if set(d.dimensionName.data for d in e.dimensions).issuperset(dims)})==1
    assert len({e for e in layout.contents_type.elements if set(d.dimensionName.data for d in e.dimensions)==set(dims)}) == 1
    if isinstance(layout, dlt.PrimitiveLayoutAttr | dlt.ConstantLayoutAttr):
        assert len(dims)==0
        return []
    elif isinstance(layout, dlt.DenseLayoutAttr):
        node = typing.cast(dlt.DenseLayoutAttr, layout)
        dim = node.dimension.dimensionName.data
        assert dim in dims
        return [dim] + get_dim_order(node.child, dims-{dim})
    elif isinstance(layout, dlt.MemberLayoutAttr):
        node = typing.cast(dlt.MemberLayoutAttr, layout)
        return get_dim_order(node.child, dims)
    elif isinstance(layout, dlt.StructLayoutAttr):
        node = typing.cast(dlt.StructLayoutAttr, layout)
        children = [c for c in node.children if any({d.dimensionName.data for d in e.dimensions}==dims for e in c.contents_type.elements)]
        assert len(children)==1
        return get_dim_order(children[0], dims)
    elif isinstance(layout, dlt.IndexingLayoutAttr):
        node = typing.cast(dlt.IndexingLayoutAttr, layout)
        direct_dims = {d.dimensionName.data for d in node.directChild.contents_type.all_dimension_attributes()} & dims
        return get_dim_order(node.directChild, direct_dims) + get_dim_order(node.indexedChild, dims - direct_dims)
    elif isinstance(layout, dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr):
        node = typing.cast(dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr, layout)
        i_dims = [d.dimensionName.data for d in node.dimensions]
        assert dims.issuperset(i_dims)
        return i_dims + get_dim_order(node.child, dims.difference(i_dims))
    elif isinstance(layout, dlt.ArithReplaceLayoutAttr):
        node = typing.cast(dlt.ArithReplaceLayoutAttr, layout)
        replacements = [r for r in node.replacements if r.outer_dimension.dimensionName.data in dims]
        assert len(replacements)==1
        outer = replacements[0].outer_dimension.dimensionName.data
        assert outer in dims
        inner = replacements[0].inner_dimension.dimensionName.data
        assert inner not in dims
        return get_dim_order(node.child, (dims - {outer})|{inner})
    else:
        raise NotImplementedError()

def is_layout_sparse(layout: dlt.Layout) -> bool:
    layout_nodes = __flatten(layout)
    for node in layout_nodes:
        if isinstance(node, dlt.IndexingLayoutAttr):
            return True
    return False

def count_layout_sparse(layout: dlt.Layout) -> int:
    layout_nodes = __flatten(layout)
    count = 0
    for node in layout_nodes:
        if isinstance(node, dlt.IndexingLayoutAttr):
            count += 1
    return count


def is_layout_sparse_dim(layout: dlt.Layout, dims: list[str]) -> bool:
    layout_nodes = __flatten(layout)
    for node in layout_nodes:
        if isinstance(node, dlt.UnpackedCOOLayoutAttr):
            coo_node = typing.cast(dlt.UnpackedCOOLayoutAttr, node)
            for d in coo_node.dimensions:
                if d.dimensionName.data in dims:
                    return True
        elif isinstance(node, dlt.SeparatedCOOLayoutAttr):
            coo_node = typing.cast(dlt.SeparatedCOOLayoutAttr, node)
            for d in coo_node.dimensions:
                if d.dimensionName.data in dims:
                    return True
    return False

def layout_has_airth_replace(layout: dlt.Layout) -> bool:
    layout_nodes = __flatten(layout)
    for node in layout_nodes:
        if isinstance(node, dlt.ArithReplaceLayoutAttr):
            return True
    return False

def layout_airth_replaced(layout: dlt.Layout) -> list[set[str]]:
    r = []
    layout_nodes = __flatten(layout)
    for node in layout_nodes:
        if isinstance(node, dlt.ArithReplaceLayoutAttr):
            arith_node = typing.cast(dlt.ArithReplaceLayoutAttr, node)
            r.append({p.outer_dimension.dimensionName.data for p in arith_node.replacements})
    return r