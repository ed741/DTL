import csv
import sys

from xdsl.dialects import builtin
from xdsl.dialects.experimental import dlt
from xdsl.transforms.experimental.dlt.generate_dlt_layouts import LayoutGenerator, ReifyConfig
from xdsl.transforms.experimental.dlt.layout_graph import LayoutGraph

results = {}
table_path = f"layouts_table.csv"
with open(table_path, "r") as f:
    csv_reader = csv.reader(f, delimiter=",")
    csv_reader.__next__()
    for row in csv_reader:
        results[int(row[0])] = int(row[1])
print(f"Read table from {table_path}")

reify_config = ReifyConfig(
    dense=True,
    unpacked_coo_buffer_options=frozenset([0]),
    separated_coo_buffer_options=frozenset([0]),
    separated_coo_buffer_index_options=frozenset([builtin.i32]),
    coo_minimum_dims=1,
    arith_replace=False,
    force_arith_replace_immediate_use=True,
    permute_structure_size_threshold=-1,
    members_first=True,
)
previous_generated_layouts = {}

for n in range(int(sys.argv[1])):
    if n in results:
        continue
    dims = [dlt.DimensionAttr(str(i+1),i+1) for i in range(n)]
    dlt_type = dlt.TypeType([([],dims,builtin.f32)])
    ptr_type = dlt.PtrType(dlt_type, base=True, identity=f"test_{n}")
    allocate = dlt.AllocOp(ptr_type, {},[])
    layout_graph = LayoutGraph()
    layout_graph.add_ssa_value(allocate.res)


    generator = LayoutGenerator(layout_graph, {ptr_type.identification:reify_config})
    generator._generated_layouts = previous_generated_layouts
    layouts = generator.generate_mappings()
    print(f" for n = {n}: |layouts| = {len(layouts)}")
    results[n] = len(layouts)


print(results)

with open(table_path, "w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(
        [
            "n",
            "layouts",
        ]
    )

    for n, layouts in results:
        csv_writer.writerow(
            [
                str(n), str(layouts)
            ]
        )
print(f"Written table to {table_path}")
