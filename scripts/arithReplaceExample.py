from dtl.visualise import LayoutPlotter
from xdsl.dialects import builtin
from xdsl.dialects.experimental import dlt

a = dlt.PrimitiveLayoutAttr(builtin.f32)
a = dlt.AbstractLayoutAttr([dlt.AbstractChildAttr([dlt.MemberAttr("abstract_a_parts", "part_a")], [], a)])
a = dlt.MemberLayoutAttr(a, "hidden", "a")


b = dlt.PrimitiveLayoutAttr(builtin.f32)
b = dlt.AbstractLayoutAttr([dlt.AbstractChildAttr([dlt.MemberAttr("abstract_b_parts", "part_b")], [], b)])
b = dlt.MemberLayoutAttr(b, "hidden", "b")

s = dlt.StructLayoutAttr([a, b])
d = dlt.DenseLayoutAttr(s, dlt.DimensionAttr("common_dim", 16))

r = dlt.ArithReplaceLayoutAttr(d, [
    dlt.ArithReplacementAttr(
        dlt.DimensionAttr("dim_a", 16),
        dlt.DimensionAttr("common_dim", 16),
        dlt.MemberAttr("hidden","a")),
    dlt.ArithReplacementAttr(
        dlt.DimensionAttr("dim_b", 16),
        dlt.DimensionAttr("common_dim", 16),
        dlt.MemberAttr("hidden","b")),
])
print(str(r.contents_type))
LayoutPlotter.plot_layout({str(r.contents_type):r}, name="Arith_replace_example_layout")
