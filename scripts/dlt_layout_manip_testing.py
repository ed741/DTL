from xdsl.dialects import builtin
from xdsl.dialects.experimental import dlt
from xdsl.transforms.experimental.dlt.layout_manipulation import Manipulator
from dtl.visualise import LayoutPlotter

def t1():
    parent1_1 = dlt.PrimitiveLayoutAttr(builtin.f32)
    parent1_1 = dlt.DenseLayoutAttr(parent1_1, dlt.DimensionAttr("n1p1", 1))

    parent1_2 = dlt.PrimitiveLayoutAttr(builtin.i32)
    parent1_2 = dlt.DenseLayoutAttr(parent1_2, dlt.DimensionAttr("n1p1", 1))

    parent2 = dlt.PrimitiveLayoutAttr(builtin.i32)
    parent2 = dlt.DenseLayoutAttr(parent2, dlt.DimensionAttr("n1p2", 1))

    parent = dlt.AbstractLayoutAttr([
        ([],[dlt.DimensionAttr("n2p1",2), dlt.DimensionAttr("n3",3)], parent1_1),
        ([],[dlt.DimensionAttr("n2p1",2), dlt.DimensionAttr("n3",3)], parent1_2),
        ([],[dlt.DimensionAttr("n2p2",2), dlt.DimensionAttr("n3",3)], parent2),
    ])
    parent = dlt.DenseLayoutAttr(parent, dlt.DimensionAttr("n4", 4))
    parent = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n5",5)], parent)])
    parent = dlt.DenseLayoutAttr(parent, dlt.DimensionAttr("n6", 6))

    child1_1 = dlt.PrimitiveLayoutAttr(builtin.f32)
    child1_1 = dlt.DenseLayoutAttr(child1_1, dlt.DimensionAttr("n1p1", 1))
    child1_2 = dlt.PrimitiveLayoutAttr(builtin.i32)
    child1_2 = dlt.DenseLayoutAttr(child1_2, dlt.DimensionAttr("n1p1", 1))
    child1 = dlt.StructLayoutAttr([child1_1, child1_2])
    child1 = dlt.DenseLayoutAttr(child1, dlt.DimensionAttr("n3", 3))
    child1 = dlt.DenseLayoutAttr(child1, dlt.DimensionAttr("n2p1", 2))
    # child1 = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n2p1",2)], child1)])
    #
    # child2 = dlt.PrimitiveLayoutAttr(builtin.i32)
    # child2 = dlt.DenseLayoutAttr(child2, dlt.DimensionAttr("n1p2", 1))
    # child2 = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n2p2",2)], child2)])
    #
    # child = dlt.AbstractLayoutAttr([
    #     ([],[dlt.DimensionAttr("n3",3)], child1),
    #     ([],[dlt.DimensionAttr("n3",3)], child2),
    # ])
    # # child = dlt.StructLayoutAttr([child1, child2])
    # # child = dlt.DenseLayoutAttr(child, dlt.DimensionAttr("n3", 3))
    #
    # child = dlt.DenseLayoutAttr(child, dlt.DimensionAttr("n4", 4))
    # child = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n5",5)], child)])
    return child1, parent, set(), {dlt.DimensionAttr("n6", 6), dlt.DimensionAttr("n5", 5), dlt.DimensionAttr("n4", 4)}


def t2():
    parent1_1 = dlt.PrimitiveLayoutAttr(builtin.f32)
    parent1_1 = dlt.DenseLayoutAttr(parent1_1, dlt.DimensionAttr("n1p1", 1))

    parent1_2 = dlt.PrimitiveLayoutAttr(builtin.i32)
    parent1_2 = dlt.DenseLayoutAttr(parent1_2, dlt.DimensionAttr("n1p1", 1))

    parent2 = dlt.PrimitiveLayoutAttr(builtin.i32)
    parent2 = dlt.DenseLayoutAttr(parent2, dlt.DimensionAttr("n1p2", 1))

    parent = dlt.AbstractLayoutAttr([
        ([],[dlt.DimensionAttr("n2p1",2), dlt.DimensionAttr("n3",3)], parent1_1),
        ([],[dlt.DimensionAttr("n2p1",2), dlt.DimensionAttr("n3",3)], parent1_2),
        ([],[dlt.DimensionAttr("n2p2",2), dlt.DimensionAttr("n3",3)], parent2),
    ])
    parent = dlt.DenseLayoutAttr(parent, dlt.DimensionAttr("n4", 4))
    parent = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n5",5)], parent)])
    parent = dlt.DenseLayoutAttr(parent, dlt.DimensionAttr("n6", 6))

    child1_1 = dlt.PrimitiveLayoutAttr(builtin.f32)
    child1_1 = dlt.DenseLayoutAttr(child1_1, dlt.DimensionAttr("n1p1", 1))
    child1_2 = dlt.PrimitiveLayoutAttr(builtin.i32)
    child1_2 = dlt.DenseLayoutAttr(child1_2, dlt.DimensionAttr("n1p1", 1))
    child1 = dlt.StructLayoutAttr([child1_1, child1_2])
    child1 = dlt.DenseLayoutAttr(child1, dlt.DimensionAttr("n3", 3))
    # child1 = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n2p1",2)], child1)])
    #
    # child2 = dlt.PrimitiveLayoutAttr(builtin.i32)
    # child2 = dlt.DenseLayoutAttr(child2, dlt.DimensionAttr("n1p2", 1))
    # child2 = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n2p2",2)], child2)])
    #
    # child = dlt.AbstractLayoutAttr([
    #     ([],[dlt.DimensionAttr("n3",3)], child1),
    #     ([],[dlt.DimensionAttr("n3",3)], child2),
    # ])
    # # child = dlt.StructLayoutAttr([child1, child2])
    # # child = dlt.DenseLayoutAttr(child, dlt.DimensionAttr("n3", 3))
    #
    # child = dlt.DenseLayoutAttr(child, dlt.DimensionAttr("n4", 4))
    # child = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n5",5)], child)])
    return child1, parent, set(), {dlt.DimensionAttr("n6", 6),        dlt.DimensionAttr("n5", 5), dlt.DimensionAttr("n4", 4), dlt.DimensionAttr("n2p1", 2) }


def t3():
    parent1_1 = dlt.PrimitiveLayoutAttr(builtin.f32)
    parent1_1 = dlt.DenseLayoutAttr(parent1_1, dlt.DimensionAttr("n1p1", 1))

    parent1_2 = dlt.PrimitiveLayoutAttr(builtin.i32)
    parent1_2 = dlt.DenseLayoutAttr(parent1_2, dlt.DimensionAttr("n1p1", 1))

    parent2 = dlt.PrimitiveLayoutAttr(builtin.i32)
    parent2 = dlt.DenseLayoutAttr(parent2, dlt.DimensionAttr("n1p2", 1))

    parent = dlt.AbstractLayoutAttr([
        ([],[dlt.DimensionAttr("n2p1",2), dlt.DimensionAttr("n3",3)], parent1_1),
        ([],[dlt.DimensionAttr("n2p1",2), dlt.DimensionAttr("n3",3)], parent1_2),
        ([],[dlt.DimensionAttr("n2p2",2), dlt.DimensionAttr("n3",3)], parent2),
    ])
    parent = dlt.DenseLayoutAttr(parent, dlt.DimensionAttr("n4", 4))
    parent = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n5",5)], parent)])
    parent = dlt.DenseLayoutAttr(parent, dlt.DimensionAttr("n6", 6))

    child1_1 = dlt.PrimitiveLayoutAttr(builtin.f32)
    child1_1 = dlt.DenseLayoutAttr(child1_1, dlt.DimensionAttr("n1p1", 1))
    child1_2 = dlt.PrimitiveLayoutAttr(builtin.i32)
    child1_2 = dlt.DenseLayoutAttr(child1_2, dlt.DimensionAttr("n1p1", 1))
    child1 = dlt.StructLayoutAttr([child1_1, child1_2])
    child1 = dlt.DenseLayoutAttr(child1, dlt.DimensionAttr("n3", 3))
    child1 = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n2p1",2)], child1)])
    #
    child2 = dlt.PrimitiveLayoutAttr(builtin.i32)
    child2 = dlt.DenseLayoutAttr(child2, dlt.DimensionAttr("n1p2", 1))
    child2 = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n2p2",2)], child2)])
    child2 = dlt.DenseLayoutAttr(child2, dlt.DimensionAttr("n3", 3))
    #
    # child = dlt.AbstractLayoutAttr([
    #     ([],[dlt.DimensionAttr("n3",3)], child1),
    #     ([],[dlt.DimensionAttr("n3",3)], child2),
    # ])
    child = dlt.StructLayoutAttr([child1, child2])
    # # child = dlt.DenseLayoutAttr(child, dlt.DimensionAttr("n3", 3))
    #
    child = dlt.DenseLayoutAttr(child, dlt.DimensionAttr("n4", 4))
    # child = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n5",5)], child)])
    return child, parent, set(), {dlt.DimensionAttr("n6", 6), dlt.DimensionAttr("n5", 5), }


def t4():
    parent = dlt.PrimitiveLayoutAttr(builtin.f32)
    parent = dlt.DenseLayoutAttr(parent, dlt.DimensionAttr("n4", 4))
    parent = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n5",5)], parent)])
    parent = dlt.DenseLayoutAttr(parent, dlt.DimensionAttr("n6", 6))

    child = dlt.PrimitiveLayoutAttr(builtin.f32)
    child = dlt.DenseLayoutAttr(child, dlt.DimensionAttr("n4", 4))
    child = dlt.DenseLayoutAttr(child, dlt.DimensionAttr("n5", 5))
    child = dlt.DenseLayoutAttr(child, dlt.DimensionAttr("n6", 6))
    # child = dlt.AbstractLayoutAttr([([],[dlt.DimensionAttr("n5",5)], child)])
    return child, parent, set(), {}



c, p, m, d = t2()
LayoutPlotter.plot_layout({"parent": p, "child": c}, view=True)
embedded = Manipulator.embed_layout_in(c, p, m, d, set(), False)

nodes = [embedded]
while nodes:
    print(' ||||| '.join([str(n) for n in nodes]))
    nodes = [n for c in nodes for n in c.get_children()]

LayoutPlotter.plot_layout({"parent": p, "child": c, "new": embedded}, view=True)
