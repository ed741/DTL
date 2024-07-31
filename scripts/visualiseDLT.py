import functools
import typing
from io import StringIO
from typing import Collection

import graphviz

from xdsl.dialects import func
from xdsl.ir import Attribute
from xdsl.printer import Printer
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.experimental import dlt


class LayoutPlotter:

    @staticmethod
    def plot_layout(
        layouts: dict[str|StringAttr, dlt.Layout|dlt.PtrType],
        *,
        name="layout",
        entry_points: Collection[str|StringAttr] = None,
        view=False,
        coalesce_duplicates=True,
        coalesce_terminals=False,
        **kwargs,
    ):
        """Render dlt layout and write to a file.
        """
        if entry_points is None:
            entry_points = []
        entry_points = [e.data if isinstance(e, StringAttr) else str(e) for e in entry_points]

        graph = graphviz.Digraph(name, **kwargs)
        seen = {}
        for name, root in layouts.items():
            if isinstance(name, StringAttr):
                name = name.data
            label_name = name
            if isinstance(root, dlt.PtrType):
                if root.identification.data != name:
                    label_name = name + " : " + root.identification.data
                root = root.layout
            node_name = LayoutPlotter._plot_layout(
                root,
                graph,
                seen,
                coalesce_duplicates=coalesce_duplicates,
                coalesce_terminals=coalesce_terminals
            )
            if name in entry_points:
                font_size = f"{18}"
            else:
                font_size = f"{14}"
            graph.node(name, label=label_name, shape="invhouse", fontsize=font_size)
            graph.edge(name, node_name, style="dashed")

        graph.render(view=view)

    @staticmethod
    def _plot_layout(
        layout_node: dlt.Layout,
        graph: graphviz.Digraph,
        seen: dict[dlt.Layout, str],
        coalesce_duplicates=True,
        coalesce_terminals=False,
    ):
        if coalesce_duplicates and layout_node in seen:
            if not LayoutPlotter._is_terminal(layout_node) or coalesce_terminals:
                return seen[layout_node]

        name = str(id(layout_node))
        seen[layout_node] = name
        node_atters, children = LayoutPlotter.get_label(layout_node)
        graph.node(name, **node_atters)
        for child_layout, edge_label, sub_parent in children:
            childName = LayoutPlotter._plot_layout(
                child_layout,
                graph,
                seen,
                coalesce_duplicates=coalesce_duplicates,
                coalesce_terminals=coalesce_terminals,
            )
            edge_start = name if sub_parent == "" else f"{name}:{sub_parent}"
            graph.edge(edge_start, childName, label=edge_label)
        return name

    @staticmethod
    def _is_terminal(layout_node: dlt.Layout) -> bool:
        return len(layout_node.get_children()) == 0


    @functools.singledispatch
    @staticmethod
    def get_label(layout_node: dlt.Layout) -> tuple[dict[str, str], list[tuple[dlt.Layout, str, str]]]:
        raise NotImplementedError

    @get_label.register
    @staticmethod
    def _(layout_node: dlt.PrimitiveLayoutAttr) -> tuple[dict[str, str], list[tuple[dlt.Layout, str, str]]]:
        return {"label": _escape_text(_print_to_str(layout_node.base_type)), "shape":"diamond"}, []

    @get_label.register
    @staticmethod
    def _(layout_node: dlt.DenseLayoutAttr) -> tuple[dict[str, str], list[tuple[dlt.Layout, str, str]]]:
        return {"label": _escape_text(_print_to_str(layout_node.dimension)), "shape":"box"}, [(layout_node.child, "", "")]

    @get_label.register
    @staticmethod
    def _(layout_node: dlt.MemberLayoutAttr) -> tuple[dict[str, str], list[tuple[dlt.Layout, str, str]]]:
        return {"label": _escape_text(_print_to_str(layout_node.member_specifier)), "shape":"parallelogram"}, [(layout_node.child, "", "")]

    @get_label.register
    @staticmethod
    def _(layout_node: dlt.ArithReplaceLayoutAttr) -> tuple[dict[str, str], list[tuple[dlt.Layout, str, str]]]:
        replacements = []
        for r in layout_node.replacements:
            replacements.append(f"{_print_to_str(r.outer_dimension)} & {_print_to_str(r.inner_member)}")
        return {"label": _escape_text(_print_to_str(layout_node.inner_dimension()) + " \n " + "\n".join(replacements)), "shape":"box"}, [(layout_node.child, "", "")]

    @get_label.register
    @staticmethod
    def _(layout_node: dlt.StructLayoutAttr) -> tuple[dict[str, str], list[tuple[dlt.Layout, str, str]]]:
        labels = []
        children_edges = []
        for idx, child in enumerate(layout_node.children):
            labels.append(f"<f{idx}> {idx}")
            children_edges.append((child, "", f"f{idx}"))
        return {"label": "|".join(labels), "shape":"record"}, children_edges

    @get_label.register
    @staticmethod
    def _(layout_node: dlt.AbstractLayoutAttr) -> tuple[dict[str, str], list[tuple[dlt.Layout, str, str]]]:
        labels = []
        children_edges = []
        for idx, child in enumerate(layout_node.children):
            member_record = "{" + ", ".join([_print_to_str(m) for m in child.member_specifiers]) + "}"
            dim_record = "[" + ", ".join([_print_to_str(d) for d in child.dimensions]) + "]"
            label = f"<f{idx}> {_escape_text(member_record)} {_escape_text(dim_record)}"
            labels.append(label)
            children_edges.append((child.child, "", f"f{idx}"))
        return {"label": "|".join(labels), "shape":"record"}, children_edges

    @get_label.register
    @staticmethod
    def _(layout_node: dlt.IndexingLayoutAttr) -> tuple[dict[str, str], list[tuple[dlt.Layout, str, str]]]:
        children_edges = []
        label  = "<f0> sparse | <f1> direct"
        children_edges.append((layout_node.indexedChild, "", "f0"))
        children_edges.append((layout_node.directChild, "", "f1"))
        return {"label": label, "shape":"record"}, children_edges

    @get_label.register
    @staticmethod
    def _(layout_node: dlt.UnpackedCOOLayoutAttr) -> tuple[dict[str, str], list[tuple[dlt.Layout, str, str]]]:
        label = _escape_text("UpCOO: ["+ ",".join([_print_to_str(dim) for dim in layout_node.dimensions]) + "]")
        return {"label": label, "shape":"doubleoctagon"}, [(layout_node.child, "", "")]


class IterationPlotter:

    def __init__(self, iteration_ops: set[dlt.IterateOp] = None,):


        self.iteration_ops = iteration_ops
        self.bodies = {}
        self.body_parts = {}
        for op in iteration_ops:
            parent_body_name = IterationPlotter._parent_name_for(op)

            self.bodies[op] = parent_body_name
            self.body_parts.setdefault(parent_body_name, []).append(op)
        for op in iteration_ops:
            body_name = IterationPlotter._body_name_for(op)
            if body_name not in self.body_parts:
                self.body_parts[body_name] = []

    @staticmethod
    def _parent_name_for(op: dlt.IterateOp) -> str | None:
        while (parent_op := op.parent_op()) is not None:
            op = parent_op
            if isinstance(op, dlt.IterateOp):
                return IterationPlotter._body_name_for(op)
            if isinstance(op, func.FuncOp):
                return "func." + op.sym_name.data + ".body"
        return None

    @staticmethod
    def _body_name_for(op: dlt.IterateOp) -> str:
        if op.identification.data == "":
            return str(id(op))+".body"
        else:
            return op.identification.data + ".body"

    def plot(self,
            iteration_orders: dict[str | StringAttr, dlt.IterationOrder | dlt.IterateOp],
            *,
            plot_name="iteration",
            view=False,
            **kwargs,
    ):
        """Render dlt iterations and write to a file.
        """
        _iteration_orders = {}
        for name, order in iteration_orders.items():
            if isinstance(name, StringAttr):
                name = name.data
            if isinstance(order, dlt.IterateOp):
                order = order.order
            _iteration_orders[name] = order
        iteration_orders = _iteration_orders

        graph = graphviz.Digraph(plot_name, **kwargs)
        for body_name, body_ops in self.body_parts.items():
            label = body_name
            graph.node(body_name, label=label, shape="box")

        for op in self.iteration_ops:
            if op.identification.data in iteration_orders:
                root = iteration_orders[op.identification.data]
            else:
                root = op.order

            child = self._plot_node(root, graph, op)
            parent = IterationPlotter._parent_name_for(op)
            graph.edge(parent, child)

        graph.render(view=view)


    @functools.singledispatchmethod
    def _plot_node(self, iteration_node: dlt.IterationOrder, graph: graphviz.Digraph, op: dlt.IterateOp) -> str:
        raise NotImplementedError(type(iteration_node))

    @_plot_node.register
    def _(self, iteration_node: dlt.BodyIterationOrderAttr, graph: graphviz.Digraph, op: dlt.IterateOp) -> str:
        name = IterationPlotter._body_name_for(op)
        return name

    @_plot_node.register
    def _(self, iteration_node: dlt.NestedIterationOrderAttr, graph: graphviz.Digraph, op: dlt.IterateOp) -> str:
        name = IterationPlotter._body_name_for(op) + str(id(iteration_node))
        extent_idx = iteration_node.extent_index.data
        extent = op.extents.data[extent_idx]
        label = _print_to_str(extent)
        dims = [ptr.type.identification.data + ":" + ",".join([dim.dimensionName.data for dim in dim_spec.data[extent_idx]]) for ptr, dim_spec in zip(op.tensors, op.dimensions) if len(dim_spec.data[extent_idx]) > 0]
        if len(dims) > 0:
            label += "\n" + "\n".join(dims)

        kwargs = {}
        if op not in self.iteration_ops:
            kwargs["color"] = "grey"
        graph.node(name, label=label, shape="invhouse", **kwargs)
        child = self._plot_node(iteration_node.child, graph, op)
        graph.edge(name, child)
        return name

    @_plot_node.register
    def _(self, iteration_node: dlt.AbstractIterationOrderAttr, graph: graphviz.Digraph, op: dlt.IterateOp) -> str:
        name = IterationPlotter._body_name_for(op) + str(id(iteration_node))
        label = "Abstract:"
        for extent_idx in iteration_node.extent_indices:
            extent_idx = extent_idx.data
            extent = op.extents.data[extent_idx]
            ex_label = _print_to_str(extent)
            dims = [ptr.type.identification.data + ":" + ",".join([dim.dimensionName.data for dim in dim_spec.data[extent_idx]]) for ptr, dim_spec in zip(op.tensors, op.dimensions) if len(dim_spec.data[extent_idx]) > 0]
            if len(dims) > 0:
                ex_label += "\n" + "\n".join(dims)
            label += "\n" + ex_label

        kwargs = {}
        if op not in self.iteration_ops:
            kwargs["color"] = "grey"
        graph.node(name, label=label, shape="invhouse", **kwargs)
        child = self._plot_node(iteration_node.child, graph, op)
        graph.edge(name, child)
        return name



def _escape_text(text: str) -> str:
    return text.replace("[", "\\[").replace("]", "\\]").replace("{", "\\{").replace("}", "\\}").replace("|", "\\|").replace("<", "\\<").replace(">", "\\>")


def _print_to_str(attr: Attribute) -> str:
    res = StringIO()
    printer = Printer(print_generic_format=False, stream=res)
    if isinstance(attr, dlt.DimensionAttr):
        attr = typing.cast(dlt.DimensionAttr, attr)
        attr.internal_print_parameters(printer)
    elif isinstance(attr, dlt.MemberAttr):
        attr = typing.cast(dlt.MemberAttr, attr)
        attr.internal_print_parameters(printer)
    elif isinstance(attr, dlt.Extent):
        attr = typing.cast(dlt.Extent, attr)
        attr.internal_print_extent(printer)
    else:
        printer.print(attr)
    return res.getvalue()
