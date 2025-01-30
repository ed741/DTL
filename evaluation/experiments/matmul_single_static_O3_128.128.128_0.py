import sys
import typing

from evaluation.evaluateExpriment import generate_data
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import PtrType
import evaluation.evaluateTools as et

experiment_name = "matmul/single_static_O3_128.128.128_0"
target_layouts = {"A": "R_0", "B": "R_1", "C": "R_2"}


A_layouts = {}
B_layouts = {}
C_layouts = {}
def gen_func(ptr_map: dict[str, PtrType])-> dict[str,str]:
    A_ptr = ptr_map["A"]
    B_ptr = ptr_map["B"]
    C_ptr = ptr_map["C"]

    CI_node = et.get_layout_node_for_dim(C_ptr.layout, "dim_4")
    CK_node = et.get_layout_node_for_dim(C_ptr.layout, "dim_5")

    CSparseIBuffer = 0
    if isinstance(CI_node, dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr):
        CSparseIBuffer = typing.cast(dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr, CI_node).buffer_scaler.data
    CSparseKBuffer = 0
    if isinstance(CK_node, dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr):
        CSparseKBuffer = typing.cast(dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr, CK_node).buffer_scaler.data

    COrder = [{"dim_4":"I", "dim_5":"K"}[d] for d in et.get_dim_order(C_ptr.layout, {"dim_4", "dim_5"})]

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

    return {
            "CIsSparse": str(int(et.is_layout_sparse(C_ptr.layout))),
            "CCountSparse": str(int(et.count_layout_sparse(C_ptr.layout))),
            "CSparseI": str(int(et.is_layout_sparse_dim(C_ptr.layout, ["dim_4"]))),
            "CSparseK": str(int(et.is_layout_sparse_dim(C_ptr.layout, ["dim_5"]))),
            "CSparseIUp": str(int(isinstance(CI_node, dlt.UnpackedCOOLayoutAttr))),
            "CSparseIS": str(int(isinstance(CI_node, dlt.SeparatedCOOLayoutAttr))),
            "CSparseKUp": str(int(isinstance(CK_node, dlt.UnpackedCOOLayoutAttr))),
            "CSparseKS": str(int(isinstance(CK_node, dlt.SeparatedCOOLayoutAttr))),
            "CSparseIBuffer": str(CSparseIBuffer),
            "CSparseKBuffer": str(CSparseKBuffer),
            "COrderIK": str(int(COrder == ["I", "K"])),
            "COrderKI": str(int(COrder == ["K", "I"])),
            "AloIDX": str(A_layout_idx),
            "BloIDX": str(B_layout_idx),
            "CloIDX": str(C_layout_idx),
            }

generate_data(experiment_name,
              test_key_def= [("layout", int), ("order", int)],
              program_keys = ["layout","order"],
              layout_targets = target_layouts,
              layout_gen=gen_func,
              layout_export=[
                  "CIsSparse",
                  "CCountSparse",
                  "CSparseI",
                  "CSparseK",
                  "CSparseIUp",
                  "CSparseIS",
                  "CSparseKUp",
                  "CSparseKS",
                  "CSparseIBuffer",
                  "CSparseKBuffer",
                  "COrderIK",
                  "COrderKI",
                  "AloIDX",
                  "BloIDX",
                  "CloIDX",
              ],
              instances=(100,1000),
              )
