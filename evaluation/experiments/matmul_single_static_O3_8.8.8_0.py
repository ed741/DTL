import sys

from evaluation.evaluateExpriment import generate_data
from xdsl.dialects.experimental.dlt import PtrType
import evaluation.evaluateTools as et

experiment_name = "matmul/single_static_O3_8.8.8_0"
target_layouts = {"A": "R_0", "B": "R_1", "C": "R_2"}

def gen_func(ptr_map: dict[str, PtrType])-> dict[str,str]:
    ptr = ptr_map["C"]
    AOrder = [{"dim_0": "I", "dim_1": "J"}[d] for d in et.get_dim_order(ptr_map["A"].layout, {"dim_0", "dim_1"})]
    BOrder = [{"dim_2": "J", "dim_3": "K"}[d] for d in et.get_dim_order(ptr_map["B"].layout, {"dim_2", "dim_3"})]
    COrder = [{"dim_4": "I", "dim_5": "K"}[d] for d in et.get_dim_order(ptr_map["C"].layout, {"dim_4", "dim_5"})]
    # if AOrder == ["J","I"] and BOrder == ["K","J"] and COrder == ["K", "I"] and not et.is_layout_sparse(ptr.layout):
    #     exit()
    return {"CIsSparse": str(int(et.is_layout_sparse(ptr.layout))),
            "CCountSparse": str(int(et.count_layout_sparse(ptr.layout))),
            "CSparseI": str(int(et.is_layout_sparse_dim(ptr.layout, ["dim_4"]))),
            "CSparseK": str(int(et.is_layout_sparse_dim(ptr.layout, ["dim_5"])))
            }

generate_data(experiment_name,
              test_key = [("layout",int),("order",int)],
              program_keys = ["layout","order"],
              layout_targets = target_layouts,
              layout_gen=gen_func,
              layout_export=[
                  "CIsSparse",
                  "CCountSparse",
                  "CSparseI",
                  "CSparseK",
              ],
              instances=(100,1000),
              )
