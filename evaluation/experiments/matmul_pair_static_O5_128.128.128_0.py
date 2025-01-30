import sys

from evaluation.evaluateExpriment import generate_data
from xdsl.dialects.experimental.dlt import PtrType
import evaluation.evaluateTools as et

experiment_name = "matmul/pair_static_O5_128.128.128_0"
target_layouts = {"A": "R_0", "B": "R_0", "C": "R_3"}

def gen_func(ptr_map: dict[str, PtrType])-> dict[str,str]:
    c_ptr = ptr_map["C"]
    ab_ptr = ptr_map["A"]
    assert ab_ptr == ptr_map["B"]
    replacement_list = et.layout_airth_replaced(ab_ptr.layout)


    return {
            # "CIsSparse": str(int(et.is_layout_sparse(c_ptr.layout))),
            # "CCountSparse": str(int(et.count_layout_sparse(c_ptr.layout))),
            # "CSparseI": str(int(et.is_layout_sparse_dim(c_ptr.layout, ["dim_4"]))),
            # "CSparseK": str(int(et.is_layout_sparse_dim(c_ptr.layout, ["dim_5"]))),
            "ABArith": str(int(et.layout_has_airth_replace(ab_ptr.layout))),
            "ABArithCount": str(int(len(replacement_list))),
            "ABArithIJ": str(int({"dim_0", "dim_2"} in replacement_list)),
            "ABArithIK": str(int({"dim_0", "dim_3"} in replacement_list)),
            "ABArithJJ": str(int({"dim_1", "dim_2"} in replacement_list)),
            "ABArithJK": str(int({"dim_1", "dim_3"} in replacement_list)),
            }

generate_data(experiment_name,
              test_key_def= [("layout", int), ("order", int)],
              program_keys = ["layout","order"],
              layout_targets = target_layouts,
              layout_gen=gen_func,
              layout_export=[
                  # "CIsSparse",
                  # "CCountSparse",
                  # "CSparseI",
                  # "CSparseK",
                  "ABArith",
                  "ABArithCount",
                  "ABArithIJ",
                  "ABArithIK",
                  "ABArithJJ",
                  "ABArithJK",
              ],
              instances=(100,1000),
              )
