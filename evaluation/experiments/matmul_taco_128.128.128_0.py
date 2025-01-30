import re

from evaluation.evaluateExpriment import Stat, generate_data

experiment_name = "matmul_taco/single_O3_128.128.128_0"

def gen_func(stat: Stat)-> dict[str,str]:
    trial = [r for r in stat.entries if r["repeats"]==-1]
    assert len(trial)==1
    trial = trial[0]
    return {
        "Correct":str(int(trial["correct"])),
        "TrialFinished":str(int(trial["finished"])),}


def tests_func(stat: Stat, path: str) -> dict[str, str]:
    with open(f"{path}/lib_taco_benchmark.so.cpp", "r") as f:

        tensors = ["a", "b", "c"]
        # map = {f"{t.capitalize()}Sparse": "0" for t in tensors}
        map = {}
        for line in f:
            pass
            for t in tensors:
                search_str = r"\s* Format\s*f_"+t+r"\(\{(Sparse|Dense),(Sparse|Dense)}.*"
                match = re.search(search_str, line)
                if match:
                    sparse = "Sparse" in [match.group(1), match.group(2)]
                    key = f"{t.capitalize()}Sparse"
                    assert key not in map
                    map[key] = str(int(sparse))
        for t in tensors:
            if f"{t.capitalize()}Sparse" not in map:
                assert f"{t.capitalize()}Sparse" in map
            assert map[f"{t.capitalize()}Sparse"] in ["0", "1"]
    return map

generate_data(experiment_name,
              test_key_def= [("taco_layout", int)],
              program_keys = ["taco_layout"],
              layout_targets = None,
              layout_gen=None,
              layout_export=[],
              instances=(100,1000),
              min_wait_time=1,
              stat_gen=gen_func,
              stat_export=["Correct", "TrialFinished"],
              tests_gen=tests_func,
              tests_export=["ASparse", "BSparse", "CSparse"],
              )
