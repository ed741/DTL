from evaluation.evaluateExpriment import Stat, generate_data

experiment_name = "matmul_taco/single_O3_8.8.8_0"

def gen_func(stat: Stat)-> dict[str,str]:
    trial = [r for r in stat.entries if r["repeats"]==-1]
    assert len(trial)==1
    trial = trial[0]
    return {
        "Correct":str(int(trial["correct"])),
        "TrialFinished":str(int(trial["finished"])),}


generate_data(experiment_name,
              test_key = [("taco_layout",int)],
              program_keys = ["taco_layout"],
              layout_targets = None,
              layout_gen=None,
              layout_export=[],
              instances=(100,1000),
              min_wait_time=1,
              stat_gen=gen_func,
              stat_export=["Correct", "TrialFinished"],
              explore_dump_codes=False,
              )
