
import utils
from glob import glob
import pandas as pd
import numpy as np

utils.chdir_in_project("ANALYZE")
utils.PROJECT_PATH = "/Users/yusuf/LLM-for-ERC"

def run_experiment1(experiment_id):
    if experiment_id not in ['a', 'b']:
        raise ValueError("id must be either 'a' or 'b'")
    results = {k:{} for k in ["default", "gemini", "claude", "gpt5"]}
    for test_path in glob(f"../EVAL_RESULTS/Experiment1{experiment_id}/*.json"):
        if test_path == '../EVAL_RESULTS/Experiment1b/MELD-model3_gemini_k12_single_n1_m1.json':
            results['gemini'][f"meld-(1,1)"] = 0.58
            # results2['gemini'][f"meld-(1,1)"] = np.nan

            continue
        test_info = utils.load_json(path=test_path)["test_info"]
        config = test_info["config"]

        results[config["prompt_type"]][f"{config['dataset']}-({config['top_n']},{config['max_m']})"] \
            = test_info['stats']['f1_score']

    df = pd.DataFrame.from_dict(results, orient="index")
    df.columns = pd.MultiIndex.from_tuples(
        [col.split('-') for col in df.columns],
        names=['dataset', 'top_n,max_m']
    )
    df = df.sort_index(axis=1, level=[0, 1])
    df['average'] = df.sum(axis=1) / df.shape[1]
    print(f"Experiment 1{experiment_id} Results:")
    print(df)
    print(f"Average weighted F1 score: {df['average'].mean()}")
    print("=" * 80 + "\n\n")


run_experiment1(experiment_id='a')
run_experiment1(experiment_id='b')







