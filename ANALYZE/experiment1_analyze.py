
import utils
from glob import glob
import pandas as pd
import numpy as np

utils.chdir_in_project("ANALYZE")
utils.PROJECT_PATH = "/Users/yusuf/LLM-for-ERC"


results2 = {k:{} for k in ["default", "gemini", "claude", "gpt5"]}
for test_path in glob("../EVAL_RESULTS/Experiment1/*.json"):
    test_info = utils.load_json(path=test_path)["test_info"]
    config = test_info["config"]

    results2[config["prompt_type"]][f"{config['dataset']}-({config['top_n']},{config['max_m']})"] \
        = test_info['stats']['f1_score']





df = pd.DataFrame.from_dict(results2, orient="index")
df.columns = pd.MultiIndex.from_tuples(
    [col.split('-') for col in df.columns],
    names=['dataset', 'top_n,max_m']
)
df = df.sort_index(axis=1, level=[0, 1])
df['average'] = df.sum(axis=1) / df.shape[1]
df