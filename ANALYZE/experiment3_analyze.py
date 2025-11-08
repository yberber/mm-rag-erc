import utils
from glob import glob
import pandas as pd

utils.set_pandas_display_options()
utils.chdir_in_project("ANALYZE")
utils.PROJECT_PATH = "/Users/yusuf/LLM-for-ERC"

pd.options.display.float_format = "{:.3f}".format


results = {k: {} for k in ['meld', 'iemocap']}

for test_path in glob("../EVAL_RESULTS/Experiment3/*.json"):
    test_info = utils.load_json(path=test_path)["test_info"]
    config = test_info["config"]

    results[config["dataset"]][config["max_k"]] = test_info["stats"]["f1_score"]

df_a = pd.DataFrame.from_dict(results, orient="index")
df_a.sort_index(inplace=True, axis=1)
df_a.loc["averaged"] = (df_a.loc["meld"] + df_a.loc["iemocap"]) / 2
df_a.columns.name = "max_k"

print("="*80)
print("Experiment 3A results:")
print(df_a)




results = {k: {} for k in ['meld', 'iemocap']}

for test_path in glob("../EVAL_RESULTS/Experiment3b/*.json"):
    test_info = utils.load_json(path=test_path)["test_info"]
    config = test_info["config"]

    results[config["dataset"]][config["max_k"]] = test_info["stats"]["f1_score"]

df_b = pd.DataFrame.from_dict(results, orient="index")
df_b.sort_index(inplace=True, axis=1)
df_b.loc["averaged"] = (df_b.loc["meld"] + df_b.loc["iemocap"]) / 2
df_b.columns.name = "max_k"
print("="*80)
print("Experiment 3A results:")
print(df_b)


pd.concat([df_a[df_b.columns], df_b], keys=["included", "excluded"], axis=1)

pd.j