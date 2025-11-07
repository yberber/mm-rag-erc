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

df = pd.DataFrame.from_dict(results, orient="index")
df.sort_index(inplace=True, axis=1)
df.loc["averaged"] = (df.loc["meld"] + df.loc["iemocap"]) / 2
df.columns.name = "k"
df