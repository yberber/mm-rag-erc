
import utils
from glob import glob
import pandas as pd

pd.options.display.float_format = "{:.3f}".format

utils.set_pandas_display_options()



results = {k: {} for k in ['meld', 'iemocap']}

for test_path in glob("../EVAL_RESULTS/Experiment2b/*.json"):
    test_info = utils.load_json(path=test_path)["test_info"]
    config = test_info["config"]

    if config['example_type'] in ['flow', 'hybrid'] and config['use_detailed_example']:
        example_type = config['example_type'] + "V2"
    else:
        example_type = config['example_type']
    config['example_type'] = example_type

    sub_results = results[config["dataset"]]


    if config["example_type"] not in sub_results.keys():
        sub_results[config["example_type"]] = {}

    sub_results[config["example_type"]][f'({config["top_n"]}, {config["max_m"]})'] = test_info['stats']['f1_score']


column_ordered =  ['(0, 0)', '(1, 1)', '(2, 1)', '(3, 1)', '(1, 3)', '(1, 5)', '(1, 7)', '(2, 3)', '(2, 5)', '(2, 7)', '(3, 3)', '(3, 5)', '(3, 7)']
index_ordered = ["single", "flow", "flowV2", "hybrid", "hybridV2"]

meld = pd.DataFrame.from_dict(results["meld"], orient='index')[column_ordered].loc[index_ordered]
iemocap = pd.DataFrame.from_dict(results["iemocap"], orient='index')[column_ordered].loc[index_ordered]

# ✅ Set descriptive labels for index and columns
meld.index.name = "Example Type"
meld.columns.name = "(top_n, max_m)"

iemocap.index.name = "Example Type"
iemocap.columns.name = "(top_n, max_m)"

average = (meld + iemocap)/2

print(meld, "\n")
print(iemocap, "\n")
print(average, "\n")


