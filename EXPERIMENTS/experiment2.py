import llm_code_eval



def get_updated_config(config, dataset, max_k, top_n, example_type, max_m, use_detailed_example):
    c = config.copy()
    c["dataset"] = dataset
    c["max_k"] = max_k
    c["top_n"] = top_n
    c["example_type"] = example_type
    c["max_m"] = max_m
    c["use_detailed_example"] = use_detailed_example
    return c

config = {
    "dataset": "meld",
    "max_k": 12,
    "example_type": "single",
    "top_n": 1,
    "max_m": 1,
    "use_detailed_example": "False",
    "limit": 20,
    "model_id": 0,
    "split": "dev",
    "save": True,
    "prompt_type": "gemini",
    "experiment_id": 2
}

datasets = ["iemocap", "meld"]
max_k = 12
top_n_list = [1,2,3]
max_m_list = [3, 5, 7]
split = "dev"
db_types = ["single", "flow", "hybrid"]
use_detailed_example = [True, False]

config_list = []
for dataset in datasets:
    for top_n in top_n_list:
        for db_type in db_types:
            if db_type == "single":
                max_m = 1
                example_detailed_flag = False
                c = get_updated_config(config, dataset, max_k, top_n, db_type, max_m, example_detailed_flag)
                config_list.append(c)
            else:
                for max_m in max_m_list:
                    for example_detailed_flag in use_detailed_example:
                        c = get_updated_config(config, dataset, max_k, top_n, db_type, max_m, example_detailed_flag)
                        config_list.append(c)


# no examples provided
special_case_iemocap = get_updated_config(config, "iemocap", max_k=12, top_n=0, example_type="single", max_m=0, use_detailed_example=False)
special_case_meld = get_updated_config(config, "meld", max_k=12, top_n=0, example_type="single", max_m=0, use_detailed_example=False)

config_list.clear()
config_list.append(special_case_iemocap)
config_list.append(special_case_meld)

for config in config_list:
    llm_code_eval.main(config)
