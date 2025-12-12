from PRELIMINARY_EXPERIMENTS import llm_code_eval


def get_updated_config(config, dataset, max_k):
    c = config.copy()
    c["dataset"] = dataset
    c["max_k"] = max_k
    return c


config = {
    "dataset": "meld",
    "max_k": 12,
    "example_type": "hybrid",
    "top_n": 2,
    "max_m": 7,
    "use_detailed_example": True,
    "limit": None,
    "model_id": 1,
    "split": "dev",
    "save": True,
    "prompt_type": "gemini",
    "experiment_id": "3a"
}

datasets = ["iemocap", "meld"]
k_values = [0, 1, 3, 5, 7, 10, 12, 15, 17, 20]

config_list = []

print("=" * 80)
print(f"Start with Experiment 3a")
config["experiment_id"] = "3a"
for dataset in datasets:
    for k in k_values:
        c = get_updated_config(config, dataset, k)
        config_list.append(c)

for config in config_list:
    llm_code_eval.main(config)




print("=" * 80)
print(f"Start with Experiment 3b")
config["experiment_id"] = "3b"
config['exclude_current_from_history'] = True
datasets = ["iemocap", "meld"]
k_values = [1, 7, 12, 20]

for dataset in datasets:
    for k in k_values:
        c = get_updated_config(config, dataset, k)
        config_list.append(c)
for config in config_list:
    llm_code_eval.main(config)

