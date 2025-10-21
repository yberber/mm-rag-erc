import llm_code_eval

config = {
    "dataset": "meld",
    "max_k": 12,
    "example_type": "single",
    "top_n": 1,
    "max_m": 1,
    "use_detailed_example": "False",
    "limit": 50,
    "model_id": 1,
    "split": "dev",
    "save": True,
    "prompt_type": "default",
    "experiment_id": 2
}

datasets = ["iemocap", "meld"]
prompt_types = ["default", "gemini", "claude", "gpt5"]

configs = []

for dataset in datasets:
    for prompt_type in prompt_types:
        config["dataset"] = dataset
        config["prompt_type"] = prompt_type
        configs.append(config.copy())


for config in configs:
    llm_code_eval.main(config)
    print("*****************\n\n")