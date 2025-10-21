import llm_code_eval

config = {
    "dataset": "meld",
    "max_k": 12,
    "example_type": "hybrid",
    "top_n": 2,
    "max_m": 5,
    "use_detailed_example": True,
    "limit": None,
    "model_id": 1,
    "split": "dev",
    "save": True,
    "prompt_type": "default",
    "experiment_id": 1
}

datasets = ["iemocap", "meld"]
prompt_types = [ "gemini", "claude", "gpt5", "default"]

configs = []

for dataset in datasets:
    for prompt_type in prompt_types:
        config["dataset"] = dataset
        config["prompt_type"] = prompt_type
        configs.append(config.copy())


for config in configs:
    llm_code_eval.main(config)
    print("*****************\n")