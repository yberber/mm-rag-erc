from PRELIMINARY_EXPERIMENTS import llm_code_eval

config = {
    "dataset": "meld",
    "max_k": 12,
    "example_type": "hybrid",
    "top_n": 2,
    "max_m": 5,
    "use_detailed_example": True,
    "limit": None,
    "model_id": 3,
    "split": "dev",
    "save": True,
    "prompt_type": "default",
    "experiment_id": "1b"
}

datasets = ["iemocap", "meld"]
prompt_types = [ "gemini", "claude", "gpt5", "default"]

configs = []

print("=" * 80)
print(f"Start with Experiment 1a")
config["experiment_id"] = "1a"
for dataset in datasets:
    for prompt_type in prompt_types:
        config["dataset"] = dataset
        config["prompt_type"] = prompt_type
        configs.append(config.copy())
for config in configs:
    llm_code_eval.main(config)
    print("*****************\n")


print("=" * 80)
print(f"Start with Experiment 1b")
config["experiment_id"] = "1b"
config["experiment_id"] = "1b"
for dataset in datasets:
    for prompt_type in prompt_types:
        config["dataset"] = dataset
        config["prompt_type"] = prompt_type
        configs.append(config.copy())
for config in configs:
    llm_code_eval.main(config)
    print("*****************\n")