import llm_code_eval


def get_updated_config(config, dataset, speaker_characteristics_prompt):
    c = config.copy()
    c["dataset"] = dataset
    c["speaker_characteristics"] = speaker_characteristics_prompt
    return c


config = {
    "dataset": "meld",
    "max_k": 20,
    "example_type": "hybrid",
    "top_n": 2,
    "max_m": 7,
    "use_detailed_example": True,
    "limit": None,
    "model_id": 1,
    "split": "dev",
    "save": True,
    "prompt_type": "gemini",
    "speaker_characteristics": None,
    "experiment_id": 4
}

datasets = ["meld", "iemocap"]
speaker_characteristic_prompts = ["default", "alt1", "alt2"]

config_list = []

for dataset in datasets:
    for speaker_characteristic_prompt in speaker_characteristic_prompts:
        c = get_updated_config(config, dataset, speaker_characteristic_prompt)
        config_list.append(c)

for config in config_list:
    llm_code_eval.main(config)



config["experiment_id"] = "4b"
config["speaker_characteristics"] = "default-no-audio"
config_list.clear()
for dataset in datasets:
    c = config.copy()
    c["dataset"] = dataset
    config_list.append(c)

for config in config_list:
    llm_code_eval.main(config)
