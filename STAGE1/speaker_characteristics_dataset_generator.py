




import llm_character_extraction
from utils import get_vectordb_path_from_attributes


datasets = ["iemocap", "meld"]
prompt_types = ["default", "alt1", "alt2"]

config = {
    'dataset_name': 'meld',
    'max_k': 20,
    'limit': 25,
    'model_id': 3,
    'splits': ['train', 'dev', 'test'],
    'prompt_type': 'default'
}


config_list = []
for dataset in datasets:
    for prompt_type in prompt_types:

        temp_config = config.copy()
        temp_config['dataset_name'] = dataset
        temp_config['prompt_type'] = prompt_type
        config_list.append(temp_config)


for config in config_list:
    print(f"config: {config}")
    llm_character_extraction.main(config)
    print("*********************\n\n\n")

