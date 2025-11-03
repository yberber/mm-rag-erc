




from STAGE1 import llm_character_extraction, parallel_llm_character_extraction

from utils import get_vectordb_path_from_attributes
import asyncio  # 1. Import asyncio

datasets = ["meld"]
# prompt_types = ["default", "alt1", "alt2"]
prompt_types = ["default-no-audio"]

config = {
    'dataset_name': ['meld', 'iemocap'],
    'max_k': 20,
    'limit': None,
    'model_id': 2,
    'splits': ['train','dev'],
    'prompt_type': 'default'
}


config_list = []
for dataset in datasets:
    for prompt_type in prompt_types:

        temp_config = config.copy()
        temp_config['dataset_name'] = dataset
        temp_config['prompt_type'] = prompt_type
        config_list.append(temp_config)


# config['prompt_type'] = 'default'
# config['splits'] = None
# for dataset in datasets:
#     config['dataset_name'] = dataset
#     config_list.append(config.copy())


# for config in config_list[6:]:
#     print(f"config: {config}")
#     llm_character_extraction.main(config)
#     print("*********************\n\n\n")

async def main():
    for config in config_list:
        print(f"config: {config}")
        # 3. Use 'await' to properly call the async function
        await parallel_llm_character_extraction.main(config)
        print("*********************\n\n\n")

# 4. Use asyncio.run() to start the main async function
if __name__ == "__main__":
    asyncio.run(main())
