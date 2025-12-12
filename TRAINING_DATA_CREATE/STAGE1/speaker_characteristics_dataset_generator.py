




from TRAINING_DATA_CREATE.STAGE1 import llm_character_extraction, parallel_llm_character_extraction

from utils import get_vectordb_path_from_attributes
import asyncio  # 1. Import asyncio

datasets = ["meld", "iemocap"]
# prompt_types = ["default", "alt1", "alt2"]
# prompt_types = ["default-no-audio"]


config = {
    'dataset_name': ['meld'],
    'max_k': 20,
    'limit': None,
    'model_id': 2,
    'splits': ['train','dev'],
    'prompt_type': 'default'
}


config_list = []
for dataset in datasets:
    temp_config = config.copy()
    temp_config['dataset_name'] = dataset
    config_list.append(temp_config)

async def main():
    for config in config_list:
        print(f"config: {config}")
        await parallel_llm_character_extraction.main(config)
        print("*********************\n\n\n")


if __name__ == "__main__":
    asyncio.run(main())
