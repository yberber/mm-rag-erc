"""Entry-point script for Phase 1 speaker-characteristic generation.

Configures and launches :mod:`parallel_character_extraction` for each
dataset specified in the ``config`` dict.  Edit the ``config`` block to
adjust the model, prompt type, window size, and dataset(s) before running.

Key configuration options:

- ``dataset_name``: list of datasets to process (``"meld"``, ``"iemocap"``).
- ``model_id``: 0 = Ollama, 1 = HuggingFace, 2 = Gemini-flash, 3 = Gemini-lite.
- ``prompt_type``: ``"default"`` | ``"alt1"`` | ``"alt2"`` | ``"default-no-audio"``.
- ``splits``: which splits to annotate (test split is never used in Phase 1).
- ``limit``: set to a small integer for a quick sanity-check run.

Usage::

    python -m src.training_data_creation.phase1.generate_speaker_characteristics
"""

import asyncio

from src.training_data_creation.phase1 import parallel_character_extraction

datasets = ["meld", "iemocap"]

# we can use different prompt settings for ablations studies such as
# prompt_types = ["default", "alt1", "alt2"] # to use different prompt template
# prompt_types = ["default-no-audio"] # to use prompt with no audio information

# TODO: remove limit and use advanced model for generating high quality and large dataset
config = {
    'dataset_name': ['meld'],
    'max_k': 20,
    # set to 10 for testing
    'limit': 10,
    # model id determines which model to llm model to use.
    # 0 for llama3.1-8b via ollama, 1 for llama3.1-8b via hf,
    # 2 for gemini-2.5-flash via vertexai, 3 for gemini-2.5-pro via vertexai."
    'model_id': 0,
    # test split is never used in phase 1
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
        await parallel_character_extraction.main(config)
        print("*********************\n\n\n")


if __name__ == "__main__":
    asyncio.run(main())
