import utils
import os
import json
import data_process

vectordb_path = utils.get_vectordb_path_from_attributes("hybrid", max_m=7)
TRAINING_SET_CONFIGS = {"dataset": "iemocap", "max_k":20, "top_n":2, "split":["train", "dev", "test"],
          "vectordb_path": vectordb_path, "use_detailed_example": True,
          "example_refinement_level": 1, "save_as": "no"}

def create_phase2_training_data_for(dataset_name):
    output_dir = os.path.join(utils.PROJECT_PATH, f"TRAINING_DATA/PHASE2/{dataset_name.upper()}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory for {dataset_name.upper()}: {output_dir}")

    tmp_configs = {**TRAINING_SET_CONFIGS, "dataset": dataset_name}
    processed_dataset = data_process.main(tmp_configs)  # dict: {"train": [...], "dev": [...], "test": [...]}


    for split_name, split_data in processed_dataset.items():
        output_file_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for item in split_data:
                # Write each item as a JSON string on its own line
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Successfully created: {output_file_path}")




for dataset in ["iemocap", "meld"]:
    create_phase2_training_data_for(dataset)