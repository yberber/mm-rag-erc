import utils
import os
import json


def create_phase1_training_data_for(dataset_name):
    output_dir = os.path.join(utils.PROJECT_PATH, f"TRAINING_DATA/PHASE1/{dataset_name.upper()}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory for {dataset_name.upper()}: {output_dir}")

    train, dev = utils.get_stage1_training_set(dataset_name, splits=["train", "dev"])

    df = utils.get_meld_iemocap_datasets_as_dataframe(splits=["train", "dev"], return_only_dataset=dataset_name)

    df.set_index("idx", inplace=True)
    valid_indices = df[(df.isna().sum(axis=1) == 0)].index.tolist()
    len(valid_indices)

    new_train = [entry for entry in train if entry['iden'] in valid_indices]
    new_dev = [entry for entry in dev if entry['iden'] in valid_indices]

    assert len(train)  + len(dev) == len(df)
    assert len(new_train) + len(new_dev) == len(valid_indices)
    assert len(valid_indices) < len(df)

    for split_name, split_data in zip(["train", "dev"], [new_train, new_dev]):
        output_file_path = os.path.join(output_dir, f"{split_name}.jsonl")

        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for item in split_data:
                # Write each item as a JSON string on its own line
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Successfully created: {output_file_path}")




for dataset in [ "iemocap", "meld"]:
    create_phase1_training_data_for(dataset)