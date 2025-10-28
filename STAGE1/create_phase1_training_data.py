import utils
import os
import json



output_dir = os.path.join(utils.PROJECT_PATH, "TRAINING_DATA/PHASE1/IEMOCAP")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

dataset = utils.load_json(relative_path_from_project="STAGE1/data/IEMOCAP-model2_default_k20_train-dev-test_size10039.json")["dataset"]
train, dev, test = dataset["train"], dataset["dev"], dataset["test"]

_, iemocap_df = utils.get_meld_iemocap_datasets_as_dataframe()

iemocap_df.set_index("idx", inplace=True)
valid_indices = iemocap_df[(iemocap_df.isna().sum(axis=1) == 0)].index.tolist()
len(valid_indices)

new_train = [entry for entry in train if entry['iden'] in valid_indices]
new_dev = [entry for entry in dev if entry['iden'] in valid_indices]
new_test = [entry for entry in test if entry['iden'] in valid_indices]

assert len(train) + len(test) + len(dev) == len(iemocap_df)
assert len(new_train) + len(new_test) + len(new_dev) == len(valid_indices)
assert len(valid_indices) < len(iemocap_df)

for split_name, split_data in zip(["train", "dev", "test"], [new_train, new_dev, new_test]):
    output_file_path = os.path.join(output_dir, f"{split_name}.jsonl")

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for item in split_data:
            # Write each item as a JSON string on its own line
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Successfully created: {output_file_path}")