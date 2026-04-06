"""Assemble the Phase 2 fine-tuning dataset (JSONL) with RAG demonstrations.

Calls :func:`~src.helper.build_prompting_dataset.main` with a fixed
configuration (hybrid vector store, window size 7, top-2 demonstrations,
detailed examples, refinement level 1) for each dataset and writes the
resulting split data as JSONL files to ``data/training/stage2/<DATASET>/``.

Each JSONL line contains the fields ``input`` (all prompt variables),
``target`` (the gold emotion label), and ``idx``.

Edit the ``TRAINING_SET_CONFIGS`` dict at the top of the file to adjust
the RAG retrieval parameters before running.

Usage::

    python src/training_data_creation/phase2/build_dataset.py
"""

import json
from src.helper import utils, build_prompting_dataset
from src.config import paths

vectordb_path = utils.get_vectordb_path_from_attributes("hybrid", max_m=7)
TRAINING_SET_CONFIGS = {"dataset": "iemocap", "max_k":20, "top_n":2, "split":["train", "dev", "test"],
          "vectordb_path": vectordb_path, "use_detailed_example": True,
          "example_refinement_level": 1, "save_as": "no"}

def create_phase2_training_data_for(dataset_name):
    output_dir = paths.TRAINING_STAGE2_DIR / dataset_name.upper()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory for {dataset_name.upper()}: {output_dir}")

    tmp_configs = {**TRAINING_SET_CONFIGS, "dataset": dataset_name}
    processed_dataset = build_prompting_dataset.main(tmp_configs)  # dict: {"train": [...], "dev": [...], "test": [...]}


    for split_name, split_data in processed_dataset.items():
        output_file_path = output_dir / f"{split_name}.jsonl"

        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for item in split_data:
                # Write each item as a JSON string on its own line
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Successfully created: {output_file_path}")




for dataset in ["iemocap", "meld"]:
    create_phase2_training_data_for(dataset)