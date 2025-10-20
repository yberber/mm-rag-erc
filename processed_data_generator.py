

import data_process
from utils import get_vectordb_path_from_attributes


datasets = ["iemocap", "meld"]
max_k = 12
top_n = [1,2,3]
max_m = 3
split = "dev"
db_types = ["single", "flow", "hybrid"]
use_detailed_example = True

config_list = []
for dataset in datasets:
    for db_type in db_types:
        for n in top_n:
            vectordb_path = get_vectordb_path_from_attributes(db_type, max_m=max_m)
            config = {"dataset": dataset, "max_k": max_k, "top_n": n, "vectordb_path": vectordb_path,
                      "split": split, "use_detailed_example": str(use_detailed_example)}
            config_list.append(config)

for config in config_list:
    print(f"config: {config}")
    data_process.main(config)
    print("*********************\n\n\n")

