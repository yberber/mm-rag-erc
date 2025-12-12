import json
import glob
import pandas as pd
import os

def get_f1_score(file_pattern, dataset_name):
    files = glob.glob(file_pattern)
    if not files:
        return "N/A"
    latest_file = max(files, key=os.path.getmtime)
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
            score = data['stats']['f1_weighted']
            return f"{score * 100:.2f}"
    except Exception as e:
        return f"Error: {e}"

# ---------------------------------------------------------
# 1. Define Paths with your Specific Names
# ---------------------------------------------------------
paths = {
    "Ours (w/ Full Context)": {
        "IEMOCAP": "EVAL_FINAL/STAGE1_2-DEFAULT-r16/IEMOCAP/QLORA/checkpoint-750/IEMOCAP/test/results_*.json",
        "MELD":    "EVAL_FINAL/STAGE1_2-DEFAULT-r16/MELD/QLORA/checkpoint-750/MELD/test/results_*.json"
    },
    "Ours (w/o Audio)": {
        # Has RAG, but NO Audio
        "IEMOCAP": "EVAL_FINAL/ABLATION/NO_AUDIO/STAGE1_2-DEFAULT-r16/IEMOCAP/QLORA/checkpoint-750/IEMOCAP/test/results_*.json",
        "MELD":    "EVAL_FINAL/ABLATION/NO_AUDIO/STAGE1_2-DEFAULT-r16/MELD/QLORA/checkpoint-750/MELD/test/results_*.json"
    },
    "Ours (w/o RAG)": {
        # Has Audio, but NO RAG
        "IEMOCAP": "EVAL_FINAL/ABLATION/NO_RAG/STAGE1_2-DEFAULT-r16/IEMOCAP/QLORA/checkpoint-750/IEMOCAP/test/results_*.json",
        "MELD":    "EVAL_FINAL/ABLATION/NO_RAG/STAGE1_2-DEFAULT-r16/MELD/QLORA/checkpoint-750/MELD/test/results_*.json"
    },
    "Ours (w/o Audio&RAG)": {
        # Text Only
        "IEMOCAP": "EVAL_FINAL/ABLATION/NO_AUDIO_RAG/STAGE1_2-DEFAULT-r16/IEMOCAP/QLORA/checkpoint-750/IEMOCAP/test/results_*.json",
        "MELD":    "EVAL_FINAL/ABLATION/NO_AUDIO_RAG/STAGE1_2-DEFAULT-r16/MELD/QLORA/checkpoint-750/MELD/test/results_*.json"
    }
}

# ---------------------------------------------------------
# 2. Extract Scores
# ---------------------------------------------------------
data_map = {}

for name, dataset_paths in paths.items():
    data_map[name] = {
        "IEMOCAP": get_f1_score(dataset_paths["IEMOCAP"], "IEMOCAP"),
        "MELD": get_f1_score(dataset_paths["MELD"], "MELD")
    }

# Add Base Model manually
data_map["Base Model (w/ Full Context)"] = {
    "IEMOCAP": "65.00",
    "MELD": "55.00"
}

# ---------------------------------------------------------
# 3. Build Ordered List
# ---------------------------------------------------------
order = [
    "Base Model (w/ Full Context)",
    "Ours (w/o Audio&RAG)",
    "Ours (w/o Audio)",
    "Ours (w/o RAG)",
    "Ours (w/ Full Context)"
]

ordered_rows = []
for name in order:
    row = {"Model Configuration": name}
    row["IEMOCAP (F1)"] = data_map[name]["IEMOCAP"]
    row["MELD (F1)"] = data_map[name]["MELD"]
    ordered_rows.append(row)

df = pd.DataFrame(ordered_rows)

# ---------------------------------------------------------
# 4. Display
# ---------------------------------------------------------
print("\n" + "="*60)
print("INFERENCE ABLATION RESULTS (Weighted F1 Scores %)")
print("="*60)
print(df.to_string(index=False, col_space=15, justify='center'))
print("="*60)