import os
import json
import glob
import matplotlib.pyplot as plt
import re
import numpy as np



plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13
})
# ==========================================
# CONFIGURATION
# ==========================================
# Update this path to point to your 'STAGE1_2-DEFAULT-r16' folder
BASE_PATH = "EVAL_FINAL/STAGE1_2-DEFAULT-r16"

# The names of the 3 model folders
MODELS = ["COMBINED", "MELD-Only", "IEMOCAP-Only"]

# The datasets evaluated
EVAL_DATASETS = ["MELD", "IEMOCAP"]

# Mappings for the fancy headers
TITLE_MAPPING = {
    "BOTH": "Final model trained on BOTH datasets (MELD + IEMOCAP)",
    "MELD": "Final model trained on MELD",
    "IEMOCAP": "Final model trained on IEMOCAP"
}
SUB_TITLE_SUFFIX = "\n(Llama-3.1-8B-Instruct with trained Stage-1 adapter, fine-tuned)"


def parse_checkpoint_number(folder_name):
    """Extracts the number from 'checkpoint-300'."""
    match = re.search(r'checkpoint-(\d+)', folder_name)
    return int(match.group(1)) if match else None


def get_data_for_model(base_path, model_name):
    """Parses directory structure to get scores."""
    model_dir = os.path.join(base_path, model_name, "QLORA")

    data = {
        "MELD": {"checkpoints": [], "accuracy": [], "f1": []},
        "IEMOCAP": {"checkpoints": [], "accuracy": [], "f1": []}
    }

    if not os.path.exists(model_dir):
        print(f"Warning: Directory not found: {model_dir}")
        return data

    ckpt_folders = [f for f in os.listdir(model_dir) if f.startswith("checkpoint-")]
    ckpt_folders.sort(key=parse_checkpoint_number)

    for ckpt in ckpt_folders:
        step = parse_checkpoint_number(ckpt)
        ckpt_path = os.path.join(model_dir, ckpt)

        for eval_set in EVAL_DATASETS:
            results_path = os.path.join(ckpt_path, eval_set, "test", "*.json")
            json_files = glob.glob(results_path)

            if json_files:
                with open(json_files[0], 'r') as f:
                    try:
                        content = json.load(f)
                        stats = content.get("stats", {})
                        acc = stats.get("accuracy")
                        f1 = stats.get("f1_weighted")

                        if acc is not None and f1 is not None:
                            data[eval_set]["checkpoints"].append(step)
                            data[eval_set]["accuracy"].append(acc*100)
                            data[eval_set]["f1"].append(f1*100)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {json_files[0]}")
    return data


def plot_model_performance(model_name, data):
    """Generates the figure with detailed headers and best score annotations."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Construct the descriptive title
    # main_title = TITLE_MAPPING.get(model_name, f"Model: {model_name}")
    # full_title = main_title + SUB_TITLE_SUFFIX
    # fig.suptitle(full_title, fontsize=14, weight='bold', y=0.98)

    subplots = [
        (axes[0], data["MELD"], "MELD Evaluation"),
        (axes[1], data["IEMOCAP"], "IEMOCAP Evaluation")
    ]

    for ax, dataset_data, title in subplots:
        x = dataset_data["checkpoints"]
        acc = dataset_data["accuracy"]
        f1 = dataset_data["f1"]

        if not x:
            ax.text(0.5, 0.5, "No Data Found", ha='center', va='center')
            ax.set_title(title)
            continue

        # Plot Weighted F1 (Solid Line)
        ax.plot(x, f1, label='Weighted F1', color='#1f77b4', linestyle='-', linewidth=2, marker='o', markersize=4)

        # Plot Accuracy (Dotted Line)
        ax.plot(x, acc, label='Accuracy', color='#ff7f0e', linestyle=':', linewidth=2, marker='s', markersize=4)

        # --- NEW: Find and Annotate Best F1 ---
        max_f1 = max(f1)
        max_idx = f1.index(max_f1)
        best_ckpt = x[max_idx]

        # 1. Highlight the best point on the chart with a red star
        ax.plot(best_ckpt, max_f1, marker='*', color='red', markersize=12, label='Best F1')

        # 2. Add text box in bottom right corner
        stats_text = f"Best Weighted F1: {max_f1:.2f}\n(at step {best_ckpt})"
        ax.text(0.95, 0.05, stats_text,
                transform=ax.transAxes,  # Relative to axes (0,0 is bottom-left, 1,1 is top-right)
                fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
        # --------------------------------------

        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel("Checkpoints (Steps)", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make room for the large title
    plt.show()


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(BASE_PATH):
        print(f"Error: Base path '{BASE_PATH}' does not exist.")
        print("Please update the 'BASE_PATH' variable inside the script.")
    else:
        for model in MODELS:
            print(f"Generating plot for: {model}...")
            model_data = get_data_for_model(BASE_PATH, model)
            plot_model_performance(model, model_data)