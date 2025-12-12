import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

# Exact paths provided
FILE_PATHS = {
    "MELD": "EVAL_FINAL/STAGE1_2-DEFAULT-r16/MELD/QLORA/checkpoint-750/MELD/test/results_Dec-04-2025_19_56_29.json",
    "IEMOCAP": "EVAL_FINAL/STAGE1_2-DEFAULT-r16/IEMOCAP/QLORA/checkpoint-750/IEMOCAP/test/results_Dec-04-2025_19:24:07.json"
}

# Label mappings defined in your thesis (Section 4.1.1 & 4.1.2)
LABELS = {
    "MELD": ['joyful', 'sad', 'neutral', 'angry', 'surprised', 'fearful', 'disgusted'],
    "IEMOCAP": ['joyful', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
}


# ---------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------

def load_data(file_path):
    """Parses your specific JSON format to extract truth and predictions."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return [], []

    with open(file_path, 'r') as f:
        data = json.load(f)

    y_true = []
    y_pred = []

    # Iterate through the "results" list in your JSON
    for item in data.get('results', []):
        y_true.append(item['ground_truth'])
        y_pred.append(item['model_output'])

    return y_true, y_pred


def plot_cm(y_true, y_pred, classes, dataset_name, cmap='Blues'):
    """Generates and saves a normalized confusion matrix."""
    if not y_true:
        print(f"Skipping {dataset_name} due to missing data.")
        return

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Normalize by row (True Label) -> percentages
    # Add epsilon to avoid division by zero if a class is missing in GT
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    # Plot setup
    plt.figure(figsize=(8, 6.5))
    sns.set_context("paper", font_scale=1.2)  # Academic look

    # Create Heatmap
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Scale'})

    plt.title(f'{dataset_name} Confusion Matrix', fontsize=14, pad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save file (filename matches your LaTeX code)
    filename = f"{dataset_name.lower()}_confusion_matrix.png"
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    # print(f"Saved {filename}")
    # plt.close()
    plt.show()


# ---------------------------------------------------------
# 3. Main Execution
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Generating Confusion Matrices...")

    # Process MELD
    print(f"Processing MELD from: {FILE_PATHS['MELD']}")
    y_true_meld, y_pred_meld = load_data(FILE_PATHS['MELD'])
    plot_cm(y_true_meld, y_pred_meld, LABELS['MELD'], "MELD", cmap='Blues')

    # Process IEMOCAP
    print(f"Processing IEMOCAP from: {FILE_PATHS['IEMOCAP']}")
    y_true_iem, y_pred_iem = load_data(FILE_PATHS['IEMOCAP'])
    plot_cm(y_true_iem, y_pred_iem, LABELS['IEMOCAP'], "IEMOCAP", cmap='Reds')
    # plot_cm(y_true_iem, y_pred_iem, LABELS['IEMOCAP'], "IEMOCAP", cmap='Blues')


    print("\nDone! Images are saved in the current directory.")