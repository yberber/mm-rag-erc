import os
import pandas as pd
from glob import glob
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any, List

# Assuming 'utils' is in a location Python can find
from utils import (get_meld_iemocap_datasets_as_dataframe,
                   load_json, set_pandas_display_options, save_as_json)

# --- Configuration ---
# Define paths relative to the script for robustness. No more os.chdir!
SCRIPT_DIR = Path(__file__).parent.resolve()
CACHE_DIR = SCRIPT_DIR.parent / "vectorstore" / "caches"
TOP_N = 3

set_pandas_display_options()


def load_and_prepare_data() -> (pd.DataFrame, Dict[str, str]):
    """
    Loads MELD and IEMOCAP datasets, prepares the test split,
    and creates a comprehensive mapping from utterance index to emotion.
    """
    print("Loading and preparing datasets...")
    df_meld, df_iemocap = get_meld_iemocap_datasets_as_dataframe()

    # Create the test DataFrame
    df_meld_test = df_meld.loc[df_meld["split"] == "test", ['idx', 'mapped_emotion']]
    df_meld_test["dataset"] = "MELD"

    df_iemocap_test = df_iemocap.loc[
        (df_iemocap["split"] == "test") & (df_iemocap['erc_target']), ['idx', 'mapped_emotion']]
    df_iemocap_test["dataset"] = "IEMOCAP"

    df_test = pd.concat([df_meld_test, df_iemocap_test], axis=0, ignore_index=True)

    # Create the idx -> emotion mapping from the full datasets
    idx_to_emotion = pd.concat([df_meld[['idx', 'mapped_emotion']], df_iemocap[['idx', 'mapped_emotion']]])
    idx_to_emotion_map = pd.Series(idx_to_emotion.mapped_emotion.values, index=idx_to_emotion.idx).to_dict()

    print(f"Prepared test set with {len(df_test)} utterances.")
    print(f"Created emotion map with {len(idx_to_emotion_map)} entries.")
    return df_test, idx_to_emotion_map


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Calculates accuracy and weighted F1-score and returns them in a dict."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"accuracy": 0.0, "weighted_f1": 0.0}

    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {
        "accuracy": round(accuracy, 4),
        "weighted_f1": round(weighted_f1, 4)
    }


def analyze_cache(cache_path: Path, df_test: pd.DataFrame, idx_to_emotion_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Analyzes a single cache file and returns a dictionary of performance metrics.
    """
    cache_name = cache_path.stem
    print(f"\nAnalyzing cache: {cache_name}")
    cache_data = load_json(path=str(cache_path))

    # --- Use Pandas for efficient data processing ---
    # Create a temporary DataFrame for analysis
    df_analysis = df_test.copy()

    # Map each test idx to its list of top N similar training idxs
    df_analysis['sim_idx_list'] = df_analysis['idx'].map(cache_data)

    # Convert the list of similar idxs to a list of emotions
    # Pad with None if a similar utterance is missing or has no emotion
    df_analysis['sim_emotions'] = df_analysis['sim_idx_list'].apply(
        lambda idx_list: [idx_to_emotion_map.get(idx) for idx in idx_list[:TOP_N]] if isinstance(idx_list,
                                                                                                 list) else [None] * TOP_N
    )

    # Split the list of emotions into separate columns for easy access
    pred_cols = [f'pred_emotion_@{i + 1}' for i in range(TOP_N)]
    df_analysis[pred_cols] = pd.DataFrame(df_analysis['sim_emotions'].tolist(), index=df_analysis.index)

    # --- Calculate metrics for each position and dataset ---
    results = {}
    df_meld_results = df_analysis[df_analysis['dataset'] == 'MELD']
    df_iemocap_results = df_analysis[df_analysis['dataset'] == 'IEMOCAP']

    assert len(df_meld_results) + len(df_iemocap_results) == len(df_analysis)

    for i in range(1, TOP_N + 1):
        pred_col = f'pred_emotion_@{i}'

        # Calculate metrics for each subset
        meld_metrics = calculate_metrics(df_meld_results['mapped_emotion'], df_meld_results[pred_col])
        iemocap_metrics = calculate_metrics(df_iemocap_results['mapped_emotion'], df_iemocap_results[pred_col])
        total_metrics = calculate_metrics(df_analysis['mapped_emotion'], df_analysis[pred_col])

        # Store results in a structured dictionary
        results[f'MELD_Acc_@{i}'] = meld_metrics['accuracy']
        results[f'MELD_F1_@{i}'] = meld_metrics['weighted_f1']
        results[f'IEMOCAP_Acc_@{i}'] = iemocap_metrics['accuracy']
        results[f'IEMOCAP_F1_@{i}'] = iemocap_metrics['weighted_f1']
        results[f'Total_Acc_@{i}'] = total_metrics['accuracy']
        results[f'Total_F1_@{i}'] = total_metrics['weighted_f1']

    return {cache_name: results}


def main():
    """Main function to run the entire analysis pipeline."""
    df_test, idx_to_emotion_map = load_and_prepare_data()

    cache_paths = sorted(list(CACHE_DIR.glob("*.json")))
    if not cache_paths:
        print(f"❌ Error: No cache files found in {CACHE_DIR}")
        return

    all_results = {}
    for path in cache_paths:
        result = analyze_cache(path, df_test, idx_to_emotion_map)
        all_results.update(result)

    # --- Final Output ---
    # 1. The structured dictionary
    print("\n\n" + "=" * 80)
    print("✅ Analysis Complete. Results as a structured dictionary:")
    print("=" * 80)
    # A bit of formatting for cleaner printing

    save_as_json("../EVALUATION_RESULTS/retrieval_final_emotion_analysis.json", all_results)

    # 2. The summary table (DataFrame)
    print("\n\n" + "=" * 80)
    print("📊 Results Summary Table")
    print("=" * 80)
    df_summary = pd.DataFrame.from_dict(all_results, orient='index')

    # Create a MultiIndex by splitting the column names
    df_summary.columns = pd.MultiIndex.from_tuples(
        [col.split('_') for col in df_summary.columns],
        names=['dataset', 'metric', 'position']
    )


    top_1_result = df_summary.xs('@1', level='position', axis=1).sort_values(('Total', 'F1'), ascending=False)
    print(top_1_result)




if __name__ == "__main__":
    main()