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

    df_meld_test = df_meld.loc[df_meld["split"] == "test", ['idx', 'mapped_emotion']]
    df_meld_test["dataset"] = "MELD"

    df_iemocap_test = df_iemocap.loc[
        (df_iemocap["split"] == "test") & (df_iemocap['erc_target']), ['idx', 'mapped_emotion']]
    df_iemocap_test["dataset"] = "IEMOCAP"

    df_test = pd.concat([df_meld_test, df_iemocap_test], axis=0, ignore_index=True)

    idx_to_emotion = pd.concat([df_meld[['idx', 'mapped_emotion']], df_iemocap[['idx', 'mapped_emotion']]])
    idx_to_emotion_map = pd.Series(idx_to_emotion.mapped_emotion.values, index=idx_to_emotion.idx).to_dict()

    print(f"Prepared test set with {len(df_test)} utterances.")
    print(f"Created emotion map with {len(idx_to_emotion_map)} entries.")
    return df_test, idx_to_emotion_map


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Calculates accuracy and weighted F1-score and returns them in a dict."""
    if y_true.empty or y_pred.dropna().empty:
        return {"accuracy": 0.0, "weighted_f1": 0.0}

    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {
        "accuracy": round(accuracy, 4),
        "weighted_f1": round(weighted_f1, 4)
    }


def analyze_cache(cache_path: Path, df_test: pd.DataFrame, idx_to_emotion_map: Dict[str, str]) -> Dict[str, Any]:
    """Analyzes a single cache file and returns a dictionary of performance metrics."""
    cache_name = cache_path.stem
    cache_data = load_json(path=str(cache_path))
    df_analysis = df_test.copy()

    df_analysis['sim_idx_list'] = df_analysis['idx'].map(cache_data)
    df_analysis['sim_emotions'] = df_analysis['sim_idx_list'].apply(
        lambda idx_list: [idx_to_emotion_map.get(idx) for idx in idx_list[:TOP_N]] if isinstance(idx_list,
                                                                                                 list) else [None] * TOP_N
    )

    pred_cols = [f'pred_emotion_@{i + 1}' for i in range(TOP_N)]
    df_analysis[pred_cols] = pd.DataFrame(df_analysis['sim_emotions'].tolist(), index=df_analysis.index)

    results = {}
    df_meld_results = df_analysis[df_analysis['dataset'] == 'MELD']
    df_iemocap_results = df_analysis[df_analysis['dataset'] == 'IEMOCAP']

    for i in range(1, TOP_N + 1):
        pred_col = f'pred_emotion_@{i}'
        for dataset_name, df_subset in [('MELD', df_meld_results), ('IEMOCAP', df_iemocap_results),
                                        ('Total', df_analysis)]:
            metrics = calculate_metrics(df_subset['mapped_emotion'], df_subset[pred_col])
            results[f'{dataset_name}_Acc_@{i}'] = metrics['accuracy']
            results[f'{dataset_name}_F1_@{i}'] = metrics['weighted_f1']

    return {cache_name: results}


def run_analysis_pipeline(cache_paths: List[Path], df_test: pd.DataFrame,
                          idx_to_emotion_map: Dict[str, str]) -> pd.DataFrame:
    """Runs the full analysis for a given test set and returns a summary DataFrame."""
    all_results = {}
    print(f"\nRunning analysis on a test set of size {len(df_test)}...")
    for path in cache_paths:
        # A simple print to show progress
        print(f"  -> Analyzing cache: {Path(path).stem}")
        result = analyze_cache(path, df_test, idx_to_emotion_map)
        all_results.update(result)

    df_summary = pd.DataFrame.from_dict(all_results, orient='index')
    df_summary.columns = pd.MultiIndex.from_tuples(
        [col.split('_') for col in df_summary.columns],
        names=['dataset', 'metric', 'position']
    )
    return df_summary


def display_results(df_summary: pd.DataFrame, title: str):
    """Formats and prints the top-1 results table."""
    print("\n\n" + "=" * 80)
    print(f"📊 {title}")
    print("=" * 80)

    # Select only top-1 results and sort by the most important metric
    top_1_results = df_summary.xs('@1', level='position', axis=1).sort_values(('Total', 'F1'), ascending=False)

    # Apply styling to highlight the best score in each column
    styled_df = top_1_results.style.highlight_max(axis=0, color='lightgreen').format("{:.2f}")

    print(styled_df.to_string())


def main():
    """Main function to run the entire analysis pipeline."""
    df_test, idx_to_emotion_map = load_and_prepare_data()

    cache_paths = sorted(list(CACHE_DIR.glob("*.json")))
    if not cache_paths:
        print(f"❌ Error: No cache files found in {CACHE_DIR}")
        return

    # --- 1. Run analysis on the COMPLETE test set ---
    df_summary_all = run_analysis_pipeline(cache_paths, df_test, idx_to_emotion_map)
    display_results(df_summary_all, "Top-1 Retrieval Performance (All Emotions)")

    # --- 2. Run analysis EXCLUDING the 'neutral' emotion ---
    df_test_no_neutral = df_test[df_test['mapped_emotion'] != 'neutral'].copy()
    df_summary_no_neutral = run_analysis_pipeline(cache_paths, df_test_no_neutral, idx_to_emotion_map)
    display_results(df_summary_no_neutral, "Top-1 Retrieval Performance (Excluding 'neutral')")


if __name__ == "__main__":
    main()