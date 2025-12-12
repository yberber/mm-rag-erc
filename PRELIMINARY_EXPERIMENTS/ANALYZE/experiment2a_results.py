import pandas as pd
from glob import glob
from pathlib import Path
from sklearn.metrics import f1_score
from typing import Dict, Any, List

from utils import (get_meld_iemocap_datasets_as_dataframe,
                   load_json, set_pandas_display_options, save_as_json)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
CACHE_DIR = SCRIPT_DIR.parent.parent / "vectorstore" / "caches"
TOP_N = 3

set_pandas_display_options()


def load_and_prepare_data() -> (pd.DataFrame, pd.DataFrame, Dict[str, str]):
    """
    Loads datasets, prepares dev and train splits, and creates an emotion map.
    The train split is indexed for fast conversational window lookups.
    """
    print("Loading and preparing datasets...")
    df_meld, df_iemocap = get_meld_iemocap_datasets_as_dataframe()
    df_meld["dataset"] = "MELD"
    df_iemocap["dataset"] = 'IEMOCAP'

    # --- Prepare Dev Data ---
    # MODIFICATION: Using 'dev' split instead of 'test'
    df_meld_dev = df_meld.loc[df_meld["split"] == "dev", ['idx', 'mapped_emotion', 'dataset']]
    df_iemocap_dev = df_iemocap.loc[
        (df_iemocap["split"] == "dev") & (df_iemocap['erc_target']), ['idx', 'mapped_emotion', 'dataset']]
    df_dev = pd.concat([df_meld_dev, df_iemocap_dev], axis=0, ignore_index=True)
    print(f"Prepared dev set with {len(df_dev)} utterances.")

    # --- Prepare Train Data (for reconstructing emotion flows) ---
    df_meld_train = df_meld.loc[df_meld['split'] == 'train', ['dialog_idx', 'idx', 'mapped_emotion', 'dataset']]
    df_iemocap_train = df_iemocap.loc[
        (df_iemocap['split'] == 'train') & (df_iemocap['mapped_emotion'] != 'unknown'), ['dialog_idx', 'idx',
                                                                                         'mapped_emotion', 'dataset']]
    df_train = pd.concat([df_meld_train, df_iemocap_train], ignore_index=True)
    df_train["turn_idx"] = df_train.groupby(['dataset', 'dialog_idx']).cumcount()
    df_train = df_train.set_index('idx', drop=False)  # Set idx as index for fast .loc lookups
    print(f"Prepared train set with {len(df_train)} utterances for context lookups.")

    # --- Create Emotion Map ---
    idx_to_emotion_map = pd.Series(
        pd.concat([df_meld['mapped_emotion'], df_iemocap['mapped_emotion']]).values,
        index=pd.concat([df_meld['idx'], df_iemocap['idx']]).values
    ).to_dict()
    print(f"Created emotion map with {len(idx_to_emotion_map)} entries.")

    return df_dev, df_train, idx_to_emotion_map


def get_predominant_emotions(retrieved_idx: str, db_type: str, window_size: int, df_train: pd.DataFrame,
                             idx_to_emotion_map: dict) -> List[str]:
    """Finds the most frequent emotion(s) in the context window of a retrieved utterance."""
    if db_type == 'single':
        emotion = idx_to_emotion_map.get(retrieved_idx)
        return [emotion] if emotion else []
    try:
        final_utterance = df_train.loc[retrieved_idx]
        dialog_idx, dataset, turn_idx_end = final_utterance['dialog_idx'], final_utterance['dataset'], final_utterance[
            'turn_idx']
        dialog_df = df_train[(df_train['dialog_idx'] == dialog_idx) & (df_train['dataset'] == dataset)]
        turn_idx_start = max(0, turn_idx_end - window_size + 1)
        window_df = dialog_df[(dialog_df['turn_idx'] >= turn_idx_start) & (dialog_df['turn_idx'] <= turn_idx_end)]
        return window_df['mapped_emotion'].mode().tolist()
    except (KeyError, IndexError):
        return []


def calculate_metrics_predominant(y_true: pd.Series, y_pred_lists: pd.Series) -> float:
    """Calculates weighted F1-score where the prediction is a list of predominant emotions."""
    # MODIFICATION: Only calculating F1-Score
    y_true_list = y_true.tolist()
    y_pred_lists_list = y_pred_lists.tolist()

    # If match, use true_label (rewards model). If no match, default to first in list.
    single_preds = [
        (true_label if pred_list and true_label in pred_list else (pred_list[0] if pred_list else None))
        for true_label, pred_list in zip(y_true_list, y_pred_lists_list)
    ]
    weighted_f1 = f1_score(y_true_list, single_preds, average='weighted', zero_division=0)
    return round(weighted_f1, 4)


def run_analysis_pipeline_predominant(cache_paths: List[Path], df_dev: pd.DataFrame, df_train: pd.DataFrame,
                                      idx_to_emotion_map: Dict[str, str]):
    """Runs the analysis based on predominant emotions."""
    all_results = {}
    print(f"\nRunning analysis based on PREDOMINANT emotions (dev set size: {len(df_dev)})...")

    for path in cache_paths:
        cache_name = path.stem
        print(f"  -> Analyzing cache: {cache_name}")
        parts = cache_name.split('_')
        db_type = parts[2]
        window_size = int(parts[3]) if len(parts) > 3 else 1
        cache_data = load_json(path=str(path))
        df_analysis = df_dev.copy()
        df_analysis['sim_idx_list'] = df_analysis['idx'].map(cache_data)

        results = {}
        for i in range(1, TOP_N + 1):
            df_analysis[f'pred_emotions_@{i}'] = df_analysis['sim_idx_list'].apply(
                lambda idx_list: get_predominant_emotions(idx_list[i - 1], db_type, window_size, df_train,
                                                          idx_to_emotion_map) if (
                        isinstance(idx_list, list) and len(idx_list) >= i) else []
            )

            # MODIFICATION: Removed 'Total'
            for dataset_name, df_subset in [('MELD', df_analysis[df_analysis['dataset'] == 'MELD']),
                                            ('IEMOCAP', df_analysis[df_analysis['dataset'] == 'IEMOCAP'])]:
                # MODIFICATION: Only getting F1
                metric = calculate_metrics_predominant(df_subset['mapped_emotion'], df_subset[f'pred_emotions_@{i}'])
                results[f'{dataset_name}_F1_@{i}'] = metric  # Storing only F1

        all_results[cache_name] = results

    df_summary = pd.DataFrame.from_dict(all_results, orient='index')
    df_summary.columns = pd.MultiIndex.from_tuples(
        [col.split('_') for col in df_summary.columns], names=['dataset', 'metric', 'position']
    )
    return df_summary, all_results


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculates weighted F1-score and returns it as a float."""
    # MODIFICATION: Only calculating F1-Score
    if y_true.empty or y_pred.dropna().empty:
        return 0.0
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return round(weighted_f1, 4)


def analyze_cache(cache_path: Path, df_dev: pd.DataFrame, idx_to_emotion_map: Dict[str, str]) -> Dict[str, Any]:
    """Analyzes a single cache file and returns a dictionary of performance metrics."""
    cache_name = cache_path.stem
    cache_data = load_json(path=str(cache_path))
    df_analysis = df_dev.copy()

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
        # MODIFICATION: Removed 'Total'
        for dataset_name, df_subset in [('MELD', df_meld_results), ('IEMOCAP', df_iemocap_results)]:
            # MODIFICATION: Only getting F1
            metric = calculate_metrics(df_subset['mapped_emotion'], df_subset[pred_col])
            results[f'{dataset_name}_F1_@{i}'] = metric  # Storing only F1

    return {cache_name: results}


def run_analysis_pipeline(cache_paths: List[Path], df_dev: pd.DataFrame,
                          idx_to_emotion_map: Dict[str, str]):
    """Runs the full analysis for a given dev set and returns a summary DataFrame."""
    all_results = {}
    print(f"\nRunning analysis on a dev set of size {len(df_dev)}...")
    for path in cache_paths:
        print(f"  -> Analyzing cache: {Path(path).stem}")
        result = analyze_cache(path, df_dev, idx_to_emotion_map)
        all_results.update(result)

    df_summary = pd.DataFrame.from_dict(all_results, orient='index')
    df_summary.columns = pd.MultiIndex.from_tuples(
        [col.split('_') for col in df_summary.columns],
        names=['dataset', 'metric', 'position']
    )
    return df_summary, all_results


import numpy as np  # Make sure to import numpy at the top of your script


def display_results(df_summary: pd.DataFrame, title: str):
    """
    Formats and prints the top-1 results as a single, combined pivot table
    with MELD and IEMOCAP side-by-side.
    """
    print("\n\n" + "=" * 80)
    print(f"📊 {title} (Top-1 Weighted F1-Scores)")
    print("=" * 80)

    # 1. Select only top-1 results
    top_1_results = df_summary.xs('@1', level='position', axis=1)

    # F1 is the only metric, so drop that level to simplify
    top_1_results.columns = top_1_results.columns.droplevel('metric')

    # 2. Parse the index to get 'strategy' and 'm' (window size)
    df = top_1_results.reset_index().rename(columns={'index': 'vectorstore'})

    def get_strategy(name):
        if 'single' in name: return 'single'
        if 'flow' in name: return 'flow'
        if 'hybrid' in name: return 'hybrid'
        return 'unknown'

    def get_m(name):
        if 'single' in name: return 1
        try:
            # Get the last part of the name (e.g., '7', '10')
            return int(name.split('_')[-1])
        except (ValueError, IndexError):
            return pd.NA

    df['strategy'] = df['vectorstore'].apply(get_strategy)
    df['m'] = df['vectorstore'].apply(get_m)

    # 3. Create the individual pivot tables
    row_order = ['single', 'flow', 'hybrid']
    col_order = sorted(df['m'].unique())

    pivot_meld = df.pivot(index='strategy', columns='m', values='MELD')
    pivot_meld = pivot_meld.loc[row_order].reindex(columns=col_order)

    pivot_iemocap = df.pivot(index='strategy', columns='m', values='IEMOCAP')
    pivot_iemocap = pivot_iemocap.loc[row_order].reindex(columns=col_order)

    # 4. Concatenate the tables side-by-side with keys
    df_combined = pd.concat([pivot_meld, pivot_iemocap], axis=1, keys=['MELD', 'IEMOCAP'])

    # 5. Print the final combined table
    print(df_combined.to_string(float_format="%.2f", na_rep="-"))

def main():
    """Main function to run all three analysis pipelines."""

    df_dev, df_train, idx_to_emotion_map = load_and_prepare_data()

    cache_paths = sorted(list(CACHE_DIR.glob("*.json")))
    if not cache_paths:
        print(f"❌ Error: No cache files found in {CACHE_DIR}")
        return

    # --- 1. Run analysis on the COMPLETE dev set ---
    df_summary_all, results_dict_all = run_analysis_pipeline(cache_paths, df_dev, idx_to_emotion_map)
    display_results(df_summary_all, "Performance (Final Utterance Emotion)")

    # --- 2. Run analysis EXCLUDING 'neutral' ---
    df_dev_no_neutral = df_dev[df_dev['mapped_emotion'] != 'neutral'].copy()
    df_summary_no_neutral, results_dict_no_neutral = run_analysis_pipeline(cache_paths, df_dev_no_neutral,
                                                                           idx_to_emotion_map)
    display_results(df_summary_no_neutral, "Performance (Final Utterance, Excl. 'neutral')")

    # --- 3. Run analysis based on PREDOMINANT emotion ---
    df_summary_predominant, results_dict_predominant = run_analysis_pipeline_predominant(cache_paths, df_dev, df_train,
                                                                                         idx_to_emotion_map)
    display_results(df_summary_predominant, "Performance (Based on Predominant Emotion)")


if __name__ == "__main__":
    main()