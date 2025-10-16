import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import set_pandas_display_options, emotion_mapper_ori_to_conv

set_pandas_display_options()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=
        "Add categorical features (intensity, pitch, articulation rate) to IEMOCAP CSVs."
    )
    p.add_argument(
        "--csv_in",
        type=Path,
        default=Path("../BENCMARK_DATASETS/iemocap_erc_acoustic_features.csv"),
        help="Input CSV with acoustic features added",
    )
    p.add_argument(
        "--csv_out",
        type=Path,
        default=Path("../BENCMARK_DATASETS/iemocap_erc_with_categories.csv"),
        help="Output CSV path with categorical columns appended",
    )
    p.add_argument(
        "--categories",
        type=int,
        default=3,
        choices=(2, 3, 5),
        help="Number of categories to split into (2, 3, or 5)",
    )
    p.add_argument(
        "--split_col",
        type=str,
        default="split",
        help="Column name indicating dataset split",
    )
    p.add_argument(
        "--train_value",
        type=str,
        default="train",
        help="Value used in split column for the train set",
    )
    p.add_argument(
        "--gender_col",
        type=str,
        default="speaker",
        help="Column containing gender markers (e.g., 'F' or 'M') for pitch thresholds",
    )
    return p.parse_args()


def _safe_values(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy()
    vals = vals[np.isfinite(vals)]
    return vals


def quantile_thresholds(values: np.ndarray, qs: Sequence[float]) -> List[float]:
    if values.size == 0:
        return []
    return [float(np.quantile(values, q)) for q in qs]


def digitize_labels(values: pd.Series, thresholds: Sequence[float], labels: Sequence[str]) -> pd.Series:
    """Return a Series of string labels (object dtype), handling NaNs gracefully.

    We assign plain strings first to avoid pandas categorical slice-assignment issues,
    then convert to ordered categoricals at the caller if desired.
    """
    if thresholds is None or len(thresholds) == 0:
        return pd.Series([None] * len(values), index=values.index, dtype=object)

    bins = np.array(thresholds, dtype=float)
    arr = pd.to_numeric(values, errors="coerce").to_numpy()
    out = np.empty(arr.shape[0], dtype=object)
    finite = np.isfinite(arr)
    if finite.any():
        idx = np.digitize(arr[finite], bins=bins, right=False)
        idx = np.clip(idx, 0, len(labels) - 1)
        out[finite] = [labels[i] for i in idx]
    out[~finite] = None
    return pd.Series(out, index=values.index, dtype=object)


# Gender normalization is unnecessary for IEMOCAP (values are 'M'/'F').


def labels_for(feature: str, n: int) -> List[str]:
    # Use generic labels across all features, ordered low->high
    if n == 2:
        return ["low", "high"]
    if n == 3:
        return ["low", "medium", "high"]
    if n == 5:
        return ["very low", "low", "medium", "high", "very high"]
    return [f"bin_{i}" for i in range(n)]


def thresholds_for_feature(
    train_df: pd.DataFrame,
    feature: str,
    n: int,
    gender_col: str,
) -> Dict[str, List[float]]:
    """Return thresholds dict. For pitch, keys per gender ('F','M'); otherwise 'ALL'."""
    if n == 2:
        qs = [0.5]
    elif n == 3:
        qs = [0.25, 0.75]
    else:  # n == 5
        qs = [0.2, 0.4, 0.6, 0.8]

    if feature == "pitch_mean_hz":
        out: Dict[str, List[float]] = {}
        for g in ("F", "M"):
            gvals = _safe_values(train_df.loc[train_df[gender_col] == g, feature])
            if gvals.size == 0:
                # Fallback to overall if empty
                gvals = _safe_values(train_df[feature])
            out[g] = quantile_thresholds(gvals, qs)
        return out
    else:
        vals = _safe_values(train_df[feature])
        return {"ALL": quantile_thresholds(vals, qs)}


def apply_categories(
    df: pd.DataFrame,
    thresholds: Dict[str, List[float]],
    labels: Sequence[str],
    feature: str,
    gender_col: str,
) -> pd.Series:
    if feature == "pitch_mean_hz":
        # Build as object series first, then convert to categorical once
        out = pd.Series([None] * len(df), index=df.index, dtype=object)
        for g in ("F", "M"):
            mask = df[gender_col] == g
            if mask.any():
                out.loc[mask] = digitize_labels(df.loc[mask, feature], thresholds[g], labels)
        return pd.Series(pd.Categorical(out, categories=list(labels), ordered=True), index=df.index)
    else:
        labs = digitize_labels(df[feature], thresholds["ALL"], labels)
        return pd.Series(
            pd.Categorical(labs, categories=list(labels), ordered=True), index=labs.index
        )


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv_in)

    # Ensure required columns exist
    required_cols = {
        "intensity_mean_db",
        "pitch_mean_hz",
        "articulation_rate_syll_per_s",
        args.split_col,
        args.gender_col,
    }
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    train_df = df[df[args.split_col] == args.train_value]
    train_df = train_df[["dialog_id", "turn_id", "speaker", "gender", "intensity_mean_db", "pitch_mean_hz", "articulation_rate_syll_per_s"]]
    if train_df.empty:
        raise ValueError(f"No rows found where {args.split_col} == {args.train_value}")

    features = [
        "intensity_mean_db",
        "pitch_mean_hz",
        "articulation_rate_syll_per_s",
    ]

    thresholds_map: Dict[str, Dict[str, List[float]]] = {}
    label_map: Dict[str, List[str]] = {}
    for feat in features:
        thresholds_map[feat] = thresholds_for_feature(train_df, feat, args.categories, args.gender_col)
        label_map[feat] = labels_for(feat, args.categories)

    # Apply categories to full dataframe using train-derived thresholds
    df["intensity_level"] = apply_categories(
        df, thresholds_map["intensity_mean_db"], label_map["intensity_mean_db"], "intensity_mean_db", args.gender_col
    )
    df["pitch_level"] = apply_categories(
        df, thresholds_map["pitch_mean_hz"], label_map["pitch_mean_hz"], "pitch_mean_hz", args.gender_col
    )
    df["rate_level"] = apply_categories(
        df,
        thresholds_map["articulation_rate_syll_per_s"],
        label_map["articulation_rate_syll_per_s"],
        "articulation_rate_syll_per_s",
        args.gender_col,
    )

    assert np.all([(e in emotion_mapper_ori_to_conv.keys()) for e in df['emotion'].unique().tolist()])
    df["mapped_emotion"] = df['emotion'].map(emotion_mapper_ori_to_conv)

    df["idx"] = "i_" + df["dialog_idx"].astype(str) + "_" + df["turn_idx"].astype(str)

    # Save
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_out, index=False)

    # Print a compact summary of thresholds used
    def fmt_thr(d: Dict[str, List[float]]) -> str:
        return ", ".join(f"{k}:{[round(x,2) for x in v]}" for k, v in d.items())

    print("Saved:", args.csv_out)
    for feat in features:
        print(f"{feat} thresholds ->", fmt_thr(thresholds_map[feat]))
        print(f"{feat} labels ->", label_map[feat])


if __name__ == "__main__":
    main()
