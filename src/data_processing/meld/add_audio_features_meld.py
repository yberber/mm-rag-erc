"""Extend the MELD benchmark CSV with Parselmouth acoustic features.

Reads the stage-1 MELD CSV (output of ``init_meld_dataset``), locates the
corresponding WAV file for every utterance, and computes seven low-level
acoustic features using Parselmouth/Praat:

- Mean and std intensity (dB)
- Mean, std, and range of fundamental frequency / pitch (Hz)
- Articulation rate (syllables per second of phonation time)
- Mean harmonics-to-noise ratio (dB)

The enriched CSV is written to the stage-2 path defined in
``src/config/paths``.

Usage::

    python -m src.data_processing.meld.add_audio_features_meld \\
        --csv  data/benchmark/meld/meld_erc_init.csv \\
        --root /path/to/MELD.Raw \\
        --out  data/benchmark/meld/meld_erc_with_audio.csv
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from src.data_processing.syllable_nuclei import speech_rate
from src.config import paths
import parselmouth as pm
from tqdm import tqdm
import warnings




def compute_features(sound_path: Path) -> Dict[str, Optional[float]]:
    """Compute acoustic features for a single WAV file using Parselmouth.

    Args:
        sound_path (Path): Path to a WAV audio file.

    Returns:
        Dict[str, Optional[float]]: Dictionary with keys:
            ``intensity_mean_db``, ``intensity_std_db``, ``pitch_mean_hz``,
            ``pitch_std_hz``, ``pitch_range_hz``,
            ``articulation_rate_syll_per_s``, ``hnr_mean_db``.
            Values are ``float`` or ``nan`` when the feature cannot be
            computed (e.g. unvoiced audio).
    """

    snd = pm.Sound(str(sound_path))

    intensity = snd.to_intensity(time_step=0.01, minimum_pitch=100)
    inten_vals = intensity.values.T.reshape(-1)
    inten_mask = np.isfinite(inten_vals)
    inten_vals_f = inten_vals[inten_mask]
    # Energy-weighted mean intensity per Praat
    mean_intensity = float(pm.praat.call(intensity, "Get mean", 0, 0, "energy")) if inten_vals_f.size else np.nan
    std_intensity = float(np.std(inten_vals_f)) if inten_vals_f.size else np.nan

    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    freqs = pitch.selected_array["frequency"]
    valid_pitch = np.isfinite(freqs) & (freqs > 0)
    pitch_vals = freqs[valid_pitch]
    mean_pitch = float(np.mean(pitch_vals)) if pitch_vals.size else np.nan
    std_pitch = float(np.std(pitch_vals)) if pitch_vals.size else np.nan
    range_pitch = float(np.max(pitch_vals) - np.min(pitch_vals)) if pitch_vals.size else np.nan

    times = pitch.xs()
    dt = np.median(np.diff(times)) if len(times) > 1 else 0.01
    phonation_time = float(np.sum(valid_pitch) * dt)

    # Articulation rate via syllable_nuclei.speech_rate
    try:
        sr = speech_rate(str(sound_path))
        articulation_rate = float(sr.get("articulation rate(nsyll / phonationtime)", np.nan))
    except Exception:
        articulation_rate = np.nan

    # Mean HNR (dB)
    harm = pm.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    mean_hnr = float(pm.praat.call(harm, "Get mean", 0, 0))

    return {
        "intensity_mean_db": mean_intensity,
        "intensity_std_db": std_intensity,
        "pitch_mean_hz": mean_pitch,
        "pitch_std_hz": std_pitch,
        "pitch_range_hz": range_pitch,
        "articulation_rate_syll_per_s": articulation_rate,
        "hnr_mean_db": mean_hnr,
    }


def split_to_folder(split_value: str) -> str:
    """Map a split name to the corresponding MELD audio subfolder name.

    Args:
        split_value (str): Split label (``"train"``, ``"dev"``, or
            ``"test"``).

    Returns:
        str: The audio subdirectory name inside the MELD root, e.g.
            ``"train_audio_splits"``.  Defaults to ``"dev_audio_splits"``
            for unrecognised values.
    """
    s = str(split_value).strip().lower()
    if s in {"train"}:
        return "train_audio_splits"
    if s in {"dev", "valid", "validation", "val"}:
        return "dev_audio_splits"
    if s in {"test", "testing"}:
        return "test_audio_splits"
    # Default to dev if unknown
    return "dev_audio_splits"


def audio_path_for_row(root: Path, split: str, dialog_id: int, turn_id: int) -> Path:
    """Construct the expected WAV path for a MELD utterance.

    Args:
        root (Path): MELD raw-data root directory.
        split (str): Dataset split (``"train"``, ``"dev"``, or ``"test"``).
        dialog_id (int): MELD dialogue ID.
        turn_id (int): MELD utterance/turn ID within the dialogue.

    Returns:
        Path: Expected path to the WAV file, e.g.
            ``<root>/audio/train_audio_splits/dia3_utt5.wav``.
    """
    folder = split_to_folder(split)
    fname = f"dia{int(dialog_id)}_utt{int(turn_id)}.wav"
    return root / "audio" / folder / fname


def main():
    parser = argparse.ArgumentParser(description="Extend meld_erc.csv with Parselmouth audio features")
    parser.add_argument("--csv", type=Path, default=paths.MELD_BENCHMARK_STAGE1_FILE_PATH)
    parser.add_argument("--root", type=Path, default=paths.MELD_RAW_DATA_DIR)
    parser.add_argument("--out", type=Path, default=paths.MELD_BENCHMARK_STAGE2_FILE_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of rows for quick tests")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit).copy()

    feature_cols = [
        "intensity_mean_db",
        "intensity_std_db",
        "pitch_mean_hz",
        "pitch_std_hz",
        "pitch_range_hz",
        "articulation_rate_syll_per_s",
        "hnr_mean_db",
    ]
    feats = {k: [] for k in feature_cols}

    root = args.root
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting audio features (MELD)"):
        wav_path = audio_path_for_row(root, row["split"], row["dialog_id"], row["turn_id"])
        if not wav_path.exists():
            # print(
            #     f"Warning: audio file not found: {wav_path} (split={row['split']}, dialog_id={row['dialog_id']}, turn_id={row['turn_id']})"
            # )
            warnings.warn(f"Warning: audio file not found: {wav_path}")
            for k in feature_cols:
                feats[k].append(np.nan)
            continue
        try:
            f = compute_features(wav_path)
        except Exception:
            f = {k: np.nan for k in feature_cols}
        for k in feature_cols:
            feats[k].append(f.get(k, np.nan))

    for k, v in feats.items():
        df[k] = v

    # Round acoustic feature columns to at most 4 decimals
    df[feature_cols] = df[feature_cols].round(4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
