"""Extend the IEMOCAP benchmark CSV with Parselmouth acoustic features.

Reads the stage-1 IEMOCAP CSV (output of ``init_iemocap_dataset``), locates
the WAV file for every utterance in the original IEMOCAP directory tree, and
computes the same seven acoustic features as the MELD counterpart:
intensity (mean/std), pitch (mean/std/range), articulation rate, and mean HNR.

Unlike the MELD version this script raises ``FileNotFoundError`` immediately
when a WAV is missing, because IEMOCAP audio paths are deterministic.

Usage::

    python -m src.data_processing.iemocap.add_audio_features_iemocap \\
        --csv  data/benchmark/iemocap/iemocap_erc_init.csv \\
        --root /path/to/IEMOCAP_full_release \\
        --out  data/benchmark/iemocap/iemocap_erc_with_audio.csv
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import parselmouth as pm
from tqdm import tqdm

from src.config.paths import IEMOCAP_RAW_DATA_DIR
from src.data_processing.syllable_nuclei import speech_rate
from src.config import paths


def compute_features(sound_path: Path) -> Dict[str, Optional[float]]:
    """Compute acoustic features for a single WAV file using Parselmouth.

    Args:
        sound_path (Path): Path to the WAV audio file.

    Returns:
        Dict[str, Optional[float]]: Dictionary with keys
            ``intensity_mean_db``, ``intensity_std_db``, ``pitch_mean_hz``,
            ``pitch_std_hz``, ``pitch_range_hz``,
            ``articulation_rate_syll_per_s``, ``hnr_mean_db``.
            Values are ``float`` or ``nan`` when the feature cannot be
            computed (e.g. unvoiced audio).
    """

    snd = pm.Sound(str(sound_path))

    intensity = snd.to_intensity(time_step=0.01, minimum_pitch=75)
    inten_vals = intensity.values.T.reshape(-1)
    inten_times = intensity.xs()
    inten_mask = np.isfinite(inten_vals)
    inten_vals_f = inten_vals[inten_mask]
    # Energy-weighted mean intensity per Praat
    mean_intensity = float(pm.praat.call(intensity, "Get mean", 0, 0, "energy")) if inten_vals_f.size else np.nan
    std_intensity = float(np.std(inten_vals_f)) if inten_vals_f.size else np.nan

    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
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


def audio_path_for_row(root: Path, session: str, dialog_id: str, turn_id: str) -> Path:
    """Construct the expected WAV path for an IEMOCAP utterance.

    Args:
        root (Path): IEMOCAP root directory.
        session (str): Session name (e.g. ``"Session1"``).
        dialog_id (str): Dialogue ID (e.g. ``"Ses01F_impro01"``).
        turn_id (str): Turn ID (e.g. ``"Ses01F_impro01_F000"``).

    Returns:
        Path: Expected path following the IEMOCAP directory convention:
            ``<root>/<session>/sentences/wav/<dialog_id>/<turn_id>.wav``.
    """
    return root / session / "sentences" / "wav" / dialog_id / f"{turn_id}.wav"


def main():
    parser = argparse.ArgumentParser(description="Extend iemocap_erc.csv with Parselmouth audio features")
    parser.add_argument("--csv",
                        type=Path,
                        default=paths.IEMOCAP_BENCHMARK_STAGE1_FILE_PATH,)
    parser.add_argument("--root",
                        type=Path,
                        default=paths.IEMOCAP_RAW_DATA_DIR,)
    parser.add_argument("--out",
                        type=Path,
                        default=paths.IEMOCAP_BENCHMARK_STAGE2_FILE_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of rows for quick tests")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit).copy()

    # Only collect acoustic feature columns; do not store paths/existence
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
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting audio features"):
        wav_path = audio_path_for_row(root, row["session"], row["dialog_id"], row["turn_id"])
        if not wav_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {wav_path} (session={row['session']}, dialog_id={row['dialog_id']}, turn_id={row['turn_id']})"
            )
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
