"""Convert MELD MP4 video files to mono 16 kHz WAV audio files.

MELD distributes its media as MP4 files split across ``train_splits/``,
``dev_splits/``, and ``test_splits/``.  This script calls ``ffmpeg`` to
strip the audio track from each MP4 and write a WAV file to a parallel
``audio/`` directory (``train_audio_splits/``, etc.) that is required by
``add_audio_features_meld``.

A special case is handled for the test split: when a ``final_videos_test``
version of a clip exists it is preferred over the original, because those
files contain corrected audio.

Usage::

    python -m src.data_processing.meld.convert_meld_mp4_to_wav \\
        --root /path/to/MELD.Raw [--overwrite]

Requires ``ffmpeg`` to be installed and available on ``PATH``.
"""

import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple
import os

from tqdm import tqdm
from src.config import paths


SPLITS = [
    ("train_splits", "audio/train_audio_splits"),
    ("dev_splits", "audio/dev_audio_splits"),
    ("test_splits", "audio/test_audio_splits"),
]


def collect_mp4_jobs(root: Path) -> List[Tuple[Path, Path]]:
    """Collect all (source MP4, target WAV) conversion jobs for MELD.

    For the test split, prefers ``final_videos_test<stem>.mp4`` over the
    plain ``dia*.mp4`` when the final version exists.

    Args:
        root (Path): MELD root directory containing ``train_splits/``,
            ``dev_splits/``, and ``test_splits/`` subdirectories.

    Returns:
        List[Tuple[Path, Path]]: List of ``(mp4_path, wav_path)`` pairs
            ready to be passed to :func:`ffmpeg_to_wav`.
    """
    jobs: List[Tuple[Path, Path]] = []
    for in_name, out_name in SPLITS:
        in_split = root / in_name
        out_split = root / out_name
        # if not in_split.exists():
        #     continue
        out_split.mkdir(parents=True, exist_ok=True)
        # every record follows the pattern "dia*.mp4"
        # except 132 records in test split which has the pattern "final_videos_testdia*.mp4"
        # Note that the final video records fixes some issues in the initial records
        pattern = "dia*.mp4"
        if in_name == "test_splits":
            for mp4 in in_split.glob(pattern):
                wav_name = mp4.stem + ".wav"
                final_record_path = in_split/ str("final_videos_test" + mp4.stem + ".mp4")
                if os.path.isfile(final_record_path):
                    mp4 = final_record_path
                wav_path = out_split / wav_name
                jobs.append((mp4, wav_path))
        else:
            for mp4 in in_split.glob(pattern):
                wav_name = mp4.stem + ".wav"
                wav_path = out_split / wav_name
                jobs.append((mp4, wav_path))
        if in_name == "test_splits":
            final_pattern = "final_videos_testdia*.mp4"
            for mp4 in in_split.glob(final_pattern):
                wav_name = mp4.stem + ".wav"
                wav_path = out_split / wav_name

    return jobs


def ffmpeg_to_wav(src: Path, dst: Path, overwrite: bool = False) -> int:
    """Convert a single MP4 to a mono 16 kHz WAV via ffmpeg.

    Args:
        src (Path): Source MP4 file.
        dst (Path): Destination WAV file path.
        overwrite (bool): If ``True``, pass ``-y`` to ffmpeg to overwrite
            existing files.  Defaults to ``False``.

    Returns:
        int: ffmpeg process return code (``0`` on success).
    """
    # Build ffmpeg command: mono, 16 kHz, wav container; no video
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    args += ["-y" if overwrite else "-n"]
    args += [
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst),
    ]
    proc = subprocess.run(args)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert MELD .mp4 files to mono 16kHz .wav with tqdm progress"
    )
    p.add_argument(
        "--root",
        type=Path,
        default=paths.MELD_RAW_DATA_DIR,
        help="Root containing train_splits/dev_splits/test_splits",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing WAV files if present",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.overwrite = True
    jobs = collect_mp4_jobs(args.root)
    if not jobs:
        print("No MP4 files found under:", args.root)
        return

    errors: List[Tuple[Path, Path, int]] = []
    for src, dst in tqdm(jobs, desc="Converting MP4 to WAV"):
        rc = ffmpeg_to_wav(src, dst, overwrite=args.overwrite)
        if rc != 0:
            errors.append((src, dst, rc))

    print(f"Converted {len(jobs) - len(errors)} / {len(jobs)} files.")
    if errors:
        print("Errors (showing up to 10):")
        for src, dst, rc in errors[:10]:
            print(f"  rc={rc} src={src} -> dst={dst}")


if __name__ == "__main__":
    main()
