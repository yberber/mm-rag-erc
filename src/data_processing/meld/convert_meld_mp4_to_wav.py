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
