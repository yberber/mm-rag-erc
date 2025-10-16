
import argparse  # renamed from create_IEMOCAP_dataset.py
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


EMO_MAP = {
    "ang": "anger",
    "hap": "happiness",
    "sad": "sadness",
    "neu": "neutral",
    "exc": "excitement",
    "fru": "frustration",
    "fea": "fear",
    "sur": "surprise",
    "dis": "disgust",
    "oth": "other",
    "xxx": "unknown",
}



EVAL_LINE_RE = re.compile(
    r"^\[(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\]"  # [start - end]
    r"\s+(?P<turn>[A-Za-z0-9_]+)\s+"  # turn id
    r"(?P<emo>[a-z]{3})\s+"  # emotion code
    r"\[(?P<val>[-\d\.]+)\s*,\s*(?P<act>[-\d\.]+)\s*,\s*(?P<dom>[-\d\.]+)\]\s*$"
)


TRANS_LINE_RE = re.compile(
    r"^(?P<turn>[A-Za-z0-9_]+)\s+\[(?P<start>\d+(?:\.\d+)?)\-(?P<end>\d+(?:\.\d+)?)\]:\s*(?P<text>.*)\s*$"
)


@dataclass
class Utterance:
    session: str
    dialog_id: str
    turn_id: str
    speaker: str
    start: float
    end: float
    emotion_code: str
    emotion: str
    utterance: str
    dialog_type: str  # impro or script
    marker_gender: str  # F or M from the dialog id
    turn_idx: int


def parse_transcriptions(path: Path) -> Dict[str, str]:
    """Return mapping turn_id -> utterance text."""
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = TRANS_LINE_RE.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        out[d["turn"]] = d.get("text", "").strip()
    return out


def parse_emoeval(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        m = EVAL_LINE_RE.match(line)
        if not m:
            continue
        d = m.groupdict()
        rows.append(
            {
                "turn_id": d["turn"],
                "start": float(d["start"]),
                "end": float(d["end"]),
                "emotion_code": d["emo"],
            }
        )
    return rows


def infer_dialog_meta(dialog_id: str) -> Dict[str, str]:
    parts = dialog_id.split("_")
    dialog_type = "impro" if any("impro" in p for p in parts) else ("script" if any("script" in p for p in parts) else "")
    marker_gender = dialog_id[5] if len(dialog_id) > 5 else ""
    return {"dialog_type": dialog_type, "marker_gender": marker_gender}


def infer_turn_meta(turn_id: str) -> Dict[str, str]:
    speaker = turn_id.split("_")[-1][0] if "_" in turn_id else ""
    dialog_id = "_".join(turn_id.split("_")[:-1]) if "_" in turn_id else turn_id
    return {"speaker": speaker, "dialog_id": dialog_id}


def session_name_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("Session"):
            return part
    return ""


def collect_dialog(dialog_eval: Path, dialog_trans: Optional[Path]) -> List[Utterance]:
    eval_rows = parse_emoeval(dialog_eval)
    trans_map = parse_transcriptions(dialog_trans) if dialog_trans else {}
    session = session_name_from_path(dialog_eval)
    dialog_id = dialog_eval.stem
    meta = infer_dialog_meta(dialog_id)

    # Sort by start time to get cross-speaker conversation order
    eval_rows.sort(key=lambda r: (r["start"], r["end"]))

    utterances: List[Utterance] = []
    for order, r in enumerate(eval_rows):
        tmeta = infer_turn_meta(r["turn_id"])  # adds speaker, dialog_id
        utterances.append(
            Utterance(
                session=session,
                dialog_id=tmeta["dialog_id"],
                turn_id=r["turn_id"],
                speaker=tmeta["speaker"],
                start=r["start"],
                end=r["end"],
                emotion_code=r["emotion_code"],
                emotion=EMO_MAP.get(r["emotion_code"], r["emotion_code"]),
                utterance=trans_map.get(r["turn_id"], ""),
                dialog_type=meta["dialog_type"],
                marker_gender=meta["marker_gender"],
                turn_idx=order,
            )
        )
    return utterances


def iterate_files(root: Path) -> Iterable[tuple[Path, Path]]:
    for session_dir in sorted(root.glob("Session*")):
        eval_dir = session_dir / "dialog" / "EmoEvaluation"
        tr_dir = session_dir / "dialog" / "transcriptions"
        if not eval_dir.exists():
            continue
        for eval_file in sorted(eval_dir.glob("*.txt")):
            tr_file = tr_dir / eval_file.name
            yield eval_file, tr_file


def build_dataframe(root: Path) -> pd.DataFrame:
    rows: List[Utterance] = []
    for eval_file, tr_file in iterate_files(root):
        rows.extend(collect_dialog(eval_file, tr_file if tr_file.exists() else None))

    df = pd.DataFrame(
        [
            {
                "dataset": "IEMOCAP",
                "session": u.session,
                "dialog_id": u.dialog_id,
                "turn_id": u.turn_id,
                "turn_idx": u.turn_idx,
                "speaker": u.speaker,
                "start": u.start,
                "end": u.end,
                "emotion_code": u.emotion_code,
                "emotion": u.emotion,
                "utterance": u.utterance,
                "dialog_type": u.dialog_type,
                "gender": u.speaker,
                "marker_gender": u.marker_gender
            }
            for u in rows
        ]
    )

    # Ensure stable ordering by dialog then utterance order
    df.sort_values(["session", "dialog_id", "turn_idx"], inplace=True, ignore_index=True)

    # Add dialog_idx: global zero-based index across all sessions
    unique_dialogs = (
        df[["session", "dialog_id"]]
        .drop_duplicates()
        .sort_values(["session", "dialog_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    dialog_idx_map: Dict[tuple, int] = {
        (row.session, row.dialog_id): i for i, row in unique_dialogs.iterrows()
    }
    df["dialog_idx"] = [dialog_idx_map[(s, d)] for s, d in zip(df["session"], df["dialog_id"]) ]

    # Assign split: train/dev/test
    allowed_sessions = [s for s in df["session"].unique() if s != "Session5"]
    s14_dialogs = (
        df[df["session"].isin(allowed_sessions)][["session", "dialog_id"]]
        .drop_duplicates()
        .sort_values(["session", "dialog_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    # First 108 dialogs -> train, remaining -> dev
    split_dialogs = {}
    for i, row in s14_dialogs.iterrows():
        split_dialogs[(row.session, row.dialog_id)] = "train" if i < 108 else "dev"

    def decide_split(session: str, dialog_id: str) -> str:
        if session == "Session5":
            return "test"
        return split_dialogs.get((session, dialog_id), "dev")

    df["split"] = [decide_split(s, d) for s, d in zip(df["session"], df["dialog_id"]) ]

    # ERC target flag (keep all rows for context)
    allowed_emos = {"ang", "hap", "sad", "neu", "exc", "fru"}
    df["erc_target"] = df["emotion_code"].isin(allowed_emos)

    # Reorder columns so dialog_idx follows dialog_id; split as first column
    desired = [
        # "dataset",
        "split",
        "session",
        "dialog_id",
        "dialog_idx",
        "turn_id",
        "turn_idx",
        "speaker",
        # "start",
        # "end",
        "emotion_code",
        "emotion",
        "erc_target",
        "utterance",
        "dialog_type",
        "gender",
    ]
    df = df[desired]
    return df


def main():
    parser = argparse.ArgumentParser(description="Create a pandas DataFrame for IEMOCAP ERC from dialog annotations")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/yusuf/Data/IEMOCAP_full_release"),
        help="Path to IEMOCAP root (contains Session1..Session5)",
    )
    parser.add_argument("--out", type=Path, default=Path("../BENCMARK_DATASETS/iemocap_erc.csv"), help="Where to save the CSV")
    parser.add_argument("--to-parquet", type=Path, default=None, help="Optional path to also save as Parquet")
    args = parser.parse_args()

    df = build_dataframe(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    if args.to_parquet:
        args.to_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.to_parquet, index=False)

    # Small summary for sanity check
    by_emo = df["emotion_code"].value_counts().to_dict()
    print(f"Rows: {len(df)} | Dialogs: {df['dialog_id'].nunique()} | Sessions: {df['session'].nunique()}")
    print(f"Emotions counts: {by_emo}\n")

    print(f"Saved {len(df)} rows to {args.out}")



if __name__ == "__main__":
    main()
