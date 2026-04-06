import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.config import paths

# Define male and female characters in separate lists for clarity
male_characters = [
    "Joey", "Ross", "Chandler", "Tag", "Doug", "Frank", "Pete", "Mark", "Richard", 
    "Gunther", "Paul", "Danny", "Mr. Treeger", "Mr. Geller", "Mike", "Eric", "Man",
    "Guy", "Gary", "David", "Dr. Green", "Chip", "Barry", "Earl", "Steve",
    "Mr. Tribbiani", "Joshua", "Dr. Long", "Julio", "Russell", "Ben", "Cliff", "Bob",
    "Policeman", "Stanley", "Duncan", "Issac", "Robert", "Mischa", "Roger",
    "Mr. Heckles", "Dr. Baldhara", "Joey/Drake", "Tommy", "Mr. Franklin", "Larry",
    "Jim", "Kyle", "Fireman No. 3", "Young Ethan", "Dr. Zane", "Joey's Hand Twin",
    "Carl", "Max", "The Fireman", "Rick", "Doctor Connelly", "Mr. Burgin",
    "Dr. Rhodes", "Dr. Miller", "Wayne", "Allesandro", "Dr. Drake Remoray", "Jester",
    "Mr. Waltham", "Bobby", "Drew", "Dr. Franzblau", "Tony", "Liam", "Alan",
    "Mr. Zelner", "Fireman #1", "Tom", "Hoshi", "Sick Bastard", "Mr. Posner",
    "Dr. Ledbetter", "Jake", "Jason", "A Waiter", "Dr. Leedbetter", "Raymond",
    "Guru Saj", "Sergei", "Stu", "Gary Collins", "The Cigarette Guy", "Frank Sr.",
    "Robbie", "Vince", "Burt", "Hombre Man", "Janitor", "Emeril", "Boy in the Cape",
    "Dr. Stryker Ramoray", "Stevens", "Peter", "Paolo", "Waiter", "The Singing Man",
    "Dr. Harad", "Jay Leno", "Fireman No. 1", "Santos", "Gerston", "Dr. Oberman",
    "The Waiter", "Marc", "Dr. Wesley", "The Conductor", "The Dry Cleaner", 
    "The Security Guard", "Mr. Kaplan", "Student", "Director"
]

female_characters = [
    "Rachel", "Phoebe", "Monica", "Janice", "Carol", "Emily", "Mona", "Joanna",
    "Susan", "Lydia", "Kate", "Chloe", "Jill", "Dina", "Dana", "Janine", "Bonnie",
    "Leslie", "Charlie", "Kristen", "Cassie", "Kim", "Katie", "Kathy", "Ursula",
    "Estelle", "Isabella", "Krista", "Mrs. Bing", "Cecilia", "Shelley", "Megan",
    "Jade", "Sarah", "Mrs. Green", "The Smoking Woman", "Nancy", "Mindy", "Alice",
    "Lorraine", "Fake Monica", "Kristin", "Brenda", "Girl", "Melissa",
    "Phoebe Sr.", "Jen", "Kori", "Whitney", "Angela", "Trudie Styler",
    "Woman On Train", "Ronni", "Paula", "Lauren", "Molly", "Female Clerk",
    "Jeannine", "Jane", "Ginger", "Sophie", "Annabelle", "Ms. McKenna", "Joanne",
    "Marjorie", "Waitress", "The Hot Girl", "Jessica Lockhart", "Erin", "Caitlin",
    "Frannie", "Hope", "Bernice", "Kiki", "The Woman", "Rachel/actress",
    "Mrs. Burgin", "A Female Student", "Cookie", "Female Student", "Stephanie",
    "Mrs. Tedlock", "Aunt Lillian", "Phoebe/Waitress", "Mrs. Waltham", "Mrs. Chatracus",
    "Nurse #1", "Nurse #2", "Helena", "Mrs. Tribbiani", "Mrs. Lynch", "Elizabeth", "Julie", 
    "Woman", "Mrs. Geller", "Phoebe Sr", "Evil Bitch", "The Stripper", "Tour Guide", "Nurse",
    "The Casting Director", "Flight Attendant", "The Cooking Teacher"

]

# Create a gender mapping function
def get_gender(speaker: str) -> str:
    s_clean = speaker.strip()
    if s_clean in male_characters:
        return "M"
    if s_clean in female_characters:
        return "F"
    return "U"

def read_split(root: Path, split: str) -> pd.DataFrame:
    filename = {
        "train": "train_sent_emo.csv",
        "dev": "dev_sent_emo.csv",
        "test": "test_sent_emo.csv",
    }[split]
    df = pd.read_csv(root / filename)
    df["split"] = split
    return df


def clean_special_characters(df: pd.DataFrame):
    special_char_conversion_pairs = [('', "'"), ('', ""), ('', ""),
                                     ('', ""), ('', " "), ('', " ")]
    for pair in special_char_conversion_pairs:
        df['utterance'] = df['utterance'].str.replace(*pair)


def build_meld_dataframe(root: Path) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = [read_split(root, s) for s in ("train", "dev", "test")]
    raw = pd.concat(dfs, ignore_index=True)

    # Ensure expected columns exist
    required = [
        "split",
        "Season",
        "Episode",
        "Dialogue_ID",
        "Utterance_ID",
        "Speaker",
        "Emotion",
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in MELD CSVs: {missing}")

    season = pd.to_numeric(raw["Season"], errors="coerce").astype("Int64")
    episode = pd.to_numeric(raw["Episode"], errors="coerce").astype("Int64")
    dialog_id = pd.to_numeric(raw["Dialogue_ID"], errors="coerce").astype("Int64")

    utt_id = pd.to_numeric(raw["Utterance_ID"], errors="coerce").astype("Int64")
    # We do not include timing columns in the output per request



    gender_column = raw["Speaker"].apply(get_gender).astype(str)



    out = pd.DataFrame(
        {
            "split": raw["split"],
            "season": season,
            "episode": episode,
            # dialog_id is the original Dialogue_ID from MELD
            "dialog_id": dialog_id,
            # Keep MELD Utterance_ID as turn_id (per dialog index in MELD files)
            "turn_id": utt_id.astype("Int64"),
            "speaker": raw["Speaker"].astype(str),
            "emotion": raw["Emotion"].astype(str),
            "gender": gender_column,
            "utterance": raw["Utterance"].astype(str)
        }
    )

    clean_special_characters(out)

    # Keep original row order within each split and ensure split order: train -> dev -> test
    split_order = {"train": 0, "dev": 1, "test": 2}
    out["_split_order"] = out["split"].map(split_order).astype(int)
    # Stable sort by split only; preserves original file order within each split
    out.sort_values(["_split_order"], kind="mergesort", inplace=True)
    out.reset_index(drop=True, inplace=True)

    # Compute dialog_idx continuous across splits (train -> dev -> test)
    # Use a composite key (split, dialog_id) to avoid collisions across splits
    out["_dialog_key"] = out["split"].astype(str) + "|" + out["dialog_id"].astype(str)
    codes, _ = pd.factorize(out["_dialog_key"], sort=False)
    out["dialog_idx"] = codes.astype(int)

    # New turn_idx: 0-based index per (split, dialog_id) in preserved order
    out["turn_idx"] = out.groupby("_dialog_key").cumcount()

    # Final column order
    out.drop(columns=["_split_order", "_dialog_key"], inplace=True)
    out = out[["split", "season", "episode", "dialog_id", "dialog_idx", "turn_id", "turn_idx", "speaker", "gender", "emotion", "utterance"]]
    return out


def main():
    parser = argparse.ArgumentParser(description="Initialize MELD ERC CSV with standardized columns")
    parser.add_argument(
        "--root",
        type=Path,
        default=paths.MELD_RAW_DATA_DIR,
        help="Path to MELD folder (contains *_sent_emo.csv files)",
    )

    parser.add_argument("--out", type=Path, default=paths.MELD_BENCHMARK_STAGE1_FILE_PATH, help="Where to save the consolidated CSV")
    args = parser.parse_args()

    df = build_meld_dataframe(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
