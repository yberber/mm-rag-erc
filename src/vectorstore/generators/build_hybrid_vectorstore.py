import pandas as pd
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.helper.utils import anonymize_speakers_in_dialog, mapped_emotion_to_id
from src.config import paths, constants

# --- Configuration Constants ---
VECTOR_STORE_BASE_PATH = paths.VECTORSTORE_DB_DIR
COLLECTION_NAME_PREFIX = "meld_iemocap_hybrid"  # Changed for clarity
EMBEDDING_MODEL = constants.EMBEDDING_MODEL

MELD_DATA_PATH = paths.MELD_BENCHMARK_FINAL_FILE_PATH
IEMOCAP_DATA_PATH = paths.IEMOCAP_BENCHMARK_FINAL_FILE_PATH


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a vector store with conversational flow, emphasizing the last utterance."
    )
    parser.add_argument(
        "--num_utterances",
        type=int,
        default=7,
        help="Number of historical utterances to include in each page content window.",
    )
    return parser.parse_args()


def load_and_prepare_meld(file_path: Path) -> pd.DataFrame:
    """Loads and prepares the MELD dataset."""
    print("Loading and preparing MELD dataset...")
    df = pd.read_csv(file_path)
    df = df.loc[df["split"] == "train", ["utterance", "speaker", "emotion", "mapped_emotion", "dialog_idx", "turn_idx", "idx"]]
    return df


def load_and_prepare_iemocap(file_path: Path) -> pd.DataFrame:
    """Loads and prepares the IEMOCAP dataset."""
    print("Loading and preparing IEMOCAP dataset...")
    df = pd.read_csv(file_path)
    df = df[df["mapped_emotion"] != 'unknown']
    df = df.loc[
        df["split"] == "train", ["utterance", "speaker", "emotion", "mapped_emotion", "dialog_idx", "turn_idx", "idx", "erc_target"]]
    # Anonymize IEMOCAP speakers directly
    df['speaker'] = df['speaker'].map({'M': 'Speaker_M', 'F': 'Speaker_F'}).fillna(df['speaker'])
    return df


def format_window_content_hybrid(dialog_df: pd.DataFrame, turn_id: int, window_size: int) -> Tuple[str, List[int]]:
    """
    Formats a window of conversation for hybrid retrieval, repeating the last utterance.

    Args:
        dialog_df (pd.DataFrame): The DataFrame for a single dialogue.
        turn_id (int): The current turn index to format up to.
        window_size (int): The number of utterances to include in the window.

    Returns:
        Tuple[str, List[int]]: A tuple of (page_content, emotion_flow).
    """
    start_index = max(0, turn_id - window_size + 1)
    dialog_history = dialog_df.iloc[start_index: turn_id + 1]

    page_content_lines = []
    emotion_flow_ids = []

    for _, row in dialog_history.iterrows():
        line = f"{row['speaker']}: {row['utterance']}"
        page_content_lines.append(line)
        emotion_id = mapped_emotion_to_id[row["mapped_emotion"]]
        emotion_flow_ids.append(emotion_id)

    # Repeat the last utterance to give it more weight
    last_utterance_text = f"{dialog_df.loc[turn_id, 'speaker']}: {dialog_df.loc[turn_id, 'utterance']}"
    page_content_lines.append(f"\nThe target utterance is: {last_utterance_text}")

    return "\n".join(page_content_lines), emotion_flow_ids


def process_dialogue_group(
        vector_store: Chroma,
        dialog_group: pd.DataFrame,
        window_size: int,
        anonymize: bool = False,
        filter_column: str = None
):
    """
    Processes a group of dialogues and adds their content windows to the vector store.
    """
    if anonymize:
        dialog_group = anonymize_speakers_in_dialog(dialog_group)

    # Ensure turns are sorted correctly within the dialogue
    # dialog_group = dialog_group.sort_values(by="turn_idx").reset_index(drop=True)
    dialog_group = dialog_group.reset_index(drop=True)

    assert (range(len(dialog_group)) == dialog_group.index.values).all(), "Turn indices are not contiguous."

    page_contents = []
    metadatas = []

    for turn_id, row in dialog_group.iterrows():
        if (filter_column and not row[filter_column]) or turn_id+1 <= window_size//2:
            continue

        # Use the modified hybrid formatting function
        page_content, emotion_flow = format_window_content_hybrid(dialog_group, turn_id, window_size)
        metadata = {
            "emotion_flow": str(emotion_flow),
            "final_emotion": row["mapped_emotion"],
            "idx": row["idx"]
        }
        page_contents.append(page_content)
        metadatas.append(metadata)

    if page_contents:
        vector_store.add_texts(texts=page_contents, metadatas=metadatas)


def main():
    """Main function to orchestrate the creation of the hybrid (repetition) vector store."""
    args = parse_arguments()
    window_size = args.num_utterances

    print(f"--- Starting Vector Store Creation for Hybrid Retrieval (Window Size: {window_size}) ---")

    vector_store_path = VECTOR_STORE_BASE_PATH / f"{COLLECTION_NAME_PREFIX}_{window_size}"

    # Ensure the parent directory for the vector store exists
    vector_store_path.parent.mkdir(parents=True, exist_ok=True)

    if vector_store_path.exists():
        print(f"SKIPPING: Vector store already exists at '{vector_store_path}'.")
        print("To re-create it, please delete the directory first.")
        return

    try:
        # 1. Initialize Vector Store
        print(f"Initializing vector store at: {vector_store_path}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = Chroma(
            collection_name=f"{COLLECTION_NAME_PREFIX}_{window_size}",
            embedding_function=embeddings,
            persist_directory=str(vector_store_path),
        )

        # 2. Load and process data
        df_meld = load_and_prepare_meld(MELD_DATA_PATH)
        df_iemocap = load_and_prepare_iemocap(IEMOCAP_DATA_PATH)

        # 3. Populate vector store
        print("Populating vector store with MELD dialogues...")
        for _, dialog_group in tqdm(df_meld.groupby("dialog_idx"), desc="MELD Dialogues"):
            process_dialogue_group(vector_store, dialog_group, window_size, anonymize=True)

        print("Populating vector store with IEMOCAP dialogues...")
        for _, dialog_group in tqdm(df_iemocap.groupby("dialog_idx"), desc="IEMOCAP Dialogues"):
            # process_dialogue_group(vector_store, dialog_group, window_size)
            process_dialogue_group(vector_store, dialog_group, window_size, filter_column="erc_target")


        print(f"\n✅ Successfully created the hybrid (repetition) vector store!")
        print(f"Total documents in store: {vector_store._collection.count()}")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Data file not found. Please check paths in the script.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    # This allows the script to be run from any directory
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    main()
