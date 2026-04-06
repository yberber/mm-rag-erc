import pandas as pd
from pathlib import Path
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

from src.config import paths, constants

# --- Configuration Constants ---
# Path to the directory where the vector store will be saved.
VECTOR_STORE_PATH = paths.VECTORSTORE_DB_DIR / "meld_iemocap_single"
# Name of the collection within the ChromaDB.
COLLECTION_NAME = "meld_iemocap_single"
# Hugging Face model to use for embeddings.
EMBEDDING_MODEL = constants.EMBEDDING_MODEL
# Number of documents to process in each batch.
BATCH_SIZE = 128
# Paths to the source data files.
MELD_DATA_PATH = paths.MELD_BENCHMARK_FINAL_FILE_PATH
IEMOCAP_DATA_PATH = paths.IEMOCAP_BENCHMARK_FINAL_FILE_PATH


def load_and_prepare_data() -> pd.DataFrame:
    """
    Loads, filters, and combines the MELD and IEMOCAP datasets.

    Returns:
        pd.DataFrame: A unified DataFrame containing the training utterances.
    """
    print("Loading and preparing MELD and IEMOCAP datasets...")

    # Load and filter MELD dataset
    df_meld = pd.read_csv(MELD_DATA_PATH)
    df_meld = df_meld.loc[df_meld["split"] == "train", ["utterance", "emotion", "mapped_emotion", "idx"]]

    # Load and filter IEMOCAP dataset
    df_iemocap = pd.read_csv(IEMOCAP_DATA_PATH)
    # df_iemocap = df_iemocap.loc[
    #     (df_iemocap["erc_target"]) & (df_iemocap["split"] == "train"),
    #     ["utterance", "emotion", "mapped_emotion", "idx"],
    # ]
    df_iemocap = df_iemocap.loc[
        (df_iemocap["mapped_emotion"] != "unknown") & (df_iemocap["split"] == "train"),
        ["utterance", "emotion", "mapped_emotion", "idx"],
    ]

    # Combine the datasets
    combined_df = pd.concat([df_meld, df_iemocap], ignore_index=True)
    print(f"Successfully loaded and combined {len(combined_df)} training utterances.")
    return combined_df


def initialize_vector_store(
        vector_store_path: Path, collection_name: str, embedding_model: str
) -> Chroma:
    """
    Initializes a new, empty Chroma vector store.

    Args:
        vector_store_path (Path): The directory to persist the store.
        collection_name (str): The name of the collection.
        embedding_model (str): The Hugging Face model for embeddings.

    Returns:
        Chroma: An initialized Chroma vector store instance.
    """
    print(f"Initializing a new vector store at: {vector_store_path}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(vector_store_path),
    )
    # Ensure the collection is empty before population
    if vector_store._collection.count() > 0:
        print(f"Collection '{collection_name}' already exists. Resetting...")
        vector_store._collection.delete(ids=vector_store._collection.get()['ids'])

    return vector_store


def populate_vector_store(vector_store: Chroma, data: pd.DataFrame, batch_size: int):
    """
    Adds documents to the vector store in batches.

    Args:
        vector_store (Chroma): The Chroma instance to populate.
        data (pd.DataFrame): The DataFrame containing the data.
        batch_size (int): The number of documents per batch.
    """
    print(f"Populating vector store with {len(data)} documents...")
    num_batches = (len(data) // batch_size) + 1

    for i in tqdm(range(0, len(data), batch_size), desc="Adding documents", total=num_batches):
        batch = data.iloc[i: i + batch_size]

        texts_to_add = batch["utterance"].tolist()
        metadatas_to_add = [
            {"final_emotion": row["mapped_emotion"], "idx": row["idx"]}
            for _, row in batch.iterrows()
        ]

        vector_store.add_texts(texts=texts_to_add, metadatas=metadatas_to_add)

    print("Vector store population complete.")


def main():
    """
    Main function to orchestrate the creation of the single-utterance vector store.
    """
    print("--- Starting Vector Store Creation for Single Utterances ---")

    # Ensure the parent directory for the vector store exists
    VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if VECTOR_STORE_PATH.exists():
        print(f"SKIPPING: Vector store already exists at '{VECTOR_STORE_PATH}'.")
        print("To re-create it, please delete the directory first.")
        return

    try:
        # 1. Load and prepare data
        utterance_data = load_and_prepare_data()

        # 2. Initialize the vector store
        vector_store = initialize_vector_store(
            VECTOR_STORE_PATH, COLLECTION_NAME, EMBEDDING_MODEL
        )

        # 3. Populate the vector store with data
        populate_vector_store(vector_store, utterance_data, BATCH_SIZE)

        print("\n✅ Successfully created the single-utterance vector store!")
        print(f"Total documents in store: {vector_store._collection.count()}")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Data file not found. Please check the paths.")
        print(e)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Correctly set the base path relative to the script's location
    # This makes the script runnable from any directory.
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    main()