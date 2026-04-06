"""Pre-compute and cache similarity lookups for all vector stores.

For each utterance in the combined MELD + IEMOCAP benchmark datasets,
performs a similarity search against a given ChromaDB vector store and
stores the ranked list of similar training-utterance indices in a JSON file.

This avoids repeated online vector-store queries at training and evaluation
time — the :class:`~src.helper.build_prompting_dataset.DemonstrationCreatorViaCache`
class reads these cache files directly.

One cache JSON file is produced per vector store, named after the collection
(e.g. ``artifacts/vectorstores/caches/meld_iemocap_hybrid_7.json``).

Usage::

    # Cache a single vector store
    python -m src.vectorstore.caching.cache_similar_utterances \\
        --vectorstore_name meld_iemocap_hybrid_7 --top_n 10

    # Cache all uncached vector stores
    python -m src.vectorstore.caching.cache_similar_utterances --top_n 10
"""

import os
import argparse

import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

from src.helper.utils import  (save_as_json, collection_exists_and_not_empty,
                               meld_mapped_valid_emotion_set, iemocap_mapped_valid_emotion_set)
from src.config import paths, constants


def get_vectordb_instance(path_to_db):
    embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDING_MODEL)
    collection_name = path_to_db.stem
    collection_exists_and_not_empty(path_to_db, collection_name, throw_exception=True)
    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=path_to_db,
    )
    print(f"The vectordb {collection_name} at {path_to_db} has {vector_db._collection.count()} documents")
    return vector_db


def get_query(group_df, turn_idx, window_size=None, type="single"):
    """Build the similarity-search query string for a given utterance.

    Args:
        group_df (pd.DataFrame): Dialogue DataFrame with ``"speaker"`` and
            ``"utterance"`` columns.
        turn_idx (int): Zero-based turn index within the dialogue.
        window_size (int, optional): Number of utterances in the context
            window; required for ``"flow"`` and ``"hybrid"`` types.
        type (str): Store type — ``"single"``, ``"flow"``, or ``"hybrid"``.

    Returns:
        str: Query string formatted to match the document style of the
            target vector store.

    Raises:
        Exception: If ``type`` is not recognised.
    """
    # print(f"len group_df is :{len(group_df)} and turn_idx is: {turn_idx}")
    if type == "single":
        row = group_df.iloc[turn_idx]
        return row["utterance"]
    elif type in ["flow", "hybrid"]:
        group_df = group_df.iloc[max(0, turn_idx - window_size + 1):turn_idx+1]
        page_content_lines = []
        for _, row in group_df.iterrows():
            line = f"{row['speaker']}: {row['utterance']}"
            page_content_lines.append(line)
        if type == "hybrid":
            # Repeat the last utterance to give it more weight
            last_utterance_text = f"{group_df.iloc[-1]['speaker']}: {group_df.iloc[-1]['utterance']}"
            page_content_lines.append(f"\nThe target utterance is: {last_utterance_text}")

        return "\n".join(page_content_lines)
    else:
        raise Exception(f"Unknown type: {type}")




def get_sim_utterance_idx(db, query, top_n, idx=None, valid_emotions=None):
    """Retrieve the top-N similar utterance indices for a query.

    Searches the top-50 results in the vector store and filters out
    documents from the same dialogue as the query utterance.

    Args:
        db (Chroma): The ChromaDB instance to search.
        query (str): The query string.
        top_n (int): Maximum number of results to return.
        idx (str, optional): Index of the query utterance, used to exclude
            same-dialogue results.
        valid_emotions (set[str], optional): If provided, only return
            results whose ``final_emotion`` metadata is in this set.

    Returns:
        list[str]: Up to ``top_n`` utterance index strings.
    """
    out = db.similarity_search(query, k=50)
    if valid_emotions:
        out_idx = [o.metadata["idx"] for o in out if o.metadata["final_emotion"] in valid_emotions]
    else:
        out_idx = [o.metadata["idx"] for o in out]
    if idx:
        idx_prefix = idx[:idx.rfind('_')+1]
        out_idx = [o for o in out_idx if idx_prefix not in o]
        out_idx = [o for o in out_idx if o != idx]
    out_idx = out_idx[:top_n]
    return out_idx



def get_sim_utterance_idx_for_dataset(db, top_n, db_type, dataset, dataset_name, window_size=None, filter_column=None, valid_emotions=None):
    similar_utterance_idx = {}
    for dialog_idx, dialog_df in tqdm(dataset.groupby("dialog_idx"), desc=f"Processing {dataset_name} dataset with {db._collection.name} vectorstore"):
        for unit in dialog_df.to_dict(orient="records"):
            if filter_column and not unit[filter_column]:
                continue
            unit_idx = unit["idx"]
            turn_idx = unit["turn_idx"]
            query = get_query(dialog_df, turn_idx, type=db_type, window_size=window_size)
            value = get_sim_utterance_idx(db, query, top_n, unit_idx, valid_emotions=valid_emotions)
            similar_utterance_idx[unit_idx] = value
    return similar_utterance_idx


def get_sim_utterance_idx_for_benchmark_datasets(db, top_n, db_type, window_size=None):
    similar_utterance_idx = {}

    df_meld = pd.read_csv(paths.MELD_BENCHMARK_FINAL_FILE_PATH)[["dialog_idx", "turn_idx", "speaker", "utterance", "idx", "mapped_emotion"]]
    df_iemocap = pd.read_csv(paths.IEMOCAP_BENCHMARK_FINAL_FILE_PATH)[["dialog_idx", "turn_idx", "speaker", "utterance", "idx", "erc_target", "mapped_emotion"]]
    # df_iemocap = df_iemocap[df_iemocap["erc_target"]].drop(columns="erc_target")

    similar_utterance_idx.update(get_sim_utterance_idx_for_dataset(db, top_n, db_type, df_meld, "MELD", window_size=window_size, valid_emotions=meld_mapped_valid_emotion_set))
    similar_utterance_idx.update(get_sim_utterance_idx_for_dataset(db, top_n, db_type, df_iemocap, "IEMOCAP", window_size=window_size, filter_column="erc_target", valid_emotions=iemocap_mapped_valid_emotion_set))

    return similar_utterance_idx

def cache_similar_utterances(vectorstore_name, top_n):
    """Cache similarity results for one or all vector stores.

    Args:
        vectorstore_name (str | None): Name of the vector store to cache
            (e.g. ``"meld_iemocap_hybrid_7"``).  If ``None``, all stores
            that do not yet have a cache file are processed.
        top_n (int): Number of similar utterances to store per query.
    """
    if vectorstore_name is None:
        vectorstore_names = next(os.walk(paths.VECTORSTORE_DB_DIR))[1]
        os.makedirs(paths.VECTORSTORE_CACHE_DIR, exist_ok=True)
        cached_vectorstores = [Path(json_file).stem for json_file in
                               os.listdir(paths.VECTORSTORE_CACHE_DIR) if json_file.endswith(".json")]
        vectorstores_to_cache = set(vectorstore_names).difference(set(cached_vectorstores))
    else:
        vectorstores_to_cache = {vectorstore_name}

    for vectorstore_name in tqdm(vectorstores_to_cache, desc=f"Caching vectorstores: {vectorstores_to_cache}"):
        db = get_vectordb_instance(paths.VECTORSTORE_DB_DIR / vectorstore_name)
        db_type = vectorstore_name.split("_")[2]
        if db_type in ['flow', 'hybrid']:
            window_size = int(vectorstore_name.split("_")[3])
        else:
            window_size = None
        similar_utterance_idx = get_sim_utterance_idx_for_benchmark_datasets(db, top_n=top_n, db_type=db_type, window_size=window_size)
        save_as_json(paths.VECTORSTORE_CACHE_DIR / f"{vectorstore_name}.json", similar_utterance_idx)

    print("Done")

def main():
    parser = argparse.ArgumentParser(description="Data processing script for LLM input")
    parser.add_argument("--vectorstore_name", type=str, default=None,
                        help="Name of the vectorstore to use for cache. If None, cache with all vectorstores "
                             "whose has no cache. (Default: None)")
    parser.add_argument("--top_n", type=int, default=10, help="Number of similar utterances to cache. (Default: 10)")
    args = parser.parse_args()

    cache_similar_utterances(args.vectorstore_name, top_n=args.top_n)


if __name__ == "__main__":
    main()


