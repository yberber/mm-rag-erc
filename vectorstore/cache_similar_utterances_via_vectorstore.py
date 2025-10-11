import os
import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils import set_pandas_display_options, save_as_json
import argparse


set_pandas_display_options()


def get_vectordb_instance(path_to_db):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        collection_name="meld_iemocap_simple",
        embedding_function=embeddings,
        persist_directory=path_to_db,
    )
    return vector_db


def get_sim_utterance_idx(db, utterance, max_k, idx=None):
    out = db.similarity_search(utterance, k=max_k+1)
    out_idx = [o.metadata["idx"] for o in out]
    if idx:
        out_idx = [o for o in out_idx if o != idx]
    out_idx = out_idx[:max_k]
    return out_idx
    # example_text = ""
    # for sim_doc in out:
    #     example_text += f"< {sim_doc.page_content} : {sim_doc.metadata['emotion']} >\n"
    # return example_text.strip('\n')


def get_sim_utterance_idx_for_meld(db, max_k):
    similar_utterance_idx = {}
    df_meld = pd.read_csv("../meld_erc_with_categories.csv")[:10]
    for unit in tqdm(df_meld.to_dict(orient="records"), desc="Processing meld data"):
        key = unit["idx"]
        value = get_sim_utterance_idx(db, unit["utterance"], max_k, unit["idx"])
        similar_utterance_idx[key] = value
    return similar_utterance_idx

def get_sim_utterance_idx_for_iemocap(db, max_k):
    similar_utterance_idx = {}
    df_iemocap = pd.read_csv("../iemocap_erc_with_categories.csv")[:10]
    for unit in tqdm(df_iemocap.to_dict(orient="records"), desc="Processing iemocap data"):
        if not unit["erc_target"]:
            continue
        key = unit["idx"]
        value = get_sim_utterance_idx(db, unit["utterance"], max_k, unit["idx"])
        similar_utterance_idx[key] = value
    return similar_utterance_idx



def cache_similar_utterances(vectorstore_name):
    if vectorstore_name is None:
        vectorstore_names = next(os.walk('vectorstore_data'))[1]
        cached_vectorstores = [json_file[:-(len(".json"))] for json_file in os.listdir("caches/") if json_file.endswith(".json")]
        vectorstores_to_cache = set(vectorstore_names).difference(set(cached_vectorstores))
    else:
        vectorstores_to_cache = {vectorstore_name}

    for vectorstore_name in tqdm(vectorstores_to_cache, desc=f"Caching vectorstores: {vectorstores_to_cache}"):
        db = get_vectordb_instance(f"vectorstore_data/{vectorstore_name}")

        similar_utterance_idx = {}
        similar_utterance_idx.update(get_sim_utterance_idx_for_meld(db, 10))
        similar_utterance_idx.update(get_sim_utterance_idx_for_iemocap(db, 10))

        save_as_json(f"caches/{vectorstore_name}.json", similar_utterance_idx)

    print("Done")

def main():
    parser = argparse.ArgumentParser(description="Data processing script for LLM input")
    parser.add_argument("--vectorstore_name", type=str, default=None,
                        help="Name of the vectorstore to use for cache. If None, cache with all vectorstores "
                             "whose has no cache. (Default: None)")
    args = parser.parse_args()

    if "vectorstore" not in os.getcwd():
        os.chdir("vectorstore")

    cache_similar_utterances(args.vectorstore_name)


if __name__ == "__main__":
    main()


