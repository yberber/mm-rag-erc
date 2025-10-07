import os
import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pickle


# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)
pd.set_option('display.precision', 2)


def get_vectordb_instance(path_to_db="../../vectorstore_data/simple/meld_iemocap_simple_db"):
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
    df_meld = pd.read_csv("../../meld_erc_with_categories.csv")
    for unit in tqdm(df_meld.to_dict(orient="records"), desc="Processing meld data"):
        key = unit["idx"]
        value = get_sim_utterance_idx(db, unit["utterance"], max_k, unit["idx"])
        similar_utterance_idx[key] = value
    return similar_utterance_idx

def get_sim_utterance_idx_for_iemocap(db, max_k):
    similar_utterance_idx = {}
    df_iemocap = pd.read_csv("../../iemocap_erc_with_categories.csv")
    for unit in tqdm(df_iemocap.to_dict(orient="records"), desc="Processing iemocap data"):
        if not unit["erc_target"]:
            continue
        key = unit["idx"]
        value = get_sim_utterance_idx(db, unit["utterance"], max_k, unit["idx"])
        similar_utterance_idx[key] = value
    return similar_utterance_idx


def main():
    if "vectorstore/simple" not in os.getcwd():
        os.chdir("vectorstore/simple")

    similar_utterance_idx = {}
    db = get_vectordb_instance()

    similar_utterance_idx.update(get_sim_utterance_idx_for_meld(db, 10))
    similar_utterance_idx.update(get_sim_utterance_idx_for_iemocap(db, 10))

    with open("similar_utterance_idx_top10.pkl", "wb") as f:
        pickle.dump(similar_utterance_idx, f)


if __name__ == "__main__":
    main()


