from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import os
from tqdm import tqdm

def create_vectorstore(persist_directory):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma(
        collection_name="meld_iemocap_simple",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vector_store.reset_collection()


    df_meld = pd.read_csv("../../meld_erc_with_categories.csv")
    df_meld["idx"] = "m_" + df_meld["dialog_idx"].astype(str) + "_" + df_meld["turn_idx"].astype(str)
    df_meld = df_meld[df_meld["split"] == "train"][["utterance", "emotion", "idx"]]


    df_iemocap = pd.read_csv("../../iemocap_erc_acoustic_features.csv")
    df_iemocap["idx"] = df_iemocap["dialog_idx"].astype(str) + "_" + df_iemocap["turn_idx"].astype(str)
    df_iemocap = "i_" + df_iemocap[(df_iemocap["erc_target"]) & (df_iemocap["split"] == "train")][["utterance", "emotion", "idx"]]

    df = pd.concat([df_meld, df_iemocap], ignore_index=True)

    batch_size = 128

    for i in tqdm(range(0, len(df), batch_size), desc="Adding texts to vector store"):
        batch = df.iloc[i:i+batch_size]
        vector_store.add_texts(batch["utterance"],
                               metadatas=[{"emotion": row["emotion"], "idx": row["idx"]}
                                          for row in batch.to_dict(orient="records")])


def main():
    persist_directory = "vectorstore_data/simple/meld_iemocap_simple_db"
    if os.path.isdir(persist_directory):
        print("The simple vectorstore db has already been created!!!")
        return
    create_vectorstore(persist_directory)



if __name__ == "__main__":
    if "vectorstore/simple" not in os.getcwd():
        os.chdir("vectorstore/simple")
    main()