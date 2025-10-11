from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import os
from tqdm import tqdm
from utils import set_pandas_display_options, anonymize_speakers_in_dialog, original_emotion_to_id
import argparse


set_pandas_display_options()



def format_page_content(df_dialog, turn_idx, num_utterance_per_content, return_emotion_flow=True):
    dialog_hist = df_dialog[max(0, turn_idx-num_utterance_per_content+1): turn_idx+1]

    page_content_text = ""
    emotion_flow = []
    for unit in dialog_hist.to_dict(orient="records"):
        speaker_id = unit["speaker"]
        utterance = unit["utterance"]
        emotion = unit["emotion"]

        page_content_text += (speaker_id + ": " + utterance + "\n")

        emotion_id = original_emotion_to_id[emotion]
        emotion_flow.append(emotion_id)

    page_content_text = page_content_text.strip("\n")
    if return_emotion_flow:
        return page_content_text, emotion_flow
    else:
        return page_content_text





def create_vectorstore(persist_directory, num_utterance_per_content):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma(
        collection_name="meld_iemocap_advanced",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    vector_store.reset_collection()

    df_meld = pd.read_csv("../../meld_erc_with_categories.csv")
    df_meld = df_meld[df_meld["split"] == "train"][["utterance", "speaker", "emotion", "dialog_idx", "turn_idx", "idx"]]

    df_iemocap = pd.read_csv("../../iemocap_erc_with_categories.csv")
    # df_iemocap = df_iemocap[(df_iemocap["erc_target"]) & (df_iemocap["split"] == "train")][["utterance", "speaker","emotion", "idx"]]
    df_iemocap = df_iemocap[df_iemocap["split"] == "train"][["utterance", "speaker", "emotion", "dialog_idx", "turn_idx", "idx", "erc_target"]]
    df_iemocap["speaker"].map({'M': 'Speaker_A', 'F': 'Speaker_B'})  # anonymize speaker names

    len(df_meld.groupby("dialog_idx"))
    len(df_iemocap.groupby("dialog_idx"))

    for dialog_idx, df_dialog in tqdm(df_meld.groupby("dialog_idx"), desc="Adding texts from MELD to advance vector store"):
        df_dialog = anonymize_speakers_in_dialog(df_dialog)
        assert (range(len(df_dialog)) == df_dialog.turn_idx.values).all()
        page_contents = []
        metadatas = []
        for row in df_dialog.to_dict(orient="records"):
            turn_idx = row["turn_idx"]
            page_content, emotion_flow = format_page_content(df_dialog, turn_idx, num_utterance_per_content, return_emotion_flow=True)
            metadata = {"emotion_flow": str(emotion_flow), "emotion": row["emotion"], "idx": row["idx"]}

            page_contents.append(page_content)
            metadatas.append(metadata)

        vector_store.add_texts(page_contents, metadatas=metadatas)

    for dialog_idx, df_dialog in tqdm(df_iemocap.groupby("dialog_idx"), desc="Adding texts from IEMOCAP to advance vector store"):
        # not necessary for IEMOCAP because it has only 2 different speakers which are 'M' and 'F'
        # df_dialog = anonymize_speakers_in_dialog(df_dialog)
        assert (range(len(df_dialog)) == df_dialog.turn_idx.values).all()
        page_contents = []
        metadatas = []
        for row in df_dialog.to_dict(orient="records"):
            if not row["erc_target"]:
                continue
            turn_idx = row["turn_idx"]
            page_content, emotion_flow = format_page_content(df_dialog, turn_idx, num_utterance_per_content)
            metadata = {"emotion_flow": str(emotion_flow), "emotion": row["emotion"], "idx": row["idx"]}

            page_contents.append(page_content)
            metadatas.append(metadata)

        vector_store.add_texts(page_contents, metadatas=metadatas)


def main():
    parser = argparse.ArgumentParser(
        description="Create a vectordb using with page contents having the conversation flow")
    parser.add_argument("--num_utterance_per_content", type=int, default=3,
                        help="Number of utterance per page content")
    args = parser.parse_args()

    if "vectorstore/flow" not in os.getcwd():
        os.chdir("vectorstore/flow")
    persist_directory = f"../vectorstore_data/meld_iemocap_advanced_{args.num_utterance_per_content}_db"
    if os.path.isdir(persist_directory):
        print("The flow vectorstore db has already been created!!!")
        return

    create_vectorstore(persist_directory, args.num_utterance_per_content)
    print(f"Advanced vectorstore with page contents having the conversation flow with {args.num_utterance_per_content} utteranced is created successfully!")


if __name__ == "__main__":

    main()

