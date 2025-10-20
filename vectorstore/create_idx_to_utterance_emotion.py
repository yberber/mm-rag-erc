import pandas as pd
import os
from utils import set_pandas_display_options, save_as_json

set_pandas_display_options()


def main():
    if "vectorstore" not in os.getcwd():
        os.chdir("vectorstore")

    df_meld = pd.read_csv("../BENCMARK_DATASETS/meld_erc_with_categories.csv")
    df_iemocap = pd.read_csv("../BENCMARK_DATASETS/iemocap_erc_with_categories.csv")


    df_meld = df_meld[df_meld["split"] == "train"][["idx", "utterance", "speaker", "mapped_emotion"]]
    df_iemocap = df_iemocap[df_iemocap["split"] == "train"][["idx", "utterance", "speaker", "mapped_emotion"]]
    df_train = pd.concat([df_meld, df_iemocap], ignore_index=True)
    df_train.set_index("idx", inplace=True)
    df_train.loc["i_107_5":"i_107_55"]


    idx_to_utterance_emotion = {}


    meld_idx_to_utterance_emotion = {idx: (utterance, emotion) for idx, utterance, emotion in df_meld[["idx", "utterance", "emotion"]].values}
    iemocap_idx_to_utterance_emotion = {idx: (utterance, emotion) for idx, utterance, emotion in df_iemocap[df_iemocap["erc_target"]][["idx", "utterance", "emotion"]].values}

    idx_to_utterance_emotion.update(meld_idx_to_utterance_emotion)
    idx_to_utterance_emotion.update(iemocap_idx_to_utterance_emotion)

    save_as_json("idx_to_utterance_emotion.json", idx_to_utterance_emotion)



if __name__ == "__main__":
    main()
