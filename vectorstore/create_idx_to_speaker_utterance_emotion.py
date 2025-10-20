import pandas as pd
import os
from utils import (set_pandas_display_options, save_as_json,
                   get_meld_iemocap_datasets_as_dataframe, anonymize_speakers_in_dialog, save_dataframe_as_json)

set_pandas_display_options()


def main():
    if "vectorstore" not in os.getcwd():
        os.chdir("vectorstore")

    df_meld = pd.read_csv("../BENCMARK_DATASETS/meld_erc_with_categories.csv")
    df_iemocap = pd.read_csv("../BENCMARK_DATASETS/iemocap_erc_with_categories.csv")


    df_meld = df_meld.groupby("dialog_idx", group_keys=False).apply(
        lambda df: anonymize_speakers_in_dialog(df, use_letters=True), include_groups=False
    )

    df_iemocap["speaker"] = "Speaker_" + df_iemocap["speaker"]


    df_meld = df_meld[["idx", "speaker", "utterance", "mapped_emotion"]]
    df_iemocap = df_iemocap[["idx", "speaker", "utterance", "mapped_emotion"]]
    df_train = pd.concat([df_meld, df_iemocap], ignore_index=True)
    df_train.set_index("idx", inplace=True)

    save_dataframe_as_json("idx_to_speaker_utterance_emotion_df.json", df_train)



if __name__ == "__main__":
    main()
