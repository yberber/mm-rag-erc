import pandas as pd
import os
from utils import set_pandas_display_options
import pickle

set_pandas_display_options()

if "vectorstore" not in os.getcwd():
    os.chdir("vectorstore")

df_meld = pd.read_csv("../meld_erc_with_categories.csv")
df_iemocap = pd.read_csv("../iemocap_erc_with_categories.csv")

df_meld["idx"]
df_meld["utterance"]
df_meld["emotion"]

idx_to_utterance_emotion = {}

meld_idx_to_utterance_emotion = {idx: (utterance, emotion) for idx, utterance, emotion in df_meld[["idx", "utterance", "emotion"]].values}
iemocap_idx_to_utterance_emotion = {idx: (utterance, emotion) for idx, utterance, emotion in df_iemocap[df_iemocap["erc_target"]][["idx", "utterance", "emotion"]].values}

idx_to_utterance_emotion.update(meld_idx_to_utterance_emotion)
idx_to_utterance_emotion.update(iemocap_idx_to_utterance_emotion)

with open("idx_to_utterance_emotion.pkl", "wb") as f:
    pickle.dump(idx_to_utterance_emotion, f)

