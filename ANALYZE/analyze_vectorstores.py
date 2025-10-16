import pandas as pd

from utils import (chdir_in_project, set_pandas_display_options,
                   get_meld_iemocap_datasets_as_dataframe,
                   load_json)
import os
from glob import glob
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

set_pandas_display_options()

chdir_in_project("ANALYZE")
os.getcwd()

df_meld, df_iemocap = get_meld_iemocap_datasets_as_dataframe()

df_meld_test = df_meld[df_meld["split"] == "test"]
df_meld_test = df_meld_test[['idx', 'mapped_emotion']]

df_iemocap_test = df_iemocap[(df_iemocap["split"] == "test") & (df_iemocap['erc_target'])]
df_iemocap_test = df_iemocap_test[['idx', 'mapped_emotion']]

df_test = pd.concat([df_meld_test, df_iemocap_test], axis=0, ignore_index=True)
# df_test.info()

df_test_splitter = len(df_meld_test)

assert len(df_test) == len(df_meld_test) + len(df_iemocap_test)

df_meld = df_meld[['idx', 'mapped_emotion']]
df_iemocap = df_iemocap[['idx', 'mapped_emotion']]


idx_to_emotion = pd.Series(df_meld.mapped_emotion.values, index=df_meld.idx).to_dict()
idx_to_emotion.update(pd.Series(df_iemocap.mapped_emotion.values, index=df_iemocap.idx).to_dict())


top_n = 3


for cache_path in glob("../vectorstore/caches/*.json"):
    cache_similar_utterance_idx = load_json(path=cache_path)
    cache_name = Path(cache_path).stem
    print(f" ****** Statistics with vectorstore: {cache_name} ******")

    similar_utterance_emotions_lists = [[] for i in range(top_n)]

    target_emotion_list = df_test["mapped_emotion"].values

    for row in df_test.to_dict('records'):
        idx = row['idx']
        mapped_emotion = row['mapped_emotion']

        similar_utterance_idx = cache_similar_utterance_idx[idx][:3]
        similar_utterance_emotions = [idx_to_emotion[idx] for idx in similar_utterance_idx]

        for i, emotion in enumerate(similar_utterance_emotions):
            similar_utterance_emotions_lists[i].append(emotion)

    for i, emotion_list in enumerate(similar_utterance_emotions_lists, start=1):
        meld_average = round(accuracy_score(target_emotion_list[:df_test_splitter], emotion_list[:df_test_splitter]), 2)
        meld_weighted_f1 = round(f1_score(target_emotion_list[:df_test_splitter], emotion_list[:df_test_splitter], average='weighted'), 2)

        iemocap_average = round(accuracy_score(target_emotion_list[df_test_splitter:], emotion_list[df_test_splitter:]), 2)
        iemocap_weighted_f1 = round(f1_score(target_emotion_list[df_test_splitter:], emotion_list[df_test_splitter:], average='weighted'), 2)

        total_average = round(accuracy_score(target_emotion_list, emotion_list), 2)
        total_weighted_f1 = round(f1_score(target_emotion_list, emotion_list, average='weighted'), 2)

        print(f"{i}. result statistics")

        print(f"Meld => average: {meld_average}, weighted f1: {meld_weighted_f1}")
        print(f"IEMOCAP => average: {iemocap_average}, weighted f1: {iemocap_weighted_f1}")
        print(f"TOTAL => average: {total_average}, weighted f1: {total_weighted_f1}")


