"""Create the utterance-index-to-metadata JSON used by the demonstration formatter.

Combines the final MELD and IEMOCAP DataFrames, anonymises speaker names,
and saves a JSON-indexed DataFrame to
``artifacts/vectorstores/utterance_index_mapping.json``.

The resulting file is loaded by
:meth:`~src.helper.build_prompting_dataset.DemonstrationCreatorViaCache.get_idx_to_speaker_utterance_emotion_df`
to look up the speaker, utterance text, and mapped emotion for each retrieved
demonstration index.

Must be run after both ``extend_meld_categories`` and
``extend_iemocap_categories`` have produced the final benchmark CSVs.

Usage::

    python -m src.vectorstore.caching.create_idx_to_metadata
"""

import os

import pandas as pd

from src.helper.utils import (anonymize_speakers_in_dialog, save_dataframe_as_json)
from src.config import paths

def main():
    if "vectorstore" not in os.getcwd():
        os.chdir("vectorstore")

    df_meld = pd.read_csv(paths.MELD_BENCHMARK_FINAL_FILE_PATH)
    df_iemocap = pd.read_csv(paths.IEMOCAP_BENCHMARK_FINAL_FILE_PATH)


    df_meld = df_meld.groupby("dialog_idx", group_keys=False).apply(
        lambda df: anonymize_speakers_in_dialog(df, use_letters=True), include_groups=False
    )

    df_iemocap["speaker"] = "Speaker_" + df_iemocap["speaker"]


    df_meld = df_meld[["idx", "speaker", "utterance", "mapped_emotion"]]
    df_iemocap = df_iemocap[["idx", "speaker", "utterance", "mapped_emotion"]]
    df_train = pd.concat([df_meld, df_iemocap], ignore_index=True)
    df_train.set_index("idx", inplace=True)

    save_dataframe_as_json(paths.VECTORSTORE_INDEX_PATH, df_train)



if __name__ == "__main__":
    main()
