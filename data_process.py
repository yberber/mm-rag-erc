import argparse
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from prompts import EMOTION_RECOGNITION_PROMPT
from tqdm import tqdm
import os
import json
from utils import (set_pandas_display_options, meld_emotion_set_mapped, iemocap_emotion_set_mapped,
                   meld_emotion_mapper, iemocap_emotion_mapper, emotion_mapper_union)
import pickle
from langchain_ollama.llms import OllamaLLM


set_pandas_display_options()



# meld_emotion_set = ["joyful", "sad", "neutral", "angry", "surprised", "fearful", "disgusted"]
# iemocap_emotion_set = ["joyful", "sad", "neutral", "angry", "excited", "frustrated"]
#
# meld_emotion_mapper = {"joy": "joyful", "sadness": "sad", "neutral": "neutral",
#                        "anger": "angry", "surprise": "surprised", "fear": "fearful", "disgust": "disgusted"}
# iemocap_emotion_mapper = {"happiness": "joyful", "sadness": "sad", "neutral": "neutral", "anger": "angry",
#                           "excitement": "excited", "frustration": "frustrated"}
# emotion_mapper_union = meld_emotion_mapper.copy()
# emotion_mapper_union.update(iemocap_emotion_mapper)


with open("vectorstore/idx_to_utterance_emotion.pkl", 'rb') as f:
    idx_to_utterance_emotion_dict = pickle.load(f)

with open("vectorstore/simple/similar_utterance_idx_top10.pkl", 'rb') as f:
    similar_utterance_idx_top10 = pickle.load(f)


def abstacted_audio_text(row):
    return (f"{row['rate_level']} speech rate, {row['pitch_level']} "
            f"pitch, and {row['intensity_level']} intensity")

def create_history_context(conversation, current_utterance_idx, max_k):
    context = ""
    conversation_history = conversation[max(0, current_utterance_idx - max_k):current_utterance_idx+1]
    for unit in conversation_history.to_dict(orient="records"):
        context += f"{unit['speaker']}: {unit['utterance']}\n"
    return context.strip("\n")

def get_vectordb_instance(path_to_db="vectorstore_data/simple/meld_iemocap_simple_db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        collection_name="meld_iemocap_simple",
        embedding_function=embeddings,
        persist_directory=path_to_db,
    )
    return vector_db


def get_examples_cached(cache_dict, top_n, idx, idx_to_utterance_emotion_dict, emotion_set):
    examples_idx = cache_dict[idx]
    example_text = ""
    emotion_count = 0
    for ex_idx in examples_idx:
        utterance, emotion = idx_to_utterance_emotion_dict[ex_idx]
        emotion = emotion_mapper_union.get(emotion, None)
        if emotion not in emotion_set:
            continue
        example_text += f"< {utterance} : {emotion} >"
        emotion_count += 1
        if emotion_count >= top_n:
            break
    return example_text.strip('\n')


def get_examples(db, utterance, max_k, idx=None):
    out = db.similarity_search(utterance, k=max_k+1)
    if idx:
        out = [o for o in out if o.metadata["idx"] != idx]
    out = out[:max_k]
    example_text = ""
    for sim_doc in out:
        example_text += f"< {sim_doc.page_content} : {sim_doc.metadata['emotion']} >\n"
    return example_text.strip('\n')

def process_dataset_meld(dataset="meld", max_k=12, top_n=1):
    # input_prompts = {'train': [], 'test': [], 'dev': []}
    targets = {'train': [], 'test': [], 'dev': []}
    identifiers = {'train': [], 'test': [], 'dev': []}

    inputs = {'train': [], 'test': [], 'dev': []}

    if dataset == "meld":
        df = pd.read_csv("meld_erc_with_categories.csv")
        emotion_set = meld_emotion_set_mapped
        emotion_mapper = meld_emotion_mapper
    elif dataset == "iemocap":
        df = pd.read_csv("iemocap_erc_with_categories.csv")
        emotion_set = iemocap_emotion_set_mapped
        emotion_mapper = iemocap_emotion_mapper

    else:
        raise ValueError("Invalid dataset name")

    emotion_set_text = ", ".join(emotion_set)

    # df["idx"] = ("m" if dataset=="meld" else "i") + "_" + df["dialog_idx"].astype(str) + "_" + df["turn_idx"].astype(str) # TODO


    df.fillna({"intensity_level":"medium"}, inplace=True)
    df.fillna({"pitch_level":"medium"}, inplace=True)
    df.fillna({"rate_level":"medium"}, inplace=True)

    df["mapped_emotion"] = df["emotion"].map(emotion_mapper)

    # df.isna().sum()
    # df = df[["utterance", "emotion", "idx", "split", "dialog_idx", "turn_idx", "speaker", "rate_level", "pitch_level", "intensity_level"]]

    db = get_vectordb_instance()

    df["abstracted_audio"] = df.apply(abstacted_audio_text, axis=1)
    for split, df_split in df.groupby(by="split"):
        for conv_idx, df_conv in tqdm(df_split.groupby(by="dialog_idx"), desc=f"Processing conversations in split {split}"):
            df_conv["speaker_idx"] = df_conv.groupby("speaker").ngroup()
            for unit in df_conv.to_dict(orient="records"):
                if dataset.lower() == "iemocap" and not unit["erc_target"]:
                    continue
                idx = unit["idx"]
                history_context = create_history_context(df_conv, unit["turn_idx"], max_k)
                speaker_id = unit["speaker"]
                utterance = unit["utterance"]
                audio_features = unit["abstracted_audio"]
                # example = get_examples(db, utterance, top_n, idx)
                example = get_examples_cached(similar_utterance_idx_top10, top_n, idx, idx_to_utterance_emotion_dict,
                                              emotion_set)
                # input = EMOTION_RECOGNITION_PROMPT.format(top_n_rag_examples=example, history=history_context,
                #                                   speaker_id=speaker_id, utterance=utterance,
                #                                   audio_features=audio_features, emotion_set=emotion_set_text)
                target = unit["mapped_emotion"]

                inputs[split].append({"history_context":history_context, "utterance":utterance,
                                      "audio_features":audio_features, "speaker_id":speaker_id, "example":example})

                # input_prompts[split].append(input)
                targets[split].append(target)
                identifiers[split].append(idx)

    data_path = f'PROCESSED_DATASET/{dataset.upper()}/k{max_k}_n{top_n}'

    os.makedirs(data_path, exist_ok=True)
    for split in ['train', 'test', 'dev']:
        with open(f'{data_path}/{split}.json', 'w') as file:
            for input, target, identifier in zip(inputs[split], targets[split], identifiers[split]):
                file.write(json.dumps({'input': input, 'target': target, 'identifier': identifier}) + '\n')
    # for split in ['train', 'test', 'dev']:
    #     with open(f'{data_path}/{split}.json', 'w') as file:
    #         for input, target, identifier in zip(input_prompts[split], targets[split], identifiers[split]):
    #             file.write(json.dumps({'input': input, 'target': target, 'identifier': identifier}) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Data processing script for LLM input")
    parser.add_argument("--dataset", type=str, default="iemocap", help="Dataset name to process, either meld or iemocap")
    parser.add_argument("--max_k", type=int, default=12, help="Window size for the history")
    parser.add_argument("--top_n", type=int, default=1, help="Number of utterance-emotion samples to retrieve for each llm input")
    args = parser.parse_args()

    process_dataset_meld(dataset=args.dataset, max_k=args.max_k, top_n=args.top_n)

if __name__ == '__main__':
    main()