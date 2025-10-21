import os.path

import pandas as pd
import json
from functools import wraps
import time
import re
import chromadb
from glob import glob


# PROJECT_PATH = "/Users/yusuf/LLM-for-ERC"
PROJECT_PATH = "/gpfs/bwfor/home/hd/hd_hd/hd_ux323/LLM-for-ERC"

def set_pandas_display_options():
    # Permanently changes the pandas settings
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.width', 500)
    # pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.precision', 2)


meld_emotion_set_original = ["joy", "sadness", "neutral", "anger", "surprise", "fear", "disgust"]
iemocap_emotion_set_original = ["happiness", "sadness", "neutral", "anger", "excitement", "frustration",
                                "unknown", "surprise", "fear", "other", "disgust"]

# union_emotion_set_original = sorted(list(set(meld_emotion_set_original).union(set(iemocap_emotion_set_original))))
union_emotion_set_original = ['anger', 'disgust','excitement', 'fear', 'frustration',
                     'happiness', 'joy', 'neutral', 'other', 'sadness', 'surprise', 'unknown']

id_to_original_emotion = {i: e for i, e in enumerate(union_emotion_set_original)}
original_emotion_to_id = {e: i for i, e in enumerate(union_emotion_set_original)}

union_emotion_set_mapped = ['angry', 'disgusted','excited', 'fearful', 'frustrated',
                     'joyful', 'neutral', 'sad', 'surprised']

id_to_mapped_emotion = {i: e for i, e in enumerate(union_emotion_set_mapped)}
mapped_emotion_to_id = {e: i for i, e in enumerate(union_emotion_set_mapped)}


meld_emotion_mapper = {"joy": "joyful", "sadness": "sad", "neutral": "neutral",
                       "anger": "angry", "surprise": "surprised", "fear": "fearful", "disgust": "disgusted"}
iemocap_emotion_mapper = {"happiness": "joyful", "sadness": "sad", "neutral": "neutral", "anger": "angry",
                          "excitement": "excited", "frustration": "frustrated"}

meld_mapped_valid_emotion_set = set(["joyful", "sad", "neutral", "angry", "surprised", "fearful", "disgusted"])
iemocap_mapped_valid_emotion_set = set(["joyful", "sad", "neutral", "angry", "excited", "frustrated"])

emotion_mapper_ori_to_conv = {
    'anger': 'angry',
    'disgust': 'disgusted',
    'excitement': 'excited',
    'fear': 'fearful',
    'frustration': 'frustrated',
    'happiness': 'joyful',
    'joy': 'joyful',
    'neutral': 'neutral',
    'other': 'unknown',
    'sadness': 'sad',
    'surprise': 'surprised',
    'unknown': 'unknown'
}

meld_emotion_set_mapped = ['joyful', 'sad', 'neutral', 'angry', 'surprised', 'fearful', 'disgusted']
assert meld_emotion_set_mapped == list(meld_emotion_mapper.values())

iemocap_emotion_set_mapped = ['joyful', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
assert iemocap_emotion_set_mapped == list(iemocap_emotion_mapper.values())

emotion_mapper_union = meld_emotion_mapper.copy()
emotion_mapper_union.update(iemocap_emotion_mapper)

emotion_set = ['joyful', 'sad', 'neutral', 'angry', 'excited', 'surprised', 'frustrated', 'fearful', 'disgusted']

def get_mapped_emotion_set(dataset=None):
    if dataset is None:
        return emotion_set
    elif dataset.lower() == 'meld':
        return meld_emotion_set_mapped
    elif dataset.lower() == 'iemocap':
        return iemocap_emotion_set_mapped
    else:
        raise ValueError('Dataset must be either None, or "meld" or "iemocap"')

from enum import Enum
class EmotionExtractionStrategy(Enum):
    FirstMentionedValidLabel = 1
    OneMentionedValidLabel = 2



def extract_emotion_from_llm_output(output_text, valid_emotions=emotion_set,
                                    approach:EmotionExtractionStrategy=EmotionExtractionStrategy.FirstMentionedValidLabel):
    output_text = output_text.lower()
    if approach == EmotionExtractionStrategy.FirstMentionedValidLabel:
        first_mentioned_emotion = None
        first_mentioned_emotion_pos = len(output_text)
        for emotion in valid_emotions:
            if emotion in output_text:
                emotion_pos = output_text.index(emotion)
                if emotion_pos < first_mentioned_emotion_pos:
                    first_mentioned_emotion_pos = emotion_pos
                    first_mentioned_emotion = emotion
        if first_mentioned_emotion is None:
            return "NoValidEmotionFound"
        return first_mentioned_emotion

    elif approach == EmotionExtractionStrategy.OneMentionedValidLabel:
        mentioned_emotion = None
        for emotion in valid_emotions:
            if emotion in output_text:
                if mentioned_emotion is not None:
                    return "MultipleValidEmotionsFound"
                    raise ValueError(f"More than one valid emotion was mentioned in the output: {output_text}")
                mentioned_emotion = emotion
        if mentioned_emotion is None:
            return "NoValidEmotionFound"
            raise ValueError(f"No valid emotion was mentioned in the output: {output_text}")
        else:
            return mentioned_emotion
    else:
        raise ValueError(f"Unknown emotion extraction strategy: {approach}")


def load_json_multiline(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def check_path_exist_from_prefix(path=None, relative_path_from_project=None):
    path = get_path(path=path, relative_path_from_project=relative_path_from_project)
    return len(glob(path + "*")) > 0


def dump_json_test_result(result, path=None, relative_path_from_project=None, add_datetime_to_filename=False, verbose=True):
    path = get_path(path=path, relative_path_from_project=relative_path_from_project)

    if add_datetime_to_filename:
        now = time.strftime("%b-%d-%Y_%H:%M:%S")
        now += ".json"
        now = "_" + now
        path = re.sub(".json", now, path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False, sort_keys=False)
    if verbose:
        print("Wrote results to {}".format(path))

def load_json_test_result(path):
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded

def save_as_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to {path}")


def save_dataframe_as_json(path, df, orient="index"):
    df.to_json(path, orient=orient, indent=4)

def load_dataframe_from_json(path=None, relative_path_from_project=None, orient="index"):
    if path is None == relative_path_from_project is None:
        print(f'one argument should be None, other non-None. But you gave {path} and {relative_path_from_project}')
    if relative_path_from_project:
        path = os.path.join(PROJECT_PATH, relative_path_from_project)
    loaded = pd.read_json(path, orient=orient)
    return loaded


def get_path(path=None, relative_path_from_project=None):
    if path is None == relative_path_from_project is None:
        print(f'one argument should be None, other non-None. But you gave {path} and {relative_path_from_project}')
    if relative_path_from_project:
        path = os.path.join(PROJECT_PATH, relative_path_from_project)
    return path

def makedirs(path=None, relative_path_from_project=None):
    path = get_path(path=path, relative_path_from_project=relative_path_from_project)
    os.makedirs(path, exist_ok=True)


def load_json(path=None, relative_path_from_project=None):
    path = get_path(path=path, relative_path_from_project=relative_path_from_project)
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        elapsed = round(te - ts, 2)
        print(f"func {func.__name__} took {elapsed} seconds")
        wrapper.elapsed_time = elapsed
        return result
    return wrapper


def anonymize_speakers_in_dialog(df_dialog, use_letters=False):
    unique_speakers = df_dialog['speaker'].unique()
    mapping = {name: f"Speaker_{chr(i+65) if use_letters else i}" for i, name in enumerate(unique_speakers)}
    df_dialog['speaker'] = df_dialog['speaker'].map(mapping)
    # Anonymize names within the 'utterance' column
    # We loop through the original names and their anonymized versions
    for name, anonymized_name in mapping.items():
        # Create a case-insensitive regular expression to find the whole word
        pattern = fr"\b{re.escape(name)}\b"

        # Apply the replacement to the utterance column
        df_dialog['utterance'] = df_dialog['utterance'].str.replace(
            pattern,
            anonymized_name,
            regex=True,
            case=False  # This makes the replacement case-insensitive
        )
    return df_dialog


def chromadb_collection_exists(persist_directory: str, collection_name: str) -> bool:
    """
    Checks if a ChromaDB collection exists in the given directory.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    collections = client.list_collections()
    return any(collection.name == collection_name for collection in collections)


def collection_exists_and_not_empty(persist_directory: str, collection_name: str, throw_exception=False) -> bool:
    """
    Checks if a ChromaDB collection exists in a directory and is not empty.

    Args:
        persist_directory: The path to the ChromaDB persist directory.
        collection_name: The name of the collection to check.

    Returns:
        True if the collection exists and contains one or more items, False otherwise.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    collections = client.list_collections()
    if any(collection.name == collection_name for collection in collections):
        collection = client.get_collection(name=collection_name)
        res = collection.count() > 0
        if throw_exception and not res :
            raise ValueError(f"Collection {collection_name} is empty in {persist_directory}")
        return res
    if throw_exception:
        raise ValueError(f"Collection {collection_name} does not exist in {persist_directory}. The list of collections at {persist_directory} is: {collections}")
    return False

def chdir_in_project(path):
    os.chdir(os.path.join(PROJECT_PATH, path))


def get_dataset_as_dataframe(dataset_name, splits=None, columns=None):
    if dataset_name.lower() == "meld":
        meld_path = os.path.join(PROJECT_PATH, "BENCMARK_DATASETS", "meld_erc_with_categories.csv")
        df = pd.read_csv(meld_path)
        df["erc_target"] = True
    elif dataset_name.lower() == "iemocap":
        iemocap_path = os.path.join(PROJECT_PATH, "BENCMARK_DATASETS", "iemocap_erc_with_categories.csv")
        df = pd.read_csv(iemocap_path)
    else:
        raise ValueError("Invalid dataset name")

    if isinstance(splits, str):
        splits = [splits]
    if splits is not None:
         df = df[df["split"].isin(splits)]
    if columns is not None:
        df = df[columns]
    return df



    if type(splits) is str:
        return meld_df[meld_df['split']==splits], iemocap_df[iemocap_df['split'] == splits]

    if type(splits) is list and len(splits):
        return meld_df[meld_df['split'].isin(splits)], iemocap_df[iemocap_df['split'].isin(splits)]

    raise Exception('splits must be either None, or str, or a non-empty list')


def get_meld_iemocap_datasets_as_dataframe(splits=None):
    meld_path = os.path.join(PROJECT_PATH, "BENCMARK_DATASETS", "meld_erc_with_categories.csv")
    iemocap_path = os.path.join(PROJECT_PATH, "BENCMARK_DATASETS", "iemocap_erc_with_categories.csv")
    meld_df = pd.read_csv(meld_path)
    iemocap_df = pd.read_csv(iemocap_path)
    if splits is None:
        return meld_df, iemocap_df

    if type(splits) is str:
        return meld_df[meld_df['split']==splits], iemocap_df[iemocap_df['split'] == splits]

    if type(splits) is list and len(splits):
        return meld_df[meld_df['split'].isin(splits)], iemocap_df[iemocap_df['split'].isin(splits)]

    raise Exception('splits must be either None, or str, or a non-empty list')


def get_vectordb_path_from_attributes(db_type: str, max_m: int = None):
    if db_type.lower() == "single":
        return "vectorstore/vectorstore_db/meld_iemocap_single"
    elif db_type.lower() in ["flow", "hybrid"]:
        return f"vectorstore/vectorstore_db/meld_iemocap_{db_type}_{max_m}"
    else:
        raise ValueError(f"Unknown db_type: {db_type}")

def str2bool (val):
    """Convert a string representation of truth to true  or false.
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    if isinstance(val, bool):
        return val
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


# eval_folder = "EVAL_RESULTS/"
#
# default_test_results_path = eval_folder + "MELD-k12_n1_Oct-08-2025_18:49:59.json"
# gemini_test_results_path = eval_folder + "MELD-k12_n1_Oct-08-2025_19:07:22.json"
# claude_test_results_path = eval_folder + "MELD-k12_n1_Oct-08-2025_19:19:36.json"
# gpt5_test_results_path = eval_folder + "MELD-k12_n1_Oct-08-2025_19:33:05.json"
#
# default_test_info, default_test_results = load_json_test_result(default_test_results_path).values()
# predictions, targets = [], []
# for unit in default_test_results:
#     pred = unit["pred"]
#     extracted_pred = extract_emotion_from_llm_output(pred)
#     if extracted_pred in ["MultipleValidEmotionsFound", "NoValidEmotionFound"]:
#         extracted_pred = "neutral"
#     actual = unit["actual"]
#
#     predictions.append(extracted_pred)
#     targets.append(actual)
#
# set(predictions)
# set(targets)
# set(targets).issubset(set(predictions))
#
# from collections import Counter
# Counter(predictions)
#
# from sklearn.metrics import classification_report
# print(classification_report(targets, predictions))
#
# def get_report_from_test_path(test_path):
#     test_info, test_results = load_json_test_result(test_path).values()
#     predictions, targets = [], []
#     set(targets)
#     for unit in test_results:
#         pred = unit["pred"]
#         extracted_pred = extract_emotion_from_llm_output(pred, valid_emotions=meld_emotion_set_mapped)
#         if extracted_pred in ["MultipleValidEmotionsFound", "NoValidEmotionFound"]:
#             extracted_pred = "neutral"
#
#         actual = unit["actual"]
#
#         predictions.append(extracted_pred)
#         targets.append(actual)
#     return classification_report(targets, predictions, zero_division=0)
#
# print(get_report_from_test_path(default_test_results_path))
# print(get_report_from_test_path(gemini_test_results_path))
# print(get_report_from_test_path(claude_test_results_path))
# print(get_report_from_test_path(gpt5_test_results_path))

