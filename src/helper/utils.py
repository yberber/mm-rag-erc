import os.path
from pathlib import Path

import pandas as pd
import json
from functools import wraps
import time
import re
import chromadb
from glob import glob

from src.config import paths

PROJECT_PATH = str(paths.PROJECT_PATH)

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
    if dataset is None or dataset.lower() == 'both':
        return emotion_set
    elif dataset.lower() == 'meld':
        return meld_emotion_set_mapped
    elif dataset.lower() == 'iemocap':
        return iemocap_emotion_set_mapped
    else:
        raise ValueError('Dataset must be either None, or "both", or "meld" or "iemocap"')

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


def get_path(path=None, relative_path_from_project=None):
    if (path is None) == (relative_path_from_project is None):
        raise ValueError(f"Provide exactly one of path or relative_path_from_project, got {path} and {relative_path_from_project}")
    if relative_path_from_project is not None:
        return (PROJECT_PATH / relative_path_from_project).resolve()
    return Path(path).resolve()


def load_json_multiline(path):
    target_path = Path(path)
    data = []
    with target_path.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def check_path_exist_from_prefix(path=None, relative_path_from_project=None):
    path = get_path(path=path, relative_path_from_project=relative_path_from_project)
    return len(glob(str(path) + "*")) > 0



def dump_json_test_result(result, path=None, relative_path_from_project=None, add_datetime_to_filename=False, verbose=True):
    path = str(get_path(path=path, relative_path_from_project=relative_path_from_project))

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
    target_path = Path(path)
    with target_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded


def save_as_json(path, data):
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to {target_path}")



def save_dataset_as_json(path, dataset):
    target_dir = Path(path)
    target_dir.mkdir(parents=True, exist_ok=True)
    for split, data in dataset.items():
        file_path = target_dir / f"{split}.json"
        with file_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    print(f"Saved dataset to {target_dir}")


def save_dataset_as_jsonl(path, dataset):
    target_dir = Path(path)
    target_dir.mkdir(parents=True, exist_ok=True)
    for split, data in dataset.items():
        file_path = target_dir / f"{split}.jsonl"
        with file_path.open("w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_dataframe_as_json(path, df, orient="index"):
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(target_path, orient=orient, indent=4)

def load_dataframe_from_json(path=None, relative_path_from_project=None, orient="index"):
    target_path = get_path(path=path, relative_path_from_project=relative_path_from_project)
    loaded = pd.read_json(target_path, orient=orient)
    return loaded

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
    os.chdir(PROJECT_PATH / path)

# for testing on validation sets
# the dataset should be created beforehand via src/training_data_creation/phase1/generate_speaker_characteristics.py
def get_idx_to_speaker_characteristics_hint(speaker_characteristics_type, dataset_name, split = "dev"):
    if speaker_characteristics_type in ["default", "alt1" ,"alt2", "default-no-audio"] and split == "dev":
        size = 1109 if dataset_name.upper() == "MELD" else 825

        data_from_model2 = load_json(path=paths.SPEAKER_CHARACTERISTICS_DIR / f"{dataset_name.upper()}-model2_default_k20_{split}_size{size}.json")["dataset"]["dev"]
        data_from_model3 = load_json(path=paths.SPEAKER_CHARACTERISTICS_DIR / f"{dataset_name.upper()}-model3_default_k20_{split}_size{size}.json")["dataset"]["dev"]

        if speaker_characteristics_type == "default":
            prefix = "reaction of potential listeners: "
        elif speaker_characteristics_type == "alt1":
            prefix = "mental state or behavior: "
        else:
            prefix = "speaker's intention or reason: "

        data_from_model2 = {d["iden"]: prefix+d["output"] for d in data_from_model2 if len(d["output"]) > 0}
        data_from_model3 = {d["iden"]: prefix+d["output"] for d in data_from_model3}
        data_from_model3.update(data_from_model2)
        return data_from_model3
    else:
        raise ValueError(f"Unknown speaker characteristics type: {speaker_characteristics_type}")



def get_stage1_training_set(dataset_name, splits=None):
    """Load a stage 1 training dataset from the speaker characteristics directory.

    Resolves the dataset file in one of two ways:
      - Exact filename: if ``dataset_name`` ends with ".json", it is treated as
        the literal filename inside ``paths.SPEAKER_CHARACTERISTICS_DIR``.
      - Short name (e.g. "meld", "iemocap"): all JSON files whose name starts
        with the given prefix (case-insensitive) are collected, and the one with
        the largest size — extracted from the ``size<N>`` suffix in the
        filename — is selected.

    All splits present in the file are returned.

    Args:
        dataset_name: Either an exact JSON filename
            (e.g. ``"MELD-model2_default_k20_train-dev_size11098.json"``)
            or a short dataset identifier (e.g. ``"meld"``, ``"iemocap"``).
        splits: Splits to return, by default all splits are returned.

    Returns:
        dict: The full dataset dictionary keyed by split name.

    Raises:
        ValueError: If no matching dataset files are found.
    """
    if dataset_name.endswith(".json"):
        file_path = paths.SPEAKER_CHARACTERISTICS_DIR / dataset_name
        if not file_path.exists():
            raise ValueError(f"Dataset file not found: {file_path}")
    else:
        candidates = [
            f for f in paths.SPEAKER_CHARACTERISTICS_DIR.glob("*.json")
            if f.name.lower().startswith(dataset_name.lower())
        ]
        if not candidates:
            raise ValueError(
                f"No datasets matching '{dataset_name}' found in "
                f"{paths.SPEAKER_CHARACTERISTICS_DIR}"
            )

        def _extract_size(path):
            """Parse the integer after the last 'size' token in the filename."""
            stem = path.stem
            if "size" in stem:
                try:
                    return int(stem.rsplit("size", 1)[-1])
                except ValueError:
                    return 0
            return 0

        file_path = max(candidates, key=_extract_size)

    dataset = load_json(file_path)["dataset"]

    total_elements = sum(len(v) for v in dataset.values() if isinstance(v, list))
    print(f"Loaded dataset: {file_path.name} with {total_elements} elements")

    if splits:
        return [dataset[split] for split in splits]

    return dataset




def get_dataset_as_dataframe(dataset_name, splits=None, columns=None):
    if dataset_name.lower() == "meld":
        df = pd.read_csv(paths.MELD_BENCHMARK_FINAL_FILE_PATH)
        df["erc_target"] = True
    elif dataset_name.lower() == "iemocap":
        df = pd.read_csv(paths.IEMOCAP_BENCHMARK_FINAL_FILE_PATH)
    else:
        raise ValueError("Invalid dataset name")

    if isinstance(splits, str):
        splits = [splits]
    if splits is not None:
         df = df[df["split"].isin(splits)]
    if columns is not None:
        df = df[columns]
    return df


def get_meld_iemocap_datasets_as_dataframe(splits=None, return_only_dataset=None):
    meld_path = paths.MELD_BENCHMARK_FINAL_FILE_PATH
    iemocap_path = paths.IEMOCAP_BENCHMARK_FINAL_FILE_PATH
    meld_df = pd.read_csv(meld_path)
    iemocap_df = pd.read_csv(iemocap_path)

    if type(splits) is str:
        meld_df = meld_df[meld_df['split']==splits]
        iemocap_df = iemocap_df[iemocap_df['split'] == splits]

    elif type(splits) is list and len(splits):
        meld_df = meld_df[meld_df['split'].isin(splits)]
        iemocap_df = iemocap_df[iemocap_df['split'].isin(splits)]

    elif splits is not None:
        raise Exception('splits must be either None, or str, or a non-empty list')

    if return_only_dataset:
        if return_only_dataset.lower() == "meld":
            return meld_df
        elif return_only_dataset.lower() == "iemocap":
            return iemocap_df

    return meld_df, iemocap_df




def get_vectordb_path_from_attributes(db_type: str, max_m: int = None):
    if db_type.lower() == "single":
        return paths.VECTORSTORE_DB_DIR / "meld_iemocap_single"
    elif db_type.lower() in ["flow", "hybrid"]:
        return paths.VECTORSTORE_DB_DIR / f"meld_iemocap_{db_type}_{max_m}"
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


def load_model_via_hf(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", max_output_tokens=10):
    from langchain_huggingface.llms import HuggingFacePipeline
    import torch
    if not torch.cuda.is_available():
        raise Exception("Cuda should be available to use the model loaded via huggingface for getting responses in a reasonable timeQ")
    print("Cuda Device Count:", torch.cuda.device_count())
    model = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": max_output_tokens, "return_full_text": False},
        # device_map="auto",
        # This line forces the model onto the GPU
        device=0,
        model_kwargs={"dtype": torch.bfloat16}, # More standard way to set dtype
    )
    model.name = f'{model_id.split("/")[1]} via HF'
    return model


def load_model_via_ollama(model_id="llama3.1:8b", max_output_tokens=10):
    from langchain_ollama.llms import OllamaLLM
    """Initialize and return the model chain."""
    model = OllamaLLM(
        model=model_id,
        num_predict=max_output_tokens)
    model.name = f'{model_id} via Ollama'
    return model


def abstacted_audio_text(row):
    return (f"{row['rate_level']} speech rate, {row['pitch_level']} "
            f"pitch, and {row['intensity_level']} intensity")


