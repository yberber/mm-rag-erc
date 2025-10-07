import pandas as pd
import json
from functools import wraps
import time
import re

def set_pandas_display_options():
    # Permanently changes the pandas settings
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.precision', 2)


meld_emotion_set_original = ["joyful", "sad", "neutral", "angry", "surprised", "fearful", "disgusted"]
iemocap_emotion_set_original = ["joyful", "sad", "neutral", "angry", "excited", "frustrated"]

meld_emotion_mapper = {"joy": "joyful", "sadness": "sad", "neutral": "neutral",
                       "anger": "angry", "surprise": "surprised", "fear": "fearful", "disgust": "disgusted"}
iemocap_emotion_mapper = {"happiness": "joyful", "sadness": "sad", "neutral": "neutral", "anger": "angry",
                          "excitement": "excited", "frustration": "frustrated"}

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
                                    approach:EmotionExtractionStrategy=EmotionExtractionStrategy.OneMentionedValidLabel):
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
                raise ValueError(f"No valid emotion was mentioned in the output: {output_text}")
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

def dump_json_test_result(path, result, add_datetime_to_filename=False, verbose=True):
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


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        elapsed = te - ts
        print(f"func {func.__name__} took {elapsed} seconds")
        wrapper.elapsed_time = elapsed
        return result
    return wrapper

