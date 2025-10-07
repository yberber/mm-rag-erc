import json
import os

from langchain_ollama.llms import OllamaLLM
from utils import (load_json_multiline, timing, get_mapped_emotion_set,
                   extract_emotion_from_llm_output, dump_json_test_result)
from prompts import EMOTION_RECOGNITION_PROMPT, EMOTION_RECOGNITION_CHAT_PROMPT
from tqdm import tqdm
import datetime
import argparse

def is_prediction_correct(prediction, expected_output, emotion_set):
    return extract_emotion_from_llm_output(prediction, emotion_set) == expected_output

@timing
def run_tests(chain, test_data:list, dataset_name:str, emotion_set:list=None, limit:int=None, verbose:bool=False):
    if emotion_set is None:
        emotion_set = get_mapped_emotion_set(dataset_name)
    if limit is not None:
        test_data = test_data[:limit]
    emotion_set_text = ",".join(emotion_set)
    predictions = []
    actuals = []
    identifiers = []

    prediction_count = 0
    correct_count = 0

    stats = {}

    start_time = datetime.datetime.now()

    for data in tqdm(test_data, desc=f"Testing on dataset {dataset_name}"):
        inp = data["input"]
        target = data["target"]
        identifier = data["identifier"]

        prediction = chain.invoke({
            "history": inp["history_context"],
            "speaker_id": inp["speaker_id"],
            "utterance": inp["utterance"],
            "audio_features": inp["audio_features"],
            "top_n_rag_examples": inp["example"],
            "emotion_set": emotion_set_text})

        predictions.append(prediction)
        actuals.append(target)
        identifiers.append(identifier)

        prediction_count += 1
        correct_count += is_prediction_correct(prediction, target, emotion_set)

        if verbose:
            print(f"{identifier}, prediction: {extract_emotion_from_llm_output(prediction, emotion_set)}, target: {target}")
            print(f"Score: {correct_count} / {prediction_count}")

    end_time = datetime.datetime.now()

    stats["correct_count"] = correct_count
    stats["prediction_count"] = prediction_count
    stats["start_time"] = start_time.strftime("%Y-%m-%d %H:%M:%S")
    stats["end_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    stats["elapsed_time"] = str(end_time - start_time).split(".")[0]

    return predictions, actuals, identifiers, stats

def main():

    parser = argparse.ArgumentParser(description="Data processing script for LLM input")
    parser.add_argument("--dataset", type=str, default="meld", help="Dataset name to process, either meld or iemocap")
    parser.add_argument("--max_k", type=int, default=12, help="Window size for the history")
    parser.add_argument("--top_n", type=int, default=1, help="Number of utterance-emotion samples to retrieve for each llm input")
    args = parser.parse_args()


    dataset = args.dataset
    max_k = args.max_k
    top_n = args.top_n

    processed_data_path = f"PROCESSED_DATASET/{dataset.upper()}/k{max_k}_n{top_n}"
    test_data_path = os.path.join(processed_data_path, "test.json")

    test_data = load_json_multiline(test_data_path)

    model = OllamaLLM(model="llama3.1:8b")

    prompt = EMOTION_RECOGNITION_PROMPT

    chain = prompt | model


    emotion_set = get_mapped_emotion_set(dataset)
    predictions, actuals, identifiers, stats = run_tests(chain, test_data, dataset_name=dataset,
                                                         emotion_set=emotion_set, limit=100, verbose=True)

    test_info = {}
    test_info["data_path"] = test_data_path
    test_info["elapsed_time_in_sec"] = round(run_tests.elapsed_time)
    test_info["prompt_type"] = prompt.__class__.__name__
    test_info["prompt_template"] = prompt.template
    test_info["used_model"] = f"{type(model).__name__}(model={model.model!r})"
    test_info["emotion_set"] = emotion_set
    test_info["stats"] = stats


    test_outputs = [{"pred":p, "actual":a, "iden":i} for p, a, i in zip(predictions, actuals, identifiers)]

    test_result = {"test_info": test_info, "test_outputs": test_outputs}


    file_name = f"{dataset.upper()}-{os.path.basename(processed_data_path)}.json"
    file_path = os.path.join("EVAL_RESULTS", file_name)

    dump_json_test_result(file_path, test_result, add_datetime_to_filename=True)


if __name__ == "__main__":
    main()

