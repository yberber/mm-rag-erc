"""
Emotion Recognition Testing Script

This script runs emotion recognition tests on processed datasets (MELD or IEMOCAP)
using LangChain and Ollama LLM models.
"""

import json
import os
import datetime
import argparse
from tqdm import tqdm

from langchain_ollama.llms import OllamaLLM
from utils import (
    load_json_multiline,
    timing,
    get_mapped_emotion_set,
    extract_emotion_from_llm_output,
    dump_json_test_result
)
from prompts import EMOTION_RECOGNITION_PROMPT, EMOTION_RECOGNITION_CHAT_PROMPT


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Emotion recognition testing script for LLM models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="meld",
        choices=["meld", "iemocap"],
        help="Dataset name to process (default: meld)"
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=12,
        help="Window size for conversation history (default: 12)"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=1,
        help="Number of RAG examples to retrieve (default: 1)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to test (default: None, tests all)"
    )
    return parser.parse_args()

def is_prediction_correct(prediction, expected_output, emotion_set):
    """
    Check if the model's prediction matches the expected output.

    Args:
        prediction: Raw model output
        expected_output: Ground truth emotion label
        emotion_set: List of valid emotion labels

    Returns:
        bool: True if prediction matches expected output
    """
    return extract_emotion_from_llm_output(prediction, emotion_set) == expected_output


@timing
def run_tests(chain, test_data, dataset_name, emotion_set=None, limit=None, verbose=False):
    """
    Run emotion recognition tests on the provided dataset.

    Args:
        chain: LangChain chain (prompt | model)
        test_data: List of test samples
        dataset_name: Name of the dataset (e.g., 'meld', 'iemocap')
        emotion_set: List of emotion labels (default: None, will be auto-detected)
        limit: Maximum number of samples to test (default: None, tests all)
        verbose: Print detailed progress information (default: False)

    Returns:
        tuple: (predictions, actuals, identifiers, stats)
    """
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
            "emotion_set": emotion_set_text
        })

        predictions.append(prediction)
        actuals.append(target)
        identifiers.append(identifier)

        prediction_count += 1
        correct_count += is_prediction_correct(prediction, target, emotion_set)

        if verbose:
            extracted_prediction = extract_emotion_from_llm_output(prediction, emotion_set)
            print(f"{identifier}, prediction: {extracted_prediction}, target: {target}")
            print(f"Score: {correct_count} / {prediction_count}")

    end_time = datetime.datetime.now()

    stats = {
        "correct_count": correct_count,
        "prediction_count": prediction_count,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_time": str(end_time - start_time).split(".")[0]
    }

    return predictions, actuals, identifiers, stats





def load_test_data(dataset, max_k, top_n):
    """Load test data from the processed dataset directory."""
    processed_data_path = f"PROCESSED_DATASET/{dataset.upper()}/k{max_k}_n{top_n}"
    test_data_path = os.path.join(processed_data_path, "test.json")
    test_data = load_json_multiline(test_data_path)
    return test_data, test_data_path, processed_data_path


def create_model_chain():
    """Initialize and return the model chain."""
    model = OllamaLLM(model="llama3.1:8b")
    prompt = EMOTION_RECOGNITION_PROMPT
    chain = prompt | model
    return chain, model, prompt


def build_test_info(test_data_path, model, prompt, emotion_set, stats):
    """Build the test information dictionary."""
    return {
        "data_path": test_data_path,
        "elapsed_time_in_sec": round(run_tests.elapsed_time),
        "prompt_type": prompt.__class__.__name__,
        "prompt_template": prompt.template,
        "used_model": f"{type(model).__name__}(model={model.model!r})",
        "emotion_set": emotion_set,
        "stats": stats
    }


def save_test_results(dataset, processed_data_path, test_info, predictions, actuals, identifiers):
    """Save test results to a JSON file."""
    test_outputs = [
        {"pred": p, "actual": a, "iden": i}
        for p, a, i in zip(predictions, actuals, identifiers)
    ]

    test_result = {
        "test_info": test_info,
        "test_outputs": test_outputs
    }

    file_name = f"{dataset.upper()}-{os.path.basename(processed_data_path)}.json"
    file_path = os.path.join("EVAL_RESULTS", file_name)
    dump_json_test_result(file_path, test_result, add_datetime_to_filename=True)


def main():
    """Main execution function."""
    args = parse_arguments()

    # Load data
    test_data, test_data_path, processed_data_path = load_test_data(
        args.dataset, args.max_k, args.top_n
    )

    # Initialize model
    chain, model, prompt = create_model_chain()

    # Run tests
    emotion_set = get_mapped_emotion_set(args.dataset)
    predictions, actuals, identifiers, stats = run_tests(
        chain,
        test_data,
        dataset_name=args.dataset,
        emotion_set=emotion_set,
        limit=args.limit,
        verbose=True
    )

    # Build and save results
    test_info = build_test_info(test_data_path, model, prompt, emotion_set, stats)
    save_test_results(args.dataset, processed_data_path, test_info, predictions, actuals, identifiers)


if __name__ == "__main__":
    main()