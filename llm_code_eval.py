"""
Emotion Recognition Testing Script

This script runs emotion recognition tests on processed datasets (MELD or IEMOCAP)
using LangChain and Ollama LLM models.
"""
import gc
import os
import datetime
import argparse
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score

from utils import (
    makedirs,
    timing,
    get_mapped_emotion_set,
    extract_emotion_from_llm_output,
    dump_json_test_result,
    load_json, str2bool,
    check_path_exist_from_prefix
)
from prompts import *

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
        "--example_type",
        type=str,
        default="single",
        choices=["single", "flow", "hybrid"],
        help="Type of RAG examples to retrieve (default: single)"
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=1,
        help="Number of RAG examples to retrieve (default: 1)"
    )
    parser.add_argument(
        "--max_m",
        type=int,
        default=1,
        help="Number of utterances in flow or hybrid demonstration (default: 1)"
    )

    parser.add_argument(
        "--use_detailed_example",
        type=str,
        default="False",
        help="Map each utterance in example to an emotion if true for the examples from flow or hybrid db"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to test (default: None, tests all)"
    )
    parser.add_argument(
        "--model_id",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="model id determines which model to llm model to use. 0 for llama3.1-8b via ollama, 1 for llama3.1-8b via hf, 2 for gemini-2.5-flash via vertexai, 3 for gemini-2.5-pro via vertexai. (Default 1)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
        help="Choose the split to process of the dataset between train, test, and dev. Default is None which processes all splits")

    parser.add_argument(
        "--save",
        type=str,
        default="True",
        help="Whether to save the results (default: True)"
    )

    parser.add_argument(
        "--experiment_id",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Determine where to save the results"
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        default="default",
        choices=["default", "gemini", "claude", "gpt5"],
        help="Determine which prompt to use"
    )
    return parser.parse_args()

def get_prompt_template(prompt_type, add_example):
    match prompt_type:
        case "default":
            return EMOTION_RECOGNITION_PROMPT if add_example else EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE
        case "gemini":
            return GEMINI_EMOTION_RECOGNITION_PROMPT if add_example else GEMINI_EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE
        case "claude":
            return CLAUDE_EMOTION_RECOGNITION_PROMPT if add_example else CLAUDE_EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE
        case "gpt5":
            return GPT5_EMOTION_RECOGNITION_PROMPT if add_example else GPT5_EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE
        case _:
            raise ValueError(f"Unknown prompt type: {prompt_type}")


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


def get_extracted_emotion(prediction, emotion_set, assign_to_invalid_emotion=None):
    extracted = extract_emotion_from_llm_output(prediction, emotion_set)
    if assign_to_invalid_emotion is not None:
        if extracted not in emotion_set:
            extracted = assign_to_invalid_emotion
    return extracted



@timing
def run_tests(chain, test_data, dataset_name, max_k, emotion_set=None, limit=None, verbose=False):
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

    emotion_set_text = ", ".join(emotion_set)

    predictions = []
    actuals = []
    identifiers = []

    prediction_count = 0
    correct_count = 0

    start_time = datetime.datetime.now()

    extracted_predictions = []

    for data in tqdm(test_data, desc=f"Testing on dataset {dataset_name}"):
        inp = data["input"]
        target = data["target"]
        identifier = data["idx"]

        # input_variables = ["demonstrations", "history", "speaker_id", "utterance", "audio_features",
                           # "candidate_emotions"],

        inp["history"] = "\n".join(inp["history"].split("\n")[-(max_k+1):])
        prediction = chain.invoke({
            **inp,
            "candidate_emotions": emotion_set_text
        })

        predictions.append(prediction)
        actuals.append(target)
        identifiers.append(identifier)

        extracted_prediction = get_extracted_emotion(prediction, emotion_set, assign_to_invalid_emotion="neutral")


        prediction_count += 1
        correct_count += int(target == extracted_prediction)


        extracted_predictions.append(extracted_prediction)


        if verbose:
            print(f"{identifier}, prediction: {prediction}, target: {target}")
            print(f"Score: {correct_count} / {prediction_count}")

    end_time = datetime.datetime.now()

    stats = {
        "accuracy_score": round(accuracy_score(actuals, extracted_predictions), 2),
        "f1_score": round(f1_score(actuals, extracted_predictions, average='weighted', zero_division=0), 2),
        "correct_count": correct_count,
        "prediction_count": prediction_count,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_time": str(end_time - start_time).split(".")[0]
    }

    return predictions, actuals, identifiers, stats


def get_data_name(dataset, max_k, example_type, top_n, max_m, use_detailed_example, use_default_k=None):
    if use_default_k is not None:
        if 0 <= max_k <= use_default_k:
            max_k = 20
        else:
            raise Exception("max_k must be between 1 and 20")
    demonstration_id =  f"{example_type}{'V2' if use_detailed_example and example_type in ['flow', 'hybrid'] else ''}_n{top_n}_m{max_m}"
    return f"{dataset.upper()}/k{max_k}_{demonstration_id}"

def load_eval_data(dataset, max_k, example_type, top_n, max_m, use_detailed_example, split):
    """Load test data from the processed dataset directory."""
    data_name = get_data_name(dataset, max_k, example_type, top_n, max_m, use_detailed_example, use_default_k=20)
    processed_data_path = f"PROCESSED_DATASET/{data_name}"
    eval_data_path = os.path.join(processed_data_path, f"{split.lower()}.json")
    eval_data = load_json(relative_path_from_project=eval_data_path)
    return eval_data, eval_data_path, processed_data_path


def create_model_chain(model_id: int, prompt_type, add_example):
    if model_id == 0:
        model = load_model_via_ollama()
    elif model_id == 1:
        model = load_model_via_hf()
    elif model_id == 2 or model_id == 3:
        raise Exception(f"Model {model_id} is not implemented.")
    else:
        raise ValueError(f'The parameter model_id should one of the [0,1,2,3], but {model_id} was given')


    prompt = get_prompt_template(prompt_type, add_example=add_example)

    chain = prompt | model
    return chain, model, prompt


def load_model_via_hf():
    from langchain_huggingface.llms import HuggingFacePipeline
    if not torch.cuda.is_available():
        raise Exception("Cuda should be available to use the model loaded via huggingface for getting responses in a reasonable timeQ")
    print("Cuda Device Count:", torch.cuda.device_count())
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 10, "return_full_text": False},
        # device_map="auto",
        # This line forces the model onto the GPU
        device=0,
        model_kwargs={"dtype": torch.bfloat16}, # More standard way to set dtype
    )
    model.name = f'{model_id.split("/")[1]} via HF'
    return model


def load_model_via_ollama():
    from langchain_ollama.llms import OllamaLLM
    """Initialize and return the model chain."""
    model_id = "llama3.1:8b"
    model = OllamaLLM(
        model=model_id,
        num_predict=10)
    model.name = f'{model_id} via Ollama'
    return model
    

def build_test_info(test_data_path, model, prompt, emotion_set, stats, args):
    """Build the test information dictionary."""
    return {
        "data_path": test_data_path,
        "elapsed_time_in_sec": round(run_tests.elapsed_time),
        "prompt_name": prompt.name,
        "prompt_template": prompt.template,
        "used_model": model.name,
        "emotion_set": str(emotion_set),
        "stats": stats,
        "config": args.__dict__,
    }


def save_test_results(test_info, predictions, actuals, identifiers, path_to_save):
    """Save test results to a JSON file."""
    test_outputs = [
        {"pred": p, "actual": a, "iden": i}
        for p, a, i in zip(predictions, actuals, identifiers)
    ]

    test_result = {
        "test_info": test_info,
        "test_outputs": test_outputs
    }

    # file_name = f"{dataset.upper()}-model{str(args.model_id)}_{args.prompt_type}_{os.path.basename(processed_data_path)}.json"
    # eval_directory = os.path.join(f"EVAL_RESULTS", '' if .experiment_id is None else f"Experiment{str(args.experiment_id)}")
    makedirs(relative_path_from_project=path_to_save[:path_to_save.rfind("/")])
    # file_path = os.path.join(path_to_save)
    dump_json_test_result(test_result, relative_path_from_project=path_to_save, add_datetime_to_filename=False)


def rel_path_to_save_results(args):
    data_name = get_data_name(args.dataset, args.max_k, args.example_type, args.top_n, args.max_m, args.use_detailed_example)
    processed_data_path = f"PROCESSED_DATASET/{data_name}"
    file_name = f"{args.dataset.upper()}-model{str(args.model_id)}_{args.prompt_type}_{os.path.basename(processed_data_path)}.json"
    eval_directory = os.path.join(f"EVAL_RESULTS", '' if args.experiment_id is None else f"Experiment{str(args.experiment_id)}")
    file_path = os.path.join(eval_directory, file_name)
    return file_path


def main(config_dict=None):
    """Main execution function."""
    chain = None
    model = None

    try:
        if config_dict is None:
            args = parse_arguments()
        else:
            args = argparse.Namespace(**config_dict)
        args.save = str2bool(args.save)
        args.use_detailed_example = str2bool(args.use_detailed_example)

        print(f"Arguments: {args}")

        # Load data
        test_data, test_data_path, processed_data_path = load_eval_data(
            args.dataset, args.max_k, args.example_type, args.top_n, args.max_m, args.use_detailed_example, args.split
        )

        path_to_save = rel_path_to_save_results(args)
        if args.save and check_path_exist_from_prefix(relative_path_from_project=path_to_save[:-5]):
            print(f"Results have already been saved at: {path_to_save[:-5]}")
            return

        # Initialize model
        chain, model, prompt = create_model_chain(args.model_id, args.prompt_type, add_example=args.top_n > 0)

        # Run tests
        candidate_emotion_set = get_mapped_emotion_set(args.dataset)
        predictions, actuals, identifiers, stats = run_tests(
            chain,
            test_data,
            max_k=args.max_k,
            dataset_name=args.dataset,
            emotion_set=candidate_emotion_set,
            limit=args.limit,
            verbose=True
        )

        # Build and save results
        test_info = build_test_info(test_data_path, model, prompt, candidate_emotion_set, stats, args)

        if args.save:
            save_test_results(test_info, predictions, actuals,
                              identifiers, path_to_save)

    finally:
        if model is not None:
            del model
        if chain is not None:
            del chain
        gc.collect()
        torch.cuda.empty_cache()
        print("Cleanup complete. GPU memory should be released.")



if __name__ == "__main__":
    main()


