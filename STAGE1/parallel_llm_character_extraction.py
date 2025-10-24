from prompts import (SPEAKER_CHARACTERISTICS_EXTRACTION_PROMPT,
                     SPEAKER_CHARACTERISTICS_EXTRACTION_PROMPT_ALT1,
                     SPEAKER_CHARACTERISTICS_EXTRACTION_PROMPT_ALT2)
from utils import (load_model_via_ollama, load_model_via_hf,
                   get_dataset_as_dataframe, set_pandas_display_options,
                   abstacted_audio_text, timing, check_path_exist_from_prefix, dump_json_test_result, makedirs)
import utils
import datetime
from tqdm import tqdm
import argparse
import asyncio
import os
import gc
import numpy as np

utils.set_pandas_display_options()




def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Emotion recognition testing script for LLM models"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="meld",
        choices=["meld", "iemocap"],
        help="Dataset name to process (default: meld)"
    )

    parser.add_argument(
        "--max_k",
        type=int,
        default=20,
        help="Window size for conversation history (default: 20)"
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
        default=0,
        choices=[0, 1, 2, 3],
        help="model id determines which model to llm model to use. 0 for llama3.1-8b via ollama, 1 for llama3.1-8b via hf, 2 for gemini-2.5-flash via vertexai, 3 for gemini-2.5-pro via vertexai. (Default 1)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=["train", "dev"],
        choices=["train", "dev", "test"],
        help="Choose the split to process of the dataset between train, test, and dev. Default is None which processes all splits")



    parser.add_argument(
        "--prompt_type",
        type=str,
        default="default",
        choices=["default", "alt1", "alt2"],
        help="Determine which prompt to use"
    )
    return parser.parse_args()



def get_prompt_template(prompt_type):
    match prompt_type:
        case "default":
            return SPEAKER_CHARACTERISTICS_EXTRACTION_PROMPT
        case "alt1":
            return SPEAKER_CHARACTERISTICS_EXTRACTION_PROMPT_ALT1
        case "alt2":
            return SPEAKER_CHARACTERISTICS_EXTRACTION_PROMPT_ALT2
        case _:
            raise ValueError(f"Unknown prompt type: {prompt_type}")


def load_model_via_vertexai(model_id: str = "gemini-2.5-flash-lite", max_output_tokens: int = None,
                            disable_thinking: bool = False):
    """
    Initializes and returns a VertexAI model instance with dynamic arguments.
    """
    from langchain_google_vertexai import VertexAI

    # 1. Build the main constructor arguments
    constructor_args = {
        "model": model_id,
        "temperature": 0
    }

    # 2. Build model_kwargs (internal model parameters)
    model_kwargs = {}
    if disable_thinking:
        model_kwargs["disable_thinking"] = True

    # Add the populated model_kwargs to the main arguments
    constructor_args["model_kwargs"] = model_kwargs

    # 3. Conditionally add other optional arguments ONLY if they are provided
    if max_output_tokens is not None:
        constructor_args["max_output_tokens"] = max_output_tokens

    # 4. Call the constructor once by unpacking the arguments dictionary
    model = VertexAI(**constructor_args)

    model.name = f"{model_id} via VertexAI"
    return model


def create_model_chain(model_id: int, prompt_type: str):
    max_output_tokens = 20
    if model_id == 0:
        model = load_model_via_ollama(max_output_tokens=max_output_tokens)
    elif model_id == 1:
        model = load_model_via_hf(max_output_tokens=max_output_tokens)
    elif model_id == 2:
        model = load_model_via_vertexai("gemini-2.5-flash", max_output_tokens=None, disable_thinking=False)
    elif model_id == 3:
        model = load_model_via_vertexai("gemini-2.5-flash-lite", max_output_tokens=max_output_tokens)

    else:
        raise ValueError(f'The parameter model_id should one of the [0,1,2,3], but {model_id} was given')

    print(f"The model {model.name} was loaded!")
    prompt = get_prompt_template(prompt_type)

    print(f"The prompt {prompt.name} was loaded!")
    print(f"Prompt template: {prompt.template}")

    chain = prompt | model
    print(f"The chain is ready!\n")
    return chain, model, prompt


def create_history_context(conversation, turn_idx, max_k):
    context = ""
    conversation_history = conversation[max(0, turn_idx - max_k):turn_idx+1]
    for unit in conversation_history.to_dict(orient="records"):
        context += f"{unit['speaker']}: {unit['utterance']}\n"
    return context.strip("\n")


def load_and_prepare_dataset(dataset_name, splits=["train", "dev"], limit=None, exclude_na=False):
    columns_to_retrive = ["split", "dialog_idx", "turn_idx", "speaker", "utterance", "mapped_emotion", "idx", "intensity_level",
         "pitch_level", "rate_level"]
    df = get_dataset_as_dataframe(dataset_name, splits=splits, columns=columns_to_retrive)

    if exclude_na:
        df["no_nan"] = (df.isna().sum(axis=1) == 0)
        df = df.dropna(axis=0)
        assert df[df["no_nan"]].isna().sum().sum() == 0
    else:
        df.fillna({"intensity_level":"medium"}, inplace=True)
        df.fillna({"pitch_level":"medium"}, inplace=True)
        df.fillna({"rate_level":"medium"}, inplace=True)
        df["no_nan"] = True
        assert df.isna().sum().sum() == 0


    if limit is not None:
        df = df[:limit]

    df["abstracted_audio"] = df.apply(abstacted_audio_text, axis=1)

    inputs = {s: [] for s in splits}
    identifiers = {s: [] for s in splits}
    emotions = {s: [] for s in splits}

    max_k = 20

    for split, df_split in df.groupby(by="split"):
        for conv_idx, df_conv in tqdm(df_split.groupby(by="dialog_idx"),
                                      desc=f"Processing conversations in split {split} for dataset {dataset_name}"):
            for unit in df_conv.to_dict(orient="records"):
                if not unit["no_nan"]:
                    continue
                idx = unit["idx"]
                history_context = create_history_context(df_conv, unit["turn_idx"], max_k)
                speaker_id = unit["speaker"]
                utterance = unit["utterance"]
                audio_features = unit["abstracted_audio"]

                inputs[split].append({"history": history_context, "utterance": utterance,
                                      "audio_features": audio_features, "speaker_id": speaker_id})
                emotions[split].append(unit["mapped_emotion"])
                identifiers[split].append(idx)

    data = {s: [] for s in splits}
    for split in splits:
        sub_data = [
            {"inputs": inp, "iden": iden, "emo": emo}
            for inp, iden, emo in zip(inputs[split], identifiers[split], emotions[split])
        ]
        data[split] = sub_data

    return data


CONCURRENCY_LIMIT = 20


async def process_data_point(data, chain, semaphore, prompt_template_str, verbose):
    """Helper coroutine to process a single data point with rate limiting."""
    async with semaphore:  # Wait to acquire a spot
        try:
            speaker_characteristics = await chain.ainvoke({**data["inputs"]})
            data["output"] = speaker_characteristics

            # Prepare verbose output
            v_out = None
            if verbose:
                input_prompt = prompt_template_str.format(**data['inputs'])
                v_out = (
                    f"Input prompt:\n{input_prompt}\n"
                    f"=> Output: {speaker_characteristics}  (idx: {data['iden']}, emotion: {data['emo']})\n"
                    "*****************************"
                )
            return data, v_out

        except Exception as e:
            print(f"Error processing idx {data['iden']}: {e}")
            data["output"] = f"ERROR: {e}"  # Log error in output
            return data, None


@timing
async def generate_characteristics_via_chain(chain, dataset, dataset_name, verbose=True):
    """
    Run emotion recognition tests on the provided dataset IN PARALLEL.

    Args:
        chain: LangChain chain (prompt | model)
        test_data: List of test samples
    """

    start_time = datetime.datetime.now()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # Get the raw prompt template string for verbose printing
    prompt_template_str = chain.steps[0].template

    for split in dataset.keys():
        tasks = []
        print(f"\nQueueing {len(dataset[split])} tasks for split '{split}'...")

        # 1. Create all tasks
        for data in dataset[split]:
            tasks.append(process_data_point(data, chain, semaphore, prompt_template_str, verbose))

        # 2. Run tasks concurrently with a progress bar
        print(f"Running tasks with concurrency limit = {CONCURRENCY_LIMIT}...")

        processed_data = []

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {split}"):
            data, verbose_output = await future
            processed_data.append(data)

            if verbose and verbose_output:
                print(verbose_output)  # Print verbose output as it completes

        # 3. Update the dataset split with the processed data
        dataset[split] = processed_data

    end_time = datetime.datetime.now()
    stats = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_time": str(end_time - start_time).split(".")[0],
        "concurrency_limit": CONCURRENCY_LIMIT
    }

    return dataset, stats


def rel_path_to_save_results(args, dataset):
    split_text = "train-dev-test" if args.splits is None else "@".join(args.splits)
    dataset_size = np.sum([len(dataset[split]) for split in dataset])
    folder = "STAGE1/data/"
    data_name = f"{args.dataset_name.upper()}-model{args.model_id}_{args.prompt_type}_k{args.max_k}_{split_text}_size{dataset_size}.json"
    file_path = os.path.join(folder, data_name)
    return file_path


def build_execution_info(model, prompt, stats, args, dataset):
    """Build the test information dictionary."""
    info = {
        "prompt_name": prompt.name,
        "prompt_template": prompt.template,
        "used_model": model.name,
        "dataset_length": {split: len(dataset[split]) for split in dataset},
    }

    info.update(stats)
    info["config"] = args.__dict__

    return info

def save_results(test_info, dataset, path_to_save):


    test_result = {
        "execution_info": test_info,
        "dataset": dataset
    }

    # file_name = f"{dataset.upper()}-model{str(args.model_id)}_{args.prompt_type}_{os.path.basename(processed_data_path)}.json"
    # eval_directory = os.path.join(f"EVAL_RESULTS", '' if .experiment_id is None else f"Experiment{str(args.experiment_id)}")
    makedirs(relative_path_from_project=path_to_save[:path_to_save.rfind("/")])
    # file_path = os.path.join(path_to_save)
    dump_json_test_result(test_result, relative_path_from_project=path_to_save, add_datetime_to_filename=False)


async def main(config_dict=None):
    """Main execution function."""
    chain = None
    model = None

    utils.chdir_in_project("STAGE1")

    try:
        if config_dict is None:
            args = parse_arguments()
        else:
            args = argparse.Namespace(**config_dict)

        print(f"Arguments: {args}")

        # Load data
        dataset = load_and_prepare_dataset(
            args.dataset_name, args.splits, args.limit
        )

        path_to_save = rel_path_to_save_results(args, dataset)
        if check_path_exist_from_prefix(relative_path_from_project=path_to_save[:-5]):
            print(f"Results have already been saved at: {path_to_save[:-5]}")
            return

        # Initialize model
        chain, model, prompt = create_model_chain(args.model_id, args.prompt_type)

        # Run tests
        dataset, stats = await generate_characteristics_via_chain(
            chain,
            dataset,
            dataset_name=args.dataset_name,
            verbose=False
        )

        # Build and save results
        execution_info = build_execution_info(model, prompt, stats, args, dataset)

        save_results(execution_info, dataset, path_to_save)


    finally:
        if model is not None:
            del model
        if chain is not None:
            del chain
        gc.collect()
        print("Cleanup complete. GPU memory should be released.")



if __name__ == "__main__":
    main()



