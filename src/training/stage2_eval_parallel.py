"""Batch evaluation script for Phase 2 (emotion recognition).

Loads a base model and an optional LoRA adapter, runs batched inference
on the Phase 2 prompting dataset (with RAG demonstrations), and reports
weighted F1 and accuracy.  Supports ablation modes (no audio, no RAG)
via ``--use_audio`` and ``--use_rag`` flags.

Results are saved under ``artifacts/eval/stage2/`` in a sub-tree
organised by ablation condition, adapter path, dataset, and split.

Usage::

    python -m src.training.stage2_eval_parallel \\
        --adapter_path artifacts/finetuning/STAGE1_2-DEFAULT-r16/.../final_checkpoint \\
        --dataset iemocap --split test [--use_audio true] [--use_rag true]
"""

import argparse
from pathlib import Path

from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.helper import utils
from src.helper.prompts import EMOTION_RECOGNITION_FINAL_PROMPT, GEMINI_EMOTION_RECOGNITION_PROMPT, \
    EMOTION_RECOGNITION_FINAL_PROMPT_NO_AUDIO_RAG, EMOTION_RECOGNITION_FINAL_PROMPT_NO_RAG, \
    EMOTION_RECOGNITION_FINAL_PROMPT_NO_AUDIO
from src.helper import build_prompting_dataset
from src.config import paths, constants


TRAINING_SET_CONFIGS = {
    # "dataset": "iemocap",
    "max_k": 20,
    "top_n": 2,
    "split": ["dev"],
    "vectordb_path": utils.get_vectordb_path_from_attributes("hybrid", max_m=7),
    "use_detailed_example": True,
    "example_refinement_level": 1,
    "save_as": "no"
}


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a single model (base or LoRA adapter)")

    parser.add_argument(
        "--base_model_id",
        type=str,
        default=constants.BASE_LLM_MODEL,
        help="The model ID from Hugging Face to use as the base model."
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Optional path to the directory of a finetuned LoRA adapter."
    )
    parser.add_argument(
        "--use_qlora",
        type=lambda x: (str(x).lower() in ['true', '1', 't']),
        default=True,
        help="Set to False to load models in 16-bit."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum number of new tokens to generate for each response."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: Limit evaluation to the first N examples for testing."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["meld", "iemocap"],
        help="which dataset to use for evaluation (iemocap or meld)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
        help="Which split to use? Default: dev"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference. Default: 16"
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level."
    )

    parser.add_argument(
        "--use_audio",
        type=lambda x: (str(x).lower() in ["true", "1", "t", "yes", "y"]),
        default=True,
        help="Whether to use abstracted audio information in the prompt",
    )

    parser.add_argument(
        "--use_rag",
        type=lambda x: (str(x).lower() in ["true", "1", "t", "yes", "y"]),
        default=True,
        help="Whether to use in context learning via RAG",
    )
    return parser.parse_args()


def save_eval_results(args, stats, results, prompt, path_to_save):
    eval_args_results = {
        "stats": stats,
        "args": vars(args),
        "prompt_info": {"prompt_name": prompt.name, "prompt_template": prompt.template},
        "training_set_configs": TRAINING_SET_CONFIGS,
        "results": results,
    }
    output_path = Path(path_to_save)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    utils.dump_json_test_result(
        eval_args_results,
        path=output_path,
        add_datetime_to_filename=True
    )


def process_inputs_in_processed_dataset(processed_dataset, split, candidate_emotions_text, use_audio=True, use_rag_in_contex=True):
    transformed = []
    for ex in processed_dataset[split]:
        input = ex['input'].copy()
        if not use_audio:
            input.pop('audio_features')
        if not use_rag_in_contex:
            input.pop('demonstrations')
        input["candidate_emotions"] = candidate_emotions_text
        transformed.append({
            "inputs": input,
            "output": ex["target"],
            "idx": ex["idx"]
        })
    return transformed



def get_eval_data(configs, use_audio=True, use_rag_in_contex=True):
    assert configs["dataset"] in ["meld", "iemocap", "both"]
    datasets_to_use = ["meld", "iemocap"] if configs["dataset"] == "both" else [configs["dataset"]]

    split_name = configs["split"][0]
    training_set = {split_name: []}

    for ds_name in datasets_to_use:
        candidate_emotions = utils.get_mapped_emotion_set(ds_name)
        candidate_emotions_text = ", ".join(candidate_emotions)

        tmp_configs = {**configs, "dataset": ds_name}
        processed_dataset = build_prompting_dataset.main(tmp_configs)

        for split in training_set.keys():
            transformed = process_inputs_in_processed_dataset(processed_dataset, split,
                                                              candidate_emotions_text,
                                                              use_audio, use_rag_in_contex)
            training_set[split].extend(transformed)

    raw_datasets = DatasetDict({
        split: Dataset.from_list(training_set[split])
        for split in training_set.keys()
    })

    print(f"Loaded {len(raw_datasets[split_name])} examples from '{split_name}' split.")
    return raw_datasets[split_name]


def load_model_and_tokenizer(model_id, use_qlora):
    print(f"Loading tokenizer for: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # CRITICAL for batch inference:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = None
    if use_qlora:
        print("Configuring QLoRA (4-bit quantization)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading base model: {model_id} {'with QLoRA' if use_qlora else 'in bfloat16'}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    return model, tokenizer


def generate_batch(model, tokenizer, prompts, max_new_tokens):
    """Generate emotion-label predictions for a batch of prompts.

    Args:
        model: The loaded causal LM (with optional LoRA adapter merged).
        tokenizer: The corresponding tokenizer (left-padded for batch
            inference).
        prompts (list[str]): List of formatted prompt strings.
        max_new_tokens (int): Maximum number of tokens to generate per
            prompt.

    Returns:
        list[str]: Decoded and stripped output strings, one per prompt.
    """
    model.eval()

    # Tokenize the batch
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True
    ).to(model.device)

    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    # Slice output to remove input prompt (handling batch dimension)
    generated_ids = outputs[:, input_length:]

    # Batch decode
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [r.strip() for r in responses]


def get_extracted_emotion(prediction, emotion_set, assign_to_invalid_emotion=None):
    extracted = utils.extract_emotion_from_llm_output(prediction, emotion_set)
    if assign_to_invalid_emotion is not None:
        if extracted not in emotion_set:
            extracted = assign_to_invalid_emotion
    return extracted


def get_intermediate_path_for_ablation_studies(args):
    if args.use_audio and args.use_rag:
        return Path("baseline")
    elif (not args.use_audio) and args.use_rag:
        return Path("ablation/no_audio")
    elif args.use_audio and (not args.use_rag):
        return Path("ablation/no_rag")
    else:
        return Path("ablation/no_audio_rag")

def build_output_path(args):
    split = args.split
    dataset_name = args.dataset

    ablation_intermediate = get_intermediate_path_for_ablation_studies(args)
    base_dir = paths.EVAL_STAGE2_DIR / ablation_intermediate
    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
        if not adapter_path.is_absolute():
            adapter_path = (paths.PROJECT_PATH / adapter_path).resolve()
        try:
            rel_adapter = adapter_path.relative_to(paths.ARTIFACTS_DIR)
        except ValueError:
            rel_adapter = adapter_path.relative_to(paths.PROJECT_PATH)
        output_dir = base_dir / rel_adapter / dataset_name.upper() / split
    else:
        output_dir = base_dir / "BASE" / dataset_name.upper() / split

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"❌ Evaluation results already exist for this model and split:\n  {output_dir}\n"
            f"Please remove or rename the folder before rerunning."
        )

    path_to_save = output_dir / "results.json"
    return path_to_save


def get_prompt_template(args):
    if "gemini" in args.adapter_path.lower():
        return GEMINI_EMOTION_RECOGNITION_PROMPT
    elif args.use_audio and args.use_rag:
        return EMOTION_RECOGNITION_FINAL_PROMPT
    elif (not args.use_audio) and args.use_rag:
        return EMOTION_RECOGNITION_FINAL_PROMPT_NO_AUDIO
    elif args.use_audio and (not args.use_rag):
        return EMOTION_RECOGNITION_FINAL_PROMPT_NO_RAG
    else:
        return EMOTION_RECOGNITION_FINAL_PROMPT_NO_AUDIO_RAG

def main():
    args = parse_arguments()
    print(f"Arguments: {args}")
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("No CUDA device found. Evaluation requires a GPU.")

    # set dataset
    assert args.dataset in ["meld", "iemocap"]
    TRAINING_SET_CONFIGS["dataset"] = args.dataset

    TRAINING_SET_CONFIGS["split"] = [args.split]
    output_path = build_output_path(args)
    print(f"Output path: {output_path}")

    # 1. Load Data
    eval_data = get_eval_data(TRAINING_SET_CONFIGS, args.use_audio,args.use_rag)
    emotion_set = utils.get_mapped_emotion_set(TRAINING_SET_CONFIGS["dataset"])
    if args.limit:
        print(f"Limiting evaluation to first {args.limit} examples.")
        eval_data = eval_data.select(range(args.limit))

    PROMPT_TEMPLATE = get_prompt_template(args)
    print(f"Prompt template name: {PROMPT_TEMPLATE.name}")
    print(f"Prompt template: {PROMPT_TEMPLATE}")


    # 2. Load Model
    print("\n--- Loading Model ---")
    model, tokenizer = load_model_and_tokenizer(args.base_model_id, args.use_qlora)

    if args.adapter_path:
        print(f"Loading LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.to(device)
        model_type = "adapter"
    else:
        print("No adapter path provided. Evaluating base model only.")
        model = model.to(device)
        model_type = "base"

    # 3. Run Evaluation
    output_path = build_output_path(args)
    print(f"\n--- Starting Evaluation (Batch Size: {args.batch_size} on Dataset {args.dataset}) ---")

    results = []
    y_true, y_pred = [], []
    correct = 0
    prediction_cnt = 0

    # Iterate in chunks
    total_samples = len(eval_data)



    for i in tqdm(range(0, total_samples, args.batch_size), desc="Eval Batches"):

        # Slicing the dataset returns a dictionary of lists (e.g. {'inputs': [..], 'output': [..]})
        batch_slice = eval_data[i: i + args.batch_size]

        # Prepare inputs and ground truths
        batch_prompts = []
        batch_ground_truths = []
        # batch_indices = [] # removed
        batch_ids = []  # <--- NEW LINE

        # Construct prompts for the batch
        for idx_in_batch, input_data in enumerate(batch_slice['inputs']):
            try:
                prompt_text = PROMPT_TEMPLATE.format(**input_data)
                batch_prompts.append(prompt_text)
                batch_ground_truths.append(batch_slice['output'][idx_in_batch])
                # batch_indices.append(idx_in_batch)
                batch_ids.append(batch_slice['idx'][idx_in_batch]) # added
            except KeyError as e:
                print(f"ERROR: Prompt formatting failed for index {i + idx_in_batch}. Key missing: {e}")
                continue

        if not batch_prompts:
            continue

        # Generate for the whole batch
        batch_outputs_raw = generate_batch(model, tokenizer, batch_prompts, args.max_new_tokens)

        # Process results
        for j, raw_output in enumerate(batch_outputs_raw):
            gt = batch_ground_truths[j]
            prompt = batch_prompts[j]

            extracted = get_extracted_emotion(
                raw_output,
                emotion_set,
                assign_to_invalid_emotion="neutral",
            )

            y_true.append(gt)
            y_pred.append(extracted)
            prediction_cnt += 1

            if extracted.lower() == gt.lower():
                correct += 1

            results.append({
                "idx": batch_ids[j], # added
                # "prompt": prompt,
                "ground_truth": gt,
                # "model_output_raw": raw_output,
                "model_output": extracted,
            })

            # Verbose printing (Print only 1st item of batch to avoid clutter, or all if verbose=2)
            if args.verbose == 1 and j == 0:
                running_acc = correct / prediction_cnt
                print(f" Batch Ex | GT: {gt} | Pred: {extracted} | Run Acc: {running_acc:.3f}")
            elif args.verbose == 2:
                running_acc = correct / prediction_cnt
                print(f"GT: {gt} | Pred: {extracted} | Raw: {raw_output} | Acc: {running_acc:.3f}")

    # 4. Metrics
    acc = round(accuracy_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS:")
    print(f"Model type: {model_type} ({'base only' if model_type == 'base' else args.adapter_path})")
    print(f"Batch Size: {args.batch_size}")
    print(f"Accuracy:   {acc}")
    print(f"Weighted F1:{f1}")
    print("=" * 80)

    stats = {
        "dataset_size": len(y_true),
        "model_type": model_type,
        "adapter_path": args.adapter_path,
        "batch_size": args.batch_size,
        "accuracy": acc,
        "f1_weighted": f1,
    }

    save_eval_results(args, stats, results, PROMPT_TEMPLATE, path_to_save=output_path)

    print("\n--- Evaluation Complete ---")
    print(f"Results saved under: {output_path}")


if __name__ == "__main__":
    main()
