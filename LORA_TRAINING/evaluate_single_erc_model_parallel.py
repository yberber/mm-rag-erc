import utils
import argparse
from datasets import DatasetDict, Dataset
import data_process
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
from prompts import EMOTION_RECOGNITION_FINAL_PROMPT, GEMINI_EMOTION_RECOGNITION_PROMPT
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import os

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
        default="meta-llama/Llama-3.1-8B-Instruct",
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

    return parser.parse_args()


def save_eval_results(args, stats, results, path_to_save):
    eval_args_results = {
        "stats": stats,
        "args": vars(args),
        "training_set_configs": TRAINING_SET_CONFIGS,
        "results": results,
    }
    utils.makedirs(relative_path_from_project=path_to_save[:path_to_save.rfind("/")])
    utils.dump_json_test_result(
        eval_args_results,
        relative_path_from_project=path_to_save,
        add_datetime_to_filename=True
    )


def get_eval_data(configs):
    assert configs["dataset"] in ["meld", "iemocap", "both"]
    datasets_to_use = ["meld", "iemocap"] if configs["dataset"] == "both" else [configs["dataset"]]

    split_name = configs["split"][0]
    training_set = {split_name: []}

    for ds_name in datasets_to_use:
        candidate_emotions = utils.get_mapped_emotion_set(ds_name)
        candidate_emotions_text = ", ".join(candidate_emotions)

        tmp_configs = {**configs, "dataset": ds_name}
        processed_dataset = data_process.main(tmp_configs)

        for split in training_set.keys():
            transformed = []
            for ex in processed_dataset[split]:
                inp = ex["input"].copy()
                inp["candidate_emotions"] = candidate_emotions_text
                transformed.append({
                    "inputs": inp,
                    "output": ex["target"],
                    "idx": ex["idx"]   # added
                })
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
    """
    Generates responses for a list of prompts in a single batch.
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


def build_output_path(args):
    split = args.split
    dataset_name = args.dataset

    if args.adapter_path:
        adapter_subpath = args.adapter_path
        if adapter_subpath.startswith("FINETUNING/"):
            adapter_subpath = adapter_subpath.split("/", 1)[1]
        output_dir = os.path.join("EVAL_FINAL", adapter_subpath, dataset_name.upper(),  split)
    else:
        output_dir = os.path.join("EVAL_FINAL", "BASE", dataset_name.upper(), split)

    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise RuntimeError(
            f"❌ Evaluation results already exist for this model and split:\n  {output_dir}\n"
            f"Please remove or rename the folder before rerunning."
        )

    path_to_save = os.path.join(output_dir, "results.json")
    return path_to_save


def main():
    args = parse_arguments()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("No CUDA device found. Evaluation requires a GPU.")

    # set dataset
    assert args.dataset in ["meld", "iemocap"]
    TRAINING_SET_CONFIGS["dataset"] = args.dataset

    TRAINING_SET_CONFIGS["split"] = [args.split]
    output_path = build_output_path(args)

    # 1. Load Data
    eval_data = get_eval_data(TRAINING_SET_CONFIGS)
    emotion_set = utils.get_mapped_emotion_set(TRAINING_SET_CONFIGS["dataset"])
    if args.limit:
        print(f"Limiting evaluation to first {args.limit} examples.")
        eval_data = eval_data.select(range(args.limit))

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

    PROMPT_TEMPLATE = GEMINI_EMOTION_RECOGNITION_PROMPT if "gemini" in args.adapter_path.lower() else EMOTION_RECOGNITION_FINAL_PROMPT
    print(f"Prompt template: {PROMPT_TEMPLATE}")

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

    save_eval_results(args, stats, results, path_to_save=output_path)

    print("\n--- Evaluation Complete ---")
    print(f"Results saved under: {output_path}")


if __name__ == "__main__":
    main()