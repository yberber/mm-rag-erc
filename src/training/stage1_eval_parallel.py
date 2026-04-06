import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
import evaluate

from src.helper import utils
from src.config import paths, constants
from src.helper.prompts import SPEAKER_CHARACTERISTICS_EXTRACTION_PROMPT

BATCH_SIZE = 16  # Adjust based on your GPU VRAM.


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a single model (base or LoRA adapter) - STAGE1")

    parser.add_argument(
        "--base_model_id",
        type=str,
        default=constants.BASE_LLM_MODEL,
        help="The model ID from Hugging Face to use as the base model."
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="artifacts/finetuning/STAGE1-DEFAULT-r16/BOTH/QLORA/final_checkpoint",
        help=(
            "Optional path to the directory of a finetuned LoRA adapter. "
            "If omitted, the base model will be evaluated."
        ),
    )
    parser.add_argument(
        "--use_qlora",
        type=lambda x: (str(x).lower() in ['true', '1', 't']),
        default=True,
        help="Set to False to load models in 16-bit (requires >VRAM)."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,  # 20 tokens should be plenty
        help="Maximum number of new tokens to generate for each response."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: Limit evaluation to the first N examples for testing."
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
        default=BATCH_SIZE,
        help="Batch size for inference. Default: 16"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["meld", "iemocap", "both"],
        help="Which dataset to use? Default: both"
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Verbosity level: 0 = silent, 1 = show GT + extracted prediction + score, 2 = show full raw info."
    )

    return parser.parse_args()


def save_eval_results(args, stats, results, path_to_save):
    # Convert args to a plain dict so it can be saved as JSON
    eval_args_results = {
        "stats": stats,
        "args": vars(args),
        "results": results,
    }
    output_path = Path(path_to_save)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    utils.dump_json_test_result(
        eval_args_results,
        path=output_path,
        add_datetime_to_filename=True
    )


def get_data_dirs(args):
    data_dirs = []
    if args.dataset in ['iemocap', 'both']:
        data_dirs.append(paths.TRAINING_STAGE1_DIR / "IEMOCAP")
    if args.dataset in ['meld', 'both']:
        data_dirs.append(paths.TRAINING_STAGE1_DIR / "MELD")
    return data_dirs


def get_eval_data(args):
    """
    Loads and prepares the data.
    """
    assert args.dataset in ["meld", "iemocap", "both"]
    data_dirs = get_data_dirs(args)

    all_datasets = []
    for data_dir in data_dirs:
        dir_path = Path(data_dir)
        data_files = {
            f"{args.split}": dir_path / f"{args.split}.jsonl",
        }

        existing_data_files = {}
        for split, path in data_files.items():
            if path.exists():
                existing_data_files[split] = str(path)
            else:
                print(f"Warning: Data file not found for split '{split}' at {path}")

        print(f"Loading data from {dir_path}...")
        all_datasets.append(load_dataset("json", data_files=existing_data_files))

    # Combine datasets if multiple were loaded
    if len(all_datasets) > 1:
        print("Combining datasets...")
        sets = [ds[args.split] for ds in all_datasets if args.split in ds]

        # --- FIX: Return the concatenated Dataset directly, not a DatasetDict ---
        if sets:
            return concatenate_datasets(sets)
        else:
            raise ValueError(f"No data found for split '{args.split}'")

    raw_datasets = all_datasets[0]
    print(f"Loaded {len(raw_datasets[args.split])} examples from '{args.split}' split for dataset {args.dataset}.")

    return raw_datasets[args.split]


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

    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
        if not adapter_path.is_absolute():
            adapter_path = (paths.PROJECT_PATH / adapter_path).resolve()
        try:
            rel_path = adapter_path.relative_to(paths.ARTIFACTS_DIR)
        except ValueError:
            rel_path = adapter_path.relative_to(paths.PROJECT_PATH)
        output_dir = paths.EVAL_STAGE1_DIR / rel_path / split
    else:
        output_dir = paths.EVAL_STAGE1_DIR / "BASE" / split

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    path_to_save = output_dir / "results.json"
    return path_to_save


def compute_metrics(predictions, references):
    print("Loading metrics (BERTScore, ROUGE, BLEU)...")

    # 1. BERTScore (Semantic Similarity)
    # Using 'roberta-large' is standard for English.
    bertscore = evaluate.load("bertscore")
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="en",
                                     model_type="roberta-large")

    # 2. ROUGE (N-gram overlap)
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=predictions, references=references)

    # 3. BLEU (Precision)
    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(predictions=predictions, references=references)

    # Aggregate BERTScore (it returns a list of scores per example)
    avg_bert_precision = sum(bert_results['precision']) / len(bert_results['precision'])
    avg_bert_recall = sum(bert_results['recall']) / len(bert_results['recall'])
    avg_bert_f1 = sum(bert_results['f1']) / len(bert_results['f1'])

    final_metrics = {
        "bertscore_f1": avg_bert_f1,
        "bertscore_precision": avg_bert_precision,
        "bertscore_recall": avg_bert_recall,
        "rouge1": rouge_results['rouge1'],
        "rouge2": rouge_results['rouge2'],
        "rougeL": rouge_results['rougeL'],
        "bleu": bleu_results['bleu']
    }

    return final_metrics


def main():
    args = parse_arguments()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("No CUDA device found. Evaluation requires a GPU.")

    # 1. Load Data
    eval_data = get_eval_data(args)

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

    # Check if results already exist to avoid overwriting (optional, you can remove this check if you want to overwrite)
    if output_path.exists():
        print(f"⚠️ Warning: Results already exist at {output_path}")
        # raise RuntimeError(f"Results file already exists: {output_path}")

    print(f"\n--- Starting Evaluation (Batch Size: {args.batch_size}) ---")

    results = []
    y_true, y_pred = [], []

    # Iterate in chunks
    total_samples = len(eval_data)

    for i in tqdm(range(0, total_samples, args.batch_size), desc="Eval Batches"):

        # Slicing the dataset returns a dictionary of lists (e.g. {'inputs': [..], 'output': [..]})
        # Because we fixed get_eval_data, eval_data is now a Dataset, so slicing works!
        batch_slice = eval_data[i: i + args.batch_size]

        # Prepare inputs and ground truths
        batch_prompts = []
        batch_ground_truths = []
        # batch_indices = []
        batch_idens = []  # <--- [CHANGE 1] List to store IDs

        # Construct prompts for the batch
        for idx_in_batch, input_data in enumerate(batch_slice['inputs']):
            try:
                prompt_text = SPEAKER_CHARACTERISTICS_EXTRACTION_PROMPT.format(**input_data)
                batch_prompts.append(prompt_text)
                batch_ground_truths.append(batch_slice['output'][idx_in_batch])
                # batch_indices.append(idx_in_batch)
                # <--- [CHANGE 2] Extract and store the 'iden'
                batch_idens.append(batch_slice['iden'][idx_in_batch])
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

            y_true.append(gt)
            y_pred.append(raw_output)

            results.append({
                # "prompt": prompt,
                "iden": batch_idens[j],
                "ground_truth": gt,
                "model_output": raw_output,
            })

            # Verbose printing (Print only 1st item of batch to avoid clutter, or all if verbose=2)
            # if args.verbose == 1 and j == 0:
            #     print(f" Batch Ex | GT: {gt} | Pred: {raw_output} | Run Acc: {running_acc:.3f}")
            if args.verbose == 2:
                print(f"Ground Truth: {gt}")
                print(f"Prediction: {raw_output}")
                print("")

    # 4. Calculate Metrics
    metrics = compute_metrics(y_pred, y_true)

    # 5. Print & Save
    print("\n" + "=" * 40)
    print("   PHASE 1 EVALUATION RESULTS   ")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")
    print("=" * 40)

    stats = {
        "dataset_size": len(y_true),
        "model_type": model_type,
        "adapter_path": args.adapter_path,
        "batch_size": args.batch_size,
        "metrics": vars(metrics) if hasattr(metrics, "__dict__") else metrics,
    }

    save_eval_results(args, stats, results, path_to_save=output_path)

    print("\n--- Evaluation Complete ---")
    print(f"Results saved under: {output_path}")


if __name__ == "__main__":
    main()
