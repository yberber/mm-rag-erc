import utils
import argparse
from datasets import DatasetDict, Dataset  # To load your .jsonl data
import data_process
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
from prompts import EMOTION_RECOGNITION_FINAL_PROMPT
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import os


TRAINING_SET_CONFIGS = {
    "dataset": "iemocap",
    "max_k": 20,
    "top_n": 2,
    "split": ["dev"],  # Will be overwritten by args.split
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
        default=10,  # Emotions are short, 10 tokens should be plenty
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
    return parser.parse_args()


def save_eval_results(args, stats, results, path_to_save):
    # Convert args to a plain dict so it can be saved as JSON
    eval_args_results = {
        "args": vars(args),
        "training_set_configs": TRAINING_SET_CONFIGS,
        "stats": stats,
        "results": results,
    }
    utils.makedirs(relative_path_from_project=path_to_save[:path_to_save.rfind("/")])
    utils.dump_json_test_result(
        eval_args_results,
        relative_path_from_project=path_to_save,
        add_datetime_to_filename=True
    )


def get_eval_data(configs):
    """
    Loads and prepares the data using the logic from your train.py.
    This function is modified to *not* tokenize, but return the raw text.
    """
    assert configs["dataset"] in ["meld", "iemocap", "both"]
    datasets_to_use = ["meld", "iemocap"] if configs["dataset"] == "both" else [configs["dataset"]]

    split_name = configs["split"][0]  # e.g. "dev", "train", "test"
    training_set = {split_name: []}

    for ds_name in datasets_to_use:
        candidate_emotions = utils.get_mapped_emotion_set(ds_name)
        candidate_emotions_text = ", ".join(candidate_emotions)

        tmp_configs = {**configs, "dataset": ds_name}

        # This calls the (dummy or real) data_process.main
        processed_dataset = data_process.main(tmp_configs)

        for split in training_set.keys():
            transformed = []
            for ex in processed_dataset[split]:
                inp = ex["input"].copy()
                inp["candidate_emotions"] = candidate_emotions_text
                transformed.append({
                    "inputs": inp,      # This is the dict for .format()
                    "output": ex["target"],  # This is the ground truth string
                })
            training_set[split].extend(transformed)

    raw_datasets = DatasetDict({
        split: Dataset.from_list(training_set[split])
        for split in training_set.keys()
    })

    print(f"Loaded {len(raw_datasets[split_name])} examples from '{split_name}' split.")
    return raw_datasets[split_name]


def load_model_and_tokenizer(model_id, use_qlora):
    """Loads the base model and tokenizer, with optional QLoRA."""
    print(f"Loading tokenizer for: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Use left padding for batch generation

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


def generate_response(model, tokenizer, prompt, max_new_tokens):
    """Generates a response from a model given a prompt."""
    model.eval()  # Set model to evaluation mode

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Store length of prompt to slice it out later
    prompt_length = input_ids.shape[1]

    with torch.no_grad():
        # Generate output
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False  # Use deterministic greedy decoding
        )

    # Decode the *newly generated tokens only*
    generated_ids = outputs[0, prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()


def get_extracted_emotion(prediction, emotion_set, assign_to_invalid_emotion=None):
    extracted = utils.extract_emotion_from_llm_output(prediction, emotion_set)
    if assign_to_invalid_emotion is not None:
        if extracted not in emotion_set:
            extracted = assign_to_invalid_emotion
    return extracted


def build_output_path(args):
    """
    Build the directory + file path where results should be stored.

    - If adapter_path is given (e.g. FINETUNING/STAGE1_2/IEMOCAP/QLORA/checkpoint-750)
      => EVAL_FINAL/STAGE1_2/IEMOCAP/QLORA/checkpoint-750/<split>/results.json
    - Else (base model)
      => EVAL_FINAL/BASE/<split>/results.json

    Throws an error if results for this model+split already exist.
    """
    split = args.split

    if args.adapter_path:
        # Drop leading "FINETUNING/" if present
        adapter_subpath = args.adapter_path
        if adapter_subpath.startswith("FINETUNING/"):
            adapter_subpath = adapter_subpath.split("/", 1)[1]

        output_dir = os.path.join("EVAL_FINAL", adapter_subpath, split)
    else:
        # Base model case
        output_dir = os.path.join("EVAL_FINAL", "BASE", split)

    # --- Check if results already exist ---
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise RuntimeError(
            f"❌ Evaluation results already exist for this model and split:\n  {output_dir}\n"
            f"Please remove or rename the folder before rerunning."
        )

    path_to_save = os.path.join(output_dir, "results.json")
    return path_to_save



def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make TRAINING_SET_CONFIGS respect the chosen split
    TRAINING_SET_CONFIGS["split"] = [args.split]

    # 1. Load Data
    eval_data = get_eval_data(TRAINING_SET_CONFIGS)
    emotion_set = utils.get_mapped_emotion_set(TRAINING_SET_CONFIGS["dataset"])
    if args.limit:
        print(f"Limiting evaluation to first {args.limit} examples.")
        eval_data = eval_data.select(range(args.limit))

    # 2. Load Model (+ optional adapter)
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
    print(f"\n--- Starting Evaluation (saving to {output_path}) ---")

    results = []
    y_true = []
    y_pred = []

    for item in tqdm(eval_data, desc="Evaluating model"):
        # Format the prompt
        try:
            prompt_text = EMOTION_RECOGNITION_FINAL_PROMPT.format(**item['inputs'])
        except KeyError as e:
            print(f"ERROR: Prompt formatting failed. Key missing: {e}")
            print(f"Prompt keys: {list(item['inputs'].keys())}")
            continue

        ground_truth = item['output']

        # Generate from the single model
        model_output_raw = generate_response(model, tokenizer, prompt_text, args.max_new_tokens)
        extracted_prediction = get_extracted_emotion(
            model_output_raw,
            emotion_set,
            assign_to_invalid_emotion="neutral"
        )

        results.append({
            "prompt": prompt_text,
            "ground_truth": ground_truth,
            "model_output_raw": model_output_raw,
            "model_output": extracted_prediction,
        })

        y_true.append(ground_truth)
        y_pred.append(extracted_prediction)

        # Debug prints
        print(f"prompt: {prompt_text}")
        print(f"ground_truth: {ground_truth}")
        print(f"model_output_raw: {model_output_raw}")
        print(f"model_output_extracted: {extracted_prediction}")
        print("\n" + "-" * 80)

    # 4. Metrics
    acc = round(accuracy_score(y_true, y_pred), 3)
    f1 = round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 3)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS:")
    print(f"Model type: {model_type} ({'base only' if model_type == 'base' else args.adapter_path})")
    print(f"Accuracy:   {acc}")
    print(f"Weighted F1:{f1}")
    print("=" * 80)

    stats = {
        "dataset_size": len(y_true),
        "model_type": model_type,
        "adapter_path": args.adapter_path,
        "accuracy": acc,
        "f1_weighted": f1,
    }

    save_eval_results(args, stats, results, path_to_save=output_path)

    print("\n--- Evaluation Complete ---")
    print(f"Results saved under (before datetime suffix): {output_path}")

    # Print a few examples
    print("\n--- Example Results ---")
    for i, res in enumerate(results[:10]):
        print(f"\nExample {i + 1}:")
        print(f"  Ground Truth:         {res['ground_truth']}")
        print(f"  Model Output (raw):   {res['model_output_raw']}")
        print(f"  Model Output (clean): {res['model_output']}")


if __name__ == "__main__":
    main()
