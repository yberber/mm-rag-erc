
import utils
import argparse
from datasets import DatasetDict, Dataset # To load your .jsonl data
import data_process
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
from prompts import EMOTION_RECOGNITION_FINAL_PROMPT
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


TRAINING_SET_CONFIGS = {
    "dataset": "iemocap",
    "max_k": 20,
    "top_n": 2,
    "split": ["dev"], # We only need the dev split for evaluation
    "vectordb_path": utils.get_vectordb_path_from_attributes("hybrid", max_m=7),
    "use_detailed_example": True,
    "example_refinement_level": 1,
    "save_as": "no"
}

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Base Model vs. Two Finetuned Adapters")

    parser.add_argument(
        "--base_model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="The model ID from Hugging Face to use as the base model."
    )
    parser.add_argument(
        "--adapter_1_path",
        type=str,
        # default="FINETUNING/PHASE2A/final_checkpoint",
        default="FINETUNING/STAGE1_2/IEMOCAP/QLORA/checkpoint-750",
        help="Path to the directory of the *first* finetuned LoRA adapter."
    )
    parser.add_argument(
        "--adapter_2_path",
        type=str,
        default="FINETUNING/STAGE1_2/IEMOCAP/QLORA/checkpoint-600",
        help="Path to the directory of the *second* finetuned LoRA adapter."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="LORA_TRAINING/PHASE2/evaluation_results.jsonl",
        help="File to save the comparison results (JSONL format)."
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
        default=10, # Emotions are short, 10 tokens should be plenty
        help="Maximum number of new tokens to generate for each response."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: Limit evaluation to the first N examples for testing."
    )
    parser.add_argument(
        "--include_base",
        type=lambda x: (str(x).lower() in ['true', '1', 't', 'yes', 'y']),
        default=False,
        help="Whether to include the base model in the comparison."
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
    utils.dump_json_test_result(eval_args_results, relative_path_from_project=path_to_save, add_datetime_to_filename=True)


def get_dev_data(configs):
    """
    Loads and prepares the 'dev' data using the logic from your train.py.
    This function is modified to *not* tokenize, but return the raw text.
    """
    assert configs["dataset"] in ["meld", "iemocap", "both"]
    datasets_to_use = ["meld", "iemocap"] if configs["dataset"] == "both" else [configs["dataset"]]

    # We only care about the 'dev' split
    # training_set = {"dev": []}
    training_set = {configs["split"][0]: []}

    for ds_name in datasets_to_use:
        candidate_emotions = utils.get_mapped_emotion_set(ds_name)
        candidate_emotions_text = ", ".join(candidate_emotions)

        tmp_configs = {**configs, "dataset": ds_name}

        # This calls the (dummy or real) data_process.main
        # We only ask for the 'dev' split
        processed_dataset = data_process.main(tmp_configs)

        for split in training_set.keys():  # Just 'dev'
            transformed = []
            for ex in processed_dataset[split]:
                inp = ex["input"].copy()
                inp["candidate_emotions"] = candidate_emotions_text
                transformed.append({
                    "inputs": inp,  # This is the dict for .format()
                    "output": ex["target"],  # This is the ground truth string
                })
            training_set[split].extend(transformed)

    raw_datasets = DatasetDict({
        split: Dataset.from_list(training_set[split])
        for split in training_set.keys()
    })

    print(f"Loaded {len(raw_datasets['dev'])} examples from 'dev' split.")
    return raw_datasets['dev']


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


# def save_eval_results(args, stats, results, path_to_save):
#     eval_args_results = {"args": args, "stats": stats, "results": results}
#
#     utils.makedirs(relative_path_from_project=path_to_save[:path_to_save.rfind("/")])
#
#     utils.dump_json_test_result(eval_args_results, relative_path_from_project=path_to_save)


def get_extracted_emotion(prediction, emotion_set, assign_to_invalid_emotion=None):
    extracted = utils.extract_emotion_from_llm_output(prediction, emotion_set)
    if assign_to_invalid_emotion is not None:
        if extracted not in emotion_set:
            extracted = assign_to_invalid_emotion
    return extracted

def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Data
    dev_data = get_dev_data(TRAINING_SET_CONFIGS)
    emotion_set = utils.get_mapped_emotion_set(TRAINING_SET_CONFIGS["dataset"])
    if args.limit:
        print(f"Limiting evaluation to first {args.limit} examples.")
        dev_data = dev_data.select(range(args.limit))


    # 2. Load Models
    # We load 3 separate copies of the model.
    # This is memory-intensive but guarantees no state pollution between models.
    # QLoRA makes this feasible on a single consumer GPU.
    base_model = None
    tokenizer = None

    print("\n--- Loading Adapter 1 ---")
    model_1, tokenizer = load_model_and_tokenizer(args.base_model_id, args.use_qlora)
    model_1 = PeftModel.from_pretrained(model_1, args.adapter_1_path)
    model_1 = model_1.to(device)
    print(f"Loaded adapter 1 from: {args.adapter_1_path}")

    if args.include_base:
        print("\n--- Loading Base Model (include_base=True) ---")
        base_model, _ = load_model_and_tokenizer(args.base_model_id, args.use_qlora)
    else:
        print("\n--- Skipping base model (include_base=False) ---")

    print("\n--- Loading Adapter 2 ---")
    model_2, _ = load_model_and_tokenizer(args.base_model_id, args.use_qlora)
    model_2 = PeftModel.from_pretrained(model_2, args.adapter_2_path)
    model_2 = model_2.to(device)
    print(f"Loaded adapter 2 from: {args.adapter_2_path}")


    # 3. Run Evaluation
    print(f"\n--- Starting Evaluation (saving to {args.output_file}) ---")

    results = []

    base_correct = 0
    adapter_1_correct = 0
    adapter_2_correct = 0
    prediction_cnt = 0
    # with open(args.output_file, 'w') as f:


    for item in tqdm(dev_data, desc="Evaluating models"):
        # Format the prompt
        try:
            prompt_text = EMOTION_RECOGNITION_FINAL_PROMPT.format(**item['inputs'])
        except KeyError as e:
            print(f"ERROR: Prompt formatting failed. Key missing: {e}")
            print(f"Prompt keys: {list(item['inputs'].keys())}")
            continue

        ground_truth = item['output']

        # Generate from base model (optional)
        if args.include_base and base_model is not None:
            base_output_raw = generate_response(base_model, tokenizer, prompt_text, args.max_new_tokens)
            extracted_prediction = get_extracted_emotion(
                base_output_raw,
                emotion_set,
                assign_to_invalid_emotion="neutral"
            )
            base_output = extracted_prediction
        else:
            base_output = None

        # Generate from adapters
        adapter_1_output = generate_response(model_1, tokenizer, prompt_text, args.max_new_tokens)
        adapter_2_output = generate_response(model_2, tokenizer, prompt_text, args.max_new_tokens)

        # Store result
        result_data = {
            "prompt": prompt_text,
            "ground_truth": ground_truth,
            "base_model_output": base_output,
            "adapter_1_output": adapter_1_output,
            "adapter_2_output": adapter_2_output,
        }
        results.append(result_data)

        # Update counters
        if args.include_base and base_output is not None:
            base_correct = (base_correct + 1) if base_output.lower() == ground_truth.lower() else base_correct

        adapter_1_correct = (adapter_1_correct + 1) if adapter_1_output.lower() == ground_truth.lower() else adapter_1_correct
        adapter_2_correct = (adapter_2_correct + 1) if adapter_2_output.lower() == ground_truth.lower() else adapter_2_correct
        prediction_cnt += 1

        # Debug prints
        print(f"prompts: {prompt_text}")
        print(f"outputs: {ground_truth}")
        if args.include_base:
            print(f"base_model_output: {base_output}, Score: {base_correct}/{prediction_cnt} => {base_correct/prediction_cnt:.3f}")
        print(f"adapter_1_output: {adapter_1_output}, Score: {adapter_1_correct}/{prediction_cnt} => {adapter_1_correct/prediction_cnt:.3f}")
        print(f"adapter_2_output: {adapter_2_output}, Score: {adapter_2_correct}/{prediction_cnt} => {adapter_2_correct/prediction_cnt:.3f}")
        print("\n" + "-" * 80)

    ground_truths = [data["ground_truth"] for data in results]
    adapter_1_outputs = [data["adapter_1_output"] for data in results]
    adapter_2_outputs = [data["adapter_2_output"] for data in results]

    # Adapter metrics
    adapter1_acc = round(accuracy_score(ground_truths, adapter_1_outputs), 3)
    adapter1_f1 = round(f1_score(ground_truths, adapter_1_outputs, average='weighted', zero_division=0), 3)
    adapter2_acc = round(accuracy_score(ground_truths, adapter_2_outputs), 3)
    adapter2_f1 = round(f1_score(ground_truths, adapter_2_outputs, average='weighted', zero_division=0), 3)

    # Base model metrics (optional)
    if args.include_base:
        base_model_outputs = [data["base_model_output"] for data in results]
        base_acc = round(accuracy_score(ground_truths, base_model_outputs), 3)
        base_f1 = round(f1_score(ground_truths, base_model_outputs, average='weighted', zero_division=0), 3)
    else:
        base_acc = None
        base_f1 = None


    print("\n" + "=" * 80)
    print("EVALUATION RESULTS:")
    if args.include_base:
        print(f"Base model accuracy:     {base_acc}, weighted-f1: {base_f1}")
    else:
        print("Base model:              skipped (include_base=False)")
    print(f"Adapter_1 accuracy:      {adapter1_acc}, weighted-f1: {adapter1_f1}")
    print(f"Adapter_2 accuracy:      {adapter2_acc}, weighted-f1: {adapter2_f1}")
    print("=" * 80)

    stats = {
        "dataset_size": prediction_cnt,
        "include_base": args.include_base,
        "base_acc": base_acc,
        "base_f1": base_f1,
        "adapter1_acc": adapter1_acc,
        "adapter1_f1": adapter1_f1,
        "adapter2_acc": adapter2_acc,
        "adapter2_f1": adapter2_f1,
    }

    save_eval_results(args, stats, results, path_to_save=args.output_file)

    print("\n--- Evaluation Complete ---")
    print(f"Results saved to {args.output_file}")


    # Print a few examples
    print("\n--- Example Results ---")
    for i, res in enumerate(results[:10]):
        print(f"\nExample {i + 1}:")
        print(f"  Ground Truth:   {res['ground_truth']}")
        print(f"  Base Model:     {res['base_model_output']}")
        print(f"  Adapter 1:      {res['adapter_1_output']}")
        print(f"  Adapter 2:      {res['adapter_2_output']}")


if __name__ == "__main__":
    main()
