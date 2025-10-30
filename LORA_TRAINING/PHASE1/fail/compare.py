import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel
import os
from tqdm import tqdm

# --- IMPORT FROM YOUR PROJECT ---
# Make sure this script is in the same directory as 'utils.py' and 'prompts.py'
import utils
from prompts import SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE
from LORA_TRAINING.PHASE1.fail.phase1_lora_training2 import load_iemocap_dataset  # Reuse your data loader

# ============================================================================
# Configuration
# ============================================================================

# --- 1. SET YOUR MODEL PATHS ---
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = os.path.join(utils.PROJECT_PATH, "FINETUNING/PHASE1b/")

# --- 2. SET NUMBER OF EXAMPLES TO TEST ---
NUM_SAMPLES = 100

# --- 3. SET GENERATION CONFIG ---
MAX_NEW_TOKENS = 150  # Max tokens to generate


# ============================================================================
# Model Loading
# ============================================================================

def load_models_and_tokenizer():
    """Load the base model, fine-tuned model, and tokenizer (16-bit LoRA version)"""

    print(f"Loading tokenizer from: {ADAPTER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading 16-bit base model: {BASE_MODEL_NAME}")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,  # <-- Use 16-bit
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()  # Set to evaluation mode

    print(f"Loading fine-tuned model (base + adapter): {ADAPTER_PATH}")
    # Load a second copy of the base model for the adapter
    ft_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,  # <-- Use 16-bit
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply the adapter
    ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_PATH)
    ft_model.eval()  # Set to evaluation mode

    print("\nModels and tokenizer loaded successfully!")
    return base_model, ft_model, tokenizer


# ============================================================================
# Generation
# ============================================================================

def format_prompt(item: dict) -> (str, str):
    """
    Format a single example from the dev dataset into a prompt.
    Returns the prompt string and the ground truth string.
    """
    prompt = SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE.format(
        history=item['inputs']['history'],
        speaker_id=item['inputs']['speaker_id'],
        utterance=item['inputs']['utterance'],
        audio_features=item['inputs']['audio_features']
    )

    ground_truth = item['output']
    return prompt, ground_truth


def generate_response(model, tokenizer, prompt_text: str) -> str:
    """Generate a response from a model given a prompt."""

    # Tokenize the prompt
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        return_attention_mask=True
    ).to(model.device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,  # Stop when EOS is generated
        )

    # Decode and strip the prompt
    decoded_full = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # This is crucial: only return the newly generated text
    response_only = decoded_full[len(prompt_text):]

    return response_only.strip()


# ============================================================================
# Main Comparison
# ============================================================================

def main():
    print("=" * 80)
    print("Starting Model Comparison")
    print("=" * 80)

    # 1. Load models
    base_model, ft_model, tokenizer = load_models_and_tokenizer()

    # 2. Load data
    print(f"\nLoading dev data...")
    _, dev_data = load_iemocap_dataset()

    # Select a few samples
    if len(dev_data) < NUM_SAMPLES:
        print(f"Warning: Dev data has fewer than {NUM_SAMPLES} samples. Using all {len(dev_data)}.")
        samples = dev_data
    else:
        samples = dev_data.select(range(NUM_SAMPLES))

    print(f"Loaded {len(samples)} samples for comparison.")

    # 3. Run comparison
    print("\n" + "=" * 80)
    print(f"RUNNING COMPARISON ON {len(samples)} SAMPLES")
    print("=" * 80 + "\n")

    for i, item in enumerate(tqdm(samples, desc="Generating responses")):
        prompt, ground_truth = format_prompt(item)

        # --- Generate from Base Model ---
        base_output = generate_response(base_model, tokenizer, prompt)

        # --- Generate from Fine-Tuned Model ---
        ft_output = generate_response(ft_model, tokenizer, prompt)

        # --- Print Comparison ---
        print(f"\n\n{'=' * 40} EXAMPLE {i + 1} {'=' * 40}")

        print("\n--- PROMPT ---")
        print(prompt)

        print("\n--- GROUND TRUTH (Expected Output) ---")
        print(ground_truth)

        print("\n--- BASE MODEL OUTPUT ---")
        print(base_output)

        print("\n--- FINE-TUNED MODEL OUTPUT ---")
        print(ft_output)

        print(f"\n{'=' * 89}")

    print("\nComparison complete!")


if __name__ == "__main__":
    main()