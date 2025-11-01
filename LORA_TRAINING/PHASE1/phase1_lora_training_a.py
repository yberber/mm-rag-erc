# import sys
# import os
#
# # Add the project root (the parent directory of this file's directory) to the Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)


import utils
from prompts import SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE
import argparse
import os
import torch # PyTorch for models and tensors




# Hugging Face libraries
from datasets import load_dataset # To load your .jsonl data
from transformers import (
    AutoModelForCausalLM, # Loads the Llama model
    AutoTokenizer,       # Loads the Llama tokenizer
    BitsAndBytesConfig,  # Configuration for 4-bit quantization (QLoRA) - needed if QLoRA is used
    TrainingArguments,   # Sets up all training parameters (epochs, lr, etc.)
    Trainer,             # Handles the actual training loop
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
# prepare_model_for_kbit_training - needed if QLoRA is used
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# utils.chdir_in_project("LORA_TRAINING/")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Phase 1: Speaker Characteristic Injection Fine-Tuning with LoRA/QLoRA")

    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="The model ID from Hugging Face to use as the base model."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False, # Changed to False as it can be provided via dict
        help="Path to the directory containing your data files (e.g., train.jsonl, dev.jsonl)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="FINETUNING/PHASE1A/",
        required=False, # Changed to False as it can be provided via dict
        help="Directory to save the fine-tuned LoRA adapter and training checkpoints."
    )
    parser.add_argument(
        "--use_qlora",
        type=lambda x: (str(x).lower() in ['true', '1', 't']),  # Allows 'True', 'true', '1', etc.
        default=True,
        help="Set to False to use standard LoRA (16-bit) instead of QLoRA (4-bit)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size. LaERC-S used 8 for Phase 1."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate. LaERC-S used 2e-4."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1536,
        help="Maximum sequence length for truncation. You specified 1536."
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (r)."
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients before updating weights."
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of evaluation steps with no improvement before stopping."
    )

    # Add other arguments if needed (e.g., warmup_ratio)
    return parser.parse_args()


def load_and_prepare_data(tokenizer, max_seq_length):
    data_path = os.path.join(utils.PROJECT_PATH, "TRAINING_DATA/PHASE1/IEMOCAP/")

    data_files = {
        "train": os.path.join(data_path, "train.jsonl"),
        "dev": os.path.join(data_path, "dev.jsonl"),
        "test": os.path.join(data_path, "test.jsonl")
    }

    print(f"Loading dta from {data_path}...")
    raw_datasets = load_dataset("json", data_files=data_files)


    def tokenize_function(batch):
        """
        Tokenizes a batch of data, formats the prompt, and masks labels/
        The model is trained only on the 'output' part of the text.
        """
        prompts_with_output = []
        prompt_lengths = []

        # Get data from the batch
        inputs_lists = batch['inputs']
        outputs = batch['output']

        for i in range(len(outputs)):

            # 1. Format the prompt
            prompt = SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE.format(**inputs_lists[i])

            # 2. Tokenize the prompt only to get its length
            # We add special tokens (like BOS) to get the tru starting length
            prompt_token_ids = tokenizer(prompt, add_special_tokens=True).input_ids
            prompt_len = len(prompt_token_ids)
            prompt_lengths.append(prompt_len)

            # 3. Create the full string for training (Prompt + Output + EOS)
            full_text = prompt + outputs[i] + tokenizer.eos_token
            prompts_with_output.append(full_text)

        # 4. Tokenize the full string with padding and truncation
        model_inputs = tokenizer(
            prompts_with_output,
            max_length=max_seq_length,
            padding=False,
            truncation=True,
            return_tensors=None
            # return_tensors="pt"
        )


        # 5. Create labels and mask the prompt
        labels_list = []
        for i in range(len(model_inputs.input_ids)):
            labels = model_inputs.input_ids[i].copy()  # Get the list
            prompt_len = prompt_lengths[i]
            mask_len = min(prompt_len, max_seq_length)
            labels[:mask_len] = [-100] * mask_len
            labels_list.append(labels)

        model_inputs["labels"] = labels_list  # Add the list of labels
        return model_inputs

    # Apply tokenization
    # We use batched=True for efficiency
    # remove columns is important to clean up old text columns
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True,
                                          batch_size=16,
                                          remove_columns=raw_datasets["train"].column_names)

    print(f"Data tokenization complete.")
    print(f"Train dataset size: {len(tokenized_datasets['train'])}")
    print(f"Dev dataset size: {len(tokenized_datasets['dev'])}")

    return tokenized_datasets




def main(config_dict=None):
    """
    Main function to run the training process.
    Accepts an optional config_dict to override command-line arguments.
    """
    print(f"{__file__} started!")
    if config_dict is None:
        # If no dict is provided, parse arguments from command line
        print("Parsing arguments from command line...")
        args = parse_arguments()
        # Check required args if parsing from command line
        # if not args.data_path:
        #     raise ValueError("--data_path is required when not providing a config_dict.")
        if not args.output_dir:
            raise ValueError("--output_dir is required when not providing a config_dict.")
    else:
        # If dict is provided, create args Namespace from it
        print("Loading arguments from config_dict...")
        # Add default values for any arguments potentially missing from the dict
        # This mirrors the defaults set in parse_arguments
        defaults = {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "use_qlora": True,
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 2e-4,
            "max_seq_length": 1536,
            "lora_r": 16,
            "lora_alpha": 32,
            "gradient_accumulation_steps": 2,
            # "data_path": None, # Ensure these exist even if None initially
            "output_dir": "FINETUNING/PHASE1/"
        }
        # Update defaults with provided config_dict, overwriting if keys exist
        defaults.update(config_dict)
        args = argparse.Namespace(**defaults)
        # Check required args if loading from dict
        # if not args.data_path:
        #     raise ValueError("data_path is required in the config_dict.")
        if not args.output_dir:
            raise ValueError("output_dir is required in the config_dict.")

    print("Effective Arguments:")
    print(vars(args)) # Print the final arguments being used
    args.output_dir = os.path.join(utils.PROJECT_PATH, args.output_dir)


    # --- 5a. Load Tokenizer ---
    print(f"Loading tokenizer for: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # **Crucial for Llama:** Set pad token = eos token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Pad sequences on the right

    # Use the efficient data collator that pads dynamically
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not Masked LM
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest"  # This ensures dynamic padding
    )

    # --- 5b. Load and Prepare Data ---
    # Call the function defined above
    tokenized_datasets = load_and_prepare_data(
        # args.data_path,
        tokenizer,
        args.max_seq_length
    )

    # --- 5c. Load Model (Conditional QLoRA or Standard LoRA) ---
    model_kwargs = {
        "device_map": "auto",           # Automatically distribute model across available GPUs
        "torch_dtype": torch.bfloat16,  # Use bfloat16 for computation/loading weights
    }
    bnb_config = None # Initialize bnb_config to None

    if args.use_qlora:
        print("Configuring QLoRA (4-bit quantization)...")
        # Configure 4-bit quantization using bitsandbytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,              # Load model in 4-bit
            bnb_4bit_quant_type="nf4",      # Use NF4 quantization type
            bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bfloat16 for speed
            bnb_4bit_use_double_quant=True, # Use double quantization for memory saving
        )
        model_kwargs["quantization_config"] = bnb_config # Add quantization config
        optimizer_type = "paged_adamw_8bit" # Optimizer for QLoRA
    else:
        print("Configuring standard LoRA (16-bit)...")
        optimizer_type = "adamw_torch" # Standard optimizer for 16-bit LoRA

    print(f"Loading base model: {args.model_id} {'with QLoRA' if args.use_qlora else 'in bfloat16'}")
    # Load the Llama 3.1 8B model, applying quantization config if enabled
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        **model_kwargs # Unpack the arguments (includes quantization_config if use_qlora is True)
    )

    # Prepare model for k-bit training *only* if using QLoRA
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # --- 5d. Configure LoRA ---
    # Define the LoRA configuration using PEFT (same for LoRA and QLoRA)
    peft_config = LoraConfig(
        r=args.lora_r,                  # LoRA rank (dimension of low-rank matrices)
        lora_alpha=args.lora_alpha,     # LoRA scaling factor
        lora_dropout=0.05,              # Dropout for LoRA layers
        # Specify which layers inside Llama 3.1 to apply LoRA to
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        task_type="CAUSAL_LM",          # Specify the task type
    )

    # Apply the LoRA configuration to the base model
    model = get_peft_model(model, peft_config)
    print("LoRA model created:")
    # Print the percentage of parameters that are actually trainable (should be small)
    model.print_trainable_parameters()

    # --- 5e. Configure Training Arguments ---
    # Set up all parameters for the Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,          # Where to save checkpoints
        num_train_epochs=args.epochs,        # Number of epochs
        per_device_train_batch_size=args.batch_size, # Batch size per GPU (train)
        per_device_eval_batch_size=args.batch_size,  # Batch size per GPU (eval)
        learning_rate=args.learning_rate,    # Learning rate
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Set optimizer based on whether QLoRA is used
        optim=optimizer_type,

        # Logging settings
        logging_dir=f"{args.output_dir}/logs", # Directory for logs (TensorBoard)
        logging_strategy="steps",            # Log metrics every N steps
        logging_steps=10,                    # Log every 10 steps

        # Evaluation settings
        eval_strategy="steps",               # Evaluate every N steps
        eval_steps=200,                      # Evaluate every 200 steps

        # Save settings
        save_strategy="steps",               # Save checkpoint every N steps
        save_steps=200,                      # Save every 100 steps
        save_total_limit=3,                  # Keep only the 3 best checkpoints to save disk space

        # Best model settings
        load_best_model_at_end=True,         # Load the best checkpoint at the end
        metric_for_best_model="eval_loss",   # Use validation loss to find the best model
        greater_is_better=False,             # Lower validation loss is better

        # Use bf16 if *not* using QLoRA (QLoRA handles precision internally)
        bf16=(not args.use_qlora),

        # Reporting
        report_to="tensorboard",             # Log results to TensorBoard

        # Reproducibility
        seed=42,
    )

    # Initialize Early Stopping Callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=0.0  # Any improvement counts
    )

    # --- 5f. Initialize Trainer ---
    # Create the Trainer object, which orchestrates the training loop
    trainer = Trainer(
        model=model,                         # The LoRA-adapted model
        args=training_args,                  # Training configuration
        train_dataset=tokenized_datasets["train"], # Training data
        eval_dataset=tokenized_datasets.get("dev"),  # Validation data (use .get to handle missing dev split)
        tokenizer=tokenizer,                 # Tokenizer (used for padding/saving)
        data_collator=data_collator,
        callbacks=[early_stopping]

    )

    # --- 5g. Start Training ---
    print("Starting training with early stopping...")
    print(f"Early stopping patience: {args.early_stopping_patience} evaluation steps")
    trainer.train()


    # --- 5h. Save Final Model ---
    # After training finishes, save the final trained LoRA adapter weights
    final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
    print(f"Training complete. Saving final LoRA adapter to {final_checkpoint_dir}")
    trainer.save_model(final_checkpoint_dir) # Saves only the adapter, not the base model

# --- 6. Script Entry Point ---
# Ensures the `main` function runs when the script is executed directly
# Now calls main() without arguments, relying on the default behavior (parse_arguments)
if __name__ == "__main__":
    main()


