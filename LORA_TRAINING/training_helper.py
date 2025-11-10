import argparse
import os
import utils
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer,
                          EarlyStoppingCallback, DataCollatorForSeq2Seq, TrainingArguments)
from peft import get_peft_model, LoraConfig, PeftModel, prepare_model_for_kbit_training
import torch
import json
from datasets import load_dataset, concatenate_datasets, DatasetDict
import datetime



def get_base_parser():
    """Gets the base argument parser with shared hyperparameters."""
    # We use add_help=False to allow subclasses to add their own help
    parser = argparse.ArgumentParser(description="Base Trainer Argument Parser", add_help=False)

    # --- Path Arguments ---
    parser.add_argument(
        '--dataset', type=str, default="both", choices=['iemocap', 'meld', 'both'],
        help='Dataset to use. "both" will combine IEMOCAP and MELD.'
    )
    parser.add_argument(
        '--iemocap_data_path', type=str, default=None,
        help='Relative path to the IEMOCAP .json(l) directory.'
    )
    parser.add_argument(
        '--meld_data_path', type=str, default=None,
        help='Relative path to the MELD .json(l) directory.'
    )
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for model. If none, it will be generated automatically')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Base model name')
    parser.add_argument(
        "--stage1_adapter_path", type=str, required=False,
        help="Path to the saved Stage-1 LoRA/QLoRA adapter (folder with adapter_model.safetensors)."
    )
    parser.add_argument("--use_qlora", type=lambda x: (str(x).lower() in ['true', '1', 't']), default=True,
                        help="Set to False to use standard LoRA (16-bit) instead of QLoRA (4-bit).")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument("--max_seq_length", type=int, default=1536,
                        help="Maximum sequence length for truncation. You specified 1536.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients before updating weights.")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Number of evaluation steps with no improvement before stopping.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for AdamW.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--eval_save_steps", type=int, default=150, help="Evaluation and save steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser


class BaseTrainer:
    """
    Abstract base class for fine-tuning, handling all shared logic.
    Subclasses must implement:
    - _get_default_args()
    - get_prompt_template()
    - get_raw_datasets()
    """

    def __init__(self, config_dict=None, stage_id=1):
        self.stage_id = stage_id
        self.phase_name = f"Phase {stage_id}"
        self.config = self._process_config(config_dict)
        self.tokenizer = None
        self.model = None
        self.tokenized_datasets = None
        self.trainer = None
        self._log_config()
        self._set_seed()


    def _get_default_args(self):
        """
        Subclasses must override this to provide their specific
        default hyperparameters (e.g., learning_rate, epochs).
        """
        raise NotImplementedError("Subclass must implement _get_default_args")

    def _process_config(self, config_dict):
        """
        Loads config from a dictionary or parses from command line,
        prioritizing subclass defaults.
        """
        base_parser = get_base_parser()

        # 1. Get defaults from the specific subclass
        subclass_defaults = self._get_default_args()

        # 2. Set these defaults in the parser
        base_parser.set_defaults(**subclass_defaults)

        if config_dict is None:
            # 3a. No dict provided: Parse from command line
            print(f"Parsing {self.phase_name} arguments from command line...")
            args = base_parser.parse_args()
        else:
            # 3b. Dict provided: Use it to override defaults
            print(f"Loading {self.phase_name} arguments from config_dict...")
            defaults = vars(base_parser.parse_args(args=[]))
            defaults.update(config_dict)
            args = argparse.Namespace(**defaults)

        # --- Resolve Paths ---

        # Resolve data paths
        args.iemocap_data_path = os.path.join(utils.PROJECT_PATH, args.iemocap_data_path)
        args.meld_data_path = os.path.join(utils.PROJECT_PATH, args.meld_data_path)
        if args.stage1_adapter_path:
            args.stage1_adapter_path = os.path.join(utils.PROJECT_PATH, args.stage1_adapter_path)

        # Handle 'both' aliases
        # if args.dataset in ['both', 'general', 'union']:
        #     args.dataset = 'both'

        # Auto-generate output_dir if not provided
        if args.output_dir is None:
            stage_name = f"STAGE{self.stage_id}"
            if self.stage_id == 2 and args.stage1_adapter_path:
                stage_name = f"STAGE1_2"
            dataset_name = args.dataset.upper()
            lora_type = 'QLORA' if args.use_qlora else 'LORA'
            args.output_dir = os.path.join("FINETUNING", stage_name, dataset_name, lora_type)

        # Ensure output_dir is an absolute path
        args.output_dir = os.path.join(utils.PROJECT_PATH, args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

        return args

    def _log_config(self):
        """Prints the effective configuration."""
        print(f"--- {self.phase_name} Configuration ---")
        for key, value in vars(self.config).items():
            print(f"{key}: {value}")
        print("-----------------------------------")

    def save_config(self, save_directory, checkpoint):
        """Saves the final config namespace to the specified directory."""
        print(f"Saving training configuration to {save_directory}...")


        config_path = os.path.join(save_directory, "training_config.json")

        # Convert Namespace to dict for JSON serialization
        config_dict = vars(self.config)
        start_time = datetime.datetime.now()

        config_dict.update({
            "stage_id": self.stage_id,
            "phase_name": self.phase_name,
            "checkpoint": checkpoint,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S")})

        print("Configuration saved successfully.")

        utils.dump_json_test_result(config_dict, config_path, add_datetime_to_filename=True)
        # Handle non-serializable types if any (e.g., Namespace)
        # For this config, all types should be serializable
        # try:
        #     with open(config_path, 'w') as f:
        #         json.dump(config_dict, f, indent=4)
        #     print("Configuration saved successfully.")
        # except Exception as e:
        #     print(f"Error saving configuration: {e}")

    def _set_seed(self):
        """Sets the random seed for reproducibility."""
        seed = self.config.seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Set seeds for numpy and random if they are used
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import random
            random.seed(seed)
        except ImportError:
            pass
        print(f"Set random seed to {seed}")

    def load_tokenizer(self):
        """Loads and configures the tokenizer."""
        print(f"Loading tokenizer for: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def load_and_prepare_data(self):
        """
        Abstract method for data loading.
        Subclasses must override this to load and tokenize their specific data.
        """
        raise NotImplementedError("Subclass must implement load_and_prepare_data")

    def get_prompt_template(self):
        """
        Abstract method
        """
        raise NotImplementedError("Subclass must implement get_prompt_template")


    def get_raw_datasets(self):
        """
        Abstract method
        """
        raise NotImplementedError("Subclass must implement get_raw_datasets")

    def make_tokenize_fn(self):
        """
        Create a tokenize function that:
          - formats prompts
          - tokenizes prompt+output
          - masks prompt tokens in the labels (for causal LM training)
        """
        PROMPT = self.get_prompt_template()
        def tokenize_function(batch):
            prompts_with_output = []
            prompt_lengths = []

            inputs_lists = batch["inputs"]
            outputs = batch["output"]

            for i in range(len(outputs)):
                # 1. Format the prompt
                prompt = PROMPT.format(**inputs_lists[i])

                # 2. Tokenize the prompt only to get its length
                prompt_token_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
                prompt_len = len(prompt_token_ids)
                prompt_lengths.append(prompt_len)

                # 3. Full training string = prompt + output + EOS
                full_text = prompt + outputs[i] + self.tokenizer.eos_token
                prompts_with_output.append(full_text)

            # 4. Tokenize the full string
            model_inputs = self.tokenizer(
                prompts_with_output,
                max_length=self.config.max_seq_length,
                padding=False,
                truncation=True,
                return_tensors=None,
            )

            # 5. Create labels and mask the prompt part
            labels_list = []
            for i in range(len(model_inputs["input_ids"])):
                labels = model_inputs["input_ids"][i].copy()
                prompt_len = prompt_lengths[i]
                mask_len = min(prompt_len, self.config.max_seq_length)
                labels[:mask_len] = [-100] * mask_len
                labels_list.append(labels)

            model_inputs["labels"] = labels_list
            return model_inputs

        return tokenize_function

    def prepare_tokenized_datasets(self):
        """
        High-level helper:
          1. Build raw datasets (MELD/IEMOCAP/both)
          2. Map tokenization + label masking
          3. Print sizes and return tokenized DatasetDict
        """
        # 1) build raw datasets
        raw_datasets = self.get_raw_datasets()

        # 2) get tokenize function bound to tokenizer + seq length
        tokenize_function = self.make_tokenize_fn()

        # 3) apply tokenization
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            batch_size=16,
            remove_columns=raw_datasets["train"].column_names,
        )

        print("Data tokenization complete.")
        print(f"Train dataset size: {len(tokenized_datasets['train'])}")
        print(f"Dev dataset size:   {len(tokenized_datasets['dev'])}")

        # longest is 1472 for stage 2, 1020 for stage 2
        # longest = 0
        # input_ids = None
        # cnt = 0
        # for split in tokenized_datasets.keys():
        #     for dev in tokenized_datasets[split]:
        #         assert  len(dev['input_ids']) == len(dev['attention_mask']) == len(dev['labels'])
        #         if len(dev['input_ids']) > longest:
        #             cnt += 1
        #             longest = len(dev['input_ids'])
        #             input_ids = dev['input_ids']
        #             print(f"new longest {longest}")
        # self.tokenizer.decode(input_ids)



        self.tokenized_datasets = tokenized_datasets
        return tokenized_datasets

    def _load_data_from_dirs(self, data_dirs):
        """Helper to load and combine datasets from one or more directories."""
        all_datasets = []
        for data_dir in data_dirs:
            if not data_dir:
                continue

            data_files = {
                "train": os.path.join(data_dir, "train.jsonl"),
                "dev": os.path.join(data_dir, "dev.jsonl"),
            }

            existing_data_files = {}
            for split, path in data_files.items():
                if os.path.exists(path):
                    existing_data_files[split] = path
                else:
                    print(f"Warning: Data file not found for split '{split}' at {path}")


            print(f"Loading data from {data_dir}...")
            all_datasets.append(load_dataset("json", data_files=existing_data_files))

        # Combine datasets if multiple were loaded
        if len(all_datasets) > 1:
            print("Combining datasets...")
            train_sets = [ds['train'] for ds in all_datasets if 'train' in ds]
            dev_sets = [ds['dev'] for ds in all_datasets if 'dev' in ds]

            combined_ds = DatasetDict()
            if train_sets:
                combined_ds['train'] = concatenate_datasets(train_sets)
            if dev_sets:
                combined_ds['dev'] = concatenate_datasets(dev_sets)

            if 'test' in all_datasets[0]:
                combined_ds['test'] = all_datasets[0]['test']

            return combined_ds

        return all_datasets[0]

    def get_data_dirs(self):
        data_dirs = []
        if self.config.dataset in ['iemocap', 'both']:
            data_dirs.append(self.config.iemocap_data_path)
        if self.config.dataset in ['meld', 'both']:
            data_dirs.append(self.config.meld_data_path)
        return data_dirs


    def load_model(self):
        """Loads the base model, applies QLoRA (if enabled), and loads adapters."""
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }
        self.optimizer_type = "adamw_torch"

        if self.config.use_qlora:
            print("Configuring QLoRA (4-bit quantization)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            self.optimizer_type = "paged_adamw_8bit"
        else:
            print("Configuring standard LoRA (16-bit)...")

        print(f"Loading base model: {self.config.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )

        if self.config.use_qlora:
            model = prepare_model_for_kbit_training(model)

        # --- Adapter Loading Logic ---
        if self.stage_id == 2 and self.config.stage1_adapter_path:
            # Phase 2: Load the trained Phase 1 adapter to continue training
            print(f"Loading adapter weights from: {self.config.stage1_adapter_path}")
            self.model = PeftModel.from_pretrained(model, self.config.stage1_adapter_path, is_trainable=True)
            print("Phase 1 adapter loaded for continued training.")
        else:
            # Phase 1: Create a new adapter
            print("Creating new adapter for Phase 1 training.")
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(model, peft_config)

        print(f"Final model for {self.phase_name}:")
        self.model.print_trainable_parameters()

    def configure_trainer(self):
        """Configures the TrainingArguments and Trainer."""
        if self.tokenizer is None:
            self.load_tokenizer()

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding="longest"  # Dynamic padding
        )

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,

            optim=self.optimizer_type,

            logging_dir=f"{self.config.output_dir}/logs",
            logging_strategy="steps",
            logging_steps=10,

            eval_strategy="steps",
            eval_steps=self.config.eval_save_steps,

            save_strategy="steps",
            save_steps=self.config.eval_save_steps,
            save_total_limit=5,

            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            bf16=(not self.config.use_qlora),
            report_to=["wandb", "tensorboard"],
            seed=self.config.seed,

        )

        callbacks = []
        if self.config.early_stopping_patience is not None and self.config.early_stopping_patience > 0:
            print(f"Enabling EarlyStopping with patience={self.config.early_stopping_patience}")
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
            )
            callbacks.append(early_stopping)
        else:
            print("EarlyStopping is disabled.")

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets.get("dev"),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks
        )

    def run(self):
        """Executes the full training pipeline."""
        try:
            self.load_tokenizer()
            # self.load_and_prepare_data()
            self.prepare_tokenized_datasets()
            self.load_model()
            self.configure_trainer()



            print(f"--- Starting {self.phase_name} Training ---")
            print(f"Early stopping patience: {self.config.early_stopping_patience} evaluation steps")

            # Check if a final checkpoint already exists
            final_checkpoint_dir = os.path.join(self.config.output_dir, "final_checkpoint")
            if os.path.exists(final_checkpoint_dir):
                print(f"Final checkpoint already exists at {final_checkpoint_dir}. Skipping training.")
                print("If you want to re-train, please delete this folder or change --output_dir.")
                return

            # Check for existing checkpoints for resumption
            checkpoint = None
            if os.path.isdir(self.config.output_dir):
                checkpoints = [
                    os.path.join(self.config.output_dir, d)
                    for d in os.listdir(self.config.output_dir)
                    if d.startswith('checkpoint-')
                ]
                if checkpoints:
                    # Get the latest checkpoint
                    checkpoint = max(checkpoints, key=os.path.getctime)
                    print("\n" + "=" * 80)
                    print(f"FOUND EXISTING CHECKPOINT: {checkpoint}")
                    print("Resuming training from this checkpoint...")
                    print("=" * 80)

            self.save_config(self.config.output_dir, checkpoint)

            # Train (with automatic resume if checkpoint exists)
            if checkpoint:
                print(f"\nResuming training from: {checkpoint}")
                train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
            else:
                print("\nStarting training from scratch...")
                train_result = self.trainer.train()

            # ----- Save metrics and trainer state -----
            print("\nCollecting and saving training metrics and state...")
            metrics = train_result.metrics
            # Log to console / integrated loggers (e.g., wandb, tensorboard)
            self.trainer.log_metrics("train", metrics)
            # Save metrics to JSON in output_dir (e.g., train_results.json)
            self.trainer.save_metrics("train", metrics)
            # Save full trainer state (optimizer, scheduler, rng, trainer_state.json, etc.)
            self.trainer.save_state()

            # ----- Save final model (for inference / later phases) -----
            print(f"\nTraining complete. Saving final adapter to {final_checkpoint_dir}")
            os.makedirs(final_checkpoint_dir, exist_ok=True)
            self.trainer.save_model(final_checkpoint_dir)

            print(f"{self.phase_name} finished successfully.")

            print(f"{self.phase_name} finished successfully.")

        except Exception as e:
            print(f"Error occurred during {self.phase_name} training: {e}")
            import traceback
            traceback.print_exc()
            print(f"{self.phase_name} finished with errors.")
