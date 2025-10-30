"""
Stage 1: Speaker Characteristics Extraction Fine-tuning with LoRA
Fine-tunes Llama-3.1-8B-Instruct to extract speaker characteristics
"""
import datasets.arrow_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, load_dataset
import json
from typing import List, Dict, Optional
import os
from dataclasses import dataclass
import argparse

import utils
# from prompts import SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE


SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE = """Now you are an expert who is good at using commonsense for reasoning.
The following conversation noted between ’### ###’ involves several speakers.
###
{history}
###
Based on the above historical utterances, please use commonsense to infer the reaction of potential listeners in < "{speaker_id}" : "{utterance}" >, said with < {audio_features} >.
Output no more than 10 words:
"""

import os
os.environ["WANDB_PROJECT"] = "stage1-speaker-extraction"
# ============================================================================
# Data Preparation for Stage 1
# ============================================================================

class Stage1DataProcessor:
    """Prepare data for Stage 1: Speaker characteristics extraction"""



    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_training_example(self, history: str, speaker_id: str,
                                utterance: str, audio_features: str,
                                response: str) -> str:
        """
        Format a single training example

        Args:
            history: Conversation history
            speaker_id: ID of the speaker
            utterance: Current utterance
            audio_features: Audio feature description
            response: Ground truth speaker characteristics (label)

        Returns:
            Full formatted text for training with EOS token
        """
        prompt = SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE.format(
            history=history,
            speaker_id=speaker_id,
            utterance=utterance,
            audio_features=audio_features
        )

        # Combine prompt and response for causal LM training
        # Add EOS token to mark the end of the sequence
        full_text = prompt + response + self.tokenizer.eos_token
        return full_text


    def prepare_dataset2(self, data: Dataset) -> Dataset:
        """
                Prepare HuggingFace Dataset from raw data.

                This version correctly masks the prompt tokens in the labels
                and uses return_tensors=None to feed lists to the collator.
                """
        print(f"Processing {len(data)} examples...")

        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for idx, item in enumerate(data):
            if idx % 500 == 0:
                print(f"  Processed {idx}/{len(data)} examples")

            # 1. Format prompt and response separately
            prompt = SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE.format(
                **item['inputs']
            )
            response_text = item['output'] + self.tokenizer.eos_token

            # 2. Tokenize the prompt *without special tokens* to get its length
            prompt_encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors=None)
            prompt_len = len(prompt_encoded['input_ids'])

            # 3. Tokenize the full combined text
            full_encoded = self.tokenizer(
                prompt + response_text,
                add_special_tokens=True,  # Adds BOS token
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None  # <-- THE FIX: Ensure output is lists
            )

            # 4. Create labels by copying input_ids
            labels = full_encoded['input_ids'].copy()

            # 5. Mask all prompt tokens (and the BOS token) by setting to -100
            # We mask 1 (for BOS) + prompt_len
            mask_len = 1 + prompt_len

            # Safety check: if the sequence was truncated *within* the prompt,
            # mask the entire thing.
            if mask_len >= len(labels):
                mask_len = len(labels)

            labels[0:mask_len] = [-100] * mask_len

            # Ensure all lists being added have the same length
            if len(full_encoded['input_ids']) != len(labels):
                # This should never happen with this logic, but good to check
                print(f"Warning: Mismatch in length for sample {idx}. Skipping.")
                continue

            all_input_ids.append(full_encoded['input_ids'])
            all_attention_masks.append(full_encoded['attention_mask'])
            all_labels.append(labels)  # Add the new labels

        print(f"Dataset prepared with {len(all_input_ids)} examples")

        # Create dataset
        return Dataset.from_dict({
            'input_ids': all_input_ids,
            'attention_mask': all_attention_masks,
            'labels': all_labels  # <-- Pass the correct labels to the Trainer
        })


    def prepare_dataset(self, data: Dataset) -> Dataset:
        """
        Prepare HuggingFace Dataset from raw data

        Args:
            data: List of dictionaries with keys:
                - history: str
                - speaker_id: str
                - utterance: str
                - audio_features: str
                - response: str (ground truth)

        Returns:
            HuggingFace Dataset ready for training
        """
        print(f"Processing {len(data)} examples...")

        formatted_texts = []
        for idx, item in enumerate(data):
            if idx % 500 == 0:
                print(f"  Processed {idx}/{len(data)} examples")

            text = self.format_training_example(
                **item['inputs'],
                response=item['output']
            )
            formatted_texts.append(text)

        # Tokenize all texts
        print("Tokenizing...")

        # Tokenize each example individually to avoid batching issues
        all_input_ids = []
        all_attention_masks = []

        for text in formatted_texts:
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            all_input_ids.append(encoded['input_ids'])
            all_attention_masks.append(encoded['attention_mask'])

        print(f"Dataset prepared with {len(all_input_ids)} examples")

        # Create dataset
        return Dataset.from_dict({
            'input_ids': all_input_ids,
            'attention_mask': all_attention_masks,
            # 'labels': all_labels
        })


# ============================================================================
# Stage 1 Trainer
# ============================================================================

class Stage1Trainer:
    """Main trainer for Stage 1"""

    def __init__(self,
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 use_4bit: bool = False):
        """
        Initialize Stage 1 trainer

        Args:
            model_name: Base model to fine-tune
            use_4bit: Use 4-bit quantization (requires bitsandbytes)
        """
        self.model_name = model_name
        self.use_4bit = use_4bit

        print("=" * 80)
        print("Initializing Stage 1 Trainer")
        print(f"Base Model: {model_name}")
        print(f"4-bit Quantization: {use_4bit}")
        print("=" * 80)

        # Load tokenizer
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")

    def get_lora_config(self) -> LoraConfig:
        """Get LoRA configuration"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA rank
            lora_alpha=32,  # Scaling parameter
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],  # Target all attention and FFN layers in Llama
            bias="none",
        )

    def load_model(self):
        """Load and prepare model with LoRA"""
        print("\nLoading base model...")

        if self.use_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        print("Base model loaded successfully")

        # Apply LoRA
        print("\nApplying LoRA...")
        lora_config = self.get_lora_config()
        model = get_peft_model(model, lora_config)

        # Enable gradient checkpointing compatibility
        model.enable_input_require_grads()

        # Print trainable parameters
        model.print_trainable_parameters()

        return model

    def get_training_arguments(self, output_dir: str, **kwargs) -> TrainingArguments:
        """
        Get training arguments

        Args:
            output_dir: Directory to save model checkpoints
            **kwargs: Override default arguments
        """
        default_args = {
            "output_dir": output_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 100,
            "eval_steps": 100,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "fp16": True,
            # "report_to": ["wandb", "tensorboard"],
            "report_to": None,
            "save_total_limit": 2,
            "logging_first_step": True,
            "gradient_checkpointing": True,
            "optim": "adamw_torch",
        }

        # Update with custom arguments
        default_args.update(kwargs)

        return TrainingArguments(**default_args)

    def train(self,
              train_data: Dataset,
              eval_data: Dataset,
              output_dir: str = "./stage1_output",
              **training_kwargs):
        """
        Train Stage 1 model

        Args:
            train_data: Training data
            eval_data: Evaluation data
            output_dir: Directory to save model
            **training_kwargs: Additional training arguments

        Returns:
            Path to saved model
        """
        print("\n" + "=" * 80)
        print("STARTING STAGE 1 TRAINING")
        print("=" * 80)

        # Load model
        model = self.load_model()

        # Prepare datasets
        print("\nPreparing training dataset...")
        processor = Stage1DataProcessor(self.tokenizer)
        train_dataset = processor.prepare_dataset2(train_data)

        print("\nPreparing evaluation dataset...")
        eval_dataset = processor.prepare_dataset2(eval_data)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're doing causal LM, not masked LM
        )

        # Training arguments
        training_args = self.get_training_arguments(output_dir, **training_kwargs)

        print("\n" + "=" * 80)
        print("Training Configuration:")
        print(f"  Output directory: {training_args.output_dir}")
        print(f"  Num epochs: {training_args.num_train_epochs}")
        print(f"  Batch size per device: {training_args.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(
            f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"  Learning rate: {training_args.learning_rate}")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Evaluation samples: {len(eval_dataset)}")
        print("=" * 80)

        # Check for existing checkpoints for resumption
        checkpoint = None
        if os.path.isdir(output_dir):
            checkpoints = [
                os.path.join(output_dir, d)
                for d in os.listdir(output_dir)
                if d.startswith('checkpoint-')
            ]
            if checkpoints:
                # Get the latest checkpoint
                checkpoint = max(checkpoints, key=os.path.getctime)
                print("\n" + "=" * 80)
                print(f"FOUND EXISTING CHECKPOINT: {checkpoint}")
                print("Resuming training from this checkpoint...")
                print("=" * 80)

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train (with automatic resume if checkpoint exists)
        if checkpoint:
            print(f"\nResuming training from: {checkpoint}")
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        else:
            print("\nStarting training from scratch...")
            train_result = trainer.train()

        # Save model
        print("\n" + "=" * 80)
        print("Training completed! Saving model...")
        print("=" * 80)

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        print(f"\nModel saved to: {output_dir}")
        print("\nFinal Training Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Evaluate
        print("\nRunning final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        print("\nFinal Evaluation Metrics:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 80)
        print("STAGE 1 TRAINING COMPLETE!")
        print(f"Model saved to: {output_dir}")
        print("You can now use this model for Stage 2 training")
        print("=" * 80)

        return output_dir

def parse_arguments():
    parser = argparse.ArgumentParser(description='Stage 1: Speaker Characteristics Fine-tuning')
    parser.add_argument('--data_dir', type=str, default="TRAINING_DATA/PHASE1/IEMOCAP/", help='Path to Training data')
    parser.add_argument('--output_dir', type=str, default='FINETUNING/PHASE1c/',
                        help='Output directory for model')
    parser.add_argument('--model_name', type=str,
                        default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Base model name')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    # parser.add_argument('--use_4bit', type=lambda x: str(x).lower() == 'true', help='Use 4-bit quantization')
    parser.add_argument('--use_4bit',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Use 4-bit quantization (QLoRA). Default: True. Disable with --no-use_4bit')

    args = parser.parse_args()
    return args

def load_iemocap_dataset():
    data_path = os.path.join(utils.PROJECT_PATH, "TRAINING_DATA/PHASE1/IEMOCAP/")

    data_files = {
        "train": os.path.join(data_path, "train.jsonl"),
        "dev": os.path.join(data_path, "dev.jsonl"),
    }
    raw_datasets = load_dataset("json", data_files=data_files)
    return raw_datasets["train"], raw_datasets["dev"]

# ============================================================================
# Main Execution
# ============================================================================

def main(config_dict=None):

    if config_dict is None:
        print("Parsing arguments from command line...")
        args = parse_arguments()
    else:
        print("Loading arguments from config_dict...")
        defaults = {
            "data_dir": "TRAINING_DATA/PHASE1/IEMOCAP/",
            "output_dir": "FINETUNING/PHASE1b/",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4,
            "use_4bit": True
        }
        defaults.update(config_dict)
        args = argparse.Namespace(**defaults)

    print("Effective Arguments:")
    print(vars(args)) # Print the final arguments being used


    args.data_dir = os.path.join(utils.PROJECT_PATH, args.data_dir)
    args.output_dir = os.path.join(utils.PROJECT_PATH, args.output_dir)

    # Load from directory structure (TRAINING_DATA/PHASE1/IEMOCAP/)
    print(f"Loading data from directory: {args.data_dir}")
    train_data, eval_data = load_iemocap_dataset()


    print(f"\nLoaded {len(train_data)} training examples")
    print(f"Loaded {len(eval_data)} evaluation examples")

    # Initialize trainer
    trainer = Stage1Trainer(
        model_name=args.model_name,
        use_4bit=args.use_4bit
    )

    # Train
    output_path = trainer.train(
        train_data=train_data,
        eval_data=eval_data,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    print("\n" + "=" * 80)
    print("ALL DONE!")
    print(f"Stage 1 model saved to: {output_path}")
    print("\nNext steps:")
    print(f"  1. Use the model at: {output_path}")
    print("  2. Run Stage 2 training with this model as the base")
    print("=" * 80)


if __name__ == "__main__":
    main()


