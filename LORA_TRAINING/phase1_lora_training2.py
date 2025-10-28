"""
Stage 1: Speaker Characteristics Extraction Fine-tuning with LoRA
Fine-tunes Llama-3.1-8B-Instruct to extract speaker characteristics
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
from typing import List, Dict, Optional
import os
from dataclasses import dataclass
from prompts import SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE

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

    def prepare_dataset(self, data: List[Dict]) -> Dataset:
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
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(data)} examples")

            text = self.format_training_example(
                **item['inputs'],
                response=item['output']
            )
            formatted_texts.append(text)

        # Tokenize all texts
        print("Tokenizing...")
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=False,  # No padding during tokenization - will be padded dynamically in batches
            max_length=self.max_length,
            return_tensors=None
        )

        # Create labels (same as input_ids for causal LM)
        tokenized['labels'] = tokenized['input_ids'].copy()

        print(f"Dataset prepared with {len(tokenized['input_ids'])} examples")

        return Dataset.from_dict(tokenized)


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
            "save_steps": 500,
            "eval_steps": 500,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "fp16": True,
            "report_to": "none",  # Change to "wandb" for W&B logging
            "save_total_limit": 2,
            "logging_first_step": True,
            "gradient_checkpointing": True,
            "optim": "adamw_torch",
        }

        # Update with custom arguments
        default_args.update(kwargs)

        return TrainingArguments(**default_args)
