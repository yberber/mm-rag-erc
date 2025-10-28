from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# ============================================================================
# LoRA Training Configuration
# ============================================================================

def get_lora_config(stage: int = 1) -> LoraConfig:
    """
    Get LoRA configuration for fine-tuning

    Args:
        stage: 1 or 2, for different stages
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha scaling
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Llama 3.1 attention and MLP modules
        bias="none",
    )


def get_training_args(stage: int, output_dir: str) -> TrainingArguments:
    """Get training arguments for each stage"""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,  # Use mixed precision
        report_to="none",  # Change to "wandb" if you use Weights & Biases
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )