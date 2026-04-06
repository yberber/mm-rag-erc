# src/training

Fine-tuning and evaluation scripts for both training phases.

## Files

| File | Purpose |
|------|---------|
| `base_trainer.py` | Abstract base class with shared training logic (QLoRA, LoRA, tokenisation, Trainer setup) |
| `phase1_finetune.py` | Phase 1 trainer: speaker-characteristic injection |
| `phase2_finetune.py` | Phase 2 trainer: emotion recognition with RAG |
| `stage1_eval_parallel.py` | Batch evaluation for Phase 1 (BERTScore / ROUGE / BLEU) |
| `stage2_eval_parallel.py` | Batch evaluation for Phase 2 (accuracy / weighted F1) |
| `run_full_stage2_evaluations.py` | Runs `stage2_eval_parallel` for every checkpoint under a directory |

---

## Phase 1 fine-tuning

```bash
python -m src.training.phase1_finetune \
    --dataset both \
    --epochs 4 \
    --lora_r 16 \
    --use_qlora true
```

Trains on data from `data/training/stage1/MELD/` and `data/training/stage1/IEMOCAP/`.
The best checkpoint (lowest eval loss) is saved to
`artifacts/finetuning/STAGE1-DEFAULT-r16/<DATASET>/QLORA/final_checkpoint/`.

---

## Phase 2 fine-tuning

```bash
python -m src.training.phase2_finetune \
    --stage1_adapter_path artifacts/finetuning/STAGE1-DEFAULT-r16/COMBINED/QLORA/final_checkpoint \
    --dataset iemocap \
    --epochs 4 \
    --lora_r 16 \
    --use_qlora true
```

Loads the Phase 1 adapter and continues training on the emotion-recognition task.
Trains on data from `data/training/stage2/`.

---

## Common training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `both` | `meld`, `iemocap`, or `both` |
| `--model_name` | LLaMA-3.1-8B-Instruct | Base HuggingFace model ID |
| `--epochs` | 4 | Number of training epochs |
| `--batch_size` | 4 | Per-device batch size |
| `--learning_rate` | 2e-4 | Learning rate |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--use_qlora` | true | 4-bit QLoRA quantisation |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation steps |
| `--early_stopping_patience` | 3 | Eval steps with no improvement before stopping |
| `--output_dir` | auto-generated | Override the checkpoint output directory |
| `--seed` | 42 | Random seed |

---

## Phase 1 evaluation

```bash
python -m src.training.stage1_eval_parallel \
    --adapter_path artifacts/finetuning/STAGE1-DEFAULT-r16/BOTH/QLORA/final_checkpoint \
    --split dev \
    --dataset both \
    [--limit 100]
```

Evaluates the Phase 1 model's free-form text output using BERTScore, ROUGE-1/2/L, and BLEU.
Results are saved to `artifacts/eval/stage1/`.

---

## Phase 2 evaluation (single adapter)

```bash
python -m src.training.stage2_eval_parallel \
    --adapter_path <path/to/checkpoint> \
    --dataset iemocap \
    --split test \
    [--use_audio true] \
    [--use_rag true] \
    [--batch_size 16]
```

Reports accuracy and weighted F1 on the test split. Results are saved to
`artifacts/eval/stage2/` under a subdirectory determined by the ablation
condition.

### Ablation conditions

| `--use_audio` | `--use_rag` | Subdirectory |
|:---:|:---:|---|
| true | true | `baseline/` |
| false | true | `ablation/no_audio/` |
| true | false | `ablation/no_rag/` |
| false | false | `ablation/no_audio_rag/` |

---

## Phase 2 evaluation (all checkpoints)

```bash
python -m src.training.run_full_stage2_evaluations \
    --finetuning_root artifacts/finetuning/STAGE1_2-DEFAULT-r16 \
    --dataset both \
    --split test \
    [--skip_base true]
```

Discovers all `checkpoint-*` directories under `--finetuning_root` and
evaluates each one by spawning `stage2_eval_parallel` as a subprocess.

---

## Checkpoint resumption

Training automatically resumes from the latest checkpoint if one already
exists in the output directory.  If a `final_checkpoint/` directory is found,
training is skipped entirely.  Delete or rename the directory to re-train.
