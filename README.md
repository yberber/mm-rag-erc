# Multimodal RAG-based Emotion Recognition in Conversation

A multimodal, retrieval-augmented approach to Emotion Recognition in Conversation (ERC) using LLaMA-3.1-8B-Instruct, evaluated on the MELD and IEMOCAP benchmark datasets.

## Overview

The system identifies the predominant emotion of an utterance within a multi-speaker conversation. It combines three key ideas:

1. **Multimodal audio-to-text abstraction** — raw audio features (pitch, intensity, speech rate) are extracted with Parselmouth/Praat and discretised into categorical descriptions (e.g. *"low speech rate, medium pitch, high intensity"*) that are fed directly into the text prompt.

2. **Two-phase fine-tuning** — the base LLM is first trained to generate commonsense speaker-state descriptions (Phase 1), then fine-tuned on the emotion-recognition task with RAG demonstrations (Phase 2).

3. **Retrieval-augmented in-context learning** — at inference time, the most semantically similar training examples are retrieved from a ChromaDB vector store and inserted as demonstrations into the prompt.

---

## Directory Structure

```
mm-rag-erc/
├── data/
│   ├── benchmark/          # Processed benchmark CSVs (MELD, IEMOCAP)
│   ├── processed/          # Optional: intermediate processed datasets
│   └── training/
│       ├── stage1/         # Phase 1 JSONL training files
│       └── stage2/         # Phase 2 JSONL training files
├── artifacts/
│   ├── finetuning/         # LoRA adapter checkpoints
│   ├── eval/               # Evaluation results (stage1, stage2)
│   ├── speaker_chars/      # Phase 1 speaker-characteristic JSON files
│   └── vectorstores/
│       ├── db/             # ChromaDB vector store directories
│       ├── caches/         # Pre-computed similarity cache JSON files
│       └── utterance_index_mapping.json
├── src/
│   ├── config/             # Paths and constants
│   ├── data_processing/    # MELD and IEMOCAP data preparation
│   ├── helper/             # Prompts, utilities, dataset builder
│   ├── vectorstore/        # Vector store creation and caching
│   ├── training_data_creation/  # Phase 1 and Phase 2 dataset assembly
│   └── training/           # Fine-tuning and evaluation scripts
└── requirements.txt
```

---

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (required for training and evaluation)
- [ffmpeg](https://ffmpeg.org/) on `PATH` (required for MELD MP4 → WAV conversion)
- [Ollama](https://ollama.com/) running locally (optional; required only if using `model_id=0` for Phase 1 annotation)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Edit `src/config/paths.py` and set the two raw-data paths at the top of the file:

```python
MELD_RAW_DATA_DIR    = "/path/to/MELD.Raw"
IEMOCAP_RAW_DATA_DIR = "/path/to/IEMOCAP_full_release"
```

All other paths are derived automatically from the project root.

---

## Pipeline

Run the steps below **in order**. All scripts are run as Python modules from the project root.

### Step 1 — Initialise benchmark CSVs

Parse the raw dataset files into standardised CSVs:

```bash
python -m src.data_processing.meld.init_meld_dataset
python -m src.data_processing.iemocap.init_iemocap_dataset
```

Output: `data/benchmark/meld/meld_erc_init.csv` and `data/benchmark/iemocap/iemocap_erc_init.csv`.

### Step 2 — Convert MELD audio (MP4 → WAV)

```bash
python -m src.data_processing.meld.convert_meld_mp4_to_wav
```

Requires `ffmpeg`. Writes WAV files to `<MELD_RAW_DATA_DIR>/audio/`.

### Step 3 — Extract acoustic features

```bash
python -m src.data_processing.meld.add_audio_features_meld
python -m src.data_processing.iemocap.add_audio_features_iemocap
```

Adds pitch, intensity, articulation rate, and HNR columns.  Output: `*_erc_with_audio.csv`.

### Step 4 — Categorise acoustic features

```bash
python -m src.data_processing.meld.extend_meld_categories
python -m src.data_processing.iemocap.extend_iemocap_categories
```

Adds `pitch_level`, `intensity_level`, `rate_level` (low/medium/high) and a unified `mapped_emotion` column.  Output: `*_erc_final.csv`.

### Step 5 — Build vector stores

```bash
python -m src.vectorstore.generators.run_vectorstore_generators
```

Creates single, flow, and hybrid ChromaDB collections under `artifacts/vectorstores/db/`.

### Step 6 — Create utterance-index metadata

```bash
python -m src.vectorstore.caching.create_idx_to_metadata
```

Writes `artifacts/vectorstores/utterance_index_mapping.json`.

### Step 7 — Cache similarity lookups

```bash
python -m src.vectorstore.caching.cache_similar_utterances --top_n 10
```

Pre-computes top-10 similar utterances for every entry in both datasets.

### Step 8 — Generate Phase 1 training data

First, generate speaker-characteristic annotations:

```bash
python -m src.training_data_creation.phase1.generate_speaker_characteristics
```

Then assemble the JSONL training files:

```bash
python src/training_data_creation/phase1/build_dataset.py
```

Output: `data/training/stage1/MELD/` and `data/training/stage1/IEMOCAP/`.

### Step 9 — Phase 1 fine-tuning

```bash
python -m src.training.phase1_finetune \
    --dataset both --epochs 4 --lora_r 16 --use_qlora true
```

The best checkpoint is saved to `artifacts/finetuning/STAGE1-DEFAULT-r16/`.

### Step 10 — Generate Phase 2 training data

```bash
python src/training_data_creation/phase2/build_dataset.py
```

Output: `data/training/stage2/MELD/` and `data/training/stage2/IEMOCAP/`.

### Step 11 — Phase 2 fine-tuning

```bash
python -m src.training.phase2_finetune \
    --stage1_adapter_path artifacts/finetuning/STAGE1-DEFAULT-r16/COMBINED/QLORA/final_checkpoint \
    --dataset iemocap --epochs 4 --lora_r 16 --use_qlora true
```

### Step 12 — Evaluation

**Phase 2 evaluation (single adapter):**

```bash
python -m src.training.stage2_eval_parallel \
    --adapter_path <path/to/checkpoint> \
    --dataset iemocap --split test
```

**Phase 2 evaluation (all checkpoints under a directory):**

```bash
python -m src.training.run_full_stage2_evaluations \
    --finetuning_root artifacts/finetuning/STAGE1_2-DEFAULT-r16 \
    --dataset both --split test
```

**Phase 1 evaluation:**

```bash
python -m src.training.stage1_eval_parallel \
    --adapter_path <path/to/checkpoint> \
    --split dev --dataset both
```

---

## Ablation Studies

`stage2_eval_parallel` supports four prompt conditions via `--use_audio` and `--use_rag` flags:

| Condition         | `--use_audio` | `--use_rag` |
|-------------------|:-------------:|:-----------:|
| Full (baseline)   | true          | true        |
| No audio          | false         | true        |
| No RAG            | true          | false       |
| No audio + no RAG | false         | false       |

Results are saved in separate subdirectories (`baseline/`, `ablation/no_audio/`, etc.) under `artifacts/eval/stage2/`.
