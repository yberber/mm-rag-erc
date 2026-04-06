# src/training_data_creation

Scripts that assemble the JSONL training files for Phase 1 and Phase 2 fine-tuning.

## Sub-packages

```
training_data_creation/
├── phase1/
│   ├── generate_speaker_characteristics.py  # Entry point: runs annotation
│   ├── parallel_character_extraction.py     # Core async annotation engine
│   └── build_dataset.py                     # Assemble JSONL from annotations
└── phase2/
    └── build_dataset.py                     # Assemble Phase 2 JSONL
```

---

## Phase 1 — Speaker-characteristic injection

### Purpose

Phase 1 training teaches the LLM to generate short commonsense descriptions
of a speaker's state, given the conversation history and audio features.
Three prompt variants are available for ablation:

| `prompt_type` | Description generated |
|--------------|----------------------|
| `default` | Likely reaction of potential listeners |
| `alt1` | Speaker's mental state or behaviour |
| `alt2` | Speaker's intention or reason |
| `default-no-audio` | Listener reaction (audio features omitted) |

### Step 1 — Generate annotations

Edit the `config` dict in `generate_speaker_characteristics.py` to set the
desired model, datasets, prompt type, and splits, then run:

```bash
python -m src.training_data_creation.phase1.generate_speaker_characteristics
```

Key config options:

| Key | Default | Description |
|-----|---------|-------------|
| `dataset_name` | `['meld']` | Datasets to annotate |
| `model_id` | `0` | 0=Ollama, 1=HuggingFace, 2=Gemini-flash, 3=Gemini-lite |
| `prompt_type` | `'default'` | Prompt variant (see table above) |
| `splits` | `['train','dev']` | Splits to annotate (test is never used) |
| `max_k` | `20` | Conversation-history window size |
| `limit` | `10` | Set `None` for a full run; use a small integer for testing |

Output JSON files are saved to `artifacts/speaker_chars/` with names encoding the dataset, model, prompt type, and size.

Alternatively, run the annotation engine directly from the command line:

```bash
python -m src.training_data_creation.phase1.parallel_character_extraction \
    --dataset_name meld --model_id 0 --prompt_type default \
    --splits train dev --max_k 20
```

### Step 2 — Build JSONL training files

```bash
python src/training_data_creation/phase1/build_dataset.py
```

Reads the largest available annotation JSON for each dataset, filters out
rows with missing audio features, and writes:

- `data/training/stage1/MELD/train.jsonl`
- `data/training/stage1/MELD/dev.jsonl`
- `data/training/stage1/IEMOCAP/train.jsonl`
- `data/training/stage1/IEMOCAP/dev.jsonl`

Each JSONL line contains `inputs` (prompt variables), `iden` (utterance ID),
and `output` (the generated speaker-characteristic text — the training target).

---

## Phase 2 — Emotion recognition

### Purpose

Phase 2 training teaches the LLM to predict the emotion label for an
utterance given conversation history, audio features, and RAG demonstrations.

### Build JSONL training files

```bash
python src/training_data_creation/phase2/build_dataset.py
```

Uses the hybrid vector store (window size 7) with top-2 demonstrations and
detailed-example formatting. Edit `TRAINING_SET_CONFIGS` in the script to
change these settings.

Output:

- `data/training/stage2/MELD/{train,dev,test}.jsonl`
- `data/training/stage2/IEMOCAP/{train,dev,test}.jsonl`

Each JSONL line contains `input` (all prompt fields including demonstrations),
`target` (gold emotion label), and `idx`.

**Prerequisites:** Vector stores and similarity caches must exist
(see `src/vectorstore/README.md`).
