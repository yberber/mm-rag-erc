# src/helper

Shared utilities, prompt templates, and the Phase 2 dataset builder used across the project.

## Files

| File | Purpose |
|------|---------|
| `prompts.py` | All LangChain `PromptTemplate` objects for emotion recognition and Phase 1 speaker-characteristic extraction. |
| `utils.py` | Emotion label sets, I/O helpers, model loaders, DataFrame utilities, and miscellaneous functions. |
| `build_prompting_dataset.py` | Builds the Phase 2 prompting dataset by combining conversation history, audio features, and RAG demonstrations. |

---

## prompts.py

Defines every prompt template used in training, evaluation, and inference. Templates come in two forms:

- A raw string constant (`*_TEMPLATE`) for direct `.format()` calls.
- A `PromptTemplate` object (`*_PROMPT`) for use in LangChain chains.

### Emotion-recognition prompts

| Name | RAG | Audio |
|------|:---:|:-----:|
| `EMOTION_RECOGNITION_FINAL_PROMPT` | yes | yes |
| `EMOTION_RECOGNITION_FINAL_PROMPT_NO_RAG` | no | yes |
| `EMOTION_RECOGNITION_FINAL_PROMPT_NO_AUDIO` | yes | no |
| `EMOTION_RECOGNITION_FINAL_PROMPT_NO_AUDIO_RAG` | no | no |
| `GEMINI_EMOTION_RECOGNITION_PROMPT` | yes | yes |
| `CLAUDE_EMOTION_RECOGNITION_PROMPT` | yes | yes |
| `GPT5_EMOTION_RECOGNITION_PROMPT` | yes | yes |

### Speaker-characteristic prompts (Phase 1)

| Name | Description requested |
|------|----------------------|
| `SPEAKER_CHARACTERISTICS_EXTRACTION_PROMPT` | Listener's reaction |
| `SPEAKER_CHARACTERISTICS_EXTRACTION_PROMPT_ALT1` | Speaker's mental state/behaviour |
| `SPEAKER_CHARACTERISTICS_EXTRACTION_PROMPT_ALT2` | Speaker's intention/reason |
| `SPEAKER_CHARACTERISTICS_EXTRACTION_PROMPT_NO_AUDIO` | Listener's reaction (no audio) |

---

## utils.py

### Emotion label sets

```python
from src.helper.utils import (
    meld_emotion_set_mapped,      # ['joyful', 'sad', 'neutral', ...]
    iemocap_emotion_set_mapped,
    emotion_mapper_ori_to_conv,   # maps raw labels to canonical labels
    get_mapped_emotion_set,       # returns label set for a given dataset
)
```

### Key functions

| Function | Description |
|----------|-------------|
| `get_dataset_as_dataframe(dataset_name, splits, columns)` | Load the final benchmark CSV as a DataFrame |
| `get_meld_iemocap_datasets_as_dataframe(splits)` | Load both datasets at once |
| `extract_emotion_from_llm_output(text, valid_emotions)` | Parse a valid emotion label from raw LLM output |
| `abstacted_audio_text(row)` | Format a row's audio levels into a prompt string |
| `get_stage1_training_set(dataset_name, splits)` | Load Phase 1 speaker-characteristic JSON |
| `get_idx_to_speaker_characteristics_hint(...)` | Load pre-generated Phase 1 hints for evaluation |
| `load_model_via_ollama(model_id, max_output_tokens)` | Load a model from a local Ollama server |
| `load_model_via_hf(model_id, max_output_tokens)` | Load a model via HuggingFace pipeline |
| `anonymize_speakers_in_dialog(df_dialog)` | Replace speaker names with anonymous IDs |
| `dump_json_test_result(result, path)` | Save a result dict to JSON |

---

## build_prompting_dataset.py

Builds the Phase 2 prompting dataset by retrieving RAG demonstrations from a ChromaDB vector store.

### Usage (as a script)

```bash
python -m src.helper.build_prompting_dataset \
    --dataset iemocap \
    --max_k 20 \
    --top_n 2 \
    --vectordb_path artifacts/vectorstores/db/meld_iemocap_hybrid_7 \
    --use_detailed_example true \
    --example_refinement_level 1 \
    --save_as jsonl
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `iemocap` | `meld` or `iemocap` |
| `--max_k` | 12 | Conversation-history window size |
| `--top_n` | 1 | Number of RAG demonstrations per prompt |
| `--vectordb_path` | single store | Path to a ChromaDB vector store |
| `--use_detailed_example` | false | Label each utterance in flow/hybrid demonstrations |
| `--example_refinement_level` | 0 | 0 = no filter; 1 = same dataset; 2 = valid emotions only |
| `--save_as` | `no` | `json`, `jsonl`, or `no` (return only) |

### Usage as a library

```python
from src.helper.build_prompting_dataset import main as build_dataset

dataset = build_dataset({
    "dataset": "iemocap",
    "max_k": 20,
    "top_n": 2,
    "split": ["train", "dev"],
    "vectordb_path": "artifacts/vectorstores/db/meld_iemocap_hybrid_7",
    "use_detailed_example": True,
    "example_refinement_level": 1,
    "save_as": "no",
})
# dataset == {"train": [...], "dev": [...]}
```
