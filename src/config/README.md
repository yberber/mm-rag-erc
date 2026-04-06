# src/config

Project-wide configuration: file-system paths and shared constants.

## Files

| File | Purpose |
|------|---------|
| `paths.py` | All directory and file paths used across the codebase. Edit `MELD_RAW_DATA_DIR` and `IEMOCAP_RAW_DATA_DIR` at the top to point to your raw-data downloads; everything else is derived automatically. |
| `constants.py` | Shared constants: embedding model name, LLM concurrency limit for Phase 1 annotation, and the base LLM model ID. |

## Editing paths

Open `paths.py` and change the two variables at the top:

```python
MELD_RAW_DATA_DIR    = "/path/to/MELD.Raw"
IEMOCAP_RAW_DATA_DIR = "/path/to/IEMOCAP_full_release"
```

No other file needs to be changed — all downstream scripts import their paths from this module.

## Key path variables

| Variable | Points to |
|----------|-----------|
| `MELD_BENCHMARK_FINAL_FILE_PATH` | Final MELD CSV (after all processing steps) |
| `IEMOCAP_BENCHMARK_FINAL_FILE_PATH` | Final IEMOCAP CSV |
| `TRAINING_STAGE1_DIR` | Phase 1 JSONL training files |
| `TRAINING_STAGE2_DIR` | Phase 2 JSONL training files |
| `SPEAKER_CHARACTERISTICS_DIR` | Phase 1 speaker-characteristic JSON outputs |
| `VECTORSTORE_DB_DIR` | ChromaDB vector store directories |
| `VECTORSTORE_CACHE_DIR` | Pre-computed similarity cache JSON files |
| `VECTORSTORE_INDEX_PATH` | Utterance-index-to-metadata JSON |
| `EVAL_STAGE1_DIR` / `EVAL_STAGE2_DIR` | Evaluation results |
