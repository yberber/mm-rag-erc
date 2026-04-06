# src/vectorstore

Scripts for building ChromaDB vector stores and pre-computing similarity caches used by the RAG retrieval system.

## Sub-packages

```
vectorstore/
├── generators/
│   ├── build_single_vectorstore.py   # Single-utterance store
│   ├── build_flow_vectorstore.py     # Conversational-flow stores
│   ├── build_hybrid_vectorstore.py   # Hybrid (repeated target) stores
│   └── run_vectorstore_generators.py # Runner: builds all stores
└── caching/
    ├── cache_similar_utterances.py   # Pre-compute similarity lookups
    └── create_idx_to_metadata.py     # Build utterance-index metadata JSON
```

---

## Vector store types

Three document styles are supported, each producing a different embedding space:

| Type | Document content | Best for |
|------|-----------------|----------|
| **Single** | One utterance | Utterance-level similarity |
| **Flow** | Sliding window of N consecutive utterances | Conversation-flow similarity |
| **Hybrid** | Flow window + target utterance repeated | Balanced context + identity |

The hybrid store with window size 7 was found to produce the best retrieval quality and is the default used in Phase 2 training.

---

## Step 1 — Build vector stores

Build all stores (single + flow × 5 sizes + hybrid × 5 sizes) in one command:

```bash
python -m src.vectorstore.generators.run_vectorstore_generators
```

Or build individual stores:

```bash
# Single-utterance store
python -m src.vectorstore.generators.build_single_vectorstore

# Flow store with a specific window size
python -m src.vectorstore.generators.build_flow_vectorstore --num_utterances 7

# Hybrid store with a specific window size
python -m src.vectorstore.generators.build_hybrid_vectorstore --num_utterances 7
```

Stores are written to `artifacts/vectorstores/db/`. Existing stores are skipped automatically.

---

## Step 2 — Create utterance-index metadata

```bash
python -m src.vectorstore.caching.create_idx_to_metadata
```

Combines both final benchmark CSVs, anonymises speaker names, and writes `artifacts/vectorstores/utterance_index_mapping.json`. This file is used by `DemonstrationCreatorViaCache` to look up speaker, utterance text, and emotion for each retrieved index.

---

## Step 3 — Cache similarity lookups

```bash
# Cache all uncached stores (recommended)
python -m src.vectorstore.caching.cache_similar_utterances --top_n 10

# Cache a single store
python -m src.vectorstore.caching.cache_similar_utterances \
    --vectorstore_name meld_iemocap_hybrid_7 --top_n 10
```

For every utterance in both datasets, queries the vector store and saves the top-N most similar training-utterance indices. Caches are written to `artifacts/vectorstores/caches/<store_name>.json`.

Caching replaces online vector-store queries at training and evaluation time, which significantly speeds up dataset construction.

---

## Embeddings

All stores use the `all-MiniLM-L6-v2` sentence-transformer model (set in `src/config/constants.py`) via `langchain-huggingface`.
