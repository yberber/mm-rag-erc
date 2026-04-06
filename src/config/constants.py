"""Project-wide constants shared across modules.

Defines the embedding model used for the RAG vector stores, the concurrency
limit for the async speaker-characteristic extraction, and the base LLM model
identifier used for fine-tuning and inference.
"""

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHARACTERISTICS_EXTRACTION_CONCURRENCY = 1
BASE_LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"