import os
import torch

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
MEDQA_DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
# No full Wikipedia dump path as we fetch on demand
WIKIPEDIA_CACHE_DIR = os.path.join(BASE_DIR, "wikipedia_cache") # To cache downloaded pages temporarily
WIKIPEDIA_TITLE_EMBEDDINGS_PATH = os.path.join(BASE_DIR, "index", "wikipedia_title_embeddings.npy")
WIKIPEDIA_TITLE_INDEX_PATH = os.path.join(BASE_DIR, "index", "wikipedia_title_faiss_index.bin")
WIKIPEDIA_TITLE_METADATA_PATH = os.path.join(BASE_DIR, "index", "wikipedia_title_metadata.json") # Stores titles/URLs of all searchable pages

# Model paths
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_TRANSFORMER_MODEL_PATH = os.path.join(BASE_DIR, "models", "sentence_transformers", SENTENCE_TRANSFORMER_MODEL.split('/')[-1])

QA_MODEL_NAME = "distilbert-base-uncased-distilled-squad"
QA_MODEL_PATH = os.path.join(BASE_DIR, "models", "qa_models", QA_MODEL_NAME.split('/')[-1])

SPACY_MODEL_NAME = "en_core_web_sm"
SPACY_MODEL_PATH = os.path.join(BASE_DIR, "models", "spacy", SPACY_MODEL_NAME)

# Retrieval settings
MAX_WIKIPEDIA_PAGES_PER_QUESTION = 5
WIKIPEDIA_SEARCH_RESULTS_TO_CONSIDER = 50 # Number of top search results to embed and re-rank

# Evaluation settings
SUBSET_EVAL_SIZE = 1000

# Other settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Uses CUDA if available, else CPU