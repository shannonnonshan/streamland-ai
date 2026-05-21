"""Configuration for all models"""

import os
from dotenv import load_dotenv

load_dotenv()


class ModelConfig:
    """Centralized configuration for model loading."""
    
    # Whisper - STT
    WHISPER_MODEL = os.getenv("WHISPER_MODEL_PATH", "shannonnonshan/streamland-whisper-ct2")
    WHISPER_USE_HF = os.getenv("WHISPER_USE_HF", "true").lower() == "true"
    
    # Embeddings - Search & Recommend
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDINGS_USE_HF = os.getenv("EMBEDDINGS_USE_HF", "true").lower() == "true"
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "embeddings.faiss")
    
    # LLM - Chat & RAG
    LLAMA_MODEL = os.getenv("LLAMA_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    LLAMA_USE_HF = os.getenv("LLAMA_USE_HF", "true").lower() == "true"
    LLAMA_DEVICE = os.getenv("LLAMA_DEVICE", "cpu")  # cpu or cuda
    
    # Moderation - Content Safety
    MODERATION_MODEL = os.getenv("MODERATION_MODEL", "detoxify")
    MODERATION_USE_HF = os.getenv("MODERATION_USE_HF", "false").lower() == "true"
    MODERATION_EN_MODEL = os.getenv("MODERATION_EN_MODEL", "s-nlp/roberta_toxicity_classifier")
    MODERATION_VI_MODEL = os.getenv("MODERATION_VI_MODEL", "cardiffnlp/twitter-xlm-roberta-base-offensive")
    MODERATION_FULL_MODEL = os.getenv("MODERATION_FULL_MODEL", MODERATION_VI_MODEL)
    MODERATION_REWRITE_MODEL = os.getenv("MODERATION_REWRITE_MODEL", "s-nlp/bart-base-detox")
    MODERATION_EMBEDDING_MODEL = os.getenv("MODERATION_EMBEDDING_MODEL", "BAAI/bge-m3")
    MODERATION_GREYZONE_LOWER = float(os.getenv("MODERATION_GREYZONE_LOWER", "0.40"))
    MODERATION_GREYZONE_UPPER = float(os.getenv("MODERATION_GREYZONE_UPPER", "0.70"))
    MODERATION_BLOCK_THRESHOLD = float(os.getenv("MODERATION_BLOCK_THRESHOLD", "0.85"))
    MODERATION_REVIEW_THRESHOLD = float(os.getenv("MODERATION_REVIEW_THRESHOLD", "0.55"))
    MODERATION_LEXICON_PATH = os.getenv("MODERATION_LEXICON_PATH")
    MODERATION_EXAMPLES_PATH = os.getenv("MODERATION_EXAMPLES_PATH")
    
    # Summarization - Text Summary
    SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "shannonnonshan/bart-summarizer")
    SUMMARIZATION_USE_HF = os.getenv("SUMMARIZATION_USE_HF", "true").lower() == "true"
    
    # Common settings
    HF_TOKEN = os.getenv("HF_TOKEN")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))

