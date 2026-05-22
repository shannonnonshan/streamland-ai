"""Configuration for all models"""

import os
from dotenv import load_dotenv

load_dotenv()


class ModelConfig:
    """Centralized configuration for model loading."""
    
    # Whisper - STT
    WHISPER_MODEL = os.getenv("WHISPER_MODEL_PATH", "openai/whisper-small")
    WHISPER_USE_HF = os.getenv("WHISPER_USE_HF", "true").lower() == "true"
    
    # Embeddings - Search & Recommend
    EMBEDDINGS_MODEL = os.getenv(
        "SEARCH_MODEL_PATH",
        os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    EMBEDDINGS_USE_HF = os.getenv("EMBEDDINGS_USE_HF", "true").lower() == "true"
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "embeddings.faiss")
    FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "embeddings.meta.json")
    SEARCH_CORPUS_PATH = os.getenv("SEARCH_CORPUS_PATH", "data/search_corpus.jsonl")
    
    # LLM - Chat & RAG
    LLAMA_MODEL = os.getenv("LLAMA_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    LLAMA_USE_HF = os.getenv("LLAMA_USE_HF", "true").lower() == "true"
    LLAMA_DEVICE = os.getenv("LLAMA_DEVICE", "cpu")  # cpu or cuda

    # Chatbot - Conversational QA
    CHATBOT_MODEL = os.getenv("CHATBOT_MODEL_PATH", "shannonnonshan/streamland-chatbot")
    CHATBOT_USE_HF = os.getenv("CHATBOT_USE_HF", "true").lower() == "true"
    CHATBOT_BASE_MODEL = os.getenv(
        "CHATBOT_BASE_MODEL",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    )
    
    # Moderation - Content Safety
    MODERATION_MODEL = os.getenv("MODERATION_MODEL", "detoxify")
    MODERATION_USE_HF = os.getenv("MODERATION_USE_HF", "false").lower() == "true"
    
    # Summarization - Text Summary
    SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "shannonnonshan/bart-summarizer")
    SUMMARIZATION_USE_HF = os.getenv("SUMMARIZATION_USE_HF", "true").lower() == "true"
    
    # Common settings
    HF_TOKEN = os.getenv("HF_TOKEN")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))

    # Replicate configuration (for backend-forwarding to Replicate-hosted model)
    REPLICATE_USE = os.getenv("REPLICATE_USE", "false").lower() == "true"
    REPLICATE_MODEL = os.getenv("REPLICATE_MODEL", "shannonnonshan/streamland-whisper")
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
