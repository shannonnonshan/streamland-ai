"""
Models module - Contains model loaders and interfaces
Supports: Whisper (STT), Embeddings, Llama, Moderation, Summarization
"""

from .base import BaseModel
from .whisper import WhisperModel
from .embeddings import EmbeddingModel

__all__ = [
    "BaseModel",
    "WhisperModel",
    "EmbeddingModel",
]
