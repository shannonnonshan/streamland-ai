"""Embedding Model Interface (all-MiniLM-L6-v2)"""

from models.base import BaseModel


class EmbeddingModel(BaseModel):
    """Embedding model for search and recommendations."""
    
    def __init__(self, model_path=None, from_hf=True):
        super().__init__("embeddings", model_path or "sentence-transformers/all-MiniLM-L6-v2", from_hf)
    
    def embed(self, text):
        """Generate embeddings for text."""
        raise NotImplementedError("embed() must be implemented")
    
    def embed_batch(self, texts):
        """Generate embeddings for multiple texts."""
        raise NotImplementedError("embed_batch() must be implemented")
