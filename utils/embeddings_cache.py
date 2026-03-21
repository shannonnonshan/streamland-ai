"""FAISS embeddings cache for vector search"""

import os


class EmbeddingsCache:
    """Manage embeddings index with FAISS."""
    
    def __init__(self, index_path="embeddings.faiss"):
        self.index_path = index_path
        self.index = None
        self.metadata = []
    
    def create_index(self, embeddings):
        """Create FAISS index from embeddings."""
        try:
            import faiss
            import numpy as np
            
            vectors = np.array(embeddings).astype('float32')
            self.index = faiss.IndexFlatL2(vectors.shape[1])
            self.index.add(vectors)
            
            return self.index
        except ImportError:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")
    
    def search(self, query_embedding, k=5):
        """Search k nearest neighbors."""
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        import numpy as np
        query = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query, k)
        
        return {
            "indices": indices[0].tolist(),
            "distances": distances[0].tolist(),
            "results": [self.metadata[i] for i in indices[0] if i < len(self.metadata)]
        }
    
    def save_index(self):
        """Save index to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        import faiss
        faiss.write_index(self.index, self.index_path)
    
    def load_index(self):
        """Load index from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        import faiss
        self.index = faiss.read_index(self.index_path)
