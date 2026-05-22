"""FAISS-backed search index utilities."""

import json
import os
import threading
from typing import Any, Dict, List

import numpy as np


class SearchIndex:
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def load(self) -> None:
        if self.index is not None:
            return

        with self._lock:
            if self.index is not None:
                return

            if not os.path.exists(self.index_path):
                raise FileNotFoundError(
                    f"FAISS index file not found: {self.index_path}. "
                    "Run build_search_index.py to create it."
                )

            try:
                import faiss
            except ImportError as exc:
                raise ImportError("FAISS not installed. Run: pip install faiss-cpu") from exc

            self.index = faiss.read_index(self.index_path)

            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r", encoding="utf-8") as handle:
                    self.metadata = json.load(handle)
            else:
                self.metadata = []

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            self.load()

        if k <= 0:
            raise ValueError("top_k must be greater than 0")

        query = np.array([query_embedding], dtype="float32")
        scores, indices = self.index.search(query, k)

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices[0].tolist()):
            if idx < 0:
                continue

            meta = self.metadata[idx] if idx < len(self.metadata) else {"id": idx}
            results.append(
                {
                    "rank": rank + 1,
                    "score": float(scores[0][rank]),
                    "metadata": meta,
                }
            )

        return results
