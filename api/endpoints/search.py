"""Search endpoint using Embeddings + FAISS"""

import asyncio

from fastapi import APIRouter, HTTPException, Depends

from api.dependencies import get_embeddings_model
from utils.config import ModelConfig
from utils.search_index import SearchIndex

router = APIRouter(prefix="/search", tags=["search"])
search_index = SearchIndex(
    index_path=ModelConfig.FAISS_INDEX_PATH,
    metadata_path=ModelConfig.FAISS_METADATA_PATH,
)


@router.post("")
async def search(query: str, top_k: int = 5, model=Depends(get_embeddings_model)):
    """Search content using semantic embeddings."""
    try:
        embedding = await asyncio.to_thread(model.embed, query)
        vector = embedding[0]
        candidate_k = max(top_k * 5, top_k)
        results = await asyncio.to_thread(search_index.search, vector, candidate_k)

        def _priority(item):
            meta = item.get("metadata", {})
            status = str(meta.get("status", "")).upper()
            if status == "LIVE":
                return 1
            if status == "ENDED":
                return 2
            return 3

        results.sort(
            key=lambda item: (
                _priority(item),
                -float(item.get("score", 0.0)),
                str(item.get("metadata", {}).get("id", "")),
            )
        )
        return {
            "status": "success",
            "query": query,
            "results": results[:top_k],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
