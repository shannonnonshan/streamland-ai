"""Search endpoint using Embeddings + FAISS"""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/search", tags=["search"])


@router.post("")
async def search(query: str, model=None):
    """Search content using semantic embeddings."""
    if not model or model.model_type != "embeddings":
        raise HTTPException(status_code=400, detail="Embeddings model not available")
    
    try:
        embedding = model.embed(query)
        # TODO: Search FAISS index
        return {
            "status": "success",
            "query": query,
            "results": []  # Placeholder
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
