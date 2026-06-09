"""
api/endpoints/admin.py — Endpoint reload FAISS index không cần restart server.
"""

from fastapi import APIRouter, HTTPException
from utils.search_index import SearchIndex
from utils.config import ModelConfig
import logging

router = APIRouter(prefix="/admin", tags=["admin"])
log = logging.getLogger(__name__)


@router.post("/reload-index")
async def reload_index() -> dict:
    """
    Reload FAISS index từ disk — dùng sau khi rebuild_index.py chạy xong.
    Không cần restart server.
    """
    try:
        from api.dependencies import get_search_index
        index: SearchIndex = get_search_index()
        index.load()
        log.info("FAISS index reloaded: %d vectors", index.index.ntotal if index.index else 0)
        return {
            "status": "ok",
            "vectors": index.index.ntotal if index.index else 0,
        }
    except Exception as e:
        log.error("Failed to reload index: %s", e)
        raise HTTPException(status_code=500, detail=str(e))