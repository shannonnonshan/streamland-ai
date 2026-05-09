"""Transcribe endpoint using Whisper"""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from typing import Dict, Tuple

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi import HTTPException

from utils.config import ModelConfig
from utils.replicate_client import run_model_async

from api.dependencies import get_whisper_model

router = APIRouter(prefix="/transcribe", tags=["transcribe"])
logger = logging.getLogger(__name__)
HEARTBEAT_INTERVAL_SECONDS = 5
TEMP_FILE_CACHE_TTL_SECONDS = 3600  # 1 hour


# Global cache: {content_hash: (file_path, created_timestamp)}
_temp_file_cache: Dict[str, Tuple[str, float]] = {}
_cache_lock = asyncio.Lock()


async def _get_or_create_temp_file(file_content: bytes, original_filename: str) -> str:
    """
    Get cached temp file if content already exists, otherwise create new temp file.
    Returns: path to temp file
    """
    # Calculate content hash
    content_hash = hashlib.sha256(file_content).hexdigest()

    async with _cache_lock:
        now = time.time()

        # Clean expired cache entries
        expired_hashes = [
            h
            for h, (_, timestamp) in _temp_file_cache.items()
            if now - timestamp > TEMP_FILE_CACHE_TTL_SECONDS
        ]
        for h in expired_hashes:
            path, _ = _temp_file_cache.pop(h)
            try:
                if os.path.exists(path):
                    os.remove(path)
                logger.debug("Cleaned expired temp file: %s", path)
            except Exception as e:
                logger.warning("Failed to clean expired temp file %s: %s", path, e)

        # Check if we have this content cached
        if content_hash in _temp_file_cache:
            file_path, _ = _temp_file_cache[content_hash]
            if os.path.exists(file_path):
                logger.info("Reusing cached temp file for hash %s: %s", content_hash[:8], file_path)
                return file_path

        # Create new temp file
        _, ext = os.path.splitext(original_filename or "")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".wav")
        temp_file.write(file_content)
        temp_file.close()
        file_path = temp_file.name

        _temp_file_cache[content_hash] = (file_path, now)
        logger.info("Created new temp file: %s (hash: %s)", file_path, content_hash[:8])

        return file_path


async def _run_replicate_transcribe(temp_path: str, language: str) -> dict:
    if not ModelConfig.REPLICATE_API_TOKEN:
        logger.error("REPLICATE_API_TOKEN is not set")
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN not configured on server")

    model_id = ModelConfig.REPLICATE_MODEL
    if ":" not in model_id:
        model_id = model_id + ":latest"

    input_payload = {"audio": temp_path}
    if language:
        input_payload["language"] = language

    logger.info("Calling Replicate model: %s with audio: %s", model_id, temp_path)

    try:
        result = await run_model_async(model_id, input_payload)
        logger.info("Replicate call succeeded: %s", model_id)
    except RuntimeError as e:
        logger.error("Replicate call failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "success", "model": model_id, "result": result}


@router.post("")
async def transcribe(file: UploadFile = File(...), language: str = Form("")):
    """Transcribe audio file using Whisper model (local or Replicate)."""
    temp_path = None

    async def generate():
        nonlocal temp_path
        try:
            # Read file content
            file_content = await file.read()
            file_size_bytes = len(file_content)

            logger.info(
                "Transcribe request received: filename=%s, content_type=%s, size=%d bytes",
                file.filename,
                file.content_type,
                file_size_bytes,
            )

            yield (
                json.dumps(
                    {
                        "status": "processing",
                        "filename": file.filename,
                        "message": "Starting transcription...",
                        "processing": True,
                    }
                ).encode()
                + b"\n"
            )

            # Get or create cached temp file
            temp_path = await _get_or_create_temp_file(file_content, file.filename)

            logger.info("Using temp file: %s, starting transcription...", temp_path)

            started_at = time.monotonic()

            if ModelConfig.REPLICATE_USE:
                transcribe_task = asyncio.create_task(_run_replicate_transcribe(temp_path, language))
            else:
                model = get_whisper_model()
                transcribe_task = asyncio.create_task(asyncio.to_thread(model.transcribe, temp_path))

            # Keep streaming heartbeat messages while waiting for final result.
            while not transcribe_task.done():
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                if transcribe_task.done():
                    break

                elapsed_seconds = int(time.monotonic() - started_at)
                yield (
                    json.dumps(
                        {
                            "status": "processing",
                            "filename": file.filename,
                            "message": "Transcription still running...",
                            "processing": True,
                            "elapsed_seconds": elapsed_seconds,
                        }
                    ).encode()
                    + b"\n"
                )

            result = await transcribe_task
            logger.info("Transcribe completed: filename=%s", file.filename)

            response_data = {
                "status": "success",
                "data": {
                    "filename": file.filename,
                    "result": result,
                },
                "error": None,
            }
            yield json.dumps(response_data).encode() + b"\n"

        except asyncio.CancelledError:
            logger.info("Transcribe stream cancelled by client: filename=%s", file.filename)
            raise
        except Exception as e:
            logger.exception("Transcribe failed: filename=%s", file.filename)
            error_response = {
                "status": "error",
                "data": None,
                "error": str(e),
            }
            yield json.dumps(error_response).encode() + b"\n"
        finally:
            # Note: temp files are NOT deleted here, they are kept in cache for reuse
            # and will be cleaned by TTL expiration logic
            pass

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/replicate")
async def transcribe_replicate(file: UploadFile = File(...), language: str = Form("")):
    """Transcribe using the Replicate-hosted `shannonnonshan/streamland-whisper` model.

    This endpoint uploads the received file temporarily and forwards it to
    Replicate via the SDK. It returns the raw JSON result from Replicate.
    """
    # Quick guard: ensure replicate usage enabled in config
    if not ModelConfig.REPLICATE_USE:
        raise HTTPException(status_code=400, detail="Replicate forwarding is disabled (REPLICATE_USE=false)")

    # Read file bytes
    try:
        contents = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {e}")

    # Save to temp file (reuse cache helper)
    temp_path = await _get_or_create_temp_file(contents, file.filename)
    return await _run_replicate_transcribe(temp_path, language)
