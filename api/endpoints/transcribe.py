"""Transcribe endpoint using Whisper"""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from typing import Dict, Optional, Tuple

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api.dependencies import get_whisper_model

router = APIRouter(prefix="/transcribe", tags=["transcribe"])
logger = logging.getLogger(__name__)
HEARTBEAT_INTERVAL_SECONDS = 5
TEMP_FILE_CACHE_TTL_SECONDS = 3600  # 1 hour


# Global cache: {content_hash: (file_path, created_timestamp)}
_temp_file_cache: Dict[str, Tuple[str, float]] = {}
_cache_lock = asyncio.Lock()


async def _get_or_create_temp_file(file_content: bytes, original_filename: str) -> str:
    content_hash = hashlib.sha256(file_content).hexdigest()

    async with _cache_lock:
        now = time.time()

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

        if content_hash in _temp_file_cache:
            file_path, _ = _temp_file_cache[content_hash]
            if os.path.exists(file_path):
                logger.info("Reusing cached temp file for hash %s: %s", content_hash[:8], file_path)
                return file_path

        _, ext = os.path.splitext(original_filename or "")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".wav")
        temp_file.write(file_content)
        temp_file.close()
        file_path = temp_file.name

        _temp_file_cache[content_hash] = (file_path, now)
        logger.info("Created new temp file: %s (hash: %s)", file_path, content_hash[:8])

        return file_path


@router.post("")
async def transcribe(
    file: Optional[UploadFile] = File(None),
    file_url: str = Form(""),
    language: str = Form(""),
):
    """Transcribe audio file using the local Whisper model. Accepts file_url or uploaded file."""

    async def generate():
        try:
            # Priority 1: download from URL
            if file_url:
                logger.info("Downloading audio from URL: %s", file_url)
                async with httpx.AsyncClient(timeout=300) as client:
                    response = await client.get(file_url)
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download file from URL: HTTP {response.status_code}",
                        )
                    file_content = response.content
                    filename = file_url.split("?")[0].split("/")[-1] or "audio.wav"
                logger.info("Downloaded %d bytes from URL", len(file_content))

            # Priority 2: uploaded file
            elif file is not None:
                file_content = await file.read()
                filename = file.filename or "audio.wav"
                logger.info("Received uploaded file: %s, %d bytes", filename, len(file_content))

            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either file_url or file is required",
                )

            file_size_bytes = len(file_content)
            logger.info(
                "Transcribe request ready: filename=%s, size=%d bytes",
                filename,
                file_size_bytes,
            )

            yield (
                json.dumps(
                    {
                        "status": "processing",
                        "filename": filename,
                        "message": "Starting transcription...",
                        "processing": True,
                    }
                ).encode()
                + b"\n"
            )

            temp_path = await _get_or_create_temp_file(file_content, filename)
            logger.info("Using temp file: %s, starting transcription...", temp_path)

            started_at = time.monotonic()
            model = get_whisper_model()
            transcribe_task = asyncio.create_task(asyncio.to_thread(model.transcribe, temp_path))

            while not transcribe_task.done():
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                if transcribe_task.done():
                    break

                elapsed_seconds = int(time.monotonic() - started_at)
                yield (
                    json.dumps(
                        {
                            "status": "processing",
                            "filename": filename,
                            "message": "Transcription still running...",
                            "processing": True,
                            "elapsed_seconds": elapsed_seconds,
                        }
                    ).encode()
                    + b"\n"
                )

            result = await transcribe_task
            logger.info("Transcribe completed: filename=%s", filename)

            yield (
                json.dumps(
                    {
                        "status": "success",
                        "data": {
                            "filename": filename,
                            "result": result,
                        },
                        "error": None,
                    }
                ).encode()
                + b"\n"
            )

        except asyncio.CancelledError:
            logger.info("Transcribe stream cancelled by client")
            raise
        except Exception as e:
            logger.exception("Transcribe failed")
            yield (
                json.dumps(
                    {
                        "status": "error",
                        "data": None,
                        "error": str(e),
                    }
                ).encode()
                + b"\n"
            )

    return StreamingResponse(generate(), media_type="application/x-ndjson")