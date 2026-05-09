"""Helper wrapper for calling Replicate-hosted models from the backend.

This module provides a thin wrapper around the `replicate` SDK and runs models
in a background thread so FastAPI event loop is not blocked.
"""
import os
import asyncio
from typing import Any, Dict


def _ensure_replicate_installed():
    try:
        import replicate  # type: ignore
        return replicate
    except Exception as e:
        raise RuntimeError("replicate SDK is not installed. Install with: pip install replicate") from e


def run_model(model_id: str, input_payload: Dict[str, Any]) -> Any:
    """Run a Replicate model synchronously (intended to be called via thread).

    model_id should include a version (e.g. owner/model:latest). The function
    raises RuntimeError if the SDK isn't available or if the call fails.
    """
    replicate = _ensure_replicate_installed()

    # Ensure version suffix
    if ":" not in model_id:
        model_id = model_id + ":latest"

    # The replicate.run helper will upload local file paths automatically.
    try:
        return replicate.run(model_id, input=input_payload)
    except Exception as e:
        raise RuntimeError(f"Replicate run failed: {e}") from e


async def run_model_async(model_id: str, input_payload: Dict[str, Any]) -> Any:
    """Async wrapper using asyncio.to_thread to avoid blocking the event loop."""
    return await asyncio.to_thread(run_model, model_id, input_payload)
