"""Shared runtime registry for loaded model instances."""

from typing import Dict, Any

# Populated by api.server during startup.
models: Dict[str, Any] = {}
