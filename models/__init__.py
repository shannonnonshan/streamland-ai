"""Models module - contains model loaders and interfaces.

The package avoids eager imports so optional model dependencies do not block
loading unrelated components such as moderation.
"""

from .base import BaseModel

__all__ = ["BaseModel", "WhisperModel"]


def __getattr__(name):
    if name == "WhisperModel":
        from .whisper import WhisperModel

        return WhisperModel
    raise AttributeError(f"module 'models' has no attribute {name!r}")
