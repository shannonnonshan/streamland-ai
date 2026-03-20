"""
Hugging Face Configuration and Utilities
Purpose: Utilities for managing Hugging Face interactions and configurations
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class HFConfig:
    """Configuration class for Hugging Face Hub integration."""
    
    # Load from .env or environment
    DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN", None)
    DEFAULT_HF_REPO_ID = os.getenv("HF_REPO_ID", None)
    DEFAULT_MODEL_PATH_LOCAL = os.getenv("MODEL_PATH", "models/whisper/model/whisper-finetuned")
    DEFAULT_PRIVATE = True
    
    @staticmethod
    def get_hf_token() -> Optional[str]:
        """Get HF token from .env or environment."""
        token = os.getenv("HF_TOKEN")
        if not token or token == "hf_your_hugging_face_token_here":
            return None
        return token
    
    @staticmethod
    def set_hf_token(token: str):
        """Set HF token as environment variable."""
        os.environ["HF_TOKEN"] = token
    
    @staticmethod
    def validate_repo_id(repo_id: str) -> bool:
        """Validate Hugging Face repo ID format (username/model-name)."""
        if not repo_id or "/" not in repo_id:
            return False
        parts = repo_id.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return False
        return True
    
    @staticmethod
    def info() -> Dict[str, Any]:
        """Get configuration info."""
        return {
            "repo_id": HFConfig.DEFAULT_HF_REPO_ID,
            "has_token": HFConfig.get_hf_token() is not None,
            "local_model_path": HFConfig.DEFAULT_MODEL_PATH_LOCAL,
            "private": HFConfig.DEFAULT_PRIVATE,
        }


def setup_hf_environment():
    """Setup HF environment and verify configuration."""
    print("[INFO] Hugging Face Environment Setup")
    print("-" * 50)
    
    config_info = HFConfig.info()
    
    print(f"Local Model Path: {config_info['local_model_path']}")
    print(f"HF Token Set: {'Yes' if config_info['has_token'] else 'No'}")
    print(f"Default Repo ID: {config_info['repo_id'] or '(not set)'}")
    print(f"Private by Default: {config_info['private']}")
    
    if not config_info['has_token']:
        print("\n[WARNING] No HF token found. Update .env with HF_TOKEN")
        print("[INFO] Get token from: https://huggingface.co/settings/tokens")
    
    return config_info


if __name__ == "__main__":
    setup_hf_environment()
