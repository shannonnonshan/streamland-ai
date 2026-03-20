"""
Universal Model Pusher Utility
Purpose: Push any model from local disk to Hugging Face Hub
Supports: All model types (Whisper, TTS, Wav2Vec2, etc.)
Usage: python model_pusher.py <model_name> [--public]
Config: Reads HF_TOKEN, HF_USERNAME, HF_REPO_BASE from .env
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_folder
from typing import Optional, Dict, Any

load_dotenv()


def push_model_to_hub(
    model_name: str,
    local_model_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    commit_message: Optional[str] = None,
    private: bool = True,
    username: Optional[str] = None,
    repo_base: Optional[str] = None,
) -> Dict[str, Any]:
    """Push model to Hugging Face Hub."""
    
    if not username:
        username = os.getenv("HF_USERNAME", "")
        if not username:
            raise ValueError("Set HF_USERNAME in .env")
    
    if not repo_base:
        repo_base = os.getenv("HF_REPO_BASE", "streamland")
    
    if not repo_id:
        repo_id = f"{username}/{repo_base}-{model_name}"
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token.startswith("hf_your"):
        raise ValueError("Set HF_TOKEN in .env")
    
    if not local_model_path:
        local_model_path = f"models/{model_name}/model/{model_name}-finetuned"
    
    if not os.path.exists(local_model_path):
        raise ValueError(f"Model path not found: {local_model_path}")
    
    if not commit_message:
        commit_message = f"Upload {model_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    print(f"[INFO] Pushing {model_name} to {repo_id}...")
    
    try:
        api = HfApi()
        
        # Create repo
        repo_url = api.create_repo(repo_id=repo_id, private=private, exist_ok=True, token=hf_token)
        print(f"[INFO] Repo ready: {repo_url}")
        
        # Count files
        total_files = 0
        total_size = 0
        for root, dirs, files in os.walk(local_model_path):
            total_files += len(files)
            for f in files:
                total_size += os.path.getsize(os.path.join(root, f))
        
        # Upload folder
        print(f"[INFO] Uploading {total_files} files...")
        upload_folder(
            folder_path=local_model_path,
            repo_id=repo_id,
            token=hf_token,
            commit_message=commit_message
        )
        
        size_mb = total_size / (1024 * 1024)
        print(f"[SUCCESS] {model_name} uploaded: {repo_id} ({total_files} files, {size_mb:.2f} MB)")
        
        return {"status": "success", "model_name": model_name, "repo_id": repo_id, "url": str(repo_url)}
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_pusher.py <model_name> [--public] [--username <user>] [--repo-base <base>]")
        sys.exit(1)
    
    model_name = sys.argv[1]
    is_private = "--public" not in sys.argv
    username = None
    repo_base = None
    
    for i, arg in enumerate(sys.argv[2:]):
        if arg == "--username" and i + 2 < len(sys.argv):
            username = sys.argv[i + 3]
        elif arg == "--repo-base" and i + 2 < len(sys.argv):
            repo_base = sys.argv[i + 3]
    
    try:
        result = push_model_to_hub(
            model_name=model_name,
            private=is_private,
            username=username,
            repo_base=repo_base
        )
        sys.exit(0 if result["status"] == "success" else 1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
