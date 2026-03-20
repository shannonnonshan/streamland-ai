"""
Universal Model Pusher Utility
Purpose: Reusable function to push any model to Hugging Face Hub
Can be used for Whisper, TTS, or any other model in the models/ directory
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, Repository
from typing import Optional, Dict, Any

# Load environment variables
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
    """
    Generic function to push any model to Hugging Face Hub.
    
    Args:
        model_name (str): Name of the model (e.g., "whisper", "tts", "wav2vec2")
                         If local_model_path not provided, assumes path is models/{model_name}/model/{model_name}-finetuned
        local_model_path (str, optional): Custom local path to model directory
        repo_id (str, optional): Full repo ID (username/model-name). 
                                If not provided, constructs from username, repo_base, and model_name
        commit_message (str, optional): Custom commit message
        private (bool): Whether to make repo private (default: True)
        username (str, optional): HF username. If not provided, reads from .env HF_USERNAME
        repo_base (str, optional): Base for repo name. If not provided, reads from .env HF_REPO_BASE
                                   Template: {username}/{repo_base}-{model_name}
        
    Returns:
        dict: Result information with status, repo_id, and url
    """
    
    # Determine username and repo_base from .env if not provided
    if not username:
        username = os.getenv("HF_USERNAME", "")
        if not username:
            raise ValueError(
                "Username not provided. Please set HF_USERNAME in .env or pass username parameter"
            )
    
    if not repo_base:
        repo_base = os.getenv("HF_REPO_BASE", "streamland")
    
    # Construct repo_id if not provided
    if not repo_id:
        repo_id = f"{username}/{repo_base}-{model_name}"
    
    # Verify HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token.startswith("hf_your"):
        raise ValueError(
            "HF_TOKEN not set or invalid in .env. "
            "Get token from: https://huggingface.co/settings/tokens"
        )
    
    # Determine local model path
    if not local_model_path:
        local_model_path = f"models/{model_name}/model/{model_name}-finetuned"
    
    # Verify model path exists
    if not os.path.exists(local_model_path):
        raise ValueError(f"Model path does not exist: {local_model_path}")
    
    # Default commit message
    if not commit_message:
        commit_message = f"Upload fine-tuned {model_name} model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    print(f"{'='*60}")
    print(f"[INFO] Starting Upload to Hugging Face Hub")
    print(f"{'='*60}")
    print(f"  Model Name:    {model_name}")
    print(f"  Repo ID:       {repo_id}")
    print(f"  Local Path:    {local_model_path}")
    print(f"  Private:       {private}")
    print(f"  Message:       {commit_message}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize Hugging Face API
        api = HfApi()
        
        # Create repo if it doesn't exist
        print("[1/5] Creating/accessing Hugging Face repository...")
        repo_url = api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            token=hf_token
        )
        print(f"      ✓ Repository ready: {repo_url}\n")
        
        # Clone repo locally
        print("[2/5] Cloning repository...")
        repo_local_dir = f"hf_repo_{model_name}"
        
        # Clean up if exists
        if os.path.exists(repo_local_dir):
            shutil.rmtree(repo_local_dir)
        
        repo = Repository(
            local_dir=repo_local_dir,
            clone_from=repo_url,
            token=hf_token
        )
        print(f"      ✓ Repository cloned to: {repo_local_dir}\n")
        
        # Clear existing model files (keep git and doc files)
        print("[3/5] Clearing existing model files...")
        exclude_items = {'.git', '.gitignore', '.gitattributes', 'README.md', '.gitmodules'}
        
        for item in os.listdir(repo_local_dir):
            if item not in exclude_items:
                item_path = os.path.join(repo_local_dir, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception as e:
                    print(f"      ⚠ Warning clearing {item}: {e}")
        print(f"      ✓ Old files cleared\n")
        
        # Copy model files
        print("[4/5] Copying new model files...")
        total_files = 0
        total_size = 0
        
        for item in os.listdir(local_model_path):
            src = os.path.join(local_model_path, item)
            dst = os.path.join(repo_local_dir, item)
            
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                # Count files in directory
                for root, dirs, files in os.walk(dst):
                    total_files += len(files)
                    for f in files:
                        total_size += os.path.getsize(os.path.join(root, f))
            else:
                shutil.copy2(src, dst)
                total_files += 1
                total_size += os.path.getsize(dst)
        
        # Copy README if exists
        readme_path = f"models/{model_name}/README.md"
        if os.path.exists(readme_path):
            shutil.copy2(readme_path, os.path.join(repo_local_dir, "README.md"))
            print(f"      ✓ Copied README.md")
        
        size_mb = total_size / (1024 * 1024)
        print(f"      ✓ Copied {total_files} files ({size_mb:.2f} MB)\n")
        
        # Push to hub
        print("[5/5] Pushing to Hugging Face Hub...")
        repo.push_to_hub(commit_message=commit_message)
        print(f"      ✓ Push completed!\n")
        
        # Clean up local repo
        print("[CLEANUP] Removing temporary clone...")
        shutil.rmtree(repo_local_dir)
        print(f"       ✓ Cleaned up\n")
        
        result = {
            "status": "success",
            "model_name": model_name,
            "repo_id": repo_id,
            "url": str(repo_url),
            "message": f"Model successfully pushed to {repo_url}"
        }
        
        print(f"{'='*60}")
        print(f"[SUCCESS] Model uploaded successfully!")
        print(f"{'='*60}")
        print(f"  Repository URL: {repo_url}")
        print(f"  Repo ID:        {repo_id}")
        print(f"  Files:          {total_files}")
        print(f"  Size:           {size_mb:.2f} MB")
        print(f"{'='*60}\n")
        
        return result
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"[ERROR] Upload failed!")
        print(f"{'='*60}")
        print(f"  Error: {str(e)}\n")
        print("Troubleshooting:")
        print("   1. Verify HF_TOKEN is valid: https://huggingface.co/settings/tokens")
        print("   2. Check repo_id format: username/model-name")
        print("   3. Verify model path exists:", local_model_path)
        print("   4. Check internet connection")
        print("   5. Ensure huggingface_hub is updated: pip install --upgrade huggingface_hub")
        print(f"{'='*60}\n")
        
        return {
            "status": "failed",
            "error": str(e)
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_pusher.py <model_name> [--public] [--custom-path <path>] [--username <username>] [--repo-base <base>]")
        print("\nExamples:")
        print("  python model_pusher.py whisper")
        print("  python model_pusher.py whisper --public")
        print("  python model_pusher.py tts --username myusername")
        print("  python model_pusher.py wav2vec2 --repo-base mymodels")
        print("\nFrom .env:")
        print("  HF_USERNAME=shannonnonshan")
        print("  HF_REPO_BASE=streamland")
        print("  Result: shannonnonshan/streamland-{model_name}")
        sys.exit(1)
    
    model_name = sys.argv[1]
    is_private = "--public" not in sys.argv
    
    # Parse optional arguments
    custom_path = None
    username = None
    repo_base = None
    
    try:
        i = 2
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == "--custom-path" and i + 1 < len(sys.argv):
                custom_path = sys.argv[i + 1]
                i += 2
            elif arg == "--username" and i + 1 < len(sys.argv):
                username = sys.argv[i + 1]
                i += 2
            elif arg == "--repo-base" and i + 1 < len(sys.argv):
                repo_base = sys.argv[i + 1]
                i += 2
            else:
                i += 1
    except:
        pass
    
    try:
        result = push_model_to_hub(
            model_name=model_name,
            local_model_path=custom_path,
            private=is_private,
            username=username,
            repo_base=repo_base
        )
        sys.exit(0 if result["status"] == "success" else 1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
