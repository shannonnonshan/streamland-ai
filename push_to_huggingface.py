"""
Push Model to Hugging Face Hub
Purpose: Script to upload the fine-tuned Whisper model to Hugging Face Hub
Requires: huggingface_hub package and HF_TOKEN in .env
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from huggingface_hub import HfApi, Repository

# Load environment variables from .env
load_dotenv()


def push_model_to_hub(
    local_model_path: str = "models/whisper/model/whisper-finetuned",
    repo_id: str = None,
    commit_message: str = None,
    private: bool = True
):
    """
    Push model to Hugging Face Hub.
    
    Args:
        local_model_path (str): Local path to model directory
        repo_id (str): Hugging Face repo ID (format: username/model-name)
        commit_message (str): Commit message for the push
        private (bool): Whether to make repo private
        
    Returns:
        dict: Result information
    """
    
    if not repo_id:
        print("[ERROR] repo_id must be provided (format: username/model-name)")
        return False
    
    if not commit_message:
        commit_message = f"Upload fine-tuned Whisper model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    print(f"[INFO] Starting upload to {repo_id}...")
    print(f"       Local path: {local_model_path}")
    print(f"       Private: {private}")
    
    try:
        # Initialize Hugging Face API
        api = HfApi()
        
        # Create repo if it doesn't exist
        print("[INFO] Creating repository...")
        repo_url = api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True
        )
        
        # Clone repo
        repo_local_dir = "hf_repo"
        repo = Repository(
            local_dir=repo_local_dir,
            clone_from=repo_url
        )
        
        # Copy model files
        print("[INFO] Copying model files...")
        import shutil
        
        # Clear existing model files
        for item in os.listdir(repo_local_dir):
            if item not in ['.git', '.gitignore', '.gitattributes', 'README.md']:
                item_path = os.path.join(repo_local_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        
        # Copy new model files
        for item in os.listdir(local_model_path):
            src = os.path.join(local_model_path, item)
            dst = os.path.join(repo_local_dir, item)
            
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        # Copy README.md if it exists
        if os.path.exists("models/whisper/README.md"):
            shutil.copy2("models/whisper/README.md", os.path.join(repo_local_dir, "README.md"))
        
        # Push to hub
        print("[INFO] Pushing to Hugging Face Hub...")
        repo.push_to_hub(commit_message=commit_message)
        
        print(f"[INFO] Model uploaded to {repo_url}")
        return {
            "status": "success",
            "repo_id": repo_id,
            "url": repo_url,
            "message": f"Model pushed to {repo_url}"
        }
        
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        print("\nTroubleshooting:")
        print("   1. Login: huggingface-cli login")
        print("   2. Verify repo_id format: username/model-name")
        print("   3. Check internet connection")
        print("   4. Install: pip install huggingface_hub")
        return False


if __name__ == "__main__":
    # Example usage
    
    if len(sys.argv) < 2:
        print("[ERROR] Usage: python push_to_huggingface.py <username/model-name> [--public]")
        print("\nExample: python push_to_huggingface.py myusername/whisper-finetuned")
        print("         python push_to_huggingface.py myusername/whisper-finetuned --public")
        sys.exit(1)
    
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token == "hf_your_hugging_face_token_here":
        print("[ERROR] HF_TOKEN not set in .env")
        print("[INFO] Get token from: https://huggingface.co/settings/tokens")
        print("[INFO] Update .env file with your token")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    is_private = "--public" not in sys.argv
    
    result = push_model_to_hub(
        repo_id=repo_id,
        private=is_private
    )
    
    sys.exit(0 if result else 1)
