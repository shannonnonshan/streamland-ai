"""
Setup Script - First Time Configuration
Purpose: Initialize .env file for Hugging Face authentication
"""

import os
import sys
from pathlib import Path


def setup_env():
    """Setup .env file interactively."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    print("=" * 60)
    print("StreamLand AI - Initial Setup")
    print("=" * 60)
    print()
    
    # Check if .env already exists
    if env_file.exists():
        print("[INFO] .env file already exists")
        response = input("Do you want to reconfigure? (y/n): ").strip().lower()
        if response != 'y':
            print("[INFO] Setup cancelled")
            return False
    
    # Get HF token
    print("\nHugging Face Configuration:")
    print("-" * 60)
    print("Get your token from: https://huggingface.co/settings/tokens")
    print("Create new token with 'write' permission")
    print()
    
    hf_token = input("Enter your HF token (hf_...): ").strip()
    if not hf_token.startswith("hf_"):
        print("[ERROR] Invalid HF token format. Must start with 'hf_'")
        return False
    
    # Get username and repo base
    hf_username = input("Enter your HF username: ").strip()
    if not hf_username:
        print("[ERROR] Username cannot be empty")
        return False
    
    hf_repo_base = input("Enter the repo base name (e.g. streamland): ").strip()
    if not hf_repo_base:
        print("[ERROR] Repo base name cannot be empty")
        return False
    
    print(f"\n[INFO] Repos will be created as: {hf_username}/{hf_repo_base}-<model_name>")
    print(f"       e.g. {hf_username}/{hf_repo_base}-whisper")
    
    # Create .env file
    try:
        with open(env_file, 'w') as f:
            f.write(f"# Hugging Face Configuration\n")
            f.write(f"HF_TOKEN={hf_token}\n")
            f.write(f"HF_USERNAME={hf_username}\n")
            f.write(f"HF_REPO_BASE={hf_repo_base}\n")
            f.write(f"\n")
            f.write(f"# Model Configuration\n")
            f.write(f"MODEL_PATH=models/whisper/model/whisper-finetuned\n")
            f.write(f"MODEL_USE_HF=false\n")
            f.write(f"\n")
            f.write(f"# API Configuration\n")
            f.write(f"API_HOST=0.0.0.0\n")
            f.write(f"API_PORT=8000\n")
            f.write(f"API_DEBUG=false\n")
            f.write(f"\n")
            f.write(f"# Audio Processing\n")
            f.write(f"AUDIO_SAMPLE_RATE=16000\n")
            f.write(f"AUDIO_CHUNK_LENGTH=30\n")
            f.write(f"\n")
            f.write(f"# Device Configuration\n")
            f.write(f"DEVICE=cuda:0\n")
            f.write(f"TORCH_DTYPE=float16\n")
        
        print("\n[INFO] .env file created successfully")
        print(f"[INFO] HF Username:  {hf_username}")
        print(f"[INFO] Repo base:    {hf_repo_base}")
        return True
    
    except Exception as e:
        print(f"[ERROR] Failed to create .env: {e}")
        return False


if __name__ == "__main__":
    success = setup_env()
    sys.exit(0 if success else 1)
