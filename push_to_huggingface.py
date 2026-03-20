"""
Push Model to Hugging Face Hub
Purpose: Script to upload a fine-tuned model to Hugging Face Hub.
         Uses HF_USERNAME and HF_REPO_BASE from .env to build the repo ID
         as  {HF_USERNAME}/{HF_REPO_BASE}-{model_name}
         so the same script works for any model in the project.
Requires: huggingface_hub package and HF_TOKEN / HF_USERNAME / HF_REPO_BASE in .env
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def main():
    if len(sys.argv) < 2:
        print("[ERROR] Usage: python push_to_huggingface.py <model_name> [--public]")
        print()
        print("Examples:")
        print("  python push_to_huggingface.py whisper")
        print("  python push_to_huggingface.py whisper --public")
        print()
        print("The repo is built from .env as:  HF_USERNAME/HF_REPO_BASE-<model_name>")
        print("  e.g. shannonnonshan/streamland-whisper")
        sys.exit(1)

    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token == "hf_your_hugging_face_token_here":
        print("[ERROR] HF_TOKEN not set in .env")
        print("[INFO] Get token from: https://huggingface.co/settings/tokens")
        print("[INFO] Update .env file with your token")
        sys.exit(1)

    model_name = sys.argv[1]
    is_private = "--public" not in sys.argv

    from utils.model_pusher import push_model_to_hub

    try:
        result = push_model_to_hub(
            model_name=model_name,
            private=is_private,
        )
        sys.exit(0 if result.get("status") == "success" else 1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
