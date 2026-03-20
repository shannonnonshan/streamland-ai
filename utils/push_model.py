import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
HF_REPO_BASE = os.getenv("HF_REPO_BASE", "streamland")


def push_model_to_hub(model_folder: str, local_model_path: str):
    """Push a model to Hugging Face Hub.

    The repository name is constructed as ``{HF_REPO_BASE}-{model_folder}``
    so that a single set of environment variables works for every model.

    Args:
        model_folder: Short name used as a suffix, e.g. ``"whisper"``.
        local_model_path: Path to the local directory containing model files.
    """
    missing = [v for v in ("HF_TOKEN", "HF_USERNAME") if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Required environment variable(s) not set: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in the values."
        )

    repo_name = f"{HF_REPO_BASE}-{model_folder}"
    repo_id = f"{HF_USERNAME}/{repo_name}"

    api = HfApi(token=HF_TOKEN)

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=local_model_path,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Model pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    push_model_to_hub(
        model_folder="whisper",
        local_model_path="models/whisper/model/whisper-finetuned",
    )
