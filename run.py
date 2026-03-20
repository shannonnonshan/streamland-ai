import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.utils import logging as hf_logging
import librosa
import numpy as np

load_dotenv()

hf_logging.set_verbosity_error()
torch.set_num_threads(1)

HF_USERNAME = os.getenv("HF_USERNAME")
HF_REPO_BASE = os.getenv("HF_REPO_BASE", "streamland")

LOCAL_MODEL_PATH = "models/whisper/model/whisper-finetuned"
HF_MODEL_ID = f"{HF_USERNAME}/{HF_REPO_BASE}-whisper" if HF_USERNAME else None


def test_whisper(audio_file):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = LOCAL_MODEL_PATH if os.path.isdir(LOCAL_MODEL_PATH) else HF_MODEL_ID
    if not model_id:
        raise EnvironmentError(
            "Local model not found and HF_USERNAME is not set. "
            "Copy .env.example to .env and fill in the values."
        )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    samples, sr = librosa.load(audio_file, sr=16000)

    chunk_length = 30 * sr
    chunks = [samples[i:i + chunk_length] for i in range(0, len(samples), chunk_length)]

    full_transcript = []

    for chunk in chunks:

        inputs = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device, torch_dtype)

        with torch.no_grad():
            predicted_ids = model.generate(inputs)

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        full_transcript.append(text)

    return " ".join(full_transcript)


if __name__ == "__main__":

    result1 = test_whisper("utils/data/audio/testaudio.mp3")
    print(f"[Test 1] {result1}\n")

    result2 = test_whisper("utils/data/audio/testaudio-vn.mp3")
    print(f"[Test 2] {result2}")