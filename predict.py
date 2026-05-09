"""
Cog predictor for Replicate - Load Whisper from Hugging Face
"""

import os
from typing import Optional
import cog
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from pathlib import Path


class Predictor(cog.Predictor):
    def setup(self):
        """Load model from Hugging Face on startup"""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

        hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )

        model_id = "shannonnonshan/streamland-whisper"

        print(f"Loading model: {model_id}")

        from_pretrained_kwargs = {
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }

        if hf_token:
            from_pretrained_kwargs["token"] = hf_token

        # Processor
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            token=hf_token
        )

        # Model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            **from_pretrained_kwargs
        ).to(self.device)

        # Pipeline
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            chunk_length_s=30,
        )

        print(f"✓ Loaded on {self.device}")

    @cog.input(
        "audio",
        type=cog.Path,
        description="Audio file"
    )
    @cog.input(
        "language",
        type=str,
        default="",
        description="Language code like en, vi"
    )
    def predict(
        self,
        audio: cog.Path,
        language: str = ""
    ) -> dict:

        audio_path = str(audio)

        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        print(f"Processing: {audio_path}")

        generate_kwargs = {
            "task": "transcribe"
        }

        if language:
            generate_kwargs["language"] = language

        result = self.pipe(
            audio_path,
            generate_kwargs=generate_kwargs,
            return_timestamps=True
        )

        return {
            "text": result.get("text", ""),
            "chunks": result.get("chunks", []),
            "language": language if language else "auto"
        }