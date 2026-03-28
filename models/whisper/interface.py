import warnings
import torch
import numpy as np
import librosa
from typing import Dict, Any, Union
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.utils import logging as hf_logging

from ..base import BaseModel

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
torch.set_num_threads(1)


class WhisperModel(BaseModel):
    def __init__(self, model_path: str, from_hf: bool = False):
        super().__init__(model_path=model_path, from_hf=from_hf)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._load_model()

    @property
    def model_type(self) -> str:
        return "whisper"

    # ================= INPUT =================
    def process_input(self, input_data: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(input_data, str):
            audio, _ = librosa.load(input_data, sr=16000)
        elif isinstance(input_data, np.ndarray):
            audio = input_data
        else:
            raise TypeError(f"Invalid input type: {type(input_data)}")

        # 🔥 normalize nhẹ
        audio = audio / (np.max(np.abs(audio)) + 1e-6)

        # 🔥 padding fix missing tail
        audio = np.concatenate([audio, np.zeros(int(1.0 * 16000))])

        return audio

    # ================= GENERATE =================
    def _generate(self, inputs, forced_decoder_ids):
        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs,
                max_new_tokens=80,
                num_beams=3,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                length_penalty=0.8,
                early_stopping=True,
                forced_decoder_ids=forced_decoder_ids
            )

        text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        return text

    # ================= LANGUAGE DETECT =================
    def _detect_language(self, inputs):
        with torch.no_grad():
            ids = self.model.generate(inputs, max_new_tokens=1)

        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].lower()

        if "vi" in text:
            return "vi"
        return "en"

    # ================= MERGE =================
    def _merge_text(self, prev, new):
        if not prev:
            return new

        # 🔥 remove overlap tail
        overlap_len = 30
        tail = prev[-overlap_len:]

        if tail in new:
            new = new.split(tail, 1)[-1]

        return prev + " " + new

    # ================= INFER =================
    def infer(self, processed_input: np.ndarray) -> Dict[str, Any]:
        sr = 16000
        total_len = len(processed_input)

        # ===== prepare full input (for language detect) =====
        inputs = self.processor(
            processed_input,
            sampling_rate=sr,
            return_tensors="pt"
        ).input_features.to(self.device, self.torch_dtype)

        # ===== detect language =====
        detected_lang = self._detect_language(inputs)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=detected_lang,
            task="transcribe"
        )

        # ===== short audio =====
        if total_len <= 30 * sr:
            text = self._generate(inputs, forced_decoder_ids)
            return {
                "text": text,
                "language": detected_lang,
                "model": "whisper"
            }

        # ===== long audio (chunking) =====
        chunk_size = 25 * sr
        overlap = 5 * sr
        stride = chunk_size - overlap
        min_chunk = 5 * sr

        result_text = ""
        start = 0

        while start < total_len:
            end = min(start + chunk_size, total_len)
            chunk = processed_input[start:end]

            if len(chunk) < min_chunk:
                break

            chunk_inputs = self.processor(
                chunk,
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features.to(self.device, self.torch_dtype)

            text = self._generate(chunk_inputs, forced_decoder_ids)

            if text:
                result_text = self._merge_text(result_text, text)

            if end >= total_len:
                break

            start += stride

        return {
            "text": result_text.strip(),
            "language": detected_lang,
            "model": "whisper"
        }

    # ================= LOAD =================
    def _load_model(self):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_path)

        print(f"[INFO] Loaded Whisper model: {self.model_path}")

    # ================= PUBLIC =================
    def transcribe(self, audio_file: str) -> Dict[str, Any]:
        audio = self.process_input(audio_file)
        return self.infer(audio)