import re
import os
import logging
from typing import Dict, Any, Union, List

import numpy as np
import torch
from faster_whisper import WhisperModel as FasterWhisperModel

from ..base import BaseModel

torch.set_num_threads(1)
logger = logging.getLogger(__name__)


SENTENCE_END_RE = re.compile(r"[.!?。！？]+(?:[\'\"\)\]]+)?\s*$")
MAX_TIMESTAMP_SEGMENT_SEC = 5.0


class WhisperModel(BaseModel):
    def __init__(self, model_path: str, from_hf: bool = False):
        super().__init__(model_path=model_path, from_hf=from_hf)
        self.device, self.device_label, self.compute_type = self._resolve_device()
        self._load_model()

    def _resolve_device(self):
        requested = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
        cuda_available = torch.cuda.is_available()

        if requested not in {"auto", "cpu", "cuda", "xpu", "mps", "directml"}:
            logger.warning(
                "Invalid WHISPER_DEVICE=%s. Expected one of: auto, cpu, cuda. Falling back to auto.",
                requested,
            )
            requested = "auto"

        if requested in {"xpu", "mps", "directml"}:
            logger.warning(
                "WHISPER_DEVICE=%s is not supported by faster-whisper. Falling back to auto.",
                requested,
            )
            requested = "auto"

        if requested == "cuda":
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                logger.info("Whisper device selected: cuda (%s)", device_name)
                return "cuda", f"cuda ({device_name})", self._resolve_compute_type("cuda")

            logger.warning(
                "WHISPER_DEVICE=cuda but CUDA is not available. Falling back to CPU."
            )
            return "cpu", "cpu", self._resolve_compute_type("cpu")

        if requested == "cpu":
            logger.info("Whisper device selected: cpu (WHISPER_DEVICE=cpu)")
            return "cpu", "cpu", self._resolve_compute_type("cpu")

        # Auto mode
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            logger.info("Whisper device selected: cuda (%s)", device_name)
            return "cuda", f"cuda ({device_name})", self._resolve_compute_type("cuda")

        logger.info("Whisper device selected: cpu (no CUDA available)")
        return "cpu", "cpu", self._resolve_compute_type("cpu")

    def _resolve_compute_type(self, device: str) -> str:
        requested = os.getenv("WHISPER_COMPUTE_TYPE", "").strip().lower()
        if requested:
            return requested

        if device == "cuda":
            return "float16"

        return "int8"

    @property
    def model_type(self) -> str:
        return "whisper"

    # ================= INPUT =================
    def process_input(self, input_data: Union[str, np.ndarray]) -> Union[str, np.ndarray]:
        if isinstance(input_data, str):
            return input_data
        if isinstance(input_data, np.ndarray):
            return input_data

        raise TypeError(f"Invalid input type: {type(input_data)}")

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.split()).strip()

    def _segments_to_sentences(self, segments: List[Dict[str, Any]]):
        sentences = []
        current_parts = []
        current_start = None
        current_end = None

        for segment in segments:
            text = self._normalize_text(segment.get("text", ""))
            if not text:
                continue

            if current_start is None:
                current_start = segment["start"]

            current_parts.append(text)
            current_end = segment["end"]

            if SENTENCE_END_RE.search(text):
                sentences.append({
                    "start": round(current_start, 2),
                    "end": round(current_end, 2),
                    "text": self._normalize_text(" ".join(current_parts)),
                })
                current_parts = []
                current_start = None
                current_end = None

        if current_parts and current_start is not None and current_end is not None:
            sentences.append({
                "start": round(current_start, 2),
                "end": round(current_end, 2),
                "text": self._normalize_text(" ".join(current_parts)),
            })

        return sentences

    def _to_timestamps(self, segments):
        sentences = self._segments_to_sentences(segments)
        timestamps = []

        for item in sentences:
            text = item.get("text", "").strip()
            if not text:
                continue

            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            duration = max(0.0, end - start)

            if duration <= MAX_TIMESTAMP_SEGMENT_SEC:
                timestamps.append({"start": round(start, 2), "text": text})
                continue

            words = text.split()
            chunk_count = int(np.ceil(duration / MAX_TIMESTAMP_SEGMENT_SEC))
            if len(words) < chunk_count:
                chunk_count = 1

            base_size = len(words) // chunk_count
            remainder = len(words) % chunk_count

            index = 0
            for i in range(chunk_count):
                size = base_size + (1 if i < remainder else 0)
                if size <= 0:
                    continue

                chunk_words = words[index:index + size]
                index += size

                if not chunk_words:
                    continue

                chunk_start = start + (duration * (i / chunk_count))
                timestamps.append({
                    "start": round(chunk_start, 2),
                    "text": " ".join(chunk_words),
                })

        return timestamps

    # ================= INFER =================
    def infer(self, processed_input: Union[str, np.ndarray]) -> Dict[str, Any]:
        segments, info = self.model.transcribe(
            processed_input,
            beam_size=3,
        )

        detected_lang = getattr(info, "language", None) or "en"

        segment_items = []
        text_parts = []

        for segment in segments:
            text = self._normalize_text(segment.text)
            if not text:
                continue

            segment_items.append({
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": text,
            })
            text_parts.append(text)

        full_text = " ".join(text_parts).strip()

        return {
            "text": full_text,
            "language": detected_lang,
            "model": "whisper",
            "timestamps": self._to_timestamps(segment_items),
        }

    # ================= LOAD =================
    def _load_model(self):
        self.model = FasterWhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
        )

        print(
            f"[INFO] Loaded faster-whisper model: {self.model_path} | "
            f"device={self.device_label} | compute_type={self.compute_type}"
        )

    # ================= PUBLIC =================
    def transcribe(self, audio_file: str) -> Dict[str, Any]:
        audio = self.process_input(audio_file)
        return self.infer(audio)