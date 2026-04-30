import warnings
import re
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


SENTENCE_END_RE = re.compile(r"[.!?。！？]+(?:[\'\"\)\]]+)?\s*$")
MAX_TIMESTAMP_SEGMENT_SEC = 5.0


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

    def _generate_with_timestamps(self, inputs, forced_decoder_ids):
        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs,
                max_new_tokens=80,
                num_beams=3,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                length_penalty=0.8,
                early_stopping=True,
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps=True,
            )

        decoded = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            output_offsets=True
        )[0]

        return decoded.get("text", "").strip(), decoded.get("offsets", [])

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

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.split()).strip()

    def _offsets_to_segments(self, offsets, base_time: float = 0.0):
        segments = []

        for offset in offsets or []:
            text = self._normalize_text(offset.get("text", ""))
            if not text:
                continue

            start, end = offset.get("timestamp", (0.0, 0.0))
            segments.append({
                "start": round(base_time + float(start), 2),
                "end": round(base_time + float(end), 2),
                "text": text,
            })

        return segments

    def _segments_to_sentences(self, segments):
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

        all_segments = []

        # ===== short audio =====
        if total_len <= 30 * sr:
            text, offsets = self._generate_with_timestamps(inputs, forced_decoder_ids)
            all_segments = self._offsets_to_segments(offsets)
            return {
                "text": text,
                "language": detected_lang,
                "model": "whisper",
                "timestamps": self._to_timestamps(all_segments)
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

            text, offsets = self._generate_with_timestamps(chunk_inputs, forced_decoder_ids)
            chunk_segments = self._offsets_to_segments(offsets, base_time=start / sr)

            if all_segments and chunk_segments:
                last_end = all_segments[-1]["end"]

                for segment in chunk_segments:
                    if segment["end"] <= last_end + 0.1:
                        continue

                    if segment["start"] < last_end:
                        segment["start"] = round(last_end, 2)

                    all_segments.append(segment)

            elif chunk_segments:
                all_segments.extend(chunk_segments)

            if text:
                result_text = self._merge_text(result_text, text)

            if end >= total_len:
                break

            start += stride

        return {
            "text": result_text.strip(),
            "language": detected_lang,
            "model": "whisper",
            "timestamps": self._to_timestamps(all_segments)
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