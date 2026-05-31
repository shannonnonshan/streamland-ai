import re
import os
import logging
import librosa
import soundfile as sf

from typing import Dict, Any, Union, List, Optional

import numpy as np
import torch

from faster_whisper import WhisperModel as FasterWhisperModel

from ..base import BaseModel
from silero_vad import load_silero_vad, get_speech_timestamps as _get_ts
torch.set_num_threads(1)

logger = logging.getLogger(__name__)

# =========================================================
# CONSTANTS
# =========================================================

SENTENCE_END_RE = re.compile(
    r"[.!?。！？]+(?:[\'\"\)\]]+)?\s*$"
)

MAX_TIMESTAMP_SEGMENT_SEC = 5.0

MIN_TRANSCRIPT_WORDS = 5
MIN_TRANSCRIPT_CHARS = 20

ALLOWED_LANGUAGES = {"vi", "en"}
FALLBACK_LANGUAGE = "en"

MUSIC_KEYWORDS = {
    "music", "♪", "[music]", "(music)",
    "[nhạc]", "[intro]", "[outro]",
    "applause", "[applause]", "(applause)",
    "[noise]", "[sound]", "[laughter]",
}

HALLUCINATION_PATTERNS = [
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "cảm ơn các bạn đã xem",
    "đăng ký kênh",
]

# =========================================================
# WHISPER MODEL
# =========================================================

class WhisperModel(BaseModel):

    def __init__(self, model_path: str, from_hf: bool = False):
        super().__init__(model_path=model_path, from_hf=from_hf)
        self.device, self.device_label, self.compute_type = self._resolve_device()
        self.vad_model = None
        self.get_speech_timestamps = None
        self.read_audio = None
        self._load_model()
        self._load_vad()

    # =====================================================
    # DEVICE
    # =====================================================

    def _resolve_device(self):
        requested = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
        cuda_available = torch.cuda.is_available()

        if requested not in {"auto", "cpu", "cuda", "xpu", "mps", "directml"}:
            logger.warning("Invalid WHISPER_DEVICE=%s", requested)
            requested = "auto"

        if requested in {"xpu", "mps", "directml"}:
            logger.warning("Unsupported device=%s", requested)
            requested = "auto"

        if requested == "cuda":
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                logger.info("Whisper device: cuda (%s)", device_name)
                return ("cuda", f"cuda ({device_name})", self._resolve_compute_type("cuda"))
            logger.warning("CUDA requested but unavailable")
            return ("cpu", "cpu", self._resolve_compute_type("cpu"))

        if requested == "cpu":
            logger.info("Whisper device: cpu")
            return ("cpu", "cpu", self._resolve_compute_type("cpu"))

        # auto
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            logger.info("Whisper device: cuda (%s)", device_name)
            return ("cuda", f"cuda ({device_name})", self._resolve_compute_type("cuda"))

        logger.info("Whisper device: cpu")
        return ("cpu", "cpu", self._resolve_compute_type("cpu"))

    def _resolve_compute_type(self, device: str) -> str:
        requested = os.getenv("WHISPER_COMPUTE_TYPE", "").strip().lower()
        if requested:
            return requested
        return "float16" if device == "cuda" else "int8"

    # =====================================================
    # MODEL TYPE
    # =====================================================

    @property
    def model_type(self) -> str:
        return "whisper"

    # =====================================================
    # LOAD MODEL
    # =====================================================

    def _load_model(self):
        self.model = FasterWhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info(
            "Loaded faster-whisper: %s | device=%s | compute_type=%s",
            self.model_path, self.device_label, self.compute_type,
        )

    # =====================================================
    # LOAD SILERO VAD
    # =====================================================

    def _load_vad(self):
        logger.info("Loading Silero VAD...")
        try:
            self.vad_model = load_silero_vad()
            self.get_speech_timestamps = _get_ts
            logger.info("Silero VAD loaded via silero-vad package")
        except ImportError:
            logger.warning("silero-vad package not found, falling back to torch.hub")
            self.vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            (self.get_speech_timestamps, _, self.read_audio, *_) = utils

    # =====================================================
    # INPUT
    # =====================================================

    def process_input(self, input_data: Union[str, np.ndarray]) -> Union[str, np.ndarray]:
        if isinstance(input_data, (str, np.ndarray)):
            return input_data
        raise TypeError(f"Invalid input type: {type(input_data)}")

    # =====================================================
    # NORMALIZE TEXT
    # =====================================================

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.split()).strip()

    # =====================================================
    # SPEECH-ONLY EXTRACTION
    # =====================================================
    SPEECH_PADDING_SAMPLES = int(0.2 * 16000)  # 200ms padding before each speech chunk

    def _extract_speech_only(
        self,
        audio_path: str,
        output_path: str,
        threshold: float = 0.3,
    ) -> Optional[List[Dict[str, float]]]:
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav)

        speech_timestamps = self.get_speech_timestamps(
            wav_tensor,
            self.vad_model,
            threshold=threshold,
            min_speech_duration_ms=200,
            min_silence_duration_ms=200,
            sampling_rate=16000,
        )

        if not speech_timestamps:
            logger.warning("No speech detected in: %s", audio_path)
            return None

        chunks = []
        offset_map = []
        stitched_cursor = 0.0

        for ts in speech_timestamps:
            start_sample = ts["start"]
            end_sample = ts["end"]

            # Pad start backward to preserve a little silence before speech.
            # This gives faster-whisper enough context to align timestamps correctly,
            # preventing subtitles from appearing slightly ahead of the audio.
            padded_start = max(0, start_sample - self.SPEECH_PADDING_SAMPLES)

            chunk = wav[padded_start:end_sample]
            chunks.append(chunk)

            original_start_sec = padded_start / 16000
            offset_map.append((stitched_cursor, original_start_sec))

            stitched_cursor += len(chunk) / 16000

        stitched = np.concatenate(chunks)
        sf.write(output_path, stitched, 16000)

        self._vad_offset_map = offset_map

        logger.info(
            "VAD: %d speech windows | %.1fs → %.1fs",
            len(speech_timestamps),
            len(wav) / 16000,
            len(stitched) / 16000,
        )

        return [
            {"start": ts["start"] / 16000, "end": ts["end"] / 16000}
            for ts in speech_timestamps
        ]
    
    # =====================================================
    # CORRECT TIMESTAMP — FIX: duyệt xuôi, tìm đúng block
    # =====================================================

    def _correct_timestamp(self, stitched_sec: float) -> float:
        if not hasattr(self, "_vad_offset_map"):
            return stitched_sec

        corrected = stitched_sec
        for i, (stitched_start, original_start) in enumerate(self._vad_offset_map):
            if i + 1 < len(self._vad_offset_map):
                stitched_end = self._vad_offset_map[i + 1][0]
            else:
                stitched_end = float("inf")

            if stitched_start <= stitched_sec < stitched_end:
                corrected = original_start + (stitched_sec - stitched_start)
                break

        return corrected

    # =====================================================
    # LANGUAGE VALIDATION
    # =====================================================

    def _validate_language(self, detected: str) -> str:
        lang = (detected or "").strip().lower()
        if lang in ALLOWED_LANGUAGES:
            return lang
        logger.warning(
            "Unexpected language detected: '%s' → falling back to '%s'",
            lang, FALLBACK_LANGUAGE,
        )
        return FALLBACK_LANGUAGE

    # =====================================================
    # TRANSCRIPT QUALITY VALIDATION
    # =====================================================

    def _validate_transcript(
        self,
        full_text: str,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        word_count = len(full_text.split())
        char_count = len(full_text)
        seg_count = len(segments)
        issues = []

        if word_count < MIN_TRANSCRIPT_WORDS:
            issues.append(f"too short: {word_count} words (min {MIN_TRANSCRIPT_WORDS})")
        if char_count < MIN_TRANSCRIPT_CHARS:
            issues.append(f"too few chars: {char_count} (min {MIN_TRANSCRIPT_CHARS})")
        if seg_count == 0:
            issues.append("no valid segments survived filtering")

        valid = len(issues) == 0

        if not valid:
            logger.warning("Transcript quality check failed: %s", "; ".join(issues))
        else:
            logger.info("Transcript quality OK — %d words, %d segments", word_count, seg_count)

        return {
            "valid": valid,
            "word_count": word_count,
            "segment_count": seg_count,
            "issues": issues,
        }

    # =====================================================
    # SEGMENT CLEANUP — FIX: giảm filter từ <= 2 → <= 1
    # =====================================================

    def _is_valid_segment(
        self,
        text: str,
        segment: Any,
        previous_text: Optional[str],
    ) -> bool:
        if len(text.split()) <= 1:  # FIX: từ <= 2 → <= 1
            return False

        if hasattr(segment, "no_speech_prob") and segment.no_speech_prob > 0.5:
            return False

        lower = text.lower()

        if any(kw in lower for kw in MUSIC_KEYWORDS):
            return False

        if any(p in lower for p in HALLUCINATION_PATTERNS):
            return False

        duration = float(segment.end) - float(segment.start)
        if duration < 1.0 and len(text.split()) > 6:
            return False

        if previous_text is not None and previous_text == lower:
            return False

        return True

    # =====================================================
    # SENTENCE SEGMENTS
    # =====================================================

    def _segments_to_sentences(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

        if current_parts and current_start is not None:
            sentences.append({
                "start": round(current_start, 2),
                "end": round(current_end, 2),
                "text": self._normalize_text(" ".join(current_parts)),
            })

        return sentences

    # =====================================================
    # TIMESTAMPS
    # =====================================================

    def _to_timestamps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            chunk_count = max(1, min(int(np.ceil(duration / MAX_TIMESTAMP_SEGMENT_SEC)), len(words)))
            base_size = len(words) // chunk_count
            remainder = len(words) % chunk_count
            index = 0

            for i in range(chunk_count):
                size = base_size + (1 if i < remainder else 0)
                if size <= 0:
                    continue

                chunk_words = words[index: index + size]
                index += size

                if not chunk_words:
                    continue

                chunk_start = start + duration * (i / chunk_count)
                timestamps.append({"start": round(chunk_start, 2), "text": " ".join(chunk_words)})

        return timestamps

    # =====================================================
    # INFER
    # =====================================================

    def infer(self, processed_input: Union[str, np.ndarray]) -> Dict[str, Any]:

        # STEP 1 — VAD
        base = os.path.splitext(processed_input)[0]
        speech_audio_path = f"{base}.speech.wav"

        speech_windows = self._extract_speech_only(processed_input, speech_audio_path)

        if speech_windows is None:
            logger.warning("Returning empty transcript — no speech found")
            return self._empty_result()

        # STEP 2 — WHISPER
        segments_gen, info = self.model.transcribe(
            speech_audio_path,
            beam_size=5,        # FIX: tăng từ 3 → 5 để chính xác hơn
            vad_filter=False,
        )

        # STEP 3 — LANGUAGE
        detected_lang = self._validate_language(getattr(info, "language", None) or "")

        # STEP 4 — SEGMENT CLEANUP
        segment_items: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        previous_lower: Optional[str] = None

        for segment in segments_gen:
            text = self._normalize_text(segment.text)

            if not self._is_valid_segment(text, segment, previous_lower):
                continue

            original_start = self._correct_timestamp(float(segment.start))
            original_end = self._correct_timestamp(float(segment.end))

            item = {
                "start": round(original_start, 2),
                "end": round(original_end, 2),
                "text": text,
            }

            segment_items.append(item)
            text_parts.append(text)
            previous_lower = text.lower()

        # STEP 5 — CLEANUP
        try:
            os.remove(speech_audio_path)
        except Exception:
            pass

        # STEP 6 — QUALITY CHECK
        full_text = " ".join(text_parts).strip()
        quality = self._validate_transcript(full_text, segment_items)

        if not quality["valid"]:
            logger.warning("Transcript failed quality gate — returning empty result")
            return self._empty_result(quality=quality)

        # STEP 7 — RETURN
        return {
            "text": full_text,
            "language": detected_lang,
            "model": "whisper",
            "timestamps": self._to_timestamps(segment_items),
            "quality": quality,
        }

    # =====================================================
    # EMPTY RESULT
    # =====================================================

    def _empty_result(self, quality: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "text": "",
            "language": FALLBACK_LANGUAGE,
            "model": "whisper",
            "timestamps": [],
            "quality": quality or {
                "valid": False,
                "word_count": 0,
                "segment_count": 0,
                "issues": ["no speech detected"],
            },
        }

    # =====================================================
    # PUBLIC
    # =====================================================

    def transcribe(self, audio_file: str) -> Dict[str, Any]:
        audio = self.process_input(audio_file)
        return self.infer(audio)