import re
import os
import logging
import librosa
import soundfile as sf

from typing import Dict, Any, Union, List

import numpy as np
import torch

from faster_whisper import (
    WhisperModel as FasterWhisperModel
)

from ..base import BaseModel

torch.set_num_threads(1)

logger = logging.getLogger(__name__)

# =========================================================
# CONSTANTS
# =========================================================

SENTENCE_END_RE = re.compile(
    r"[.!?。！？]+(?:[\'\"\)\]]+)?\s*$"
)

MAX_TIMESTAMP_SEGMENT_SEC = 5.0

MUSIC_KEYWORDS = {
    "music",
    "♪",
    "[music]",
    "(music)",
    "[nhạc]",
    "[intro]",
    "[outro]",
    "applause",
    "[applause]",
    "(applause)",
    "[noise]",
    "[sound]",
    "[laughter]"
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

    def __init__(
        self,
        model_path: str,
        from_hf: bool = False
    ):

        super().__init__(
            model_path=model_path,
            from_hf=from_hf
        )

        (
            self.device,
            self.device_label,
            self.compute_type
        ) = self._resolve_device()

        self.vad_model = None
        self.get_speech_timestamps = None
        self.read_audio = None

        self._load_model()
        self._load_vad()

    # =====================================================
    # DEVICE
    # =====================================================

    def _resolve_device(self):

        requested = os.getenv(
            "WHISPER_DEVICE",
            "auto"
        ).strip().lower()

        cuda_available = (
            torch.cuda.is_available()
        )

        if requested not in {
            "auto",
            "cpu",
            "cuda",
            "xpu",
            "mps",
            "directml"
        }:

            logger.warning(
                "Invalid WHISPER_DEVICE=%s",
                requested
            )

            requested = "auto"

        if requested in {
            "xpu",
            "mps",
            "directml"
        }:

            logger.warning(
                "Unsupported device=%s",
                requested
            )

            requested = "auto"

        # =================================================
        # CUDA
        # =================================================

        if requested == "cuda":

            if cuda_available:

                device_name = (
                    torch.cuda.get_device_name(0)
                )

                logger.info(
                    "Whisper device: cuda (%s)",
                    device_name
                )

                return (
                    "cuda",
                    f"cuda ({device_name})",
                    self._resolve_compute_type(
                        "cuda"
                    )
                )

            logger.warning(
                "CUDA requested but unavailable"
            )

            return (
                "cpu",
                "cpu",
                self._resolve_compute_type(
                    "cpu"
                )
            )

        # =================================================
        # CPU
        # =================================================

        if requested == "cpu":

            logger.info(
                "Whisper device: cpu"
            )

            return (
                "cpu",
                "cpu",
                self._resolve_compute_type(
                    "cpu"
                )
            )

        # =================================================
        # AUTO
        # =================================================

        if cuda_available:

            device_name = (
                torch.cuda.get_device_name(0)
            )

            logger.info(
                "Whisper device: cuda (%s)",
                device_name
            )

            return (
                "cuda",
                f"cuda ({device_name})",
                self._resolve_compute_type(
                    "cuda"
                )
            )

        logger.info(
            "Whisper device: cpu"
        )

        return (
            "cpu",
            "cpu",
            self._resolve_compute_type(
                "cpu"
            )
        )

    def _resolve_compute_type(
        self,
        device: str
    ) -> str:

        requested = os.getenv(
            "WHISPER_COMPUTE_TYPE",
            ""
        ).strip().lower()

        if requested:
            return requested

        if device == "cuda":
            return "float16"

        return "int8"

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

        print(
            f"[INFO] Loaded faster-whisper: "
            f"{self.model_path} | "
            f"device={self.device_label} | "
            f"compute_type={self.compute_type}"
        )

    # =====================================================
    # LOAD SILERO VAD
    # =====================================================

    def _load_vad(self):
        logger.info("Loading Silero VAD...")

        try:
            self.vad_model, utils = (
                torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True
                )
            )

            (
                self.get_speech_timestamps,
                _,
                self.read_audio,
                *_
            ) = utils
        except Exception as e:
            logger.warning(
                "Silero VAD unavailable (%s). Falling back to direct transcription.",
                e,
            )
            self.vad_model = None
            self.get_speech_timestamps = None
            self.read_audio = None

    # =====================================================
    # INPUT
    # =====================================================

    def process_input(
        self,
        input_data: Union[str, np.ndarray]
    ) -> Union[str, np.ndarray]:

        if isinstance(
            input_data,
            str
        ):
            return input_data

        if isinstance(
            input_data,
            np.ndarray
        ):
            return input_data

        raise TypeError(
            f"Invalid input type: "
            f"{type(input_data)}"
        )

    # =====================================================
    # NORMALIZE
    # =====================================================

    def _normalize_text(
        self,
        text: str
    ) -> str:

        return " ".join(
            text.split()
        ).strip()

    # =====================================================
    # SPEECH ONLY AUDIO
    # =====================================================

    def _extract_speech_only(
        self,
        audio_path: str,
        output_path: str,
        threshold: float = 0.5
    ) -> bool:

        if self.vad_model is None or self.get_speech_timestamps is None:
            wav, _ = librosa.load(
                audio_path,
                sr=16000,
                mono=True
            )
            sf.write(output_path, wav, 16000)
            return True

        # =========================================
        # LOAD AUDIO WITH LIBROSA
        # =========================================

        wav, sr = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        wav = torch.from_numpy(
            wav
        )

        # =========================================
        # DETECT SPEECH
        # =========================================

        speech_timestamps = (
            self.get_speech_timestamps(
                wav,
                self.vad_model,
                threshold=threshold,
                min_speech_duration_ms=500,
                min_silence_duration_ms=300,
                sampling_rate=16000
            )
        )

        if not speech_timestamps:

            logger.warning(
                "No speech detected"
            )

            return False

        # =========================================
        # BUILD SPEECH AUDIO
        # =========================================

        speech_chunks = []

        for ts in speech_timestamps:

            start = ts["start"]
            end = ts["end"]

            speech_chunks.append(
                wav[start:end]
            )

        speech_audio = torch.cat(
            speech_chunks
        )

        # =========================================
        # SAVE AUDIO
        # =========================================

        sf.write(
            output_path,
            speech_audio.numpy(),
            16000
        )

        logger.info(
            "Speech-only audio generated"
        )

        return True

    # =====================================================
    # SENTENCE SEGMENTS
    # =====================================================

    def _segments_to_sentences(
        self,
        segments: List[Dict[str, Any]]
    ):

        sentences = []

        current_parts = []
        current_start = None
        current_end = None

        for segment in segments:

            text = self._normalize_text(
                segment.get("text", "")
            )

            if not text:
                continue

            if current_start is None:
                current_start = (
                    segment["start"]
                )

            current_parts.append(text)

            current_end = (
                segment["end"]
            )

            if SENTENCE_END_RE.search(
                text
            ):

                sentences.append({
                    "start": round(
                        current_start,
                        2
                    ),
                    "end": round(
                        current_end,
                        2
                    ),
                    "text": self._normalize_text(
                        " ".join(
                            current_parts
                        )
                    ),
                })

                current_parts = []
                current_start = None
                current_end = None

        if (
            current_parts
            and current_start is not None
            and current_end is not None
        ):

            sentences.append({
                "start": round(
                    current_start,
                    2
                ),
                "end": round(
                    current_end,
                    2
                ),
                "text": self._normalize_text(
                    " ".join(current_parts)
                ),
            })

        return sentences

    # =====================================================
    # TIMESTAMPS
    # =====================================================

    def _to_timestamps(
        self,
        segments
    ):

        sentences = (
            self._segments_to_sentences(
                segments
            )
        )

        timestamps = []

        for item in sentences:

            text = item.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            start = float(
                item.get(
                    "start",
                    0.0
                )
            )

            end = float(
                item.get(
                    "end",
                    start
                )
            )

            duration = max(
                0.0,
                end - start
            )

            # =============================================
            # SHORT SEGMENT
            # =============================================

            if (
                duration
                <= MAX_TIMESTAMP_SEGMENT_SEC
            ):

                timestamps.append({
                    "start": round(
                        start,
                        2
                    ),
                    "text": text
                })

                continue

            # =============================================
            # SPLIT LONG SEGMENT
            # =============================================

            words = text.split()

            chunk_count = int(
                np.ceil(
                    duration
                    / MAX_TIMESTAMP_SEGMENT_SEC
                )
            )

            if len(words) < chunk_count:
                chunk_count = 1

            base_size = (
                len(words)
                // chunk_count
            )

            remainder = (
                len(words)
                % chunk_count
            )

            index = 0

            for i in range(chunk_count):

                size = base_size + (
                    1 if i < remainder
                    else 0
                )

                if size <= 0:
                    continue

                chunk_words = words[
                    index:index + size
                ]

                index += size

                if not chunk_words:
                    continue

                chunk_start = (
                    start
                    + (
                        duration
                        * (
                            i
                            / chunk_count
                        )
                    )
                )

                timestamps.append({
                    "start": round(
                        chunk_start,
                        2
                    ),
                    "text": " ".join(
                        chunk_words
                    ),
                })

        return timestamps

    # =====================================================
    # INFER
    # =====================================================

    def infer(
        self,
        processed_input: Union[
            str,
            np.ndarray
        ]
    ) -> Dict[str, Any]:

        use_pre_vad = os.getenv("WHISPER_USE_PRE_VAD", "true").strip().lower() == "true"
        use_pre_vad = use_pre_vad and self.vad_model is not None and self.get_speech_timestamps is not None

        transcribe_input = processed_input
        speech_audio_path = f"{processed_input}.speech.wav"

        if use_pre_vad:
            has_speech = self._extract_speech_only(
                processed_input,
                speech_audio_path
            )

            if not has_speech:
                return {
                    "text": "",
                    "language": "en",
                    "model": "whisper",
                    "timestamps": [],
                }

            transcribe_input = speech_audio_path

        # =================================================
        # TRANSCRIBE
        # =================================================

        segments, info = (
            self.model.transcribe(
                transcribe_input,
                beam_size=3,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500
                )
            )
        )

        detected_lang = (
            getattr(
                info,
                "language",
                None
            )
            or "en"
        )

        segment_items = []
        text_parts = []

        # =================================================
        # FILTER SEGMENTS
        # =================================================

        for segment in segments:

            text = (
                self._normalize_text(
                    segment.text
                )
            )

            if not text:
                continue

            # =============================================
            # SHORT
            # =============================================

            if len(text.split()) <= 2:
                continue

            # =============================================
            # LOW CONFIDENCE
            # =============================================

            if hasattr(
                segment,
                "no_speech_prob"
            ):

                if (
                    segment.no_speech_prob
                    > 0.5
                ):
                    continue

            lower_text = (
                text.lower()
            )

            # =============================================
            # MUSIC / NOISE
            # =============================================

            if any(
                kw in lower_text
                for kw in MUSIC_KEYWORDS
            ):
                continue

            # =============================================
            # HALLUCINATION
            # =============================================

            if any(
                p in lower_text
                for p in (
                    HALLUCINATION_PATTERNS
                )
            ):
                continue

            # =============================================
            # BAD TIMING
            # =============================================

            seg_duration = (
                float(segment.end)
                - float(segment.start)
            )

            words = text.split()

            if (
                seg_duration < 1.0
                and len(words) > 6
            ):
                continue

            # =============================================
            # DUPLICATE
            # =============================================

            if segment_items:

                last_text = (
                    segment_items[-1]["text"]
                    .lower()
                )

                if last_text == lower_text:
                    continue

            item = {
                "start": round(
                    float(segment.start),
                    2
                ),
                "end": round(
                    float(segment.end),
                    2
                ),
                "text": text,
            }

            segment_items.append(
                item
            )

            text_parts.append(
                text
            )

        # =================================================
        # CLEANUP
        # =================================================

        if use_pre_vad:
            try:
                os.remove(speech_audio_path)
            except Exception:
                pass

        full_text = " ".join(
            text_parts
        ).strip()

        return {
            "text": full_text,
            "language": detected_lang,
            "model": "whisper",
            "timestamps": (
                self._to_timestamps(
                    segment_items
                )
            ),
        }

    # =====================================================
    # PUBLIC
    # =====================================================

    def transcribe(
        self,
        audio_file: str
    ) -> Dict[str, Any]:

        audio = self.process_input(
            audio_file
        )

        return self.infer(audio)