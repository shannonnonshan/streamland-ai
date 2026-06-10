import os
import shutil
import re
import importlib
import logging
from typing import Any, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langdetect import detect, LangDetectException

from models.base import BaseModel
import threading
import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning)

HF_TOKEN = os.getenv("HF_TOKEN")
logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "streamland-ai",
    "summarization",
)

# Sentence-ending punctuation for post-processing trim
_SENTENCE_END_RE = re.compile(r'[.!?。！？]')
_VI_CHAR_RE = re.compile(
    r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'
    r'ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]'
)

class SummarizationModel(BaseModel):
    SUPPORTED_MODELS = {
        "vi": "VietAI/vit5-base-vietnews-summarization",
        "en": "shannonnonshan/bart-summarizer",
    }

    def __init__(self, model_path=None, from_hf=True):
        super().__init__(model_path or "shannonnonshan/bart-summarizer", from_hf)

        self.cache_dir = os.getenv("SUMMARIZATION_CACHE_DIR", DEFAULT_CACHE_DIR)
        self.device, self.device_label, self.torch_dtype, self.pipeline_device = self._resolve_device()

        self.tokenizer = None
        self.model = None
        self.vi_model = None
        self.vi_tokenizer = None
        self.vi_summarizer = None

        self.en_model = None
        self.en_tokenizer = None
        self.en_summarizer = None

        self.summarizer = None
        self.current_language = None
        self.current_pipeline_task = None
        self._model_lock = threading.Lock()
        self._vi_load_lock = threading.Lock()
        self._load_model()
        self.en_model = self.model
        self.en_tokenizer = self.tokenizer
        self.en_summarizer = self.summarizer
        self.current_language = "en"

    # =====================================================
    # DEVICE
    # =====================================================

    def _try_directml_device(self):
        try:
            torch_directml = importlib.import_module("torch_directml")
            return torch_directml.device()
        except Exception:
            return None

    def _resolve_device(self):
        requested = os.getenv("SUMMARIZATION_DEVICE", "auto").strip().lower()
        cuda_available = torch.cuda.is_available()
        xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        if requested not in {"auto", "cpu", "cuda", "xpu", "mps", "directml"}:
            logger.warning(
                "Invalid SUMMARIZATION_DEVICE=%s. Falling back to auto.", requested
            )
            requested = "auto"

        if requested == "cpu":
            logger.info("Summarization device: cpu")
            return "cpu", "cpu", torch.float32, -1

        if requested == "directml":
            dml_device = self._try_directml_device()
            if dml_device is not None:
                logger.info("Summarization device: directml")
                return dml_device, "directml", torch.float32, dml_device
            logger.warning("directml unavailable, falling back to cpu")
            return "cpu", "cpu", torch.float32, -1

        if requested == "cuda":
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                logger.info("Summarization device: cuda:0 (%s)", device_name)
                return "cuda:0", f"cuda:0 ({device_name})", torch.float16, 0
            logger.warning("CUDA unavailable, falling back to cpu")
            return "cpu", "cpu", torch.float32, -1

        if requested == "xpu":
            if xpu_available:
                logger.info("Summarization device: xpu:0")
                return "xpu:0", "xpu:0", torch.float16, "xpu:0"
            logger.warning("XPU unavailable, falling back to cpu")
            return "cpu", "cpu", torch.float32, -1

        if requested == "mps":
            if mps_available:
                logger.info("Summarization device: mps")
                return "mps", "mps", torch.float32, "mps"
            logger.warning("MPS unavailable, falling back to cpu")
            return "cpu", "cpu", torch.float32, -1

        # auto
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            logger.info("Summarization device: cuda:0 (%s)", device_name)
            return "cuda:0", f"cuda:0 ({device_name})", torch.float16, 0
        if xpu_available:
            logger.info("Summarization device: xpu:0")
            return "xpu:0", "xpu:0", torch.float16, "xpu:0"
        if mps_available:
            logger.info("Summarization device: mps")
            return "mps", "mps", torch.float32, "mps"
        dml_device = self._try_directml_device()
        if dml_device is not None:
            logger.info("Summarization device: directml")
            return dml_device, "directml", torch.float32, dml_device

        logger.info("Summarization device: cpu (no accelerator available)")
        return "cpu", "cpu", torch.float32, -1

    # =====================================================
    # LANGUAGE DETECTION
    # =====================================================

    def detect_language(self, text: str) -> str:
        """
        Detect whether text is Vietnamese or English.
        Never returns None — defaults to 'en' if uncertain.
        """
        try:
            detected = detect(text)
            if detected.startswith("vi"):
                return "vi"
            if detected.startswith("en"):
                return "en"
            logger.warning(
                "langdetect returned unsupported language '%s' (len=%d) — trying heuristic",
                detected, len(text),
            )
        except LangDetectException as e:
            logger.warning("langdetect failed: %s — trying heuristic", e)
        vi_chars = len(_VI_CHAR_RE.findall(text))
        ratio = vi_chars / max(1, len(text))
        logger.info(
            "Language heuristic: vi_char_ratio=%.3f (vi=%d, total=%d)",
            ratio, vi_chars, len(text),
        )
        if ratio > 0.05:
            return "vi"

        logger.info("Language defaulting to 'en'")
        return "en"


    
    def _switch_model_for_language(self, language):
        if language == "en":
            if self.vi_model is not None:
                del self.vi_model
                del self.vi_tokenizer
                if self.vi_summarizer:
                    del self.vi_summarizer
                self.vi_model = None
                self.vi_tokenizer = None
                self.vi_summarizer = None
                gc.collect()

            self.model = self.en_model
            self.tokenizer = self.en_tokenizer
            self.summarizer = self.en_summarizer
            self.current_language = "en"
            self.model_path = self.SUPPORTED_MODELS["en"]
            return True

        if language == "vi":
            if self.vi_model is None:
                with self._vi_load_lock:
                    if self.vi_model is None:
                        old_path = self.model_path
                        self.model_path = self.SUPPORTED_MODELS["vi"]
                        self._load_model()
                        self.vi_model = self.model
                        self.vi_tokenizer = self.tokenizer
                        self.vi_summarizer = self.summarizer
                        self.model_path = old_path

            self.model = self.vi_model
            self.tokenizer = self.vi_tokenizer
            self.summarizer = self.vi_summarizer
            self.current_language = "vi"
            self.model_path = self.SUPPORTED_MODELS["vi"]
            return True

        return False
    
    
    @property
    def model_type(self) -> str:
        return "summarization"

    # =====================================================
    # CACHE
    # =====================================================

    def _clear_model_cache(self, candidate: str):
        hf_cache = os.path.join(
            os.path.expanduser("~"),
            ".cache", "huggingface", "hub",
            f"models--{candidate.replace('/', '--')}",
        )
        if os.path.exists(hf_cache):
            shutil.rmtree(hf_cache, ignore_errors=True)

    # =====================================================
    # LOAD MODEL
    # =====================================================

    def _load_candidate(self, candidate: str, force_download: bool = False):
        candidate_lower = candidate.lower()
        is_bart_family = "bart" in candidate_lower

        tokenizer_kwargs = {
            "use_fast": is_bart_family,
            "cache_dir": self.cache_dir,
            "force_download": force_download,
        }
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "force_download": force_download,
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if HF_TOKEN:
            tokenizer_kwargs["token"] = HF_TOKEN
            model_kwargs["token"] = HF_TOKEN

        # Load tokenizer with fallbacks
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, **tokenizer_kwargs)
        except (AttributeError, TypeError) as e:
            if "'list' object has no attribute 'keys'" in str(e):
                retry_kwargs = dict(tokenizer_kwargs)
                retry_kwargs["extra_special_tokens"] = {}
                tokenizer = AutoTokenizer.from_pretrained(candidate, **retry_kwargs)
            else:
                raise
        except Exception:
            if is_bart_family and tokenizer_kwargs.get("use_fast") is True:
                retry_kwargs = dict(tokenizer_kwargs)
                retry_kwargs["use_fast"] = False
                tokenizer = AutoTokenizer.from_pretrained(candidate, **retry_kwargs)
            else:
                raise

        try:
            tokenizer.encode("test")
        except Exception:
            raise RuntimeError("Tokenizer corrupted (spiece.model issue)")

        model = AutoModelForSeq2SeqLM.from_pretrained(candidate, **model_kwargs)
        model = model.to(self.device)

        self.tokenizer = tokenizer
        self.model = model

        # Pipeline — optional, ViT5/T5 only
        if not is_bart_family:
            try:
                pipeline_kwargs = {
                    "task": "text2text-generation",
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                }
                if self.pipeline_device is not None:
                    pipeline_kwargs["device"] = self.pipeline_device
                self.summarizer = pipeline(**pipeline_kwargs)
                self.current_pipeline_task = "text2text-generation"
                logger.info("ViT5 pipeline initialized")
            except Exception as e:
                logger.warning("Pipeline unavailable (%s) — using model.generate()", e)
                self.summarizer = None
                self.current_pipeline_task = None
        else:
            self.summarizer = None
            self.current_pipeline_task = None

        self.model_path = candidate

    def _load_model(self):
        candidate = self.model_path
        try:
            self._load_candidate(candidate, force_download=False)
            logger.info("Loaded: %s | device=%s | dtype=%s",
                        candidate, self.device_label, self.torch_dtype)
            return
        except Exception as e:
            logger.warning("First load failed: %s", e)

        logger.info("Clearing cache for: %s", candidate)
        self._clear_model_cache(candidate)

        try:
            self._load_candidate(candidate, force_download=True)
            logger.info("Loaded after cache reset: %s", candidate)
        except Exception as e:
            raise RuntimeError(f"Unable to load summarization model: {e}")

    # =====================================================
    # INFERENCE
    # =====================================================

    def process_input(self, input_data: Union[str, bytes, Any]) -> str:
        if isinstance(input_data, bytes):
            return input_data.decode("utf-8", errors="ignore")
        return str(input_data)

    def infer(self, processed_input: str):
        processed_input = self._strip_instruction_wrapper(processed_input)
        
        if not processed_input or not processed_input.strip():
            return False
        
        language = self.detect_language(processed_input)
        if language is None:
            return False

        with self._model_lock:
            if not self._switch_model_for_language(language):
                return False
            result = {"summary": self.summarize(processed_input)}

        return result

    # =====================================================
    # POST-PROCESS: TRIM TO COMPLETE SENTENCE
    # =====================================================

    def _trim_to_complete_sentence(self, text: str) -> str:
        """
        Ensure the summary ends at a sentence boundary.

        Strategy:
        1. If text already ends with sentence-ending punctuation → return as-is
        2. Find the last sentence-ending punctuation → trim there
        3. If no punctuation found → return original (better than empty)
        """
        text = text.strip()
        if not text:
            return text

        # Already ends properly
        if text[-1] in '.!?。！？':
            return text

        # Find last sentence-ending punctuation
        last_match = None
        for m in _SENTENCE_END_RE.finditer(text):
            last_match = m

        if last_match:
            return text[:last_match.end()].strip()

        # No sentence boundary found — return as-is rather than empty string
        return text

    # =====================================================
    # HELPERS
    # =====================================================

    def _looks_degenerate(self, text: str) -> bool:
        if not text:
            return True

        compact = re.sub(r"\s+", "", text.lower())
        if len(compact) >= 120:
            if re.search(r"(.{3,12})\1{6,}", compact):
                return True
            if len(set(compact)) <= 10:
                return True

        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(tokens) < 12:
            return False

        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        if unique_ratio < 0.25:
            return True

        max_run = 1
        run = 1
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1]:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 1
        return max_run >= 6

    def _extractive_fallback_summary(self, source_text: str, ratio: float = 0.4) -> str:
        """
        Extractive fallback — pick leading sentences up to ratio% of input length.
        Works for both EN and VI since we split on universal sentence-end punctuation.
        """
        clean_source = re.sub(r'\s+', ' ', source_text).strip()
        if not clean_source:
            return ""

        target_chars = max(60, int(len(clean_source) * ratio))

        # Split on sentence-ending punctuation (works for both EN and VI)
        sentences = re.split(r'(?<=[.!?])\s+', clean_source)

        picked, total = [], 0
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            # Stop adding if we'd exceed target and already have something
            if total > 0 and total + len(s) > target_chars:
                break
            picked.append(s)
            total += len(s) + 1

        summary = " ".join(picked).strip()
        # Last resort: hard truncate at a sentence boundary
        if not summary:
            summary = clean_source[:target_chars].strip()
        return summary

    def _extract_summary_text(self, pipeline_output: Any) -> str:
        if not isinstance(pipeline_output, list) or not pipeline_output:
            return ""
        first = pipeline_output[0]
        if not isinstance(first, dict):
            return ""
        return (
            first.get("summary_text")
            or first.get("generated_text")
            or first.get("text")
            or ""
        )

    def _strip_instruction_wrapper(self, text: str) -> str:
        clean_text = re.sub(r"\s+", " ", (text or "")).strip()
        if not clean_text:
            return ""
        english_pattern = (
            r"summarize\s+the\s+following\s+text\s+into\s+at\s+most\s+\d+\s+key\s+points,\s*"
            r"concise\s+and\s+clear,\s*preserving\s+important\s+information\.?"
        )
        vietnamese_pattern = (
            r"t[oó]m\s+t[aă]t\s+v[aă]n\s+b[aă]n\s+sau\s+th[aà]nh\s+t[oố]i\s+\d+\s+[yý]\s+ch[ií]nh,\s*"
            r"ng[aă]n\s+g[oọ]n,\s*r[oõ]\s+r[aà]ng,\s*gi[uữ]\s+th[oô]ng\s+tin\s+quan\s+tr[oọ]ng\.?"
        )
        clean_text = re.sub(english_pattern, " ", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(vietnamese_pattern, " ", clean_text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", clean_text).strip()

    # =====================================================
    # GENERATE HELPER
    # =====================================================

    def _generate_with_model(
        self,
        prompt: str,
        max_new_tokens: int,
        min_new_tokens: int,
        no_repeat_ngram_size: int = 2,
        num_beams: int = 4,
        repetition_penalty: float = 1.0,
    ) -> str:
        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=num_beams,
                length_penalty=2.0,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                forced_eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=(num_beams > 1),  # only valid with beam search
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # =====================================================
    # SUMMARIZE
    # =====================================================

    def summarize(self, text, max_length=120, min_length=20, as_prompt=False):
        if not text or not str(text).strip():
            return ""

        text = self._strip_instruction_wrapper(str(text))
        if not text:
            return ""

        prompt = text
        adjusted_max_tokens = max(80, int(max_length))
        adjusted_min_tokens = min(max(15, int(min_length)), adjusted_max_tokens - 10)

        try:
            is_bart_model = self.current_language == "en"

            if is_bart_model:
                summary = self._generate_with_model(
                    prompt,
                    max_new_tokens=adjusted_max_tokens,
                    min_new_tokens=adjusted_min_tokens,
                    no_repeat_ngram_size=3,
                    num_beams=1,        # greedy — 4x faster on CPU
                )
            else:
                if self.summarizer is not None:
                    gen_kwargs = {
                        "max_new_tokens": adjusted_max_tokens,
                        "min_new_tokens": adjusted_min_tokens,
                        "do_sample": False,
                        "num_beams": 1,  # greedy — faster on CPU
                        "length_penalty": 2.0,
                        "repetition_penalty": 1.3,
                    }
                    result = self.summarizer(prompt, **gen_kwargs)
                    summary = self._extract_summary_text(result).strip()
                else:
                    summary = self._generate_with_model(
                        prompt,
                        max_new_tokens=adjusted_max_tokens,
                        min_new_tokens=adjusted_min_tokens,
                        no_repeat_ngram_size=3,
                        num_beams=1,
                        repetition_penalty=1.3,
                    )

        except Exception as e:
            logger.warning("Summarization failed: %s", e)
            return ""

        # Cleanup
        summary = re.sub(r'<extra_id_\d+>', '', summary).strip()
        summary = summary.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        summary = "".join(
            ch for ch in summary
            if (ch.isprintable() and not ch.isspace()) or ch in (" ", "\n", "\t")
        )
        summary = re.sub(r'\s+', ' ', summary).strip()

        # Trim to complete sentence
        summary = self._trim_to_complete_sentence(summary)

        # Guard: summary must be shorter than input
        # If model just parroted the input back, fall back to extractive
        if len(summary) >= len(prompt) * 0.9:
            logger.warning(
                "Summary not shorter than input (summary=%d, input=%d) — using extractive",
                len(summary), len(prompt),
            )
            summary = self._extractive_fallback_summary(prompt, ratio=0.4)
            summary = self._trim_to_complete_sentence(summary)

        # Degenerate retry — all languages
        if self._looks_degenerate(summary):
            logger.warning("Degenerate output (lang=%s) — retrying", self.current_language)
            try:
                retry_min_tokens = max(20, adjusted_min_tokens // 2)
                is_bart_model = "bart" in str(self.model_path).lower()

                if is_bart_model or self.summarizer is None:
                    retry_summary = self._generate_with_model(
                        prompt,
                        max_new_tokens=adjusted_max_tokens,
                        min_new_tokens=retry_min_tokens,
                        no_repeat_ngram_size=4,
                        num_beams=1,
                        repetition_penalty=1.5,
                    )
                else:
                    retry_kwargs = {
                        "max_new_tokens": adjusted_max_tokens,
                        "min_new_tokens": retry_min_tokens,
                        "do_sample": False,
                        "num_beams": 1,
                        "no_repeat_ngram_size": 4,
                        "repetition_penalty": 1.5,
                        "length_penalty": 2.0,
                    }
                    retry_result = self.summarizer(prompt, **retry_kwargs)
                    retry_summary = self._extract_summary_text(retry_result).strip()

                retry_summary = re.sub(r'<extra_id_\d+>', '', retry_summary).strip()
                retry_summary = re.sub(r'\s+', ' ', retry_summary).strip()
                retry_summary = self._trim_to_complete_sentence(retry_summary)

                if retry_summary and not self._looks_degenerate(retry_summary):
                    summary = retry_summary

            except Exception:
                pass

        # Final guardrail: extractive fallback for both languages
        if self._looks_degenerate(summary):
            logger.warning("Falling back to extractive summary (lang=%s)", self.current_language)
            summary = self._extractive_fallback_summary(text, ratio=0.4)
            summary = self._trim_to_complete_sentence(summary)

        return summary