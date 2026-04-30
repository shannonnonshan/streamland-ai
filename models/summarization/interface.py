import os
import shutil
import re
from typing import Any, Dict, Union, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langdetect import detect, LangDetectException

from models.base import BaseModel


HF_TOKEN = os.getenv("HF_TOKEN")

DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "streamland-ai",
    "summarization",
)


class SummarizationModel(BaseModel):
    SUPPORTED_MODELS = {
        "vi": "VietAI/vit5-base-vietnews-summarization",
        "en": "shannonnonshan/bart-summarizer",
    }

    def __init__(self, model_path=None, from_hf=True):
        super().__init__(model_path or "shannonnonshan/bart-summarizer", from_hf)

        self.cache_dir = os.getenv("SUMMARIZATION_CACHE_DIR", DEFAULT_CACHE_DIR)
        self.device = 0 if torch.cuda.is_available() else -1

        self.tokenizer = None
        self.model = None
        self.summarizer = None
        self.current_language = None
        self.current_pipeline_task = None

        self._load_model()

    # -------------------------
    # Language detection
    # -------------------------
    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect language with langdetect and normalize to {'vi','en'}.
        Returns: 'vi' or 'en', otherwise None.
        """
        try:
            detected_lang = detect(text)
            if detected_lang.startswith("vi"):
                return "vi"
            if detected_lang.startswith("en"):
                return "en"
            return None
        except LangDetectException:
            return None

    def _switch_model_for_language(self, language: str):
        """Switch to appropriate model for detected language."""
        if language not in self.SUPPORTED_MODELS:
            return False

        if self.current_language == language:
            print(f"[INFO] Already using model for {language}, skipping reload")
            return True

        model_path = self.SUPPORTED_MODELS[language]
        print(f"[INFO] Switching model: {self.model_path} -> {model_path}")
        self.model_path = model_path
        self._load_model()
        self.current_language = language
        return True

    @property
    def model_type(self) -> str:
        return "summarization"

    # -------------------------
    # Cache handling (safe)
    # -------------------------
    def _clear_model_cache(self, candidate: str):
        hf_cache = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "huggingface",
            "hub",
            f"models--{candidate.replace('/', '--')}",
        )
        if os.path.exists(hf_cache):
            shutil.rmtree(hf_cache, ignore_errors=True)

    # -------------------------
    # Load model
    # -------------------------
    def _load_candidate(self, candidate: str, force_download: bool = False):
        candidate_lower = candidate.lower()
        is_bart_family = "bart" in candidate_lower

        tokenizer_kwargs = {
            # BART repos often provide fast-tokenizer artifacts only.
            # Keep slow tokenizer for T5/ViT5-style models.
            "use_fast": is_bart_family,
            "cache_dir": self.cache_dir,
            "force_download": force_download,
        }

        model_kwargs = {
            "cache_dir": self.cache_dir,
            "force_download": force_download,
        }

        if HF_TOKEN:
            tokenizer_kwargs["token"] = HF_TOKEN
            model_kwargs["token"] = HF_TOKEN

        # Load tokenizer with fallback for malformed tokenizer_config.json
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, **tokenizer_kwargs)
        except (AttributeError, TypeError) as e:
            # Some custom mT5 repos store extra_special_tokens with wrong type (list instead of dict)
            if "'list' object has no attribute 'keys'" in str(e):
                retry_kwargs = dict(tokenizer_kwargs)
                retry_kwargs["extra_special_tokens"] = {}
                tokenizer = AutoTokenizer.from_pretrained(candidate, **retry_kwargs)
            else:
                raise
        except Exception:
            # If fast tokenizer fails (missing artifacts), retry with slow tokenizer once.
            if is_bart_family and tokenizer_kwargs.get("use_fast") is True:
                retry_kwargs = dict(tokenizer_kwargs)
                retry_kwargs["use_fast"] = False
                tokenizer = AutoTokenizer.from_pretrained(candidate, **retry_kwargs)
            else:
                raise

        # Validate tokenizer
        try:
            tokenizer.encode("test")
        except Exception:
            raise RuntimeError("Tokenizer corrupted (spiece.model issue)")

        # Load model (NO resume_download)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            candidate,
            **model_kwargs
        )

        if torch.cuda.is_available():
            model = model.to("cuda")

        self.tokenizer = tokenizer
        self.model = model

        pipeline_task = "summarization" if "bart" in candidate_lower else "text2text-generation"

        # Use task that matches model family output format.
        self.summarizer = pipeline(
            pipeline_task,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        self.current_pipeline_task = pipeline_task
        self.model_path = candidate

    def _load_model(self):
        candidate = self.model_path

        # Attempt 1
        try:
            self._load_candidate(candidate, force_download=False)
            print(f"[INFO] Loaded summarization model: {candidate}")
            return
        except Exception as e:
            print(f"[WARNING] First load failed: {e}")

        # Clear corrupted cache
        print("[INFO] Clearing corrupted model cache...")
        self._clear_model_cache(candidate)

        # Attempt 2
        try:
            self._load_candidate(candidate, force_download=True)
            print(f"[INFO] Loaded after cache reset: {candidate}")
        except Exception as e:
            raise RuntimeError(f"Unable to load summarization model: {e}")

    # -------------------------
    # Inference
    # -------------------------
    def process_input(self, input_data: Union[str, bytes, Any]) -> str:
        if isinstance(input_data, bytes):
            return input_data.decode("utf-8", errors="ignore")
        return str(input_data)

    def infer(self, processed_input: str) -> Union[Dict[str, Any], bool]:
        processed_input = self._strip_instruction_wrapper(processed_input)

        # Route by langdetect only:
        # - en -> shannonnonshan/bart-summarizer
        # - vi -> VietAI/vit5-base-vietnews-summarization
        language = self.detect_language(processed_input)
        if language is None:
            return False

        # Switch to appropriate model if needed
        if not self._switch_model_for_language(language):
            return False

        return {"summary": self.summarize(processed_input)}

    # -------------------------
    # Summarization
    # -------------------------
    def _looks_degenerate(self, text: str) -> bool:
        if not text:
            return True

        compact = re.sub(r"\s+", "", text.lower())
        if len(compact) >= 120:
            # Detect contiguous repeated chunks, e.g. ILLARILLARILLAR...
            if re.search(r"(.{3,12})\1{6,}", compact):
                return True

            # Very low character diversity in long output is a strong bad-signal.
            if len(set(compact)) <= 10:
                return True

        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(tokens) < 12:
            return False

        # Detect very repetitive outputs, e.g. ILLAR ILLAR ILLAR ...
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

    def _extractive_fallback_summary(self, source_text: str, ratio: float = 0.5) -> str:
        clean_source = re.sub(r'\s+', ' ', source_text).strip()
        if not clean_source:
            return ""

        target_chars = max(80, int(len(clean_source) * ratio))
        sentences = re.split(r'(?<=[.!?])\s+', clean_source)

        picked = []
        total = 0
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            picked.append(s)
            total += len(s) + 1
            if total >= target_chars:
                break

        summary = " ".join(picked).strip()
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
        """Remove common summarization instruction wrappers from prompt-like inputs."""
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

    def summarize(self, text, max_length=128, min_length=10, as_prompt=False):
        if not text or not str(text).strip():
            return ""

        text = self._strip_instruction_wrapper(str(text))
        if not text:
            return ""

        # Keep prompt plain for seq2seq summarization models.
        prompt = text

        # Keep decoding length settings simple and stable.
        adjusted_max_length = max(32, int(max_length))
        adjusted_min_length = min(max(10, int(min_length)), adjusted_max_length - 5)

        try:
            is_bart_model = "bart" in str(self.model_path).lower()

            if is_bart_model:
                # Match training/inference setup for BART to reduce generation drift.
                model_device = next(self.model.parameters()).device
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )

                input_ids = inputs["input_ids"].to(model_device)
                attention_mask = inputs["attention_mask"].to(model_device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=adjusted_max_length,
                        min_length=adjusted_min_length,
                        num_beams=4,
                        length_penalty=1.0,
                        no_repeat_ngram_size=2,
                    )

                summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            else:
                gen_kwargs = {
                    "max_length": adjusted_max_length,
                    "min_length": adjusted_min_length,
                    "do_sample": False,
                    "num_beams": 4,
                    "early_stopping": True,
                    "length_penalty": 1.0,
                }
                result = self.summarizer(prompt, **gen_kwargs)
                summary = self._extract_summary_text(result).strip()
        except Exception as e:
            print(f"[WARNING] Summarization failed: {e}")
            return ""

        # Final cleanup: remove any remaining sentinel tokens
        summary = re.sub(r'<extra_id_\d+>', '', summary).strip()

        # Normalize and keep only safe printable Unicode.
        # This avoids invalid/control characters that can break display pipelines.
        summary = summary.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        summary = "".join(
            ch for ch in summary
            if (ch.isprintable() and not ch.isspace()) or ch in (" ", "\n", "\t")
        )

        # Clean up excessive whitespace
        summary = re.sub(r'\s+', ' ', summary).strip()

        # Retry once with stricter decoding when English output is clearly degenerate.
        if self.current_language == "en" and self._looks_degenerate(summary):
            try:
                retry_min_length = max(10, adjusted_min_length // 2)
                is_bart_model = "bart" in str(self.model_path).lower()

                if is_bart_model:
                    model_device = next(self.model.parameters()).device
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    input_ids = inputs["input_ids"].to(model_device)
                    attention_mask = inputs["attention_mask"].to(model_device)

                    with torch.no_grad():
                        retry_output_ids = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=adjusted_max_length,
                            min_length=retry_min_length,
                            num_beams=4,
                            length_penalty=1.0,
                            no_repeat_ngram_size=3,
                        )

                    retry_summary = self.tokenizer.decode(
                        retry_output_ids[0], skip_special_tokens=True
                    ).strip()
                else:
                    retry_kwargs = {
                        "max_length": adjusted_max_length,
                        "min_length": retry_min_length,
                        "do_sample": False,
                        "num_beams": 2,
                        "early_stopping": True,
                        "no_repeat_ngram_size": 3,
                        "length_penalty": 1.1,
                    }
                    retry_result = self.summarizer(prompt, **retry_kwargs)
                    retry_summary = self._extract_summary_text(retry_result).strip()

                retry_summary = re.sub(r'<extra_id_\d+>', '', retry_summary).strip()
                retry_summary = ''.join(
                    char for char in retry_summary
                    if ((char.isprintable() or char in '\n\t') and ord(char) > 31) or char.isspace()
                )
                retry_summary = re.sub(r'\s+', ' ', retry_summary).strip()
                if retry_summary and not self._looks_degenerate(retry_summary):
                    summary = retry_summary
            except Exception:
                # Keep the original summary if fallback decode fails.
                pass

        # Final guardrail for English model: if output is still repetitive/garbled,
        # return a safe extractive summary from the original text.
        if self.current_language == "en" and self._looks_degenerate(summary):
            summary = self._extractive_fallback_summary(text, ratio=0.5)

        return summary