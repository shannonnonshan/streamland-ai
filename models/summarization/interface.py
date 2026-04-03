from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import shutil

HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "streamland-ai",
    "summarization",
)


class SummarizationModel:
    def __init__(self, model_path="google/mt5-small"):
        self.model_path = model_path
        self.cache_dir = os.getenv("SUMMARIZATION_CACHE_DIR", DEFAULT_CACHE_DIR)
        self.device = 0 if torch.cuda.is_available() else -1
        self.summarizer = None
        self.tokenizer = None
        self.model = None

        self._load_model()

    def _clear_cache(self):
        if os.path.exists(self.cache_dir):
            print(f"[INFO] Clearing cache: {self.cache_dir}")
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _load_from_cache(self, force_download=False):
        tokenizer_kwargs = {
            "token": HF_TOKEN if HF_TOKEN else None,
            "use_fast": False,
            "revision": "main",
            "cache_dir": self.cache_dir,
        }

        model_kwargs = {
            "token": HF_TOKEN if HF_TOKEN else None,
            "cache_dir": self.cache_dir,
        }

        if force_download:
            tokenizer_kwargs["force_download"] = True
            model_kwargs["force_download"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            **tokenizer_kwargs
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            **model_kwargs
        )

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.summarizer = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def _load_model(self):
        print(f"[INFO] Loading model: {self.model_path}")

        # ---- TRY 1: NORMAL LOAD ----
        try:
            self._load_from_cache(force_download=False)
            print("[SUCCESS] Loaded model normally")
            return

        except Exception as e:
            print(f"[WARNING] First load failed: {e}")

        # ---- TRY 2: CLEAR CACHE + FORCE DOWNLOAD ----
        try:
            self._clear_cache()
            self._load_from_cache(force_download=True)
            print("[SUCCESS] Loaded after cache reset")
            return

        except Exception as e2:
            print(f"[ERROR] Load failed completely: {e2}")
            raise

    def summarize(self, text, max_length=150, min_length=50):
        if not text or not text.strip():
            return "Empty input"

        # Improved prompt for mT5
        prompt = f"Summarize the following text concisely:\n{text}"

        try:
            if self.summarizer:
                result = self.summarizer(
                    prompt,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                return result[0]["generated_text"]

            elif self.model and self.tokenizer:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )

                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )

                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return None