"""Text Summarization Model Interface (mT5 / FLAN-T5)"""

import torch
from typing import Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging as hf_logging

from models.base import BaseModel

hf_logging.set_verbosity_error()


class SummarizationModel(BaseModel):
    """Text summarization for transcripts and multilingual content."""

    DEFAULT_MODEL = "google/mt5-small"
    # Task prefix expected by mT5/T5 sequence-to-sequence models.
    TASK_PREFIX = "summarize: "

    def __init__(self, model_path: str = None, from_hf: bool = True):
        super().__init__(
            model_path=model_path or self.DEFAULT_MODEL,
            from_hf=from_hf,
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._load_model()

    @property
    def model_type(self) -> str:
        return "summarization"

    # ================= LOAD =================
    def _load_model(self):
        # use_fast=False avoids binary spiece.model parsing errors that occur
        # with the Rust-based fast tokenizer on some mT5 / T5 checkpoints.
        self.processor = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        print(f"[INFO] Loaded Summarization model: {self.model_path}")

    # ================= INPUT =================
    def process_input(self, input_data: Union[str, bytes]) -> Dict[str, Any]:
        if isinstance(input_data, bytes):
            input_data = input_data.decode("utf-8")

        if not isinstance(input_data, str):
            raise TypeError(f"Expected str or bytes, got {type(input_data)}")

        return {"text": input_data.strip()}

    # ================= INFER =================
    def infer(self, processed_input: Dict[str, Any], max_length: int = 150) -> Dict[str, Any]:
        text = processed_input["text"]

        prefix = self.TASK_PREFIX
        inputs = self.processor(
            prefix + text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=4,
                length_penalty=0.8,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        summary = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

        return {
            "summary": summary,
            "model": self.model_type,
            "model_path": self.model_path,
        }

    # ================= PUBLIC =================
    def summarize(self, text: str, max_length: int = 150) -> Dict[str, Any]:
        """Summarize long text into a shorter version.

        Args:
            text (str): Input text to summarize.
            max_length (int): Maximum number of tokens in the summary.

        Returns:
            Dict with key ``summary`` containing the generated summary.
        """
        processed = self.process_input(text)
        return self.infer(processed, max_length=max_length)
