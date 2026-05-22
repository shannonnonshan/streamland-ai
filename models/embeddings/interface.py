import logging
import os
from typing import Any, Dict, List, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ..base import BaseModel

logger = logging.getLogger(__name__)


class EmbeddingModel(BaseModel):
    def __init__(self, model_path: str, from_hf: bool = False):
        super().__init__(model_path=model_path, from_hf=from_hf)
        self.device, self.device_label, self.dtype = self._resolve_device()
        self.use_sentence_transformer = False
        self._load_model()

    @property
    def model_type(self) -> str:
        return "embeddings"

    def _resolve_device(self):
        requested = os.getenv("EMBEDDINGS_DEVICE", "auto").strip().lower()
        cuda_available = torch.cuda.is_available()
        mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

        if requested not in {"auto", "cpu", "cuda", "mps"}:
            logger.warning(
                "Invalid EMBEDDINGS_DEVICE=%s. Expected one of: auto, cpu, cuda, mps. Falling back to auto.",
                requested,
            )
            requested = "auto"

        if requested == "cuda":
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                logger.info("Embeddings device selected: cuda (%s)", device_name)
                return "cuda", f"cuda ({device_name})", self._resolve_dtype("cuda")

            logger.warning("EMBEDDINGS_DEVICE=cuda but CUDA is not available. Falling back to CPU.")
            return "cpu", "cpu", self._resolve_dtype("cpu")

        if requested == "mps":
            if mps_available:
                logger.info("Embeddings device selected: mps")
                return "mps", "mps", self._resolve_dtype("mps")

            logger.warning("EMBEDDINGS_DEVICE=mps but MPS is not available. Falling back to CPU.")
            return "cpu", "cpu", self._resolve_dtype("cpu")

        if requested == "cpu":
            logger.info("Embeddings device selected: cpu (EMBEDDINGS_DEVICE=cpu)")
            return "cpu", "cpu", self._resolve_dtype("cpu")

        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            logger.info("Embeddings device selected: cuda (%s)", device_name)
            return "cuda", f"cuda ({device_name})", self._resolve_dtype("cuda")

        if mps_available:
            logger.info("Embeddings device selected: mps")
            return "mps", "mps", self._resolve_dtype("mps")

        logger.info("Embeddings device selected: cpu (no accelerator available)")
        return "cpu", "cpu", self._resolve_dtype("cpu")

    def _resolve_dtype(self, device: str) -> torch.dtype:
        requested = os.getenv("EMBEDDINGS_DTYPE", "").strip().lower()
        if requested:
            return self._parse_dtype(requested)

        if device in {"cuda", "mps"}:
            return torch.float16

        return torch.float32

    def _parse_dtype(self, dtype_value: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        dtype = mapping.get(dtype_value)
        if not dtype:
            logger.warning("Unknown EMBEDDINGS_DTYPE=%s. Falling back to float32.", dtype_value)
            return torch.float32
        return dtype

    def _load_model(self):
        token = os.getenv("HF_TOKEN")
        load_kwargs: Dict[str, Any] = {}
        if token:
            load_kwargs["token"] = token
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **load_kwargs)
            self.model = AutoModel.from_pretrained(self.model_path, **load_kwargs)

            self.model.eval()

            if self.device in {"cuda", "mps"} and self.dtype in {torch.float16, torch.bfloat16}:
                self.model = self.model.to(self.device, dtype=self.dtype)
            else:
                self.model = self.model.to(self.device)

            logger.info(
                "Loaded embeddings model: %s | device=%s | dtype=%s",
                self.model_path,
                self.device_label,
                self.dtype,
            )
            return
        except Exception as exc:
            logger.warning("AutoModel load failed: %s. Falling back to SentenceTransformer.", exc)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence-transformers is required for this embeddings model") from exc

        self.use_sentence_transformer = True
        self.model = SentenceTransformer(self.model_path, device=self.device)
        logger.info(
            "Loaded SentenceTransformer model: %s | device=%s",
            self.model_path,
            self.device_label,
        )

    def process_input(self, input_data: Union[str, List[str]]) -> List[str]:
        if isinstance(input_data, str):
            cleaned = input_data.strip()
            if not cleaned:
                raise ValueError("Input text cannot be empty")
            return [cleaned]

        if isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            cleaned = [item.strip() for item in input_data if item.strip()]
            if not cleaned:
                raise ValueError("Input text list cannot be empty")
            return cleaned

        raise TypeError(f"Invalid input type: {type(input_data)}")

    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def infer(self, processed_input: List[str]) -> Dict[str, Any]:
        if self.use_sentence_transformer:
            embeddings = self.model.encode(
                processed_input,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            return {
                "model": self.model_path,
                "embeddings": embeddings.tolist(),
            }

        with torch.no_grad():
            encoded = self.tokenizer(
                processed_input,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            output = self.model(**encoded)
            pooled = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings = normalized.cpu().numpy().astype(np.float32)

        return {
            "model": self.model_path,
            "embeddings": embeddings.tolist(),
        }

    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        processed = self.process_input(text)
        result = self.infer(processed)
        return result["embeddings"]
