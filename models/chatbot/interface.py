import logging
import os
from typing import Dict, Any

import torch

from ..base import BaseModel

logger = logging.getLogger(__name__)


class ChatbotModel(BaseModel):
    def __init__(self, model_path: str, from_hf: bool = False):
        super().__init__(model_path=model_path, from_hf=from_hf)
        self.device, self.device_label, self.dtype = self._resolve_device()
        self._load_model()

    @property
    def model_type(self) -> str:
        return "chatbot"

    def _resolve_device(self):
        requested = os.getenv("CHATBOT_DEVICE", "auto").strip().lower()
        cuda_available = torch.cuda.is_available()

        if requested not in {"auto", "cpu", "cuda"}:
            logger.warning(
                "Invalid CHATBOT_DEVICE=%s. Expected one of: auto, cpu, cuda. Falling back to auto.",
                requested,
            )
            requested = "auto"

        if requested == "cuda":
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                logger.info("Chatbot device selected: cuda (%s)", device_name)
                return "cuda", f"cuda ({device_name})", self._resolve_dtype("cuda")

            logger.warning("CHATBOT_DEVICE=cuda but CUDA is not available. Falling back to CPU.")
            return "cpu", "cpu", self._resolve_dtype("cpu")

        if requested == "cpu":
            logger.info("Chatbot device selected: cpu (CHATBOT_DEVICE=cpu)")
            return "cpu", "cpu", self._resolve_dtype("cpu")

        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            logger.info("Chatbot device selected: cuda (%s)", device_name)
            return "cuda", f"cuda ({device_name})", self._resolve_dtype("cuda")

        logger.info("Chatbot device selected: cpu (no CUDA available)")
        return "cpu", "cpu", self._resolve_dtype("cpu")

    def _resolve_dtype(self, device: str) -> torch.dtype:
        requested = os.getenv("CHATBOT_DTYPE", "").strip().lower()
        if requested:
            return self._parse_dtype(requested)

        if device == "cuda":
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
            logger.warning("Unknown CHATBOT_DTYPE=%s. Falling back to float32.", dtype_value)
            return torch.float32
        return dtype

    def _load_model(self) -> None:
        token = os.getenv("HF_TOKEN")
        adapter_id = self.model_path
        base_model_id = os.getenv(
            "CHATBOT_BASE_MODEL",
            "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        )
        use_unsloth = os.getenv("CHATBOT_USE_UNSLOTH", "false").lower() == "true"
        max_seq_length = int(os.getenv("CHATBOT_MAX_SEQ_LENGTH", "2048"))
        load_in_4bit = os.getenv("CHATBOT_LOAD_IN_4BIT", "false").lower() == "true"

        tokenizer_kwargs: Dict[str, Any] = {}
        if token:
            tokenizer_kwargs["token"] = token

        if use_unsloth and self.device != "cuda":
            logger.warning("Unsloth requires CUDA; falling back to transformers loader.")
            use_unsloth = False

        model_kwargs: Dict[str, Any] = {}
        if token:
            model_kwargs["token"] = token
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required for LoRA adapter loading") from exc

        if use_unsloth:
            try:
                import unsloth  # noqa: F401
                from unsloth import FastLanguageModel
            except ImportError as exc:
                raise ImportError("unsloth is required when CHATBOT_USE_UNSLOTH=true") from exc

            base_model, self.tokenizer = FastLanguageModel.from_pretrained(
                base_model_id,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                device_map={"": 0} if self.device == "cuda" else None,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = PeftModel.from_pretrained(base_model, adapter_id, **tokenizer_kwargs)

            try:
                self.model = FastLanguageModel.for_inference(self.model)
            except Exception:
                pass
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, **tokenizer_kwargs)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            if load_in_4bit:
                try:
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig
                except ImportError:
                    logger.warning(
                        "4-bit requested but bitsandbytes/quant config unavailable; falling back to full precision."
                    )
                    load_in_4bit = False

            if load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                model_kwargs["torch_dtype"] = self.dtype

            base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
            self.model = PeftModel.from_pretrained(base_model, adapter_id, **tokenizer_kwargs)
        self.model.eval()

        if self.device != "cuda":
            if self.dtype in {torch.float16, torch.bfloat16}:
                self.model = self.model.to(self.device, dtype=self.dtype)
            else:
                self.model = self.model.to(self.device)

        logger.info(
            "Loaded chatbot adapter: %s | base=%s | device=%s | dtype=%s | unsloth=%s",
            adapter_id,
            base_model_id,
            self.device_label,
            self.dtype,
            use_unsloth,
        )

    def process_input(self, input_data: str) -> str:
        if not isinstance(input_data, str):
            raise TypeError("Chatbot input must be a string")
        cleaned = input_data.strip()
        if not cleaned:
            raise ValueError("Chatbot input cannot be empty")
        return cleaned

    def infer(self, processed_input: str) -> Dict[str, Any]:
        inputs = self.tokenizer(processed_input, return_tensors="pt", truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        max_new_tokens = int(os.getenv("CHATBOT_MAX_NEW_TOKENS", "500"))
        temperature = float(os.getenv("CHATBOT_TEMPERATURE", "0.5"))
        top_p = float(os.getenv("CHATBOT_TOP_P", "0.9"))

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = text[len(processed_input) :].strip()
        answer = answer.split("User:")[0].strip()

        return {"response": answer}

    def generate(self, prompt: str) -> str:
        processed = self.process_input(prompt)
        result = self.infer(processed)
        return result["response"]
