"""
Whisper Model Wrapper
Purpose: Clean interface for loading and using fine-tuned Whisper model
Supports: Local disk loading or Hugging Face Hub remote loading
Features: Automatic device detection (GPU/CPU), audio preprocessing
"""

import warnings
import torch
import os
from typing import Dict, Any, Union
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.utils import logging as hf_logging
import librosa

from ..base import BaseModel

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
torch.set_num_threads(1)


class WhisperModel(BaseModel):
    """
    Wrapper class for fine-tuned Whisper speech recognition model.
    Handles model loading from local or Hugging Face Hub, device management, and audio processing.
    Inherits from BaseModel to support model factory pattern.
    """

    def __init__(self, model_path: str = "models/whisper/model/whisper-finetuned", 
                 from_hf: bool = False):
        """
        Initialize WhisperModel.
        
        Args:
            model_path (str): Path to model (local path or HF model ID like "username/model-name")
            from_hf (bool): If True, load from Hugging Face Hub. If False, load from local disk.
        """
        super().__init__(model_path=model_path, from_hf=from_hf)
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._load_model()

    @property
    def model_type(self) -> str:
        """Return model type."""
        return "whisper"

    def process_input(self, input_data: Union[str, np.ndarray]) -> np.ndarray:
        """
        Process and load audio input.
        
        Args:
            input_data (str or np.ndarray): Path to audio file or audio array
            
        Returns:
            np.ndarray: Audio samples at 16kHz
        """
        if isinstance(input_data, str):
            # Load from file path
            samples, sr = librosa.load(input_data, sr=16000)
            return samples
        elif isinstance(input_data, np.ndarray):
            # Resample if needed
            return input_data
        else:
            raise TypeError(f"Expected str or np.ndarray, got {type(input_data)}")

    def infer(self, processed_input: np.ndarray) -> Dict[str, Any]:
        """
        Run transcription inference on processed audio.
        
        Args:
            processed_input (np.ndarray): Audio samples at 16kHz
            
        Returns:
            Dict with 'text' and 'language' keys
        """
        sr = 16000
        chunk_length = 30 * sr
        chunks = [processed_input[i:i + chunk_length] for i in range(0, len(processed_input), chunk_length)]
        
        full_transcript = []
        
        for chunk in chunks:
            inputs = self.processor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device, self.torch_dtype)
            
            with torch.no_grad():
                predicted_ids = self.model.generate(inputs)
            
            text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            full_transcript.append(text)
        
        return {
            "text": " ".join(full_transcript),
            "language": "unknown",
            "model": "whisper"
        }

    def _load_model(self):
        """Load the model and processor from local disk or Hugging Face Hub."""
        try:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            source = "Hugging Face Hub" if self.from_hf else "local disk"
            print(f"[INFO] Model loaded from {source}: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise

    def transcribe(self, audio_file: str) -> Dict[str, Any]:
        """
        Transcribe an audio file to text.
        Backward-compatible method that uses the base class interface.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            Dict with transcription result and metadata
        """
        # Use the new process_input and infer methods
        processed = self.process_input(audio_file)
        result = self.infer(processed)
        return result
