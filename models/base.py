"""
Base Model Interface
Purpose: Abstract base class for all models in StreamLand AI
All models should inherit from this class to ensure consistent interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all StreamLand AI models.
    Ensures consistent interface across different model types (Speech-to-Text, TTS, etc.)
    """
    
    def __init__(self, model_path: str, from_hf: bool = False, **kwargs):
        """
        Initialize model.
        
        Args:
            model_path (str): Path to model (local or HF model ID)
            from_hf (bool): Whether to load from Hugging Face Hub
            **kwargs: Additional model-specific arguments
        """
        self.model_path = model_path
        self.from_hf = from_hf
        self.model = None
        self.processor = None
        self.device = None
        
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type (e.g., 'whisper', 'tts', 'wav2vec2')"""
        pass
    
    @abstractmethod
    def _load_model(self):
        """Load model and processor from disk or HF Hub."""
        pass
    
    @abstractmethod
    def process_input(self, input_data: Union[str, np.ndarray, bytes]) -> Any:
        """
        Process and validate input data.
        
        Args:
            input_data: Input to process (varies by model type)
            
        Returns:
            Processed input ready for model inference
        """
        pass
    
    @abstractmethod
    def infer(self, processed_input: Any) -> Dict[str, Any]:
        """
        Run inference on processed input.
        
        Args:
            processed_input: Output from process_input()
            
        Returns:
            Dict with model-specific outputs
        """
        pass
    
    def __call__(self, input_data: Union[str, np.ndarray, bytes]) -> Dict[str, Any]:
        """
        Convenience method for model inference.
        Calls process_input() then infer().
        """
        processed = self.process_input(input_data)
        return self.infer(processed)
    
    def info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "type": self.model_type,
            "path": self.model_path,
            "from_hf": self.from_hf,
            "device": self.device,
        }
