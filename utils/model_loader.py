"""
Model Loader / Factory
Purpose: Unified interface for loading any model type
Supports: Dynamic model loading based on type parameter
"""

import os
from typing import Dict, Any, Optional
from models.base import BaseModel


class ModelLoader:
    """Factory class for loading models."""
    
    # Available model types
    AVAILABLE_MODELS = {
        "whisper": "models.whisper.interface.WhisperModel",
        "moderation": "models.moderation.interface.ModerationModel",
        "summarization": "models.summarization.interface.SummarizationModel",
    }
    
    @staticmethod
    def load_model(model_type: str, model_path: str, from_hf: bool = False, **kwargs) -> BaseModel:
        """
        Load a model by type.
        
        Args:
            model_type (str): Type of model ('whisper', 'moderation', 'summarization')
            model_path (str): Path to model or HF model ID
            from_hf (bool): Whether to load from Hugging Face Hub
            **kwargs: Additional model-specific arguments
            
        Returns:
            BaseModel: Loaded model instance
            
        Raises:
            ValueError: If model type is not supported
            ImportError: If model class cannot be imported
        """
        model_type = model_type.lower()
        
        if model_type not in ModelLoader.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {', '.join(ModelLoader.AVAILABLE_MODELS.keys())}"
            )
        
        try:
            # Dynamically import and instantiate model
            module_path, class_name = ModelLoader.AVAILABLE_MODELS[model_type].rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            
            return model_class(model_path=model_path, from_hf=from_hf, **kwargs)
        
        except Exception as e:
            raise ImportError(f"Failed to load model '{model_type}': {e}")
    
    @staticmethod
    def from_env() -> Optional[BaseModel]:
        """
        Load model from environment variables.
        
        Environment variables:
        - MODEL_TYPE: Type of model (required)
        - MODEL_PATH: Path or HF ID (required)
        - MODEL_USE_HF: Whether to use HF Hub (optional, default: true)
        
        Returns:
            BaseModel: Loaded model or None if not configured
        """
        model_type = os.getenv("MODEL_TYPE", "").lower()
        model_path = os.getenv("MODEL_PATH", "")
        use_hf = os.getenv("MODEL_USE_HF", "true").lower() == "true"
        
        if not model_type or not model_path:
            return None
        
        try:
            return ModelLoader.load_model(model_type, model_path, from_hf=use_hf)
        except Exception as e:
            print(f"[ERROR] Failed to load model from environment: {e}")
            return None
    
    @staticmethod
    def list_available() -> Dict[str, str]:
        """Return dict of available models and their descriptions."""
        return {
            "whisper": "Speech-to-Text (STT) - Detect giọng nói",
            "moderation": "Content Moderation - Text & Image Safety",
            "summarization": "Text Summarization - Tóm tắt nội dung",
        }
