"""
Speech Processing Pipeline
Purpose: Orchestrates the complete speech-to-text processing workflow
from audio input to final transcription output.
Supports both local and Hugging Face Hub models.
"""

from models.whisper import WhisperModel


class SpeechPipeline:
    """
    Main pipeline for speech processing tasks.
    Handles end-to-end transcription workflow with flexible model loading.
    """

    def __init__(self, model_path: str = "models/whisper/model/whisper-finetuned", 
                 use_hf: bool = False):
        """
        Initialize the speech processing pipeline.
        
        Args:
            model_path (str): Path to model (local or HF repo ID)
            use_hf (bool): Load from Hugging Face Hub if True
        """
        self.model_path = model_path
        self.use_hf = use_hf
        self.whisper_model = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            self.whisper_model = WhisperModel(
                model_path=self.model_path,
                from_hf=self.use_hf
            )
            source = "Hugging Face Hub" if self.use_hf else "local disk"
            print(f"[INFO] Pipeline initialized with model from {source}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise

    def process_audio(self, audio_file: str) -> str:
        """
        Process audio file and return transcription.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            str: Transcribed text
        """
        if not self.whisper_model:
            raise RuntimeError("Model not loaded. Check initialization.")
            
        return self.whisper_model.transcribe(audio_file)
