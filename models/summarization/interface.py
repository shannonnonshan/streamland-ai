"""Text Summarization Model Interface (FLAN-T5 / BART)"""

from models.base import BaseModel


class SummarizationModel(BaseModel):
    """Text summarization for transcripts and content."""
    
    def __init__(self, model_path=None, from_hf=True):
        super().__init__("summarization", model_path or "google/flan-t5-base", from_hf)
    
    def summarize(self, text, max_length=150):
        """Summarize long text into shorter version."""
        raise NotImplementedError("summarize() must be implemented")
