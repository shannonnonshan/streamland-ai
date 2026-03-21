"""LLM Model Interface (Llama 3 8B Instruct)"""

from models.base import BaseModel


class LlamaModel(BaseModel):
    """LLM for chat QA and RAG-based responses."""
    
    def __init__(self, model_path=None, from_hf=True):
        super().__init__("llama", model_path or "meta-llama/Llama-2-7b-chat-hf", from_hf)
    
    def generate(self, prompt, max_length=512):
        """Generate text response from prompt."""
        raise NotImplementedError("generate() must be implemented")
    
    def chat(self, messages):
        """Chat with multi-turn conversation."""
        raise NotImplementedError("chat() must be implemented")
