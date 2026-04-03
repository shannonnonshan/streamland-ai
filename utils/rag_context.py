from typing import List, Dict
import re


class SummarizationRAG:
    """Lightweight RAG for summarization (keyword-based retrieval)."""

    def __init__(self):
        self.contexts = {
            "streamland_ai": """StreamLand AI is an AI platform providing transcription,
summarization, and content moderation. Key features include Whisper speech-to-text,
mT5 summarization, multi-language support, and real-time processing.""",

            "technical": """This text relates to AI/ML concepts such as transformers,
Hugging Face, NLP, neural networks, and model deployment.""",

            "general": """Summarization should be concise, preserve key ideas,
and remain easy to understand. Do not repeat information.
Use simple sentences."""
        }

    # ✅ Add context
    def add_context(self, context_id: str, content: str):
        self.contexts[context_id] = content

    # ✅ Simple keyword-based retrieval (lightweight RAG)
    def retrieve_context(self, text: str) -> str:
        text_lower = text.lower()

        if any(k in text_lower for k in ["ai", "model", "huggingface", "transformer"]):
            return self.contexts["technical"]

        if any(k in text_lower for k in ["streamland", "transcript", "audio"]):
            return self.contexts["streamland_ai"]

        return self.contexts["general"]

    # ✅ Truncate text to avoid overflow
    def truncate_text(self, text: str, max_chars: int = 2000) -> str:
        return text[:max_chars]

    # Build a prompt template compatible with mT5 text2text generation.
    def build_prompt(self, text: str) -> str:
        context = self.retrieve_context(text)
        text = self.truncate_text(text)

        prompt = f"""
You are a helpful AI assistant.

Use the context below to improve the summary.

Context:
{context}

Task:
Summarize the following text clearly and concisely.

Text:
{text}

Summary:
"""
        return prompt.strip()


# Singleton instance
_rag_instance = None


def get_summarization_rag() -> SummarizationRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = SummarizationRAG()
    return _rag_instance