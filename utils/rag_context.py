from dataclasses import dataclass


def detect_language(text: str) -> str:
    if any(c in text for c in "ăâđêôơưĂÂĐÊÔƠƯ"):
        return "vi"
    return "en"


@dataclass
class SummarizationRAGContext:
    max_points: int = 5

    def build_prompt(self, transcript: str) -> str:
        text = (transcript or "").strip()
        if not text:
            return ""

        lang = detect_language(text)

        if lang == "vi":
            instruction = (
                f"Tóm tắt văn bản sau thành tối đa {self.max_points} ý chính, "
                "ngắn gọn, rõ ràng, giữ thông tin quan trọng."
            )
        else:
            instruction = (
                f"Summarize the following text into at most {self.max_points} key points, "
                "concise and clear, preserving important information."
            )

        return f"{instruction}\n\n{text}"
    
def get_summarization_rag():
    return SummarizationRAGContext()