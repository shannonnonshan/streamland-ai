"""Chat endpoint — enhanced với video cards JSON, in-session history, context-aware."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_chatbot_model, get_embeddings_model
from utils.config import ModelConfig
from utils.search_index import SearchIndex

router = APIRouter(prefix="/chat", tags=["chat"])

search_index = SearchIndex(
    index_path=ModelConfig.FAISS_INDEX_PATH,
    metadata_path=ModelConfig.FAISS_METADATA_PATH,
)

DICT_PATTERNS = ["what is", "what does", "meaning of", "means", "define", "definition"]

SYSTEM_PROMPT = (
    "You are StreamLand's helpful AI assistant for students watching educational videos. "
    "Use the provided video context as your primary knowledge source. "
    "Answer clearly and concisely. "
    "When no relevant video is found, answer from general knowledge but note it. "
    "Match the user's language (Vietnamese or English)."
)


# ─── Pydantic models ─────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str          # "user" | "assistant"
    msg: str


class VideoCard(BaseModel):
    """Structured video card returned to frontend — no markdown parsing needed."""
    id: str
    title: str
    teacher: str = ""
    category: str = ""
    thumbnail_url: str = ""
    views: int = 0
    summary: str = ""
    link: str = ""


class ChatRequest(BaseModel):
    message: str
    # history chứa toàn bộ lượt hội thoại trong session hiện tại
    # Frontend giữ state, gửi lên mỗi lần — KHÔNG lưu server-side
    history: List[ChatMessage] = Field(default_factory=list)
    exclude_ids: List[str] = Field(default_factory=list)
    top_k: int = 5
    # Context video đang xem — dùng để ưu tiên nội dung liên quan
    current_video_id: Optional[str] = None
    current_video_title: Optional[str] = None


class ChatResponse(BaseModel):
    status: str
    message: str
    response: str
    # Trả về video cards dạng structured thay vì markdown text
    video_cards: List[VideoCard] = Field(default_factory=list)
    retrieved_ids: List[str] = Field(default_factory=list)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_teacher_from_video_id(video_id: Optional[str], search_index) -> str:
    """Lấy teacher name từ current_video_id qua FAISS metadata."""
    if not video_id:
        return ""
    try:
        if search_index.index is None:
            search_index.load()
        for meta in search_index.metadata:
            if str(meta.get("id", "")) == video_id:
                return meta.get("teacher_name") or meta.get("teacher") or ""
    except Exception:
        pass
    return ""

def is_teacher_query(q: str) -> bool:
    q_lower = q.lower()
    return any(p in q_lower for p in [
        "by this teacher", "same teacher", "more by", "videos by",
        "by the teacher", "from this teacher", "of this teacher"
    ])

def is_recommend_query(q: str) -> bool:
    """Chỉ trả video_cards khi user thật sự muốn recommend."""
    q_lower = q.lower()
    return any(p in q_lower for p in [
        "recommend", "suggest", "video hay", "gợi ý", "tìm video",
        "show me videos", "find videos", "what should i watch",
        "nên xem", "video nào", "video về", "videos about",
        "videos on", "more videos", "thêm video", "video nữa",
        "what to watch", "learning resources", "tài liệu học",
        "by this teacher", "same teacher", "more by",
    ])

def is_definition_query(q: str) -> bool:
    q_lower = q.lower().strip()
    if any(p in q_lower for p in DICT_PATTERNS):
        return True
    words = q_lower.split()
    return len(words) <= 3 and ("mean" in q_lower or "means" in q_lower)


def _format_history(history: List[ChatMessage], max_turns: int = 6) -> str:
    """Format lịch sử hội thoại — lấy max_turns turns gần nhất."""
    recent = history[-(max_turns * 2):]  # mỗi turn = 1 user + 1 assistant
    lines = []
    for item in recent:
        role = "User" if item.role == "user" else "Assistant"
        lines.append(f"{role}: {item.msg}")
    return "\n".join(lines)


def _meta_to_video_card(meta: dict) -> VideoCard:
    """Chuyển metadata từ FAISS sang VideoCard structured."""
    vid_id = str(meta.get("id", ""))
    return VideoCard(
        id=vid_id,
        title=meta.get("title") or meta.get("name") or "Untitled",
        teacher=meta.get("teacher_name") or meta.get("teacher") or "",
        category=meta.get("category") or "",
        thumbnail_url=meta.get("thumbnail") or meta.get("thumbnailUrl") or "",
        views=int(meta.get("totalViews") or meta.get("views") or 0),
        summary=meta.get("summary") or meta.get("description") or "",
        link=f"/student/video/{vid_id}" if vid_id else "",
    )


def _format_doc_for_prompt(meta: dict) -> str:
    """Format ngắn gọn để đưa vào prompt — không dùng làm card."""
    title = meta.get("title") or ""
    teacher = meta.get("teacher_name") or meta.get("teacher") or ""
    category = meta.get("category") or ""
    summary = meta.get("summary") or meta.get("description") or ""
    parts = [p for p in [title, teacher, category] if p]
    header = " | ".join(parts)
    snippet = summary[:200] if summary else ""
    return f"{header}\n{snippet}".strip()


def retrieve(
    query: str,
    embeddings_model,
    exclude_ids: Optional[List[str]],
    top_k: int,
    boost_id: Optional[str] = None,
) -> Tuple[List[str], List[str], List[VideoCard]]:
    """
    Semantic search + optional boost cho video đang xem.
    Trả về (doc_texts, ids, video_cards).
    """
    embedding = embeddings_model.embed(query)[0]
    candidate_k = max(top_k * 6, 30)
    results = search_index.search(embedding, candidate_k)

    exclude_set = set(exclude_ids or [])
    docs: List[str] = []
    ids: List[str] = []
    cards: List[VideoCard] = []

    # Nếu có video đang xem, push nó lên đầu nếu tìm thấy trong results
    if boost_id:
        for item in results:
            meta = item.get("metadata", {})
            if str(meta.get("id", "")) == boost_id:
                formatted = _format_doc_for_prompt(meta)
                if formatted:
                    docs.append(formatted)
                    ids.append(boost_id)
                    cards.append(_meta_to_video_card(meta))
                break

    for item in results:
        if len(docs) >= top_k:
            break
        meta = item.get("metadata", {})
        item_id = str(meta.get("id", ""))
        if item_id in exclude_set or item_id in ids:
            continue
        formatted = _format_doc_for_prompt(meta)
        if not formatted:
            continue
        docs.append(formatted)
        ids.append(item_id)
        cards.append(_meta_to_video_card(meta))

    return docs, ids, cards


def build_chat_prompt(
    q: str,
    history: List[ChatMessage],
    embeddings_model,
    exclude_ids: List[str],
    top_k: int,
    current_video_id: Optional[str],
    current_video_title: Optional[str],
    force_retrieve: bool = False,
) -> Tuple[str, List[str], List[VideoCard]]:
    """Build prompt đầy đủ với history + context video + retrieved docs."""

    history_text = _format_history(history)

    # Context về video đang xem
    video_context = ""
    if current_video_title:
        video_context = f"The student is currently watching: \"{current_video_title}\"\n"

    new_ids: List[str] = []
    cards: List[VideoCard] = []
    ctx = ""

    # Chỉ retrieve khi query thật sự cần recommend video
    if force_retrieve or is_recommend_query(q):
        retrieved_docs, new_ids, cards = retrieve(
            q,
            embeddings_model,
            exclude_ids,
            top_k,
            boost_id=current_video_id,
        )
        if retrieved_docs:
            ctx = "Relevant videos from StreamLand:\n" + "\n---\n".join(retrieved_docs[:top_k])
    elif current_video_title:
        # Không recommend nhưng vẫn có context video đang xem
        ctx = f"Context: Student is watching \"{current_video_title}\". Answer their question directly."
    
    # Prompt format
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{video_context}"
        f"{ctx}\n\n"
        f"Conversation so far:\n{history_text}\n"
        f"User: {q}\nAssistant:"
    )

    return prompt, new_ids, cards


def build_dictionary_prompt(word: str) -> str:
    return (
        f"You are a clear and concise teacher. "
        f"Explain \"{word}\" with: meaning, key nuances, and 2 practical examples. "
        f"Be thorough but avoid padding.\nAssistant:"
    )


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    model=Depends(get_chatbot_model),
    embeddings_model=Depends(get_embeddings_model),
) -> ChatResponse:
    """
    Chat với StreamLand AI.

    - history: toàn bộ lịch sử session gửi từ frontend (stateless server)
    - video_cards: trả về structured JSON — frontend render trực tiếp, không parse markdown
    - current_video_id / current_video_title: ưu tiên ngữ cảnh video đang xem
    """
    try:
        video_cards: List[VideoCard] = []
        new_ids: List[str] = []

        if is_definition_query(request.message):
            word = request.message
            for p in DICT_PATTERNS:
                word = word.lower().replace(p, "").strip()
            word = word.replace("?", "").strip() or request.message
            prompt = build_dictionary_prompt(word)
        elif is_teacher_query(request.message):
            teacher_name = get_teacher_from_video_id(request.current_video_id, search_index)
            enriched_message = f"Recommend videos by teacher {teacher_name}" if teacher_name else request.message
            prompt, new_ids, video_cards = build_chat_prompt(
                enriched_message,
                request.history,
                embeddings_model,
                request.exclude_ids,
                request.top_k,
                request.current_video_id,
                request.current_video_title,
                force_retrieve=True,
            )
        else:
            # General Q&A — không tự động recommend video
            prompt, new_ids, video_cards = build_chat_prompt(
                request.message,
                request.history,
                embeddings_model,
                request.exclude_ids,
                request.top_k,
                request.current_video_id,
                request.current_video_title,
                force_retrieve=False,  # chỉ retrieve nếu is_recommend_query() = True
            )

        response_text = await asyncio.to_thread(model.generate, prompt)

        # Strip markdown links khỏi text nếu đã có video_cards
        # Model được train để output markdown format — cần clean để tránh render duplicate
        import re
        if video_cards:
            response_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', response_text)
            # Bỏ các dòng chỉ chứa link /student/... hoặc http://
            response_text = re.sub(r'\n?-?\s*(https?://\S+|/student/\S+)\s*', '', response_text)
            response_text = response_text.strip()

        return ChatResponse(
            status="success",
            message=request.message,
            response=response_text,
            video_cards=video_cards,
            retrieved_ids=new_ids,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc