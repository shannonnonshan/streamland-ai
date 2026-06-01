"""Chat endpoint using chatbot model + search index."""

import asyncio
import random
from typing import List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Depends
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
    "You are StreamLand's helpful assistant. Use the provided videos as primary context. "
    "If no relevant videos are found, suggest looking for general educational videos online."
)


class ChatMessage(BaseModel):
    role: str
    msg: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    exclude_ids: List[str] = Field(default_factory=list)
    top_k: int = 5


def is_definition_query(q: str) -> bool:
    q = q.lower().strip()
    if any(p in q for p in DICT_PATTERNS):
        return True
    words = q.split()
    if len(words) <= 3 and ("mean" in q or "means" in q):
        return True
    return False


def build_dictionary_prompt(word: str) -> str:
    return (
        "You are a thoughtful teacher. Explain the meaning of \""
        + word
        + "\" in depth. Return in format: Meaning, Nuance, Examples. Assistant:"
    )


def _format_history(history: List[ChatMessage]) -> str:
    convo = ""
    for item in history[-6:]:
        role = "User" if item.role == "user" else "Assistant"
        convo += f"{role}: {item.msg}\n"
    return convo


def _format_doc(meta: dict) -> str:
    title = meta.get("title") or meta.get("name") or ""
    description = meta.get("description") or ""
    content = meta.get("content") or meta.get("text") or ""
    teacher = meta.get("teacher_name") or meta.get("teacher") or ""
    status = meta.get("status") or ""
    parts = [title, description, content]
    snippet = " ".join(part.strip() for part in parts if part and str(part).strip()).strip()
    header = " | ".join([piece for piece in [title, teacher, status] if piece])
    if header:
        return f"{header}\n{snippet}"
    return snippet


def _unique_shuffle(items: List[str]) -> List[str]:
    unique_items = list(dict.fromkeys(item for item in items if item and str(item).strip()))
    random.shuffle(unique_items)
    return unique_items


def generate_video_suggestions(video_item: dict, fallback_query: str = "") -> List[str]:
    suggestions: List[str] = []

    video_id = str(video_item.get("id", "")).strip()
    teacher_name = str(video_item.get("teacher_name") or video_item.get("teacher") or "").strip()
    title = str(video_item.get("title") or "").strip()
    title_words = title.split()
    category = str(video_item.get("category") or "").strip()
    tags = list(video_item.get("tags") or [])

    if video_id:
        suggestions.append(f"Summarize this video {video_id}")
        suggestions.append(f"What is the description of this video {video_id}?")

    if teacher_name and teacher_name != "N/A":
        suggestions.append(f"Recommend more videos by {teacher_name}")
        suggestions.append(f"Who is {teacher_name}?")

    if category and category != "N/A":
        suggestions.append(f"Recommend more videos about {category}")
        suggestions.append(f"Explain {category} in simple terms.")

    if tags:
        random.shuffle(tags)
        for tag in tags[:3]:
            suggestions.append(f"Recommend more videos about {tag}")

    if title_words:
        suggestions.append(f"What is {random.choice(title_words)}?")

    if fallback_query.strip():
        suggestions.append(f"Recommend more videos related to {fallback_query.strip()}")

    suggestions.append("Recommend more videos in general")
    return _unique_shuffle(suggestions)


def retrieve(query: str, embeddings_model, exclude_ids: Optional[List[str]], top_k: int) -> Tuple[List[str], List[str], List[dict]]:
    embedding = embeddings_model.embed(query)[0]
    candidate_k = max(top_k * 5, top_k)
    results = search_index.search(embedding, candidate_k)

    exclude_set = set(exclude_ids or [])
    docs: List[str] = []
    new_ids: List[str] = []
    metas: List[dict] = []

    for item in results:
        meta = item.get("metadata", {})
        item_id = str(meta.get("id", ""))
        if item_id in exclude_set:
            continue
        formatted = _format_doc(meta)
        if not formatted:
            continue
        docs.append(formatted)
        new_ids.append(item_id)
        metas.append(meta)
        if len(docs) >= top_k:
            break

    return docs, new_ids, metas


def build_chat_prompt(q: str, history: List[ChatMessage], embeddings_model, exclude_ids: List[str], top_k: int):
    convo = _format_history(history)
    retrieved_docs, new_ids, retrieved_meta = retrieve(q, embeddings_model, exclude_ids, top_k)

    if not retrieved_docs:
        ctx = (
            "No relevant videos found in StreamLand database for this topic. "
            "You may suggest looking for general educational videos on YouTube related to the user's query."
        )
        new_ids = []
    else:
        ctx = "\n---\n".join(retrieved_docs)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Relevant videos from StreamLand database:\n{ctx}\n\n"
        f"Conversation history:\n{convo}"
        f"User: {q}\nAssistant:"
    )

    return prompt, new_ids, retrieved_meta


@router.post("")
async def chat(
    request: ChatRequest,
    model=Depends(get_chatbot_model),
    embeddings_model=Depends(get_embeddings_model),
):
    """Chat with StreamLand context using chatbot model."""
    try:
        if is_definition_query(request.message):
            prompt = build_dictionary_prompt(request.message)
            new_ids: List[str] = []
            retrieved_meta: List[dict] = []
        else:
            prompt, new_ids, retrieved_meta = build_chat_prompt(
                request.message,
                request.history,
                embeddings_model,
                request.exclude_ids,
                request.top_k,
            )

        response = await asyncio.to_thread(model.generate, prompt)
        suggestions: List[str] = []
        if retrieved_meta:
            for item in retrieved_meta[:3]:
                suggestions.extend(generate_video_suggestions(item, request.message))
        else:
            suggestions = generate_video_suggestions({"title": request.message}, request.message)

        return {
            "status": "success",
            "message": request.message,
            "response": response,
            "retrieved_ids": new_ids,
            "suggestions": suggestions[:8],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
