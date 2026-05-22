"""Build a FAISS index for search using the embeddings model."""

import json
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.embeddings.interface import EmbeddingModel
from utils.config import ModelConfig


def _load_jsonl(path: str) -> List[Any]:
    items: List[Any] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def _load_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_corpus(path: str) -> Any:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return _load_jsonl(path)
    if ext == ".json":
        return _load_json(path)
    if ext in {".txt", ".md"}:
        return _load_txt(path)
    raise ValueError("Unsupported corpus format. Use .jsonl, .json, .txt, or .md")


def _load_postgres_corpus() -> Dict[str, Any]:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL is not set for Postgres corpus loading")

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError as exc:
        raise ImportError("psycopg2 is not installed. Run: pip install psycopg2-binary") from exc

    conn = psycopg2.connect(db_url)
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SET search_path TO streamland")

        cursor.execute(
            """
            SELECT
                ls.id,
                ls.title,
                ls.description,
                u."fullName" as teacher_name,
                ls."totalViews",
                ls.status
            FROM streamland.livestreams ls
            JOIN streamland.users u ON ls."teacherId" = u.id
            WHERE ls.status IN ('LIVE', 'ENDED')
            ORDER BY ls."totalViews" DESC
            LIMIT 1000
            """
        )
        videos = cursor.fetchall()

        cursor.execute(
            """
            SELECT
                d.id,
                d.title,
                d.description,
                u."fullName" as teacher_name
            FROM streamland.documents d
            LEFT JOIN streamland.users u ON d."teacherId" = u.id
            LIMIT 500
            """
        )
        documents = cursor.fetchall()
    finally:
        conn.close()

    return {"videos": videos, "documents": documents, "transcripts": []}


def _join_parts(parts: List[str]) -> str:
    return " ".join(part.strip() for part in parts if part and str(part).strip()).strip()


def _build_unified_corpus(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        if not raw:
            return []

        unified: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw):
            if isinstance(item, str):
                unified.append(
                    {
                        "id": idx,
                        "content": item,
                        "type": "document",
                        "status": "N/A",
                    }
                )
                continue

            if not isinstance(item, dict):
                raise ValueError("Unsupported record type in corpus list")

            content = item.get("content") or item.get("text") or item.get("body")
            if not content:
                content = _join_parts([item.get("title", ""), item.get("description", "")])

            if not content:
                raise ValueError("Corpus item missing content/text/body/title")

            unified.append(
                {
                    "id": item.get("id", idx),
                    "content": content,
                    "type": item.get("type", "document"),
                    "status": item.get("status", "N/A"),
                    **{k: v for k, v in item.items() if k not in {"content", "text", "body"}},
                }
            )
        return unified

    if isinstance(raw, dict):
        videos = raw.get("videos", [])
        documents = raw.get("documents", [])
        transcripts = raw.get("transcripts", [])

        transcript_map: Dict[Any, str] = {}
        for item in transcripts:
            if not isinstance(item, dict):
                continue
            recording_id = item.get("recordingId") or item.get("id")
            if recording_id is None:
                continue
            summary = item.get("summary") or item.get("text") or ""
            if summary:
                transcript_map[recording_id] = summary

        unified: List[Dict[str, Any]] = []

        for idx, video in enumerate(videos):
            if not isinstance(video, dict):
                continue
            recording_id = video.get("id") or video.get("recordingId") or f"video-{idx}"
            summary = transcript_map.get(recording_id, "")
            content = _join_parts([video.get("title", ""), video.get("description", ""), summary])
            if not content:
                continue
            unified.append(
                {
                    **video,
                    "id": recording_id,
                    "content": content,
                    "type": "video",
                    "status": video.get("status", "N/A"),
                }
            )

        for idx, doc in enumerate(documents):
            if not isinstance(doc, dict):
                continue
            doc_id = doc.get("id") or f"doc-{idx}"
            content = _join_parts(
                [doc.get("title", ""), doc.get("description", ""), doc.get("content", "")]
            )
            if not content:
                continue
            unified.append(
                {
                    **doc,
                    "id": doc_id,
                    "content": content,
                    "type": "document",
                    "status": "N/A",
                }
            )

        return unified

    raise ValueError("Unsupported corpus format. Expected list or dict")


def _normalize_record(record: Dict[str, Any], fallback_id: int) -> Tuple[str, Dict[str, Any]]:
    content = record.get("content") or record.get("text") or record.get("body")
    if not content:
        raise ValueError("Record missing content/text/body")
    meta = dict(record)
    meta.setdefault("id", fallback_id)
    meta["content"] = content
    return content, meta


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def main() -> None:
    corpus_path = ModelConfig.SEARCH_CORPUS_PATH
    index_path = ModelConfig.FAISS_INDEX_PATH
    metadata_path = ModelConfig.FAISS_METADATA_PATH
    batch_size = int(os.getenv("SEARCH_BATCH_SIZE", "32"))
    source = os.getenv("SEARCH_SOURCE", "file").strip().lower()

    if source == "postgres":
        raw_records = _load_postgres_corpus()
    else:
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")
        raw_records = _load_corpus(corpus_path)
    unified_corpus = _build_unified_corpus(raw_records)
    if not unified_corpus:
        raise ValueError("Corpus is empty")

    texts: List[str] = []
    metadata: List[Dict[str, Any]] = []

    for idx, record in enumerate(unified_corpus):
        text, meta = _normalize_record(record, idx)
        texts.append(text)
        metadata.append(meta)

    model = EmbeddingModel(
        model_path=ModelConfig.EMBEDDINGS_MODEL,
        from_hf=ModelConfig.EMBEDDINGS_USE_HF,
    )

    embeddings: List[List[float]] = []
    for batch in _batched(texts, batch_size):
        embeddings.extend(model.embed(batch))

    vectors = np.array(embeddings, dtype="float32")
    if vectors.ndim != 2:
        raise ValueError("Expected 2D embeddings array")

    try:
        import faiss
    except ImportError as exc:
        raise ImportError("FAISS not installed. Run: pip install faiss-cpu") from exc

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False)

    print(f"Index written: {index_path}")
    print(f"Metadata written: {metadata_path}")
    print(f"Items indexed: {len(metadata)}")


if __name__ == "__main__":
    main()
