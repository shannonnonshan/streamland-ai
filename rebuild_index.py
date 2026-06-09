"""
rebuild_index.py — Tự động rebuild FAISS index từ database mới nhất.
Chạy bằng cron job hoặc thủ công: python rebuild_index.py
"""

import os
import json
import logging
import urllib.request
import faiss
import psycopg2
import pandas as pd
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PG_HOST   = os.getenv("PG_HOST", "aws-1-ap-south-1.pooler.supabase.com")
PG_PORT   = os.getenv("PG_PORT", "6543")
PG_DB     = os.getenv("PG_DB", "postgres")
PG_SCHEMA = os.getenv("PG_SCHEMA", "streamland")
PG_USER   = os.getenv("PG_USER")
PG_PASS   = os.getenv("PG_PASS")
MONGO_URI = os.getenv("MONGODB_URL")

OUTPUT_DIR  = os.getenv("FAISS_OUTPUT_DIR", "./data")
INDEX_PATH  = os.path.join(OUTPUT_DIR, "embeddings.faiss")
META_PATH   = os.path.join(OUTPUT_DIR, "embeddings.meta.json")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
AI_PORT     = os.getenv("AI_PORT", "8000")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_server_host() -> str:
    """
    Tự detect host theo thứ tự ưu tiên:
    1. AI_SERVER_HOST env var (override thủ công)
    2. GCP metadata server → external IP của VM
    3. Fallback localhost
    """
    # 1. Override thủ công
    host = os.getenv("AI_SERVER_HOST")
    if host:
        log.info("Using AI_SERVER_HOST from env: %s", host)
        return host

    # 2. Thử lấy external IP từ GCP metadata server
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/externalIp",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            external_ip = resp.read().decode().strip()
            if external_ip:
                log.info("Detected GCP external IP: %s", external_ip)
                return external_ip
    except Exception:
        pass

    # 3. Fallback localhost
    log.info("Could not detect external IP — using localhost")
    return "localhost"


def load_postgres() -> pd.DataFrame:
    log.info("Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, database=PG_DB,
        user=PG_USER, password=PG_PASS,
    )
    query = f"""
        SELECT
            l.id,
            l.title,
            l.description,
            l.thumbnail,
            l.category,
            l."totalViews",
            u."fullName" as teacher,
            COALESCE(
                array_agg(t.name) FILTER (WHERE t.name IS NOT NULL),
                ARRAY[]::text[]
            ) as tags
        FROM {PG_SCHEMA}.livestreams l
        LEFT JOIN {PG_SCHEMA}.users u ON l."teacherId" = u.id
        LEFT JOIN {PG_SCHEMA}.tags t  ON t."livestreamId" = l.id
        WHERE l.status = 'ENDED'
        GROUP BY l.id, u."fullName", l.thumbnail
        LIMIT 5000;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    log.info("Loaded %d videos from PostgreSQL", len(df))
    return df


def load_mongo_transcripts() -> dict:
    log.info("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client["streamland"]
    docs = list(db["ai_transcript_summary"].find(
        {},
        {"summary": 1, "recordingId": 1, "_id": 0}
    ))
    client.close()
    transcript_map = {
        d["recordingId"]: d["summary"]
        for d in docs
        if d.get("recordingId") and d.get("summary")
    }
    log.info("Loaded %d transcripts from MongoDB", len(transcript_map))
    return transcript_map


def build_knowledge_base(df: pd.DataFrame, transcript_map: dict) -> list:
    knowledge_base = []
    for _, r in df.iterrows():
        summary = transcript_map.get(r.id, "")
        tags = list(r.tags or [])

        content_parts = []
        if r.title:
            content_parts += [r.title, r.title]
        if r.category:
            content_parts.append(r.category)
        if r.description:
            content_parts.append(r.description)
        if summary:
            content_parts.append(summary)
        if r.teacher:
            content_parts += [r.teacher, r.teacher]
        content_parts.extend(tags)

        knowledge_base.append({
            "id":           r.id,
            "title":        r.title or "",
            "description":  r.description or "",
            "thumbnail":    r.thumbnail or "",
            "category":     r.category or "",
            "teacher":      r.teacher or "",
            "teacher_name": r.teacher or "",
            "views":        int(r.totalViews or 0),
            "tags":         tags,
            "summary":      summary,
            "content":      " ".join(p.strip() for p in content_parts if p).strip(),
        })

    before = len(knowledge_base)
    knowledge_base = [
        item for item in knowledge_base
        if item["title"].strip() and len(item["title"].strip()) > 3 and item["id"].strip()
    ]
    log.info("Knowledge base: %d items (removed %d dirty)", len(knowledge_base), before - len(knowledge_base))
    log.info("Items with summary: %d", sum(1 for i in knowledge_base if i["summary"]))
    return knowledge_base


def build_faiss_index(knowledge_base: list) -> faiss.Index:
    log.info("Loading embedding model: %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    texts = [item["content"] or item["title"] for item in knowledge_base]
    log.info("Encoding %d texts...", len(texts))

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("FAISS index built: %d vectors (dim=%d)", index.ntotal, dim)
    return index


def save(index: faiss.Index, knowledge_base: list) -> None:
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False)
    log.info("Saved index → %s", INDEX_PATH)
    log.info("Saved metadata → %s", META_PATH)


def reload_server() -> None:
    host = get_server_host()
    url = f"http://{host}:{AI_PORT}/admin/reload-index"
    log.info("Reloading AI server at %s ...", url)
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode()
            log.info("Reload response: %s", body)
    except Exception as e:
        log.warning("Could not reload AI server: %s", e)
        log.warning("Restart the server manually to apply new index.")


def main() -> None:
    log.info("=== Starting FAISS rebuild ===")
    df = load_postgres()
    transcript_map = load_mongo_transcripts()
    knowledge_base = build_knowledge_base(df, transcript_map)

    if not knowledge_base:
        log.error("Knowledge base is empty — aborting.")
        return

    index = build_faiss_index(knowledge_base)
    save(index, knowledge_base)
    reload_server()
    log.info("=== Rebuild complete ===")


if __name__ == "__main__":
    main()