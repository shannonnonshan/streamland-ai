"""Staged content moderation for mixed EN/VI text.

The implementation keeps the requested moderation flow local to this repository
and lazy-loads optional ML dependencies so the API can start even when heavy
model weights are unavailable.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
import importlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is already a project dependency
    torch = None

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional at runtime
    pipeline = None

try:
    lingua_module = importlib.import_module("lingua")
    Language = getattr(lingua_module, "Language", None)
    LanguageDetectorBuilder = getattr(lingua_module, "LanguageDetectorBuilder", None)
except Exception:  # pragma: no cover - optional at runtime
    Language = None
    LanguageDetectorBuilder = None

try:
    sentence_transformers_module = importlib.import_module("sentence_transformers")
    SentenceTransformer = getattr(sentence_transformers_module, "SentenceTransformer", None)
except Exception:  # pragma: no cover - optional at runtime
    SentenceTransformer = None

from models.base import BaseModel
from utils.config import ModelConfig


SAFE = "SAFE"
REVIEW = "REVIEW"
BLOCK = "BLOCK"

TOKEN_RE = re.compile(r"\w+|[^\w\s]+", re.UNICODE)
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}", re.UNICODE)

VI_DIACRITICS = set("ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụýỳỷỹỵ")

BYPASS_EXPANSIONS = [
    (re.compile(r"\bđjt\b", re.IGNORECASE), "địt"),
    (re.compile(r"\bdm\b", re.IGNORECASE), "địt mẹ"),
    (re.compile(r"\bđmm\b", re.IGNORECASE), "địt mẹ mày"),
    (re.compile(r"\bu\s+r\b", re.IGNORECASE), "you are"),
    (re.compile(r"\bur\b", re.IGNORECASE), "you are"),
]

EN_TOXIC_LEXICON = {
    "idiot",
    "stupid",
    "dumb",
    "fool",
    "moron",
    "trash",
    "shit",
    "fuck",
    "bitch",
    "bastard",
    "kill",
    "hate",
    "loser",
    "asshole",
    "suck",
}

VI_TOXIC_LEXICON = {
    "địt",
    "địt mẹ",
    "đồ ngu",
    "ngu",
    "óc chó",
    "chó",
    "lồn",
    "cặc",
    "đmm",
    "dm",
    "mẹ mày",
    "thằng ngu",
    "con đĩ",
    "đồ chó",
    "đồ khốn",
}

DEFAULT_EN_MODEL = getattr(ModelConfig, "MODERATION_EN_MODEL", os.getenv("MODERATION_EN_MODEL", "s-nlp/roberta_toxicity_classifier"))
DEFAULT_VI_MODEL = getattr(ModelConfig, "MODERATION_VI_MODEL", os.getenv("MODERATION_VI_MODEL", "cardiffnlp/twitter-xlm-roberta-base-offensive"))
DEFAULT_FULL_MODEL = getattr(ModelConfig, "MODERATION_FULL_MODEL", os.getenv("MODERATION_FULL_MODEL", DEFAULT_VI_MODEL))
DEFAULT_REWRITE_MODEL = getattr(ModelConfig, "MODERATION_REWRITE_MODEL", os.getenv("MODERATION_REWRITE_MODEL", "s-nlp/bart-base-detox"))
DEFAULT_EMBEDDING_MODEL = getattr(ModelConfig, "MODERATION_EMBEDDING_MODEL", os.getenv("MODERATION_EMBEDDING_MODEL", "BAAI/bge-m3"))


@dataclass
class ModerationSpan:
    text: str
    lang: str
    span_type: str
    token_count: int
    score: float = 0.0
    categories: List[str] = field(default_factory=list)


DEFAULT_EXAMPLES = [
    {"text": "you are an idiot", "label": BLOCK, "score": 0.94, "lang": "en"},
    {"text": "mày thật sự là đồ ngu", "label": BLOCK, "score": 0.93, "lang": "vi"},
    {"text": "that was a stupid thing to say", "label": REVIEW, "score": 0.61, "lang": "en"},
    {"text": "đừng nói kiểu đó nữa", "label": SAFE, "score": 0.08, "lang": "vi"},
    {"text": "you are being annoying", "label": REVIEW, "score": 0.58, "lang": "en"},
    {"text": "địt mẹ mày", "label": BLOCK, "score": 0.97, "lang": "vi"},
]


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower().strip()
    for pattern, replacement in BYPASS_EXPANSIONS:
        text = pattern.sub(replacement, text)
    text = REPEATED_CHAR_RE.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def _word_tokens(text: str) -> List[str]:
    return [token for token in _tokenize(text) if re.search(r"\w", token, re.UNICODE)]


def _is_vietnamese_token(token: str) -> bool:
    return any(char in VI_DIACRITICS for char in token)


class _HeuristicScorer:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def __call__(self, text: str, **kwargs):
        normalized = _normalize_text(text)
        words = set(_word_tokens(normalized))
        hits = sum(1 for word in EN_TOXIC_LEXICON | VI_TOXIC_LEXICON if word in normalized)

        base = min(0.98, 0.12 + (0.18 * hits))
        if {"kill", "hate"} & words:
            base = min(0.99, base + 0.15)

        return [
            {"label": "toxicity", "score": round(base, 4)},
            {"label": "insult", "score": round(min(0.99, base * 0.85), 4)},
            {"label": "profanity", "score": round(min(0.99, base * 0.75), 4)},
            {"label": "threat", "score": round(min(0.99, base * 0.35), 4)},
        ]


class ModerationModel(BaseModel):
    """Content moderation for mixed EN/VI text."""

    def __init__(self, model_path=None, from_hf=True):
        super().__init__(model_path or "staged-moderation", from_hf)
        self.en_model_id = DEFAULT_EN_MODEL
        self.vi_model_id = DEFAULT_VI_MODEL
        self.full_model_id = DEFAULT_FULL_MODEL
        self.rewrite_model_id = DEFAULT_REWRITE_MODEL
        self.embedding_model_id = DEFAULT_EMBEDDING_MODEL

        self.greyzone_lower = float(os.getenv("MODERATION_GREYZONE_LOWER", "0.40"))
        self.greyzone_upper = float(os.getenv("MODERATION_GREYZONE_UPPER", "0.70"))
        self.block_threshold = float(os.getenv("MODERATION_BLOCK_THRESHOLD", "0.85"))
        self.review_threshold = float(os.getenv("MODERATION_REVIEW_THRESHOLD", "0.55"))

        self._lingua_detector = self._build_lingua_detector()
        self._embedding_model = None
        self._scorers: Dict[str, Any] = {}
        self._examples = list(DEFAULT_EXAMPLES)

    @property
    def model_type(self):
        return "moderation"

    def _load_model(self):
        """Pipeline components are loaded lazily on demand."""
        return None

    def process_input(self, input_data):
        if not isinstance(input_data, str):
            raise TypeError(f"Invalid input type for moderation: {type(input_data)}")
        return input_data

    def infer(self, processed_input):
        return self.moderate_text(processed_input)

    def moderate_text(self, text, rewrite: bool = True):
        return self.evaluate(text, rewrite=rewrite)

    def moderate_image(self, image_path):
        raise NotImplementedError("Image moderation is not implemented in this flow")

    def _build_lingua_detector(self):
        if LanguageDetectorBuilder is None or Language is None:
            return None

        try:
            return LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.VIETNAMESE).build()
        except Exception:
            return None

    def _get_scorer(self, model_id: str):
        if model_id in self._scorers:
            return self._scorers[model_id]

        scorer = None
        if pipeline is not None:
            try:
                device = 0 if torch is not None and torch.cuda.is_available() else -1
                scorer = pipeline(
                    "text-classification",
                    model=model_id,
                    tokenizer=model_id,
                    device=device,
                    top_k=None,
                    truncation=True,
                )
            except Exception:
                scorer = None

        if scorer is None:
            scorer = _HeuristicScorer(model_id)

        self._scorers[model_id] = scorer
        return scorer

    def _scan_lexicon(self, text: str) -> Dict[str, Any]:
        normalized = _normalize_text(text)
        hits = [token for token in sorted(EN_TOXIC_LEXICON | VI_TOXIC_LEXICON, key=len, reverse=True) if token in normalized]
        return {"normalized": normalized, "hits": hits, "safe": len(hits) == 0}

    def _token_language(self, token: str) -> str:
        if _is_vietnamese_token(token):
            return "vi"
        if token.lower() in {"you", "your", "are", "idiot", "stupid", "fuck", "shit", "kill", "hate"}:
            return "en"
        if re.search(r"[a-z]", token.lower()):
            return "en"
        return "vi"

    def _split_language_spans(self, text: str) -> Tuple[List[ModerationSpan], Dict[str, Any]]:
        normalized = _normalize_text(text)
        tokens = _word_tokens(normalized)
        if not tokens:
            return [], {"en_ratio": 0.0, "vi_ratio": 0.0, "mixed": False}

        spans: List[ModerationSpan] = []
        current_tokens: List[str] = []
        current_lang = self._token_language(tokens[0])

        for token in tokens:
            lang = self._token_language(token)
            if lang != current_lang and current_tokens:
                span_text = " ".join(current_tokens)
                spans.append(ModerationSpan(span_text, current_lang, "language", len(current_tokens)))
                current_tokens = [token]
                current_lang = lang
            else:
                current_tokens.append(token)

        if current_tokens:
            spans.append(ModerationSpan(" ".join(current_tokens), current_lang, "language", len(current_tokens)))

        # Merge single-token spans into the nearest same-language span when possible.
        index = 0
        while index < len(spans):
            if spans[index].token_count >= 2 or len(spans) == 1:
                index += 1
                continue

            merged = False
            if index > 0 and spans[index - 1].lang == spans[index].lang:
                spans[index - 1].text = f"{spans[index - 1].text} {spans[index].text}".strip()
                spans[index - 1].token_count += spans[index].token_count
                spans.pop(index)
                merged = True
            elif index + 1 < len(spans) and spans[index + 1].lang == spans[index].lang:
                spans[index + 1].text = f"{spans[index].text} {spans[index + 1].text}".strip()
                spans[index + 1].token_count += spans[index].token_count
                spans.pop(index)
                merged = True

            if not merged:
                index += 1

        en_tokens = sum(span.token_count for span in spans if span.lang == "en")
        vi_tokens = sum(span.token_count for span in spans if span.lang == "vi")
        total_tokens = max(1, len(tokens))
        stats = {
            "en_ratio": round(en_tokens / total_tokens, 4),
            "vi_ratio": round(vi_tokens / total_tokens, 4),
            "mixed": en_tokens > 0 and vi_tokens > 0,
        }
        return spans, stats

    def _build_spans(self, language_spans: List[ModerationSpan], full_text: str) -> List[ModerationSpan]:
        if not language_spans:
            return [ModerationSpan(full_text, "mixed", "full", len(_word_tokens(full_text)))]

        spans = [ModerationSpan(span.text, span.lang, span.span_type, span.token_count) for span in language_spans]
        for index in range(len(language_spans) - 1):
            left = language_spans[index]
            right = language_spans[index + 1]
            if left.lang == right.lang:
                continue

            boundary_text = f"{left.text} {right.text}".strip()
            spans.append(
                ModerationSpan(
                    boundary_text,
                    "mixed",
                    "boundary",
                    len(_word_tokens(boundary_text)),
                )
            )

        spans.append(ModerationSpan(full_text, "mixed", "full", len(_word_tokens(full_text))))
        return spans

    def _normalize_scores(self, raw_output: Any) -> Dict[str, float]:
        if isinstance(raw_output, dict):
            raw_output = [raw_output]

        if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list):
            raw_output = raw_output[0]

        if not isinstance(raw_output, list):
            return {"toxicity": float(raw_output) if isinstance(raw_output, (int, float)) else 0.0}

        label_map = {
            "toxicity": "toxicity",
            "toxic": "toxicity",
            "severe_toxic": "toxicity",
            "severe_toxicity": "toxicity",
            "insult": "insult",
            "offensive": "offensive",
            "obscene": "profanity",
            "profanity": "profanity",
            "threat": "threat",
            "hate": "hate",
            "identity_hate": "hate",
            "neutral": "toxicity",
            "clean": "toxicity",
            "safe": "toxicity",
        }

        scores: Dict[str, float] = {}
        for item in raw_output:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip().lower()
            mapped = label_map.get(label, label or "toxicity")
            score = float(item.get("score", 0.0))
            scores[mapped] = max(scores.get(mapped, 0.0), score)

        if not scores:
            scores["toxicity"] = 0.0
        elif "toxicity" not in scores:
            scores["toxicity"] = max(scores.values())

        return scores

    def _classify_span(self, span: ModerationSpan, model_id: str) -> ModerationSpan:
        scorer = self._get_scorer(model_id)
        raw_output = scorer(span.text, truncation=True)
        scores = self._normalize_scores(raw_output)
        span.score = round(float(scores.get("toxicity", max(scores.values()) if scores else 0.0)), 4)
        span.categories = [name for name, value in scores.items() if value >= 0.5 and name != "toxicity"]
        if span.score >= 0.5 and "toxicity" not in span.categories:
            span.categories.insert(0, "toxicity")
        return span

    def _score_spans(self, spans: List[ModerationSpan]) -> List[ModerationSpan]:
        scored: List[ModerationSpan] = []
        for span in spans:
            if span.span_type == "full":
                scored.append(self._classify_span(span, self.full_model_id))
            elif span.lang == "en":
                scored.append(self._classify_span(span, self.en_model_id))
            else:
                scored.append(self._classify_span(span, self.vi_model_id))
        return scored

    def _build_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        if SentenceTransformer is None:
            return None
        try:
            self._embedding_model = SentenceTransformer(self.embedding_model_id)
        except Exception:
            self._embedding_model = None
        return self._embedding_model

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        model = self._build_embedding_model()
        if model is None:
            return None
        try:
            vector = model.encode([text], normalize_embeddings=True)
            return np.asarray(vector[0], dtype=np.float32)
        except Exception:
            return None

    def _similar_examples(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vector = self._embed_text(text)
        examples = self._examples

        if query_vector is None:
            query_words = set(_word_tokens(text))
            ranked: List[Tuple[float, Dict[str, Any]]] = []
            for example in examples:
                example_words = set(_word_tokens(example["text"]))
                overlap = len(query_words & example_words)
                similarity = float(overlap) / max(1, len(example_words))
                ranked.append((similarity, example))
        else:
            ranked = []
            for example in examples:
                example_vector = self._embed_text(example["text"])
                if example_vector is None:
                    continue
                similarity = float(np.dot(query_vector, example_vector))
                ranked.append((similarity, example))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "text": example["text"],
                "label": example["label"],
                "score": round(float(example["score"]), 4),
                "lang": example["lang"],
                "similarity": round(float(similarity), 4),
            }
            for similarity, example in ranked[:top_k]
        ]

    def _apply_greyzone_nudge(self, span: ModerationSpan, current_score: float) -> Tuple[float, List[Dict[str, Any]]]:
        if not (self.greyzone_lower <= current_score <= self.greyzone_upper):
            return current_score, []

        retrieved = self._similar_examples(span.text, top_k=3)
        if not retrieved:
            return current_score, []

        toxic_votes = sum(1 for item in retrieved if item["score"] >= self.review_threshold or item["label"] in {BLOCK, REVIEW})
        adjustment = 0.1 if toxic_votes >= 2 else -0.1
        return max(0.0, min(1.0, current_score + adjustment)), retrieved

    def _fuse_scores(self, scored_spans: List[ModerationSpan]) -> Tuple[float, List[str]]:
        if not scored_spans:
            return 0.0, []

        span_scores = [span.score for span in scored_spans if span.span_type != "boundary"] or [0.0]
        full_scores = [span.score for span in scored_spans if span.span_type == "full"] or [span_scores[-1]]
        overall = 0.7 * max(span_scores) + 0.3 * max(full_scores)
        categories = sorted({category for span in scored_spans for category in span.categories})
        return round(float(overall), 4), categories

    def _decision(self, score: float) -> str:
        if score >= self.block_threshold:
            return BLOCK
        if score >= self.review_threshold:
            return REVIEW
        return SAFE

    def _rewrite_span(self, text: str, lang: str) -> str:
        replacements = [
            (r"\bidiot\b", "person"),
            (r"\bstupid\b", "unhelpful"),
            (r"\bdumb\b", "unclear"),
            (r"\bfuck\b", "heck"),
            (r"\bshit\b", "mess"),
            (r"địt mẹ", ""),
            (r"địt", ""),
            (r"đồ ngu", "không phù hợp"),
            (r"ngu", "chưa phù hợp"),
            (r"óc chó", "không phù hợp"),
        ]

        rewritten = text
        for pattern, replacement in replacements:
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        if not rewritten:
            return "[content removed]"
        if lang == "vi" and rewritten == text:
            return text
        return rewritten

    def _rewrite_review_text(self, original_text: str, scored_spans: List[ModerationSpan]) -> str:
        revised = original_text
        for span in scored_spans:
            if span.score < self.review_threshold:
                continue
            replacement = self._rewrite_span(span.text, span.lang)
            if replacement and replacement != span.text:
                revised = revised.replace(span.text, replacement)
        return re.sub(r"\s+", " ", revised).strip()

    def evaluate(self, text: str, rewrite: bool = True, skip_rag: bool = False) -> Dict[str, Any]:
        lexicon_result = self._scan_lexicon(text)
        normalized = lexicon_result["normalized"]
        if lexicon_result["safe"]:
            return {
                "status": SAFE,
                "score": 0.0,
                "categories": [],
                "toxic_word": [],
            }

        language_spans, language_stats = self._split_language_spans(normalized)
        spans = self._build_spans(language_spans, normalized)
        scored_spans = self._score_spans(spans)

        greyzone_examples: Dict[str, List[Dict[str, Any]]] = {}
        if not skip_rag:
            for span in scored_spans:
                if self.greyzone_lower <= span.score <= self.greyzone_upper:
                    nudged_score, retrieved = self._apply_greyzone_nudge(span, span.score)
                    if retrieved:
                        greyzone_examples[span.text] = retrieved
                    span.score = nudged_score

        overall_score, categories = self._fuse_scores(scored_spans)
        status = self._decision(overall_score)

        return {
            "status": status,
            "score": round(float(overall_score), 4),
            "categories": categories,
            "toxic_word": sorted(set(lexicon_result["hits"])),
        }
