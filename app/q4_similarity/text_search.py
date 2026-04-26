from __future__ import annotations
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
_CORPUS_PATH = Path(__file__).parent / "data" / "texts.json"
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TextHit(BaseModel):
    id: str
    text: str
    score: float


class TextSearchResult(BaseModel):
    query: str
    hits: list[TextHit]
    model: str = _MODEL_NAME


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    logger.info("Loading sentence-transformers model: %s", _MODEL_NAME)
    return SentenceTransformer(_MODEL_NAME)


@lru_cache(maxsize=1)
def _load_corpus() -> tuple[list[dict], np.ndarray]:
    with _CORPUS_PATH.open("r", encoding="utf-8") as f:
        items = json.load(f)
    model = _load_model()
    texts = [item["text"] for item in items]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return (items, np.asarray(embeddings, dtype=np.float32))


def search_text(query: str, top_k: int = 3) -> TextSearchResult:
    items, corpus_emb = _load_corpus()
    model = _load_model()
    q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    scores = (corpus_emb @ q_emb.T).flatten()
    top_idx = np.argsort(-scores)[:top_k]
    hits = [
        TextHit(id=items[i]["id"], text=items[i]["text"], score=float(scores[i]))
        for i in top_idx
    ]
    return TextSearchResult(query=query, hits=hits)
