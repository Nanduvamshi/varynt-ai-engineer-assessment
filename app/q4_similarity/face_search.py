from __future__ import annotations
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)
_DATA_DIR = Path(__file__).parent / "data"
_FACES_DIR = _DATA_DIR / "faces"
_INDEX_PATH = _DATA_DIR / "face_embeddings.npz"


class FaceHit(BaseModel):
    file: str
    score: float


class FaceSearchResult(BaseModel):
    query_file: str
    hits: list[FaceHit]
    model: str = "insightface/buffalo_l"
    note: Optional[str] = None


@lru_cache(maxsize=1)
def _get_face_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.2)
    return app


def _embed_image(image_path: Path) -> Optional[np.ndarray]:
    import cv2

    app = _get_face_app()
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    faces.sort(
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True
    )
    emb = faces[0].normed_embedding
    return np.asarray(emb, dtype=np.float32)


def build_index() -> tuple[list[str], np.ndarray]:
    files = sorted(
        [
            p
            for p in _FACES_DIR.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    embeddings: list[np.ndarray] = []
    kept_files: list[str] = []
    for p in files:
        emb = _embed_image(p)
        if emb is None:
            logger.warning("No face detected in %s; skipping", p.name)
            continue
        embeddings.append(emb)
        kept_files.append(p.name)
    if not embeddings:
        raise RuntimeError(f"No usable face images in {_FACES_DIR}")
    matrix = np.stack(embeddings, axis=0).astype(np.float32)
    np.savez(_INDEX_PATH, files=np.array(kept_files), embeddings=matrix)
    logger.info("Wrote face index with %d entries to %s", len(kept_files), _INDEX_PATH)
    return (kept_files, matrix)


@lru_cache(maxsize=1)
def _load_index() -> tuple[list[str], np.ndarray]:
    if not _INDEX_PATH.exists():
        return build_index()
    with np.load(_INDEX_PATH, allow_pickle=False) as npz:
        return (list(npz["files"]), npz["embeddings"].astype(np.float32))


def search_face(query_path: Path, top_k: int = 3) -> FaceSearchResult:
    files, matrix = _load_index()
    q_emb = _embed_image(query_path)
    if q_emb is None:
        return FaceSearchResult(
            query_file=query_path.name, hits=[], note="No face detected in query image."
        )
    scores = matrix @ q_emb
    top_idx = np.argsort(-scores)[:top_k]
    hits = [FaceHit(file=files[i], score=float(scores[i])) for i in top_idx]
    return FaceSearchResult(query_file=query_path.name, hits=hits)
