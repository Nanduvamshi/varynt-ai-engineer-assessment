"""FastAPI routes for Q4: text + face similarity search."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.q4_similarity.text_search import TextSearchResult, search_text
from app.q4_similarity.face_search import FaceSearchResult, search_face

router = APIRouter()


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 3


@router.post("/search/text", response_model=TextSearchResult)
def text_search_endpoint(req: TextSearchRequest) -> TextSearchResult:
    return search_text(req.query, top_k=req.top_k)


@router.post("/search/face", response_model=FaceSearchResult)
async def face_search_endpoint(
    file: UploadFile = File(..., description="Query face image (jpg/png)"),
    top_k: int = 3,
) -> FaceSearchResult:
    suffix = Path(file.filename or "query.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, .png supported")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        return search_face(tmp_path, top_k=top_k)
    finally:
        tmp_path.unlink(missing_ok=True)
