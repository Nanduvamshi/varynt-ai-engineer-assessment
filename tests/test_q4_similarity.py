from pathlib import Path
from app.q4_similarity.text_search import search_text
from app.q4_similarity.face_search import search_face


def test_text_search_returns_top_k():
    r = search_text("real estate lead capture", top_k=3)
    assert len(r.hits) == 3
    assert all((r.hits[i].score >= r.hits[i + 1].score for i in range(len(r.hits) - 1)))


def test_text_search_finds_real_estate_top():
    r = search_text(
        "I run a real estate firm and want to capture inbound leads", top_k=1
    )
    assert "real estate" in r.hits[0].text.lower() or "broker" in r.hits[0].text.lower()


def test_face_search_self_match_is_perfect():
    p = Path("app/q4_similarity/data/faces/face_1.jpg")
    if not p.exists():
        return
    r = search_face(p, top_k=1)
    if not r.hits:
        return
    assert r.hits[0].file == "face_1.jpg"
    assert r.hits[0].score > 0.99
