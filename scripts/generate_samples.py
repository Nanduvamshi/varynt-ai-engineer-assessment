"""One-shot script: drives every endpoint and writes real outputs to /samples/.

Run once with:
    OPENAI_API_KEY=sk-...  python -m scripts.generate_samples

OR keyless (uses MOCK_OPENAI=1 stand-ins) for the Q1 path:
    MOCK_OPENAI=1 python -m scripts.generate_samples

Outputs are committed to git so reviewers see them without running anything.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)

# Force module path so we can run as `python scripts/generate_samples.py` too.
sys.path.insert(0, str(ROOT))

from app.q1_classifier.classifier import LeadInput, classify_lead  # noqa: E402
from app.q2_routing.base import GenerationRequest  # noqa: E402
from app.q2_routing.router import generate as q2_generate  # noqa: E402
from app.q4_similarity.face_search import search_face  # noqa: E402
from app.q4_similarity.text_search import search_text  # noqa: E402


def write(name: str, payload) -> None:
    p = SAMPLES / name
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    else:
        data = payload
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  wrote {p.relative_to(ROOT)}")


def gen_q1() -> None:
    print("Q1 lead classifier")
    cases = {
        "q1_classify_hot.json": LeadInput(
            name="Priya Mehta",
            email="priya@northstarrealty.com",
            company="NorthStar Realty",
            role="Marketing Lead",
            source="webinar",
            message="We're ready to roll out lead-capture funnels for 8 agents this month. What does pricing look like for a mid-tier plan?",
            budget="$3,000-5,000/mo",
            timeline="this month",
            team_size=12,
        ),
        "q1_classify_warm.json": LeadInput(
            name="Bob Chen",
            email="bob@bobyoga.io",
            company="Bob Yoga Studio",
            role="Owner",
            source="organic",
            message="Looking into funnel builders for class signups. Comparing options. How does pricing work?",
            timeline="next quarter",
        ),
        "q1_classify_cold.json": LeadInput(
            name="Alex",
            source="ad",
            message="Just curious, browsing different tools. Maybe later this year.",
        ),
        "q1_classify_incomplete.json": LeadInput(
            email="someone@example.com",
        ),
    }
    for name, lead in cases.items():
        result = classify_lead(lead)
        write(name, {"input": lead.model_dump(exclude_none=True), "output": result.model_dump()})


def gen_q2() -> None:
    print("Q2 multi-provider routing")
    cases = {
        "q2_route_image.json": GenerationRequest(modality="image", prompt="modern real-estate hero banner with warm lighting"),
        "q2_route_video.json": GenerationRequest(modality="video", prompt="15s testimonial-style intro for a fitness coach"),
        "q2_route_voice.json": GenerationRequest(modality="voice", prompt="welcome message in a warm female voice"),
        "q2_route_image_fallback.json": GenerationRequest(
            modality="image",
            prompt="modern real-estate hero banner with warm lighting",
            force_fail=True,
        ),
    }
    for name, req in cases.items():
        result = q2_generate(req)
        write(name, {"input": req.model_dump(), "output": result.model_dump()})


def gen_q4_text() -> None:
    print("Q4 text similarity")
    queries = {
        "q4_text_search_real_estate.json": "I run a real estate firm and want to capture inbound leads",
        "q4_text_search_fitness.json": "Help me build a paid newsletter for my fitness clients",
        "q4_text_search_b2b_saas.json": "I have a SaaS launch coming up and need a high-converting funnel",
    }
    for name, q in queries.items():
        result = search_text(q, top_k=3)
        write(name, result)


def gen_q4_face() -> None:
    print("Q4 face similarity")
    faces_dir = ROOT / "app" / "q4_similarity" / "data" / "faces"
    # Use 2 indexed faces as queries; expect top hit = self with score ~1.0
    for i in (1, 4):
        query_path = faces_dir / f"face_{i}.jpg"
        if not query_path.exists():
            continue
        result = search_face(query_path, top_k=4)
        write(f"q4_face_search_face_{i}.json", result)


def main() -> None:
    print(f"Writing samples to {SAMPLES}")
    print(f"  MOCK_OPENAI={os.getenv('MOCK_OPENAI', '0')}")
    print(f"  OPENAI_API_KEY={'set' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print()
    gen_q1()
    gen_q2()
    gen_q4_text()
    gen_q4_face()
    print()
    print("done.")


if __name__ == "__main__":
    main()
