"""Tests for Q2 routing + Q5 fallback chain."""

from app.q2_routing.base import GenerationRequest
from app.q2_routing.router import generate


def test_image_normal_path():
    r = generate(GenerationRequest(modality="image", prompt="hello"))
    assert r.provider == "openai-dalle3"
    assert r.fallback_used is False
    assert r.asset_url.startswith("https://")


def test_image_fallback_chain_runs_when_primary_fails():
    r = generate(GenerationRequest(modality="image", prompt="hello", force_fail=True))
    assert r.provider == "stability-sdxl"
    assert r.fallback_used is True


def test_video_and_voice_chains():
    for modality, expected in [("video", "runway-gen3"), ("voice", "elevenlabs-multilingual-v2")]:
        r = generate(GenerationRequest(modality=modality, prompt="x"))
        assert r.provider == expected


def test_explicit_provider_picks_secondary_first():
    r = generate(GenerationRequest(modality="image", prompt="x", provider="stability-sdxl"))
    assert r.provider == "stability-sdxl"
