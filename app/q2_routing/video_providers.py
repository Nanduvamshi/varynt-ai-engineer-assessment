"""Mock video providers."""

import hashlib
import time

from app.q2_routing.base import BaseProvider, GenerationResult
from app.q5_resilience.retry import TransientProviderError


def _stable_id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


class MockRunwayVideoProvider(BaseProvider):
    name = "runway-gen3"
    modality = "video"

    def generate(self, prompt: str, force_fail: bool = False) -> GenerationResult:
        if force_fail:
            raise TransientProviderError("simulated 504 from runway-gen3")
        time.sleep(0.12)
        return GenerationResult(
            modality="video",
            provider=self.name,
            asset_url=f"https://cdn.varynt.example/vid/{_stable_id(prompt)}.mp4",
            duration_ms=120,
        )


class MockPikaVideoProvider(BaseProvider):
    name = "pika-1.5"
    modality = "video"

    def generate(self, prompt: str, force_fail: bool = False) -> GenerationResult:
        time.sleep(0.15)
        return GenerationResult(
            modality="video",
            provider=self.name,
            asset_url=f"https://cdn.varynt.example/vid/pika/{_stable_id(prompt)}.mp4",
            duration_ms=150,
        )
