import hashlib
import time
from app.q2_routing.base import BaseProvider, GenerationResult
from app.q5_resilience.retry import TransientProviderError


def _stable_id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


class MockOpenAIImageProvider(BaseProvider):
    name = "openai-dalle3"
    modality = "image"

    def generate(self, prompt: str, force_fail: bool = False) -> GenerationResult:
        if force_fail:
            raise TransientProviderError("simulated 503 from openai-dalle3")
        time.sleep(0.05)
        return GenerationResult(
            modality="image",
            provider=self.name,
            asset_url=f"https://cdn.varynt.example/img/{_stable_id(prompt)}.png",
            duration_ms=50,
        )


class MockStabilityImageProvider(BaseProvider):
    name = "stability-sdxl"
    modality = "image"

    def generate(self, prompt: str, force_fail: bool = False) -> GenerationResult:
        time.sleep(0.07)
        return GenerationResult(
            modality="image",
            provider=self.name,
            asset_url=f"https://cdn.varynt.example/img/sdxl/{_stable_id(prompt)}.png",
            duration_ms=70,
        )
