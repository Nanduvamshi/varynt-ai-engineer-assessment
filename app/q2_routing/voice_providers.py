import hashlib
import time
from app.q2_routing.base import BaseProvider, GenerationResult
from app.q5_resilience.retry import TransientProviderError


def _stable_id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


class MockElevenLabsVoiceProvider(BaseProvider):
    name = "elevenlabs-multilingual-v2"
    modality = "voice"

    def generate(self, prompt: str, force_fail: bool = False) -> GenerationResult:
        if force_fail:
            raise TransientProviderError("simulated 502 from elevenlabs")
        time.sleep(0.04)
        return GenerationResult(
            modality="voice",
            provider=self.name,
            asset_url=f"https://cdn.varynt.example/voice/{_stable_id(prompt)}.mp3",
            duration_ms=40,
        )


class MockOpenAITTSProvider(BaseProvider):
    name = "openai-tts-1"
    modality = "voice"

    def generate(self, prompt: str, force_fail: bool = False) -> GenerationResult:
        time.sleep(0.06)
        return GenerationResult(
            modality="voice",
            provider=self.name,
            asset_url=f"https://cdn.varynt.example/voice/openai/{_stable_id(prompt)}.mp3",
            duration_ms=60,
        )
