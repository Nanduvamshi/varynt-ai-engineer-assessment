from __future__ import annotations
import logging
from typing import Iterable
from fastapi import APIRouter, HTTPException
from app.q2_routing.base import (
    BaseProvider,
    GenerationRequest,
    GenerationResult,
    Modality,
)
from app.q2_routing.image_providers import (
    MockOpenAIImageProvider,
    MockStabilityImageProvider,
)
from app.q2_routing.video_providers import (
    MockPikaVideoProvider,
    MockRunwayVideoProvider,
)
from app.q2_routing.voice_providers import (
    MockElevenLabsVoiceProvider,
    MockOpenAITTSProvider,
)
from app.q5_resilience.retry import TransientProviderError, PermanentProviderError

logger = logging.getLogger(__name__)
router = APIRouter()
_PROVIDER_CHAINS: dict[Modality, list[BaseProvider]] = {
    "image": [MockOpenAIImageProvider(), MockStabilityImageProvider()],
    "video": [MockRunwayVideoProvider(), MockPikaVideoProvider()],
    "voice": [MockElevenLabsVoiceProvider(), MockOpenAITTSProvider()],
}


def _chain_for(modality: Modality, explicit: str | None) -> Iterable[BaseProvider]:
    chain = _PROVIDER_CHAINS[modality]
    if explicit is None:
        return chain
    explicit_provider = next((p for p in chain if p.name == explicit), None)
    if explicit_provider is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider '{explicit}' for modality '{modality}'.",
        )
    rest = [p for p in chain if p.name != explicit]
    return [explicit_provider] + rest


@router.post("/generate", response_model=GenerationResult)
def generate(req: GenerationRequest) -> GenerationResult:
    chain = list(_chain_for(req.modality, req.provider))
    last_err: Exception | None = None
    fallback_used = False
    for idx, provider in enumerate(chain):
        try:
            result = provider.generate(
                prompt=req.prompt, force_fail=req.force_fail and idx == 0
            )
            result.fallback_used = idx > 0
            return result
        except PermanentProviderError as e:
            logger.warning("Permanent error from %s: %s", provider.name, e)
            raise HTTPException(status_code=400, detail=str(e))
        except TransientProviderError as e:
            logger.warning(
                "Provider %s transient failure (%s); trying next in chain",
                provider.name,
                e,
            )
            last_err = e
            fallback_used = True
            continue
    raise HTTPException(
        status_code=503,
        detail=f"All providers in chain for modality '{req.modality}' failed: {last_err}",
        headers={"Retry-After": "10"},
    )
