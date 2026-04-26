"""Q2 — Multi-provider generation router.

Architecture (see ARCHITECTURE.md for full diagram):

  POST /generate {modality, prompt, provider?, force_fail?}
       -> select primary by modality (or honor explicit provider)
       -> wrap call with Q5 retry decorator
       -> on failure, walk fallback chain (secondary -> cached -> template)
       -> return GenerationResult with provider name + url + flags

The router does NOT know what an image/video/voice actually IS — it only
knows the BaseProvider contract. Adding a new modality = add a new provider
class + register it.
"""

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
from app.q5_resilience.retry import (
    TransientProviderError,
    PermanentProviderError,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Per-modality provider chain: primary first, then fallback order.
# In production these would be loaded from config + driven by latency/cost
# routing decisions. For Q2 the chain itself is the architectural artifact.
_PROVIDER_CHAINS: dict[Modality, list[BaseProvider]] = {
    "image": [MockOpenAIImageProvider(), MockStabilityImageProvider()],
    "video": [MockRunwayVideoProvider(), MockPikaVideoProvider()],
    "voice": [MockElevenLabsVoiceProvider(), MockOpenAITTSProvider()],
}


def _chain_for(modality: Modality, explicit: str | None) -> Iterable[BaseProvider]:
    chain = _PROVIDER_CHAINS[modality]
    if explicit is None:
        return chain
    # Reorder so the explicit provider is tried first; others remain as fallbacks.
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
            # `force_fail` only applies to the primary so we can demo the chain.
            result = provider.generate(
                prompt=req.prompt,
                force_fail=(req.force_fail and idx == 0),
            )
            result.fallback_used = idx > 0
            return result
        except PermanentProviderError as e:
            # Permanent errors typically mean bad input - surface to caller.
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
