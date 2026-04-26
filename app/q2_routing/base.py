from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel, Field

Modality = Literal["image", "video", "voice"]


class GenerationRequest(BaseModel):
    modality: Modality
    prompt: str
    provider: str | None = Field(
        None,
        description="Optional explicit provider; otherwise the router picks per modality.",
    )
    force_fail: bool = Field(
        False,
        description="Test hook: when true, the primary provider raises so the fallback chain runs. Used in samples.",
    )


class GenerationResult(BaseModel):
    modality: Modality
    provider: str
    asset_url: str
    duration_ms: int
    cached: bool = False
    fallback_used: bool = False


class BaseProvider(ABC):
    name: str
    modality: Modality

    @abstractmethod
    def generate(self, prompt: str, force_fail: bool = False) -> GenerationResult: ...
