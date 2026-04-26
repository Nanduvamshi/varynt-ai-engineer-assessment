"""Q1 — Lead classifier + personalized response generator.

Pipeline:
  1. Validate the lead JSON (Pydantic). Missing fields are *allowed* —
     the classifier itself decides how to weight them.
  2. Call OpenAI in JSON mode with CLASSIFIER_SYSTEM_PROMPT to get
     {category, confidence, reasoning, missing_signals}.
  3. Call OpenAI again with RESPONSE_SYSTEM_PROMPT, conditioned on the
     classification, to produce the personalized reply.
  4. Return a single combined response.

Resilience: each OpenAI call is wrapped with the Q5 retry decorator. If
MOCK_OPENAI=1, deterministic canned outputs are returned so the reviewer
can run the whole pipeline without a key.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError
from pydantic import BaseModel, Field

from app import config
from app.q5_resilience.retry import (
    TransientProviderError,
    PermanentProviderError,
    with_retry,
    timed,
)
from app.q1_classifier.prompts import CLASSIFIER_SYSTEM_PROMPT, RESPONSE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

Category = Literal["hot", "warm", "cold"]


class LeadInput(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    source: Optional[str] = Field(None, description="e.g., webinar, ad, organic")
    message: Optional[str] = None
    budget: Optional[str] = None
    timeline: Optional[str] = None
    team_size: Optional[int] = None


class Classification(BaseModel):
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    missing_signals: list[str] = Field(default_factory=list)


class LeadResponse(BaseModel):
    classification: Classification
    suggested_reply: str
    source: Literal["openai", "mock"] = "openai"


_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _wrap_openai_error(err: Exception) -> Exception:
    """Map OpenAI errors to the resilience layer's transient/permanent split."""
    if isinstance(err, (RateLimitError, APIConnectionError, APITimeoutError)):
        return TransientProviderError(str(err))
    if isinstance(err, APIError):
        status = getattr(err, "status_code", None)
        if status and 500 <= status < 600:
            return TransientProviderError(str(err))
        return PermanentProviderError(str(err))
    return err


@with_retry(max_attempts=3, initial_wait=0.5, max_wait=4.0, overall_timeout=20.0)
@timed
def _classify_call(lead_json: str) -> dict:
    if config.MOCK_OPENAI:
        return _mock_classify(lead_json)
    try:
        resp = _get_client().chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Lead profile (JSON):\n{lead_json}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)
    except (APIError, APIConnectionError, RateLimitError, APITimeoutError) as e:
        raise _wrap_openai_error(e) from e


@with_retry(max_attempts=3, initial_wait=0.5, max_wait=4.0, overall_timeout=20.0)
@timed
def _respond_call(lead_json: str, classification_json: str) -> str:
    if config.MOCK_OPENAI:
        return _mock_respond(lead_json, classification_json)
    try:
        resp = _get_client().chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Lead profile (JSON):\n{lead_json}\n\n"
                        f"Classification (JSON):\n{classification_json}"
                    ),
                },
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except (APIError, APIConnectionError, RateLimitError, APITimeoutError) as e:
        raise _wrap_openai_error(e) from e


def classify_lead(lead: LeadInput) -> LeadResponse:
    """End-to-end pipeline: classify -> generate reply."""
    lead_json = lead.model_dump_json(exclude_none=False)
    raw = _classify_call(lead_json)
    classification = Classification(**raw)
    reply = _respond_call(lead_json, classification.model_dump_json())
    return LeadResponse(
        classification=classification,
        suggested_reply=reply,
        source="mock" if config.MOCK_OPENAI else "openai",
    )


# --- Mock backend for keyless testing -----------------------------------------

def _mock_classify(lead_json: str) -> dict:
    """Deterministic stand-in. Heuristic only — for keyless smoke tests."""
    lead = json.loads(lead_json)
    msg = (lead.get("message") or "").lower()
    has_budget = bool(lead.get("budget"))
    has_timeline = bool(lead.get("timeline"))
    has_intent = any(w in msg for w in ("buy", "purchase", "demo", "pricing", "ready"))
    is_cold_signal = any(w in msg for w in ("just curious", "browsing", "later", "exploring"))

    missing: list[str] = []
    if not lead.get("budget"):
        missing.append("budget")
    if not lead.get("timeline"):
        missing.append("timeline")
    if not lead.get("message"):
        missing.append("message")

    if is_cold_signal or len(missing) >= 3:
        return {
            "category": "cold",
            "confidence": 0.55,
            "reasoning": "Insufficient signal: missing budget/timeline and message indicates exploration only.",
            "missing_signals": missing,
        }
    if has_intent and has_budget and has_timeline:
        return {
            "category": "hot",
            "confidence": 0.85,
            "reasoning": "Explicit buying intent in message, plus stated budget and timeline.",
            "missing_signals": missing,
        }
    return {
        "category": "warm",
        "confidence": 0.65,
        "reasoning": "Some interest signal present but at least one of budget/timeline/intent is missing.",
        "missing_signals": missing,
    }


def _mock_respond(lead_json: str, classification_json: str) -> str:
    lead = json.loads(lead_json)
    cls = json.loads(classification_json)
    name = lead.get("name") or "there"
    cat = cls["category"]
    if cat == "hot":
        return (
            f"Hi {name}, thanks for reaching out — sounds like you're ready to move. "
            "Happy to walk you through KeaBuilder live. I have Tuesday 10am or Thursday 2pm "
            "open this week — which works? "
            "Sam from KeaBuilder"
        )
    if cat == "warm":
        return (
            f"Hi {name}, glad you're looking into KeaBuilder. Most teams in your spot "
            "find our funnel templates the fastest way to evaluate fit — happy to send a "
            "short walkthrough or answer specific questions. What's most useful? "
            "Sam from KeaBuilder"
        )
    return (
        f"Hi {name}, appreciate you stopping by. To point you to the most useful resource, "
        "could you share a bit about what you're trying to build — is it a lead-capture funnel, "
        "a course, or something else? "
        "Sam from KeaBuilder"
    )
