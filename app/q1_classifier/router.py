"""FastAPI route for Q1: POST /classify."""

from fastapi import APIRouter, HTTPException

from app.q1_classifier.classifier import LeadInput, LeadResponse, classify_lead
from app.q5_resilience.retry import PermanentProviderError, TransientProviderError

router = APIRouter()


@router.post("/classify", response_model=LeadResponse)
def classify_endpoint(lead: LeadInput) -> LeadResponse:
    try:
        return classify_lead(lead)
    except PermanentProviderError as e:
        raise HTTPException(status_code=502, detail=f"Upstream rejected request: {e}")
    except TransientProviderError as e:
        raise HTTPException(
            status_code=503,
            detail=f"AI is temporarily unavailable, please retry: {e}",
            headers={"Retry-After": "5"},
        )
