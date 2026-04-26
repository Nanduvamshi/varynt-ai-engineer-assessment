import os

os.environ["MOCK_OPENAI"] = "1"
from app.q1_classifier.classifier import LeadInput, classify_lead


def test_hot_lead_classified_hot():
    lead = LeadInput(
        name="Test",
        message="Ready to buy this week. Looking for demo.",
        budget="$2k/mo",
        timeline="this week",
    )
    out = classify_lead(lead)
    assert out.classification.category == "hot"
    assert 0.0 <= out.classification.confidence <= 1.0
    assert out.suggested_reply.strip()


def test_cold_lead_when_only_email():
    lead = LeadInput(email="someone@example.com")
    out = classify_lead(lead)
    assert out.classification.category == "cold"
    assert (
        "message" in out.classification.missing_signals
        or out.classification.missing_signals
    )


def test_warm_lead_with_partial_signals():
    lead = LeadInput(
        name="Bob",
        message="Looking into funnel builders, evaluating",
        timeline="next quarter",
    )
    out = classify_lead(lead)
    assert out.classification.category == "warm"


def test_response_references_name_when_provided():
    lead = LeadInput(
        name="Priya", message="Ready to buy", budget="$5k/mo", timeline="now"
    )
    out = classify_lead(lead)
    assert "Priya" in out.suggested_reply
