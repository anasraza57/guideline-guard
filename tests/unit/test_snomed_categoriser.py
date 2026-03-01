"""
Tests for the SNOMED concept categoriser.

Verifies that rule-based classification correctly categorises
common clinical concepts from our dataset.
"""

import pytest

from src.services.snomed_categoriser import categorise_by_rules, categorise_concepts


class TestRuleBasedCategorisation:
    """Test the rule-based pattern matching."""

    @pytest.mark.parametrize("display,expected", [
        # Diagnoses
        ("Low back pain", "diagnosis"),
        ("Pain of shoulder region", "diagnosis"),
        ("Fracture of phalanx of finger", "diagnosis"),
        ("Osteoarthritis of knee", "diagnosis"),
        ("Carpal tunnel syndrome", "diagnosis"),
        ("Plantar fasciitis", "diagnosis"),
        ("Lumbago with sciatica", "diagnosis"),
        ("Backache", "diagnosis"),
        ("Lateral epicondylitis", "diagnosis"),
        ("Pain of hip region", "diagnosis"),
        ("Rotator cuff tendinitis", "diagnosis"),
        ("Gout", "diagnosis"),
        ("Fibromyalgia", "diagnosis"),

        # Referrals
        ("Referral to orthopaedic service", "referral"),
        ("Referral to physiotherapist", "referral"),
        ("Referral for further care", "referral"),
        ("Physiotherapy self-referral", "referral"),
        ("Referral to podiatry service", "referral"),
        ("Referral for nerve conduction studies", "referral"),

        # Treatments
        ("Intramuscular injection of vitamin B12", "treatment"),
        ("Injection of steroid into knee joint", "treatment"),
        ("Prescription of drug", "treatment"),
        ("Injection of steroid into shoulder joint", "treatment"),

        # Investigations
        ("Blood sample taken", "investigation"),
        ("Blood pressure recorded by patient at home", "investigation"),
        ("Point of care testing", "investigation"),
        ("Blood test requested", "investigation"),

        # Administrative
        ("eMED3 (2010) new statement issued, not fit for work", "administrative"),
        ("Informed consent for procedure", "administrative"),
        ("Weight monitoring", "administrative"),
        ("Review of medication", "administrative"),
        ("Telephone consultation", "administrative"),
        ("Smoking cessation education", "administrative"),
        ("Med3 certificate issued to patient", "administrative"),

        # Procedures
        ("Total replacement of knee joint", "procedure"),
        ("Arthroscopy of knee", "procedure"),
    ])
    def test_known_concepts(self, display, expected):
        result = categorise_by_rules(display)
        assert result == expected, f"{display!r}: expected {expected!r}, got {result!r}"

    def test_unrecognised_returns_none(self):
        result = categorise_by_rules("Something very unusual and unrecognisable")
        assert result is None


class TestCategoriseConcepts:
    """Test the full categorisation pipeline (without LLM)."""

    @pytest.mark.asyncio
    async def test_all_matched_by_rules(self):
        concepts = ["Low back pain", "Referral to physiotherapist", "Blood test requested"]
        result = await categorise_concepts(concepts, ai_provider=None)

        assert result["Low back pain"] == "diagnosis"
        assert result["Referral to physiotherapist"] == "referral"
        assert result["Blood test requested"] == "investigation"

    @pytest.mark.asyncio
    async def test_unmatched_defaults_to_other_without_llm(self):
        concepts = ["Something very unusual"]
        result = await categorise_concepts(concepts, ai_provider=None)
        assert result["Something very unusual"] == "other"
