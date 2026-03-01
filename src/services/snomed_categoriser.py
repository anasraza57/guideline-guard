"""
SNOMED concept categoriser.

Classifies clinical concepts into categories (diagnosis, treatment,
procedure, referral, investigation, other) using a two-tier approach:
1. Rule-based keyword matching for clear-cut cases
2. LLM fallback for ambiguous terms

Since the dataset has ~1,261 unique concepts, we classify each concept
once and cache the result — no need to re-classify per entry.
"""

import logging
import re

logger = logging.getLogger(__name__)

# The categories we assign to clinical entries
CATEGORIES = {
    "diagnosis",
    "treatment",
    "procedure",
    "referral",
    "investigation",
    "administrative",
    "other",
}

# ── Rule-based patterns ──────────────────────────────────────────────
# Each pattern is a tuple of (compiled_regex, category).
# Patterns are checked in order — first match wins.
# Case-insensitive matching is applied.

_PATTERNS: list[tuple[re.Pattern, str]] = []


def _p(pattern: str, category: str) -> None:
    """Helper to register a pattern."""
    _PATTERNS.append((re.compile(pattern, re.IGNORECASE), category))


# -- Referrals (check first — "referral for X-ray" is a referral, not investigation)
_p(r"\breferral\b", "referral")
_p(r"\brefer(?:red)?\s+to\b", "referral")
_p(r"\bself[- ]referral\b", "referral")

# -- Investigations
_p(r"\bx[- ]?ray\b", "investigation")
_p(r"\bmri\b", "investigation")
_p(r"\bct\s+scan\b", "investigation")
_p(r"\bultrasound\b", "investigation")
_p(r"\bscan\b", "investigation")
_p(r"\bblood\s+(sample|test|glucose)\b", "investigation")
_p(r"\bnerve\s+conduction\b", "investigation")
_p(r"\btest(ing)?\b", "investigation")
_p(r"\bscreening\b", "investigation")
_p(r"\bexamination\b", "investigation")
_p(r"\bbiopsy\b", "investigation")
_p(r"\bspirometry\b", "investigation")
_p(r"\becg\b", "investigation")
_p(r"\burinalysis\b", "investigation")
_p(r"\bdexa\b", "investigation")
_p(r"\bblood\s+pressure\s+(recorded|monitor)", "investigation")

# -- Administrative (check before treatments — "review of medication" is admin, not treatment)
_p(r"\bmed3\b", "administrative")
_p(r"\bemed3\b", "administrative")
_p(r"\bcertificate\b", "administrative")
_p(r"\bconsent\b", "administrative")
_p(r"\bconsultation\b", "administrative")
_p(r"\btelephone\b", "administrative")
_p(r"\breview\b", "administrative")
_p(r"\bassessment\b", "administrative")
_p(r"\bmonitoring\b", "administrative")
_p(r"\bsmoking\s+cessation\b", "administrative")
_p(r"\bweight\s+monitor", "administrative")
_p(r"\beducation\b", "administrative")
_p(r"\badvice\b", "administrative")
_p(r"\bclinical\s+trial\b", "administrative")

# -- Treatments (medications, injections, therapies)
_p(r"\binjection\b", "treatment")
_p(r"\bprescription\b", "treatment")
_p(r"\bprescribed\b", "treatment")
_p(r"\bibuprofen\b", "treatment")
_p(r"\bnaproxen\b", "treatment")
_p(r"\bparacetamol\b", "treatment")
_p(r"\bcodeine\b", "treatment")
_p(r"\bamitriptyline\b", "treatment")
_p(r"\bgabapentin\b", "treatment")
_p(r"\bpregabalin\b", "treatment")
_p(r"\bsteroid\b", "treatment")
_p(r"\bcorticosteroid\b", "treatment")
_p(r"\bsplint\b", "treatment")
_p(r"\borthosis\b", "treatment")
_p(r"\bphysiotherapy\b", "treatment")
_p(r"\bexercise\s+therap", "treatment")
_p(r"\bacupuncture\b", "treatment")
_p(r"\banalges", "treatment")
_p(r"\banti[- ]?inflammator", "treatment")
_p(r"\bdrug\s+therapy\b", "treatment")
_p(r"\bmedication\b", "treatment")

# -- Procedures (surgical, clinical actions)
_p(r"\bsurgery\b", "procedure")
_p(r"\bsurgical\b", "procedure")
_p(r"\boperation\b", "procedure")
_p(r"\breplacement\b", "procedure")
_p(r"\barthroscop", "procedure")
_p(r"\bexcision\b", "procedure")
_p(r"\bremoval\b", "procedure")
_p(r"\binsertion\b", "procedure")
_p(r"\baspiration\b", "procedure")
_p(r"\bmanipulation\b", "procedure")
_p(r"\bimmobilis", "procedure")
_p(r"\bsutur", "procedure")
_p(r"\bdebridement\b", "procedure")

# -- Diagnoses (conditions, symptoms, disorders)
_p(r"\bpain\b", "diagnosis")
_p(r"\bfracture\b", "diagnosis")
_p(r"\bosteoarthritis\b", "diagnosis")
_p(r"\barthritis\b", "diagnosis")
_p(r"\btendin", "diagnosis")
_p(r"\bbursitis\b", "diagnosis")
_p(r"\bfasciitis\b", "diagnosis")
_p(r"\bsyndrome\b", "diagnosis")
_p(r"\bscoliosis\b", "diagnosis")
_p(r"\bstenosis\b", "diagnosis")
_p(r"\bhernia", "diagnosis")
_p(r"\bsciatica\b", "diagnosis")
_p(r"\blumbago\b", "diagnosis")
_p(r"\bsprain\b", "diagnosis")
_p(r"\bstrain\b", "diagnosis")
_p(r"\bdislocation\b", "diagnosis")
_p(r"\bcontusion\b", "diagnosis")
_p(r"\bswelling\b", "diagnosis")
_p(r"\bstiffness\b", "diagnosis")
_p(r"\bweakness\b", "diagnosis")
_p(r"\bneuropathy\b", "diagnosis")
_p(r"\bdegenerat", "diagnosis")
_p(r"\binflammation\b", "diagnosis")
_p(r"\bdeformity\b", "diagnosis")
_p(r"\binstability\b", "diagnosis")
_p(r"\bimpingement\b", "diagnosis")
_p(r"\brupture\b", "diagnosis")
_p(r"\btear\b", "diagnosis")
_p(r"\bgout\b", "diagnosis")
_p(r"\bfibromyalgia\b", "diagnosis")
_p(r"\bosteoporosis\b", "diagnosis")
_p(r"\bbackache\b", "diagnosis")
_p(r"\bneck\s+ache\b", "diagnosis")
_p(r"\bepicondylitis\b", "diagnosis")
_p(r"\b\w+itis\b", "diagnosis")  # General -itis suffix (inflammation)
_p(r"\babrasion\b", "diagnosis")
_p(r"\babscess\b", "diagnosis")
_p(r"\bhallux\b", "diagnosis")
_p(r"\bkyphosis\b", "diagnosis")
_p(r"\blordosis\b", "diagnosis")
_p(r"\btrigger\s+finger\b", "diagnosis")
_p(r"\bdupuytren\b", "diagnosis")
_p(r"\bganglion\b", "diagnosis")
_p(r"\blaceration\b", "diagnosis")
_p(r"\blesion\b", "diagnosis")
_p(r"\blump\b", "diagnosis")
_p(r"\bmass\b", "diagnosis")
_p(r"\bcyst\b", "diagnosis")
_p(r"\bcalcification\b", "diagnosis")
_p(r"\bcontracture\b", "diagnosis")
_p(r"\bdisorder\b", "diagnosis")
_p(r"\bdisease\b", "diagnosis")
_p(r"\binjury\b", "diagnosis")
_p(r"\bwound\b", "diagnosis")
_p(r"\bburn\b", "diagnosis")
_p(r"\bbruise\b", "diagnosis")
_p(r"\bnumbness\b", "diagnosis")
_p(r"\btingling\b", "diagnosis")
_p(r"\bcrepitus\b", "diagnosis")
_p(r"\blimited\s+range\b", "diagnosis")
_p(r"\brestriction\b", "diagnosis")
_p(r"\bprolapse\b", "diagnosis")
_p(r"\boedema\b", "diagnosis")
_p(r"\bedema\b", "diagnosis")
_p(r"\beffusion\b", "diagnosis")
_p(r"\bankylos", "diagnosis")
_p(r"\b\w+pathy\b", "diagnosis")  # General -pathy suffix
_p(r"\b\w+osis\b", "diagnosis")  # General -osis suffix (but after specific terms)

# -- Diagnoses (additional)
_p(r"\bbunion\b", "diagnosis")
_p(r"\bbandy\b", "diagnosis")
_p(r"\bdysplastic\b", "diagnosis")
_p(r"\bbow[- ]?leg", "diagnosis")
_p(r"\bflat\s+foot\b", "diagnosis")
_p(r"\bclaw\s+toe\b", "diagnosis")
_p(r"\bhammer\s+toe\b", "diagnosis")
_p(r"\bleg\s+length\b", "diagnosis")

# -- Treatments (additional)
_p(r"\badministration of\b", "treatment")
_p(r"\bvaccin", "treatment")
_p(r"\bimmuni[sz]ation\b", "treatment")
_p(r"\bcontraception\b", "treatment")
_p(r"\bdressing\b", "treatment")
_p(r"\bstrapping\b", "treatment")
_p(r"\bbandage\b", "treatment")
_p(r"\btherapy\b", "treatment")
_p(r"\banti[- ]?coagulant\b", "treatment")
_p(r"\bcast\b", "treatment")
_p(r"\bsling\b", "treatment")
_p(r"\bclosure\b", "treatment")

# -- Procedures (additional)
_p(r"\barthrodesis\b", "procedure")
_p(r"\bbursectomy\b", "procedure")
_p(r"\bosteotomy\b", "procedure")
_p(r"\b\w+ectomy\b", "procedure")  # General -ectomy suffix (surgical removal)
_p(r"\b\w+otomy\b", "procedure")  # General -otomy suffix (surgical incision)
_p(r"\b\w+plasty\b", "procedure")  # General -plasty suffix (surgical repair)

# -- Administrative (additional)
_p(r"\bcare\s+plan\b", "administrative")
_p(r"\bfollow[- ]?up\b", "administrative")
_p(r"\bmanagement\b", "administrative")
_p(r"\baction\s+plan\b", "administrative")
_p(r"\bunfit\s+for\s+work\b", "administrative")
_p(r"\bfit\s+note\b", "administrative")
_p(r"\bplanning\b", "administrative")
_p(r"\bagreeing\b", "administrative")

# -- Investigations (additional)
_p(r"\bblood\s+pressure\b", "investigation")
_p(r"\bmeasurement\b", "investigation")


def categorise_by_rules(concept_display: str) -> str | None:
    """
    Try to categorise a concept using keyword rules.

    Returns the category string if matched, or None if no rule applies.
    """
    for pattern, category in _PATTERNS:
        if pattern.search(concept_display):
            return category
    return None


LLM_CATEGORISE_PROMPT = """You are a clinical coding expert. Categorise the following SNOMED CT clinical concept into exactly one of these categories:

- diagnosis: A condition, symptom, disease, or injury (e.g., "Low back pain", "Fracture of femur")
- treatment: A medication, therapy, or therapeutic intervention (e.g., "Ibuprofen", "Physiotherapy")
- procedure: A surgical or clinical procedure (e.g., "Knee replacement", "Arthroscopy")
- referral: A referral to another service or specialist (e.g., "Referral to orthopaedics")
- investigation: A test, scan, or diagnostic investigation (e.g., "X-ray of knee", "Blood test")
- administrative: Administrative actions, consultations, certificates, reviews (e.g., "Med3 certificate", "Telephone consultation")
- other: Anything that does not fit the above categories

Respond with ONLY the category name, nothing else.

Concept: "{concept_display}"
Category:"""


async def categorise_by_llm(
    concept_displays: list[str],
    ai_provider,
) -> dict[str, str]:
    """
    Categorise concepts using the LLM. Processes one at a time
    to keep responses clean and parseable.

    Returns a dict mapping concept_display → category.
    """
    results = {}
    for display in concept_displays:
        prompt = LLM_CATEGORISE_PROMPT.format(concept_display=display)
        try:
            response = await ai_provider.chat_simple(prompt)
            category = response.strip().lower()
            if category in CATEGORIES:
                results[display] = category
            else:
                logger.warning(
                    "LLM returned invalid category %r for %r, defaulting to 'other'",
                    category, display,
                )
                results[display] = "other"
        except Exception as e:
            logger.error("LLM categorisation failed for %r: %s", display, e)
            results[display] = "other"
    return results


async def categorise_concepts(
    concept_displays: list[str],
    ai_provider=None,
) -> dict[str, str]:
    """
    Categorise a list of unique concept display names.

    Uses rules first, falls back to LLM for unmatched concepts.
    Returns a dict mapping concept_display → category.
    """
    mapping: dict[str, str] = {}
    unmatched: list[str] = []

    for display in concept_displays:
        category = categorise_by_rules(display)
        if category:
            mapping[display] = category
        else:
            unmatched.append(display)

    logger.info(
        "Rule-based: %d/%d categorised, %d need LLM",
        len(mapping), len(concept_displays), len(unmatched),
    )

    if unmatched and ai_provider:
        llm_results = await categorise_by_llm(unmatched, ai_provider)
        mapping.update(llm_results)
    elif unmatched:
        logger.warning(
            "No AI provider available — %d concepts defaulting to 'other'",
            len(unmatched),
        )
        for display in unmatched:
            mapping[display] = "other"

    return mapping
