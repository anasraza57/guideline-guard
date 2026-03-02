"""
Reporting service — aggregate analytics over audit results.

Provides functions for dashboard statistics, condition breakdowns,
non-adherent case listing, and score distributions. Designed to
be extended with gold-standard validation metrics later.
"""

import json
import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.audit import AuditResult

logger = logging.getLogger(__name__)


# ── Private helpers ───────────────────────────────────────────────────


async def _load_completed_results(
    session: AsyncSession,
    job_id: int | None = None,
    include_details: bool = False,
) -> list[AuditResult]:
    """
    Shared query helper for loading completed AuditResults.

    Args:
        session: Async database session.
        job_id: Optional job ID to scope results to a batch run.
        include_details: If True, eager-loads the Patient relationship
            (needed when the response requires pat_id).

    Returns:
        List of AuditResult objects with status='completed'.
    """
    query = select(AuditResult).where(AuditResult.status == "completed")

    if job_id is not None:
        query = query.where(AuditResult.job_id == job_id)

    if include_details:
        query = query.options(selectinload(AuditResult.patient))

    result = await session.execute(query)
    return list(result.scalars().all())


# ── Public functions ──────────────────────────────────────────────────


async def get_dashboard_stats(
    session: AsyncSession,
    job_id: int | None = None,
) -> dict:
    """
    High-level summary statistics.

    Returns total audited/failed counts, mean/median/min/max adherence
    score, and failure rate. Uses SQL columns only (no JSON parsing).
    """
    base_filters: list = []
    if job_id is not None:
        base_filters.append(AuditResult.job_id == job_id)

    # Count completed
    q_completed = select(func.count(AuditResult.id)).where(
        AuditResult.status == "completed", *base_filters,
    )
    total_completed = (await session.execute(q_completed)).scalar() or 0

    # Count failed
    q_failed = select(func.count(AuditResult.id)).where(
        AuditResult.status == "failed", *base_filters,
    )
    total_failed = (await session.execute(q_failed)).scalar() or 0

    # Score aggregates (completed results with non-null scores)
    score_filters = [
        AuditResult.status == "completed",
        AuditResult.overall_score.isnot(None),
        *base_filters,
    ]

    q_stats = select(
        func.avg(AuditResult.overall_score),
        func.min(AuditResult.overall_score),
        func.max(AuditResult.overall_score),
    ).where(*score_filters)

    stats_row = (await session.execute(q_stats)).one()
    avg_score, min_score, max_score = stats_row

    # Median (no SQL standard — compute in Python)
    q_scores = (
        select(AuditResult.overall_score)
        .where(*score_filters)
        .order_by(AuditResult.overall_score)
    )
    scores = [row[0] for row in (await session.execute(q_scores)).all()]

    median_score = None
    if scores:
        n = len(scores)
        if n % 2 == 0:
            median_score = (scores[n // 2 - 1] + scores[n // 2]) / 2
        else:
            median_score = scores[n // 2]

    total = total_completed + total_failed
    failure_rate = total_failed / total if total > 0 else 0.0

    return {
        "total_audited": total_completed,
        "total_failed": total_failed,
        "failure_rate": round(failure_rate, 4),
        "score_stats": {
            "mean": round(avg_score, 4) if avg_score is not None else None,
            "median": round(median_score, 4) if median_score is not None else None,
            "min": round(min_score, 4) if min_score is not None else None,
            "max": round(max_score, 4) if max_score is not None else None,
        },
    }


async def get_condition_breakdown(
    session: AsyncSession,
    job_id: int | None = None,
    min_count: int = 1,
    sort_by: str = "count",
) -> list[dict]:
    """
    Adherence rates grouped by diagnosis term.

    Parses details_json for each completed result, extracts the
    per-diagnosis scores, and groups them by diagnosis term.

    Args:
        session: Async database session.
        job_id: Optional job ID to scope results.
        min_count: Minimum number of cases to include a diagnosis.
        sort_by: "count" (descending) or "adherence_rate" (ascending).

    Returns:
        List of dicts with diagnosis, counts, and adherence_rate.
    """
    results = await _load_completed_results(session, job_id, include_details=False)

    conditions: dict[str, dict[str, int]] = {}
    for r in results:
        if not r.details_json:
            continue
        try:
            details = json.loads(r.details_json)
        except json.JSONDecodeError:
            continue

        for ds in details.get("scores", []):
            term = ds.get("diagnosis", "Unknown")
            if term not in conditions:
                conditions[term] = {"adherent": 0, "non_adherent": 0, "errors": 0}

            score = ds.get("score")
            if score == 1:
                conditions[term]["adherent"] += 1
            elif score == -1:
                conditions[term]["non_adherent"] += 1

            if ds.get("error"):
                conditions[term]["errors"] += 1

    breakdown = []
    for term, counts in conditions.items():
        total = counts["adherent"] + counts["non_adherent"]
        if total < min_count:
            continue
        adherence_rate = counts["adherent"] / total if total > 0 else 0.0
        breakdown.append({
            "diagnosis": term,
            "total_cases": total,
            "adherent": counts["adherent"],
            "non_adherent": counts["non_adherent"],
            "errors": counts["errors"],
            "adherence_rate": round(adherence_rate, 4),
        })

    if sort_by == "adherence_rate":
        breakdown.sort(key=lambda x: x["adherence_rate"])
    else:
        breakdown.sort(key=lambda x: x["total_cases"], reverse=True)

    return breakdown


async def get_non_adherent_cases(
    session: AsyncSession,
    job_id: int | None = None,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    """
    Paginated list of non-adherent diagnoses for clinical review.

    Returns every diagnosis that scored -1, with the patient ID,
    explanation, and list of guidelines not followed.
    """
    results = await _load_completed_results(session, job_id, include_details=True)

    non_adherent = []
    for r in results:
        if not r.details_json:
            continue
        try:
            details = json.loads(r.details_json)
        except json.JSONDecodeError:
            continue

        pat_id = r.patient.pat_id if r.patient else details.get("pat_id", "Unknown")

        for ds in details.get("scores", []):
            if ds.get("score") == -1:
                non_adherent.append({
                    "pat_id": pat_id,
                    "diagnosis": ds.get("diagnosis", "Unknown"),
                    "index_date": ds.get("index_date"),
                    "explanation": ds.get("explanation", ""),
                    "guidelines_not_followed": ds.get("guidelines_not_followed", []),
                })

    total = len(non_adherent)
    start = (page - 1) * page_size
    end = start + page_size
    page_data = non_adherent[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total > 0 else 0,
        "cases": page_data,
    }


async def get_score_distribution(
    session: AsyncSession,
    job_id: int | None = None,
    bins: int = 10,
) -> dict:
    """
    Histogram of patient-level overall_score values.

    Divides the 0.0–1.0 range into equal bins and counts how many
    patients fall into each. Uses SQL columns only (no JSON parsing).
    """
    score_filters = [
        AuditResult.status == "completed",
        AuditResult.overall_score.isnot(None),
    ]
    if job_id is not None:
        score_filters.append(AuditResult.job_id == job_id)

    q_scores = select(AuditResult.overall_score).where(*score_filters)
    scores = [row[0] for row in (await session.execute(q_scores)).all()]

    if not scores:
        return {"bins": [], "total": 0}

    bin_width = 1.0 / bins
    histogram = []
    for i in range(bins):
        bin_start = round(i * bin_width, 4)
        bin_end = round((i + 1) * bin_width, 4)
        if i == bins - 1:
            # Last bin includes the right edge (1.0)
            count = sum(1 for s in scores if bin_start <= s <= bin_end)
        else:
            count = sum(1 for s in scores if bin_start <= s < bin_end)
        histogram.append({
            "bin_start": bin_start,
            "bin_end": bin_end,
            "count": count,
        })

    return {"bins": histogram, "total": len(scores)}
