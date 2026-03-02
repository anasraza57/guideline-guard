"""
Report API routes.

Read-only analytics endpoints for reviewing audit results:
dashboard stats, condition breakdowns, non-adherent cases,
and score distributions.
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_session
from src.services.reporting import (
    get_condition_breakdown,
    get_dashboard_stats,
    get_non_adherent_cases,
    get_score_distribution,
)

router = APIRouter(prefix="/reports", tags=["reports"])


# ── Response schemas ──────────────────────────────────────────────────


class ScoreStatsSchema(BaseModel):
    mean: float | None
    median: float | None
    min: float | None
    max: float | None


class DashboardResponse(BaseModel):
    total_audited: int
    total_failed: int
    failure_rate: float
    score_stats: ScoreStatsSchema


class ConditionBreakdownItem(BaseModel):
    diagnosis: str
    total_cases: int
    adherent: int
    non_adherent: int
    errors: int
    adherence_rate: float


class NonAdherentCase(BaseModel):
    pat_id: str
    diagnosis: str
    index_date: str | None
    explanation: str
    guidelines_not_followed: list[str]


class NonAdherentResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    cases: list[NonAdherentCase]


class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int


class ScoreDistributionResponse(BaseModel):
    bins: list[HistogramBin]
    total: int


# ── Endpoints ─────────────────────────────────────────────────────────


@router.get("/dashboard", response_model=DashboardResponse)
async def dashboard(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    session: AsyncSession = Depends(get_session),
):
    """High-level summary: totals, scores, failure rate."""
    return await get_dashboard_stats(session, job_id)


@router.get("/conditions", response_model=list[ConditionBreakdownItem])
async def conditions(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    min_count: int = Query(1, ge=1, description="Minimum cases to include a diagnosis"),
    sort_by: str = Query("count", description="Sort by 'count' or 'adherence_rate'"),
    session: AsyncSession = Depends(get_session),
):
    """Per-condition adherence breakdown."""
    return await get_condition_breakdown(session, job_id, min_count, sort_by)


@router.get("/non-adherent", response_model=NonAdherentResponse)
async def non_adherent(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Results per page"),
    session: AsyncSession = Depends(get_session),
):
    """Non-adherent cases for clinical review."""
    return await get_non_adherent_cases(session, job_id, page, page_size)


@router.get("/score-distribution", response_model=ScoreDistributionResponse)
async def score_distribution(
    job_id: int | None = Query(None, description="Scope to a specific batch job"),
    bins: int = Query(10, ge=2, le=100, description="Number of histogram bins"),
    session: AsyncSession = Depends(get_session),
):
    """Histogram of patient-level adherence scores."""
    return await get_score_distribution(session, job_id, bins)
