# 10. Reporting Endpoints — Explained

## What This Phase Built

Phase 7a added **read-only analytics endpoints** on top of the audit results produced by the pipeline. While Phase 6 built the "write path" (run audits, store results), this phase builds the "read path" (aggregate and present results).

**New files:**
- `src/services/reporting.py` — computation layer (4 public functions + 1 helper)
- `src/api/routes/reports.py` — thin route layer (4 GET endpoints + Pydantic schemas)
- `tests/unit/test_reporting.py` — 26 tests using in-memory SQLite

## Architecture: Service Layer Pattern

We split reporting into two layers:

```
HTTP Request → Route (reports.py) → Service (reporting.py) → Database
                  │                        │
                  │ Pydantic schemas        │ SQLAlchemy queries
                  │ Query params            │ JSON parsing
                  │ HTTP concerns           │ Business logic
```

**Why separate?**
- **Testability:** Service functions can be tested directly with a database session — no need for HTTP clients or FastAPI TestClient.
- **Extensibility:** When gold-standard validation comes (Phase 7b), we add `get_validation_metrics()` to the service and a new endpoint to the routes. The pattern is established.
- **Readability:** Routes are 5 lines each (parse params → call service → return). All logic lives in the service.

## The Four Report Functions

### 1. `get_dashboard_stats()` — High-Level Summary

**What it returns:** Total audited/failed counts, mean/median/min/max adherence score, failure rate.

**How it works:** Pure SQL aggregation — `COUNT`, `AVG`, `MIN`, `MAX` via SQLAlchemy `func`. The one exception is **median**, which has no SQL standard function, so we load all scores into Python and compute it there (perfectly fine for ~4,327 values).

**Key design choice:** Uses only SQL columns (`overall_score`, `status`), never parses `details_json`. This keeps the function fast and avoids coupling to the JSON schema.

### 2. `get_condition_breakdown()` — Per-Diagnosis Adherence

**What it returns:** For each diagnosis term: total cases, adherent count, non-adherent count, adherence rate.

**How it works:** Loads all completed AuditResults, parses each `details_json` to extract the `scores` array, groups by `diagnosis` term, and counts adherent (+1) vs non-adherent (-1) per group.

**Parameters:**
- `min_count` — filters out diagnoses with fewer than N total cases (default 1)
- `sort_by` — `"count"` (descending, default) or `"adherence_rate"` (ascending, worst-first)

**Why Python-side aggregation?** The `details_json` column is `TEXT`, not PostgreSQL `JSONB`. We can't use SQL JSON functions. But with ~4,327 results, loading them all into Python and parsing with `json.loads()` takes milliseconds. No need for the complexity of database-side JSON extraction.

### 3. `get_non_adherent_cases()` — Clinical Review List

**What it returns:** Paginated list of every diagnosis that scored -1 (non-adherent), with patient ID, explanation, and guidelines not followed.

**How it works:** Similar to condition breakdown — parses `details_json`, but filters for `score == -1` entries. Also eager-loads the `Patient` relationship via `selectinload` to get the patient's UUID (`pat_id`).

**Pagination:** Returns `page`, `page_size`, `total`, `total_pages`, and the `cases` slice. Handles edge cases like requesting a page beyond the data (returns empty `cases` with correct `total`).

**Purpose:** This is the endpoint clinicians would use to review cases flagged as non-adherent. Each case includes the explanation (from the Scorer Agent's LLM output) and the specific guidelines that weren't followed.

### 4. `get_score_distribution()` — Score Histogram

**What it returns:** A histogram dividing the 0.0–1.0 range into equal bins, with the count of patients in each bin.

**How it works:** Loads all `overall_score` values (SQL only), then bins them in Python. The last bin includes the right edge (so a score of exactly 1.0 falls in [0.9, 1.0] not nowhere).

**Parameters:**
- `bins` — number of histogram bins (default 10, range 2–100)

**Edge cases:** An empty dataset returns `{"bins": [], "total": 0}`.

## The Shared Helper: `_load_completed_results()`

All functions that parse `details_json` share the same base query: "give me all AuditResults with status='completed', optionally filtered by job_id." This helper avoids duplication:

```python
async def _load_completed_results(session, job_id=None, include_details=False):
    query = select(AuditResult).where(AuditResult.status == "completed")
    if job_id is not None:
        query = query.where(AuditResult.job_id == job_id)
    if include_details:
        query = query.options(selectinload(AuditResult.patient))
    result = await session.execute(query)
    return list(result.scalars().all())
```

The `include_details` flag controls whether to eager-load the Patient relationship (an extra SQL query). Only `get_non_adherent_cases()` needs this (to show `pat_id`); `get_condition_breakdown()` doesn't.

## Job Scoping

All 4 endpoints accept an optional `?job_id=N` query parameter. This scopes the report to a single batch job, letting you compare results across different runs. If omitted, the report covers all completed results in the database.

## Testing Strategy: In-Memory SQLite

**The problem:** Reporting functions run real SQL queries with `func.count()`, `func.avg()`, `select().where()`, etc. Mocking `session.execute()` would be extremely fragile — `get_dashboard_stats()` alone makes 4 separate execute calls, each returning different types (`scalar`, `one`, `all`).

**The solution:** We use `aiosqlite` to create an in-memory SQLite database for each test:

```python
@pytest_asyncio.fixture
async def async_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # ... yield session
```

This gives us:
- **Real SQL execution** — tests validate actual query logic, not mock behaviour
- **No external dependencies** — no PostgreSQL needed, runs anywhere
- **Speed** — 26 tests complete in under 1 second
- **Schema sync** — uses the same SQLAlchemy models, so tables match production

**Helper functions** (`_make_patient`, `_add_completed_result`, `_add_failed_result`, `_make_job`) make test setup concise — each test creates its own isolated data and verifies the output.

## Pydantic Response Schemas

Each endpoint has typed response models:

```python
class DashboardResponse(BaseModel):
    total_audited: int
    total_failed: int
    failure_rate: float
    score_stats: ScoreStatsSchema  # mean, median, min, max

class ConditionBreakdownItem(BaseModel):
    diagnosis: str
    total_cases: int
    adherent: int
    non_adherent: int
    errors: int
    adherence_rate: float
```

FastAPI uses these for:
- **Validation** — ensures service functions return the right structure
- **Serialization** — converts Python dicts to proper JSON
- **Documentation** — Swagger UI shows the exact response shape

## What's Next

Phase 7b will add **gold-standard validation**: importing the 120 manually-audited cases, running our pipeline against them, and computing agreement metrics (Cohen's kappa, accuracy, etc.). This will add a `get_validation_metrics()` function to the reporting service and a new `/api/v1/reports/validation` endpoint — the exact pattern established here.
