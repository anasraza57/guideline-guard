# GuidelineGuard

An agentic AI framework for evaluating musculoskeletal (MSK) consultation adherence to NICE clinical guidelines in primary care.

## How It Works

GuidelineGuard runs a 4-agent pipeline on each patient's clinical record:

```
Patient Record → Extractor → Query → Retriever → Scorer → Adherence Report
                     │           │         │          │
                     │           │         │          └─ LLM scores each diagnosis
                     │           │         └─ FAISS + PubMedBERT find relevant guidelines
                     │           └─ Generates search queries per diagnosis
                     └─ Groups SNOMED-coded entries by episode (diagnoses + treatments)
```

**Output:** Each patient gets an overall adherence score (0.0–1.0) and per-diagnosis scores (+1 adherent, -1 non-adherent) with explanations and guideline references.

## Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- Python 3.11+
- An OpenAI API key (for the LLM-based agents)
- ~2 GB RAM for PubMedBERT model loading

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/anasraza57/guideline-guard.git
cd guideline-guard

# 2. Copy environment template and set your API key
cp .env.example .env
# Edit .env — at minimum, set: OPENAI_API_KEY=sk-your-key-here

# 3. Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Start the database
docker compose up -d db

# 5. Run database migrations
DB_HOST=localhost alembic upgrade head

# 6. Import data into PostgreSQL
DB_HOST=localhost python3 scripts/import_data.py

# 7. Download the PubMedBERT embedding model (~440 MB, one-time download)
python3 -c "from transformers import AutoModel, AutoTokenizer; m='NeuML/pubmedbert-base-embeddings-matryoshka'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m); print('Model downloaded.')"

# 8. Build the FAISS guideline index (encodes all 1,656 guidelines with PubMedBERT)
#    Takes ~5-15 minutes on CPU — this is a one-time cost
python3 scripts/build_index.py

# 9. Start the application
make run
# (or: DB_HOST=localhost uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload)
```

> **Startup note:** The first launch takes 30–60 seconds — the server loads the PubMedBERT
> embedding model (~440 MB) and the FAISS guideline index into memory before accepting
> requests. Watch the terminal logs for "Embedding model loaded" and "Vector store ready".

> **`DB_HOST=localhost`:** When running locally, the database container is reached via
> `localhost`. Inside Docker, services communicate via the `db` hostname. The `.env` file
> defaults to `DB_HOST=db` (for Docker), so we override it for local commands.

### Verify It's Running

```bash
# Health check
curl http://localhost:8000/health

# Open interactive API docs
open http://localhost:8000/docs
```

## API Endpoints

Once running, all endpoints are documented interactively at **http://localhost:8000/docs** (Swagger UI).

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Application health check |

### Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/data/stats` | Database row counts (patients, entries, guidelines) |
| POST | `/api/v1/data/import/patients` | Import patient records from CSV |
| POST | `/api/v1/data/import/guidelines` | Import guidelines from CSV |

### Audit (Pipeline Execution)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/audit/patient/{pat_id}` | Audit a single patient |
| POST | `/api/v1/audit/batch` | Start a batch audit (all or subset of patients) |
| GET | `/api/v1/audit/jobs/{job_id}` | Check batch job status and progress |
| GET | `/api/v1/audit/jobs/{job_id}/results` | Get paginated results for a batch job |
| GET | `/api/v1/audit/results/{pat_id}` | Get all audit results for a specific patient |

### Reports (Analytics)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/reports/dashboard` | High-level summary (total audited, score stats, failure rate) |
| GET | `/api/v1/reports/conditions` | Per-condition adherence breakdown |
| GET | `/api/v1/reports/non-adherent` | Paginated list of non-adherent cases for clinical review |
| GET | `/api/v1/reports/score-distribution` | Score histogram (configurable bins) |

All report endpoints accept an optional `?job_id=N` query parameter to scope results to a specific batch run.

## Running Audits & Viewing Results

### 1. Audit a Single Patient

```bash
# Pick a patient ID from the database
curl -X POST http://localhost:8000/api/v1/audit/patient/SOME_PAT_ID
```

The response includes the patient's overall adherence score and per-diagnosis breakdown.

### 2. Run a Batch Audit

```bash
# Audit all 4,327 patients (takes a while — each patient calls the LLM)
curl -X POST http://localhost:8000/api/v1/audit/batch

# Or audit a subset (e.g. 50 patients)
curl -X POST "http://localhost:8000/api/v1/audit/batch?limit=50"

# Check progress
curl http://localhost:8000/api/v1/audit/jobs/1

# View results (paginated)
curl "http://localhost:8000/api/v1/audit/jobs/1/results?page=1&page_size=20"
```

### 3. View Analytics

After running audits, the reporting endpoints aggregate the results:

```bash
# Dashboard summary
curl http://localhost:8000/api/v1/reports/dashboard

# Which conditions have the worst adherence?
curl "http://localhost:8000/api/v1/reports/conditions?sort_by=adherence_rate"

# Cases flagged as non-adherent (for clinical review)
curl http://localhost:8000/api/v1/reports/non-adherent

# Score distribution histogram
curl http://localhost:8000/api/v1/reports/score-distribution
```

Or use the **Swagger UI** at http://localhost:8000/docs to explore all endpoints interactively with a visual interface.

## Development

```bash
# Run all tests (216 tests)
make test

# Run tests with coverage
make test-cov

# Start only the database
docker compose up -d db

# Stop all services
docker compose down

# View logs
docker compose logs -f
```

## Database

GuidelineGuard uses PostgreSQL (runs via Docker on port 5433).

| Table | Rows | Description |
|-------|------|-------------|
| `patients` | 4,327 | Anonymised MSK patients from the CrossCover trial |
| `clinical_entries` | 21,530 | SNOMED-coded clinical events (diagnoses, treatments, referrals, etc.) |
| `guidelines` | 1,656 | NICE clinical guideline documents |
| `audit_jobs` | — | Tracks batch audit processing runs |
| `audit_results` | — | Per-patient guideline adherence scores |

### Migrations

```bash
# Apply migrations
DB_HOST=localhost alembic upgrade head

# Create a new migration after model changes
DB_HOST=localhost alembic revision --autogenerate -m "description"
```

## Project Structure

```
guideline-guard/
├── src/                  # Application source code
│   ├── ai/               # AI/LLM provider abstraction (Strategy Pattern)
│   ├── agents/           # The 4 pipeline agents (Extractor, Query, Retriever, Scorer)
│   ├── api/routes/       # FastAPI route handlers (health, data, audit, reports)
│   ├── config/           # Configuration management (Pydantic Settings)
│   ├── models/           # SQLAlchemy database models
│   ├── services/         # Business logic (pipeline, data import, vector store, embedder, reporting)
│   └── utils/            # Shared utilities (logging)
├── tests/                # Test suite (216 tests)
├── data/                 # Data files — CSVs, FAISS index
├── migrations/           # Alembic database migrations
├── scripts/              # Utility scripts (data import)
├── docker-compose.yml    # PostgreSQL + app services
├── Dockerfile            # Multi-stage app build
├── Makefile              # Common commands (make help for full list)
└── PROJECT_BIBLE.md      # Complete project state, decisions, and roadmap
```

## Documentation

- **[PROJECT_BIBLE.md](PROJECT_BIBLE.md)** — Single source of truth: analysis, architecture, decisions, roadmap, progress
- **[docs/learning/](docs/learning/)** — Educational docs explaining every concept in plain English
