# Query Agent Explained

## What Does the Query Agent Do?

The Query Agent is **Stage 2** of the 4-agent pipeline. Its job is to take the diagnoses identified by the Extractor and generate targeted search queries that will be used to find relevant NICE guidelines.

Think of it as a librarian who reads a doctor's notes and writes search cards: "I need to find the guidelines about low back pain management", "I need the referral criteria for knee osteoarthritis", etc.

## Why Do We Need It?

The Extractor gives us diagnoses like "Low back pain" or "Osteoarthritis of knee". But we can't just throw those raw terms into our guideline search — we need **optimised search queries** that will find the right guideline passages.

Here's why:

### The FAISS Search Problem

Our guideline database uses **semantic similarity search**. When we search for guidelines, we:
1. Take a query text
2. Convert it to a 768-dimensional vector using PubMedBERT
3. Find the guideline vectors closest to our query vector (cosine similarity)

The problem: the search quality depends entirely on what we search for. Consider:

| Search Query | What FAISS Finds |
|---|---|
| "Low back pain" | Matches anything mentioning back pain — too broad |
| "NICE guidelines for assessment and management of low back pain and sciatica" | Matches the actual NICE guideline title — precise |
| "low back pain referral criteria and imaging recommendations" | Matches specific guideline sections about when to refer or image |

The more our query text matches how NICE guidelines are actually written, the better the search results. That's what the Query Agent optimises for.

### Real Example

For a patient diagnosed with "Low back pain", the Query Agent generates:

```
Query 1: "NICE guidelines for assessment and management of low back pain and sciatica"
Query 2: "non-specific low back pain pharmacological and non-pharmacological treatment"
Query 3: "low back pain referral criteria and imaging recommendations"
```

Each query targets a different aspect of care:
- **Query 1** finds the main guideline document
- **Query 2** finds treatment-specific sections
- **Query 3** finds referral and imaging criteria

Together, they give the Retriever Agent a comprehensive set of guideline passages to work with.

## How It Works

### The Three-Tier Approach

Just like the Extractor uses rules first and LLM second, the Query Agent has three tiers:

```
Diagnosis → [1. Template Match?] → YES → Use pre-written queries
                    ↓ NO
             [2. LLM Available?] → YES → Generate queries via LLM
                    ↓ NO
             [3. Default Queries] → Generic template queries
```

### Tier 1: Template Queries (Most Common Diagnoses)

For the most common MSK conditions, we have **hand-crafted query templates** — pre-written search queries that we know work well with our FAISS index.

```python
"low back pain": [
    "NICE guidelines for assessment and management of low back pain and sciatica",
    "non-specific low back pain pharmacological and non-pharmacological treatment",
    "low back pain referral criteria and imaging recommendations",
]
```

We have templates for ~15 common MSK conditions: low back pain, osteoarthritis, carpal tunnel syndrome, plantar fasciitis, gout, fibromyalgia, osteoporosis, fractures, shoulder pain, hip pain, sciatica, and more.

**Why templates?** Because for common conditions, we can write better queries than an LLM would generate — we know exactly how NICE guidelines are titled and structured. Templates are also free, instant, and deterministic.

**Matching logic:** The agent first tries an exact match (case-insensitive), then a substring match. So "Osteoarthritis of knee" matches the "osteoarthritis" template even though it's not an exact match.

### Tier 2: LLM Generation (Unusual Diagnoses)

For diagnoses that don't match any template (e.g., "Acquired hallux valgus" or "Dupuytren's contracture"), the agent asks the LLM to generate queries.

The prompt instructs the LLM to:
- Generate exactly 3 search queries (configurable via `max_queries_per_diagnosis`)
- Cover different aspects of care (assessment, treatment, referral)
- Use clinical terminology matching how NICE guidelines are written
- Include "NICE" or "guidelines" in at least one query

The prompt also includes **episode context** — what treatments and referrals the patient received — so the LLM can generate more targeted queries.

### Tier 3: Default Queries (Fallback)

If there's no template AND no LLM available (or the LLM fails), the agent generates generic queries using a simple template:

```
1. "NICE clinical guidelines for {diagnosis} management"
2. "{diagnosis} treatment options and recommendations"
3. "{diagnosis} referral criteria and investigations"
```

These aren't as optimised as the other tiers, but they're reasonable and always available.

## Data Flow

```
ExtractionResult (from Extractor)
│
├── Episode 1 (2024-01-15)
│   ├── Diagnosis: "Low back pain"        → Template queries (3)
│   └── Diagnosis: "Sciatica"             → Template queries (3)
│
├── Episode 2 (2024-06-01)
│   └── Diagnosis: "Hallux valgus"        → LLM queries (3)
│
└── QueryResult
    ├── pat_id: "pat-001"
    ├── total_diagnoses: 3
    ├── total_queries: 9
    └── diagnosis_queries: [
          {diagnosis: "Low back pain", queries: [...], source: "template"},
          {diagnosis: "Sciatica", queries: [...], source: "template"},
          {diagnosis: "Hallux valgus", queries: [...], source: "llm"},
        ]
```

## What Happens Next?

The `QueryResult` is passed to the **Retriever Agent** (Stage 3), which:
1. Takes each query text
2. Encodes it using PubMedBERT into a 768-dimensional vector
3. Searches the FAISS index for the top-5 most similar guideline passages
4. Returns the matched guideline texts for the Scorer Agent to evaluate

## How This Differs from Hiruni's Approach

### What Hiruni Did

Hiruni's `QueryAgent` used a local Mistral-7B model (via `llama_cpp`) to generate all queries:

```python
class QueryAgent:
    def __init__(self, model_path):
        self.llm = Llama(model_path=model_path, n_ctx=4096)

    def generate(self, extracted_data):
        prompt = f"Generate 1-3 search queries per concept..."
        output = self.llm(prompt, max_tokens=200)
        return [line.strip() for line in output.split("\n")]
```

### Problems with Hiruni's Approach

1. **Underpowered model** — Mistral-7B is a general-purpose model, not optimised for medical query generation. It doesn't know how NICE guidelines are structured or titled.
2. **Local model dependency** — requires downloading a 4GB model file and having enough RAM to run it
3. **Hardcoded model path** — `C:\Users\hirun\agentic-msk\models\mistral-7b-instruct-v0.1.Q4_K_M.gguf`
4. **No fallback** — if the LLM fails, the entire pipeline fails
5. **No optimisation for PubMedBERT** — the queries aren't tailored for semantic similarity search
6. **All queries go through LLM** — even for "Low back pain" where a template would be better and faster

### What We Improved

| Aspect | Hiruni | Ours |
|---|---|---|
| **Query source** | All LLM-generated | Templates for common, LLM for rare, defaults as fallback |
| **LLM model** | Mistral-7B (local, weak) | GPT-4o-mini via API (via provider abstraction, swappable) |
| **Query quality** | Generic medical queries | Queries tailored to NICE guideline language and PubMedBERT similarity |
| **Reliability** | No fallback — fails if LLM fails | Three-tier fallback: template → LLM → default |
| **Dependencies** | 4GB local model file | Zero (templates are in code; LLM is optional) |
| **Speed** | Slow (local inference) | Instant for templates, fast API for LLM |
| **Portability** | Hardcoded Windows path | Works anywhere |
| **Context awareness** | No episode context | Includes treatments/referrals in LLM prompt |

### Is Our Approach Better?

**Yes, unambiguously.** Our three-tier approach means:
- Common diagnoses get **expert-crafted queries** that we know match well with our FAISS index (templates)
- Rare diagnoses get **LLM-generated queries** using a much stronger model with episode context
- If everything fails, we still get **reasonable default queries** — the pipeline never breaks

The template approach is particularly clever because we can test and tune these queries against the actual FAISS index to maximise retrieval quality. An LLM doesn't have this advantage — it's guessing what queries might work, while our templates are empirically validated.

## Configuration

The Query Agent reads one setting from the environment:

| Setting | Default | Description |
|---|---|---|
| `MAX_QUERIES_PER_DIAGNOSIS` | 3 | Maximum search queries generated per diagnosis |

This is defined in `src/config/settings.py` and can be overridden via the `.env` file.
