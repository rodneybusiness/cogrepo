# CogRepo v2: Master Implementation Plan

> **Status:** Approved for implementation
> **Created:** December 2024
> **Last Updated:** December 2024
> **Estimated Token Cost:** ~$9 for complete enrichment backfill

---

## Executive Summary

CogRepo v2 transforms the conversation archive from a search tool into a **knowledge system** with:

- **10-100x faster search** via SQLite FTS5 + vector embeddings
- **True semantic search** using local embeddings (zero API cost)
- **Artifact extraction** — code, commands, solutions directly usable
- **Project grouping** — auto-detected project contexts
- **Conversation chains** — linked problem-solving journeys
- **Knowledge graph** — entity relationships across conversations
- **Secure configuration** — API keys in hidden config, not plaintext
- **Complete documentation** — LLMs can operate every aspect

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase Breakdown](#phase-breakdown)
3. [Enrichment System](#enrichment-system)
4. [Database Design](#database-design)
5. [API Key Security](#api-key-security)
6. [QC Strategy](#qc-strategy)
7. [Success Metrics](#success-metrics)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web UI Layer                             │
│  cogrepo-ui/app.py (Flask + WebSocket)                          │
│  ├── Split-pane view with advanced filters                       │
│  ├── Artifact browser                                            │
│  ├── Project views                                               │
│  └── Chain visualization                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Search Layer                              │
│  ├── database/repository.py (SQLite + FTS5)                     │
│  ├── search/hybrid_search.py (BM25 + semantic)                  │
│  └── search/embeddings.py (sentence-transformers)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Enrichment Layer                            │
│  ├── enrichment/structured_enrichment.py (Claude API)           │
│  ├── enrichment/zero_token.py (regex, stats - FREE)             │
│  ├── enrichment/artifact_extractor.py (Haiku - cheap)           │
│  └── enrichment/embeddings.py (local model - FREE)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Context Layer                              │
│  ├── context/project_inference.py (auto-grouping)               │
│  ├── context/chain_detection.py (linked conversations)          │
│  └── context/knowledge_graph.py (entity relationships)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Parser Layer                               │
│  parsers/{chatgpt,claude,gemini}_parser.py                      │
│  └── smart_parser.py (auto-detect, incremental)                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase Breakdown

### Phase 0: Foundation + Complete Enrichment

**Goal:** Secure configuration, comprehensive enrichment backfill

| Task | Deliverable | Token Cost |
|------|-------------|------------|
| 0.1 Config system | `~/.cogrepo/config.yaml` | $0 |
| 0.2 Zero-token enrichment | Code detection, links, terms | $0 |
| 0.3 Embeddings | All conversations vectorized | $0 |
| 0.4 Artifact extraction | Code, commands, solutions | ~$4 |
| 0.5 Context system | Projects, chains, graph | ~$3 |
| 0.6 Deep analysis | High-value conversations | ~$2 |
| 0.7 Documentation | Complete LLM guide | $0 |

**Total Phase 0 Cost:** ~$9

### Phase 1: Database Foundation

**Goal:** Replace JSONL with indexed SQLite

| Task | Deliverable |
|------|-------------|
| Schema design | `database/schema.sql` |
| Repository pattern | `database/repository.py` |
| Migration script | `database/migrations.py` |
| Parallel operation | Both JSONL + SQLite active |

### Phase 2: Semantic Search

**Goal:** Hybrid keyword + vector search

| Task | Deliverable |
|------|-------------|
| Embedding engine | `search/embeddings.py` |
| Vector storage | SQLite blob or ChromaDB |
| Hybrid ranking | BM25 + cosine with RRF |
| Similar API | `/api/search/similar/<id>` |

### Phase 3: Enhanced UI

**Goal:** Professional interface with new capabilities

| Feature | Description |
|---------|-------------|
| Split-pane | Side-by-side results + conversation |
| Artifact browser | Browse/copy code, commands, solutions |
| Project views | Auto-grouped by inferred project |
| Chain visualization | Follow problem-solving journeys |
| Advanced filters | Multi-dimensional, code language, domain |

### Phase 4: Production Hardening

**Goal:** Deployment-ready infrastructure

| Component | Deliverable |
|-----------|-------------|
| Production server | Gunicorn config |
| Health endpoints | `/health`, `/ready` |
| Docker | `Dockerfile`, `docker-compose.yml` |
| Backup/restore | One-click snapshots |

### Phase 5: Intelligence Layer (Optional)

**Goal:** Advanced insights

| Feature | Method |
|---------|--------|
| Auto-clustering | HDBSCAN on embeddings |
| Recommendations | Vector similarity |
| Collection summaries | On-demand Haiku |

---

## Enrichment System

### The Six Tiers

```
TIER 1: METADATA (Existing)
├── generated_title, summary, tags, topics
├── brilliance_score with factors
├── key_insights, status, future_potential
└── Cost: Already paid

TIER 2: ZERO-TOKEN (New - FREE)
├── has_code, code_languages, code_block_count
├── turn_count, question_count, question_types
├── has_links, link_domains, technical_terms
└── Cost: $0 (regex/compute only)

TIER 3: SEMANTIC (New - FREE)
├── embedding (384-dim vector)
├── cluster_id
└── Cost: $0 (local sentence-transformers)

TIER 4: ARTIFACTS (New - Transformative)
├── Code snippets, shell commands, configurations
├── API calls, SQL queries, regex patterns
├── Error solutions, best practices
└── Cost: ~$0.80 per 1K conversations (Haiku)

TIER 5: CONTEXT (New - Transformative)
├── project_id, project_name (auto-inferred)
├── chain_id, chain_position (linked conversations)
├── entities (knowledge graph nodes)
└── Cost: ~$0.60 per 1K conversations (Haiku)

TIER 6: DEEP ANALYSIS (Optional)
├── detailed_breakdown
├── related_concepts, suggested_followups
└── Cost: ~$7 per 1K (Sonnet, high-value only)
```

### Artifact Types

| Type | Description | Example |
|------|-------------|---------|
| `code_snippet` | Reusable code | async function, class definition |
| `shell_command` | Terminal commands | docker run, git commands |
| `configuration` | Config files | .env, yaml, json configs |
| `api_call` | API examples | curl commands, fetch calls |
| `sql_query` | Database queries | SELECT, JOIN patterns |
| `regex_pattern` | Regular expressions | Validation, parsing patterns |
| `error_solution` | Problem + fix pairs | CORS fix, import errors |
| `best_practice` | Advice/patterns | Design patterns, tips |

---

## Database Design

### Schema Overview

```sql
-- Core conversations
CREATE TABLE conversations (
    convo_id TEXT PRIMARY KEY,
    external_id TEXT,
    timestamp TEXT,
    source TEXT,
    raw_text TEXT,
    -- Tier 1: Metadata
    generated_title TEXT,
    summary_abstractive TEXT,
    summary_extractive TEXT,
    primary_domain TEXT,
    score INTEGER,
    score_reasoning TEXT,
    status TEXT,
    -- Tier 2: Zero-token
    has_code BOOLEAN,
    code_block_count INTEGER,
    turn_count INTEGER,
    question_count INTEGER,
    has_links BOOLEAN,
    duration_minutes INTEGER,
    -- Tier 3: Semantic
    embedding BLOB,
    cluster_id TEXT,
    -- Tier 5: Context
    project_id TEXT,
    chain_id TEXT,
    chain_position INTEGER,
    -- Meta
    enrichment_version TEXT,
    enriched_at TEXT
);

-- Full-text search
CREATE VIRTUAL TABLE conversations_fts USING fts5(
    generated_title,
    summary_abstractive,
    raw_text,
    content=conversations,
    content_rowid=rowid
);

-- Tags (many-to-many)
CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);

CREATE TABLE conversation_tags (
    conversation_id TEXT,
    tag_id INTEGER,
    FOREIGN KEY (conversation_id) REFERENCES conversations(convo_id),
    FOREIGN KEY (tag_id) REFERENCES tags(id)
);

-- Artifacts (one-to-many)
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    artifact_type TEXT,
    content TEXT,
    language TEXT,
    description TEXT,
    use_case TEXT,
    verified_working BOOLEAN,
    FOREIGN KEY (conversation_id) REFERENCES conversations(convo_id)
);

-- Projects (inferred groupings)
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,
    project_name TEXT,
    technologies TEXT,  -- JSON array
    first_conversation TEXT,
    last_conversation TEXT,
    conversation_count INTEGER,
    project_summary TEXT,
    status TEXT
);

-- Conversation chains
CREATE TABLE chains (
    chain_id TEXT PRIMARY KEY,
    title TEXT,
    starting_point TEXT,
    resolution TEXT,
    key_learnings TEXT  -- JSON array
);

-- Knowledge graph entities
CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT,
    name TEXT,
    occurrence_count INTEGER
);

CREATE TABLE entity_relationships (
    conversation_id TEXT,
    entity_id TEXT,
    relationship TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(convo_id),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);
```

---

## API Key Security

### Resolution Chain

```
1. ANTHROPIC_API_KEY environment variable (highest priority)
2. ~/.cogrepo/config.yaml → anthropic.api_key
3. ./.cogrepo/config.yaml → anthropic.api_key (project override)
4. Error with helpful setup instructions
```

### Config File Format

```yaml
# ~/.cogrepo/config.yaml
anthropic:
  api_key: sk-ant-...

enrichment:
  model: claude-3-5-haiku-20241022
  batch_size: 10

search:
  use_semantic: true
  default_limit: 50

embeddings:
  model: all-MiniLM-L6-v2
```

### Security Measures

- Config file created with `chmod 600` (owner-only)
- Located in home directory, never in git
- `.cogrepo/` added to `.gitignore`
- Never logged or displayed
- Falls back to env var for CI/CD compatibility

---

## QC Strategy

### Validation Gates

Each phase completes ONLY when:

1. **Unit tests pass** — All new code tested
2. **Integration tests pass** — End-to-end workflows verified
3. **Regression tests pass** — Existing functionality unchanged
4. **Manual spot-check** — Human verification of 10 random records
5. **Performance benchmark** — Meets targets
6. **Rollback tested** — Can undo changes cleanly

### Parallel Systems

During migration:
- Both JSONL and SQLite active simultaneously
- Results compared for parity
- Switchover only after validation gate passes
- JSONL preserved as backup indefinitely

### Verification Script

```bash
python cogrepo_verify.py --comprehensive

# Checks:
# - All tiers enriched
# - Data integrity (checksums)
# - Search results match expectations
# - Performance within targets
# - Rollback procedure functional
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Search latency | < 100ms for 10K conversations |
| Semantic relevance | > 80% accuracy in top-5 |
| Artifact accuracy | > 95% correctly extracted |
| Project detection | > 90% meaningful groupings |
| Import speed | > 100 conversations/second |
| Zero regressions | All existing features work |

---

## File Map (New Files)

```
cogrepo/
├── core/
│   └── config.py              # Updated: ~/.cogrepo support
│
├── database/
│   ├── schema.sql             # NEW: SQLite schema
│   ├── repository.py          # NEW: Data access layer
│   └── migrations.py          # NEW: JSONL → SQLite
│
├── search/
│   ├── embeddings.py          # NEW: sentence-transformers
│   └── hybrid_search.py       # NEW: BM25 + semantic
│
├── enrichment/
│   ├── zero_token.py          # NEW: Free enrichment
│   ├── artifact_extractor.py  # NEW: Code/command extraction
│   └── structured_enrichment.py  # Existing, enhanced
│
├── context/
│   ├── project_inference.py   # NEW: Auto-grouping
│   ├── chain_detection.py     # NEW: Linked conversations
│   └── knowledge_graph.py     # NEW: Entity relationships
│
├── cogrepo_setup.py           # NEW: Interactive setup
├── cogrepo_enrich.py          # NEW: Backfill enrichment
├── cogrepo_verify.py          # NEW: Validation script
│
├── docs/
│   ├── v2-planning/           # This planning documentation
│   ├── ARCHITECTURE.md        # NEW: System design
│   ├── API_REFERENCE.md       # NEW: All endpoints
│   ├── CONFIGURATION.md       # NEW: All config options
│   └── TROUBLESHOOTING.md     # NEW: Common issues
│
└── tests/
    ├── test_database.py       # NEW
    ├── test_embeddings.py     # NEW
    ├── test_artifacts.py      # NEW
    └── test_context.py        # NEW
```

---

## Next Steps

See [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) for step-by-step instructions.

See [CONTINUATION_GUIDE.md](./CONTINUATION_GUIDE.md) for resuming this project later.
