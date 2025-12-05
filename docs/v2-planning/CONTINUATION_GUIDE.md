# CogRepo v2: Continuation Guide

> **Purpose:** Enable any developer or LLM to pick up this project and continue seamlessly.
> **Last Updated:** December 2024

---

## Quick Context Reload

### What is CogRepo?

CogRepo is a knowledge system for LLM conversations. It imports, enriches, and enables searching across conversations from ChatGPT, Claude, and Gemini.

### What is CogRepo v2?

A major upgrade transforming CogRepo from a search tool into a comprehensive knowledge system with:

1. **SQLite database** replacing JSONL file scanning
2. **Semantic search** via local embeddings
3. **Artifact extraction** — code, commands, solutions directly usable
4. **Project grouping** — auto-detected project contexts
5. **Conversation chains** — linked problem-solving journeys
6. **Knowledge graph** — entity relationships

### Current Status

Check this file for the latest status update, or run:

```bash
python cogrepo_verify.py --status
```

---

## How to Resume This Project

### Step 1: Understand Where We Are

```bash
# Check current phase completion
python cogrepo_verify.py --status

# Or manually check:
cd /path/to/cogrepo

# Phase 0.1 Config: Check if config system exists
python -c "from core.config import AnthropicConfig; print(AnthropicConfig.load().has_api_key)"

# Phase 0.2 Zero-token: Check if new fields exist
python -c "
import json
with open('data/enriched_repository.jsonl') as f:
    conv = json.loads(f.readline())
    print('has_code' in conv, 'code_languages' in conv)
"

# Phase 0.3 Embeddings: Check if embeddings exist
ls data/*embeddings*.npy

# Phase 0.4 Artifacts: Check if artifacts exist
ls data/*artifacts*.jsonl
```

### Step 2: Read the Relevant Documentation

| Phase | Read This |
|-------|-----------|
| 0 | [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) Phase 0 section |
| 1 | Database section in MASTER_PLAN |
| 2 | Search section in MASTER_PLAN |
| 3+ | See phase-specific sections |

### Step 3: Run Tests Before Any Changes

```bash
# Always run tests first
pytest tests/ -v

# Backup current data
cp data/enriched_repository.jsonl data/enriched_repository.jsonl.backup.$(date +%Y%m%d)
```

### Step 4: Continue Implementation

Follow the step-by-step instructions in [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md).

---

## Key Decisions Made

These decisions were made after careful analysis. Do not change without good reason:

### 1. SQLite over PostgreSQL

**Why:** Zero deployment friction, single-file database, excellent for single-user local tool.

**Trade-off:** Can't scale to multi-user, but that's not a use case.

### 2. Local Embeddings over API Embeddings

**Why:** Zero ongoing cost, privacy (data stays local), fast (14k sentences/second).

**Model:** `all-MiniLM-L6-v2` — good balance of quality and speed.

### 3. Haiku for Extraction, Sonnet for Analysis

**Why:** Haiku is 10x cheaper, sufficient for structured extraction tasks.

**When Sonnet:** Only for deep analysis of high-value conversations (score >= 8).

### 4. Hidden Config Directory (~/.cogrepo/)

**Why:** Standard pattern (.aws, .ssh), safe (never in git), simple.

**Fallback:** Environment variable still works for CI/CD.

### 5. JSONL Preserved as Backup

**Why:** Zero-risk migration. Can always revert.

**When to Delete:** Only after 30 days of successful SQLite operation.

---

## Code Patterns to Follow

### Configuration Access

```python
# CORRECT: Use the config loader
from core.config import AnthropicConfig, get_config

config = get_config()
if config.has_api_key:
    client = Anthropic(api_key=config.anthropic.api_key)

# WRONG: Direct env access
api_key = os.getenv('ANTHROPIC_API_KEY')  # Only as fallback
```

### Database Access

```python
# CORRECT: Use repository pattern
from database.repository import ConversationRepository

repo = ConversationRepository()
conv = repo.get(convo_id)
repo.save(enriched_conv)

# WRONG: Direct SQL
conn = sqlite3.connect('data/cogrepo.db')
cursor.execute("SELECT * FROM conversations")
```

### Enrichment

```python
# CORRECT: Check for existing enrichment
if not conv.get('has_code'):  # Zero-token field missing
    metrics = zero_token_enricher.extract(conv['raw_text'])
    conv.update(metrics.to_dict())

# WRONG: Re-enrich everything
for conv in all_conversations:
    enriched = pipeline.enrich(conv)  # Wasteful if already enriched
```

### Error Handling

```python
# CORRECT: Graceful degradation
try:
    artifacts = extractor.extract(conv)
except Exception as e:
    logger.warning(f"Artifact extraction failed for {conv['convo_id']}: {e}")
    artifacts = []  # Continue without artifacts

# WRONG: Let errors propagate
artifacts = extractor.extract(conv)  # May crash entire batch
```

---

## File Locations

### Existing Files (v1)

```
cogrepo/
├── CLAUDE.md                    # LLM navigation guide
├── README.md                    # User documentation
├── models.py                    # Data models
├── app.py                       # Main Flask app (in cogrepo-ui)
├── enrichment/
│   ├── enrichment_pipeline.py   # Original enrichment
│   └── structured_enrichment.py # Improved JSON-based enrichment
├── parsers/                     # ChatGPT, Claude, Gemini parsers
├── data/
│   └── enriched_repository.jsonl  # Current data store
└── tests/                       # Test suite
```

### New Files (v2) - To Be Created

```
cogrepo/
├── core/
│   └── config.py                # UPDATED: ~/.cogrepo support
│
├── database/
│   ├── schema.sql               # NEW: SQLite schema
│   ├── repository.py            # NEW: Data access layer
│   └── migrations.py            # NEW: JSONL → SQLite
│
├── search/
│   ├── embeddings.py            # NEW: sentence-transformers
│   └── hybrid_search.py         # NEW: BM25 + semantic
│
├── enrichment/
│   ├── zero_token.py            # NEW: Free enrichment
│   └── artifact_extractor.py    # NEW: Code extraction
│
├── context/
│   ├── project_inference.py     # NEW: Auto-grouping
│   ├── chain_detection.py       # NEW: Linked conversations
│   └── knowledge_graph.py       # NEW: Entity relationships
│
├── cogrepo_setup.py             # NEW: Interactive setup
├── cogrepo_enrich.py            # NEW: Backfill enrichment
├── cogrepo_verify.py            # NEW: Validation script
│
└── docs/v2-planning/            # This documentation
```

---

## Common Tasks

### Add a New Enrichment Field

1. Add to `ZeroTokenMetrics` in `enrichment/zero_token.py` (if zero-token)
2. Add to `EnrichedConversation` in `models.py`
3. Add to SQLite schema in `database/schema.sql`
4. Add migration in `database/migrations.py`
5. Update CLAUDE.md with new field

### Add a New Artifact Type

1. Add to `EXTRACTION_PROMPT` in `enrichment/artifact_extractor.py`
2. Add to artifact types list in documentation
3. Add UI support in artifact browser

### Debug Enrichment Issues

```bash
# Check a specific conversation
python -c "
import json
with open('data/enriched_repository.jsonl') as f:
    for line in f:
        conv = json.loads(line)
        if conv['convo_id'] == 'YOUR_ID':
            print(json.dumps(conv, indent=2))
            break
"

# Check enrichment stats
python -c "
import json
from collections import Counter

stats = Counter()
with open('data/enriched_repository.jsonl') as f:
    for line in f:
        conv = json.loads(line)
        if conv.get('has_code'): stats['has_code'] += 1
        if conv.get('artifacts'): stats['has_artifacts'] += 1

print(dict(stats))
"
```

### Run Specific Phase

```bash
# Just Phase 0.2 (zero-token)
python enrichment/zero_token.py data/enriched_repository.jsonl

# Just Phase 0.4 (artifacts) for new conversations only
python -c "
from enrichment.artifact_extractor import ArtifactExtractor
import json

extractor = ArtifactExtractor()

# Find conversations without artifacts
with open('data/enriched_repository.jsonl') as f:
    for line in f:
        conv = json.loads(line)
        if not conv.get('artifacts'):
            artifacts = extractor.extract(conv['convo_id'], conv['raw_text'])
            print(f'{conv[\"convo_id\"]}: {len(artifacts)} artifacts')
"
```

---

## Troubleshooting

### "No API key configured"

```bash
# Check if config exists
cat ~/.cogrepo/config.yaml

# If not, run setup
python cogrepo_setup.py

# Or set environment variable
export ANTHROPIC_API_KEY="sk-ant-..."
```

### "Module not found: sentence_transformers"

```bash
pip install sentence-transformers
```

### "Embeddings file not found"

```bash
# Generate embeddings
python search/embeddings.py data/enriched_repository.jsonl
```

### "SQLite database locked"

```bash
# Only one process can write at a time
# Check for running processes
ps aux | grep python | grep cogrepo

# If stuck, restart
killall -9 python  # CAREFUL: kills all Python processes
```

### Tests Failing After Changes

```bash
# Run specific test to see error
pytest tests/test_enrichment.py -v -s

# Check if data format changed
python -c "
import json
with open('data/enriched_repository.jsonl') as f:
    conv = json.loads(f.readline())
    print(list(conv.keys()))
"
```

---

## Contacts and Resources

### Documentation

- [COGREPO_V2_MASTER_PLAN.md](./COGREPO_V2_MASTER_PLAN.md) — Full plan
- [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) — Step-by-step
- [ENRICHMENT_SCHEMA.md](./ENRICHMENT_SCHEMA.md) — Data schemas
- [../CLAUDE.md](../CLAUDE.md) — LLM navigation guide

### External Resources

- [Anthropic API Docs](https://docs.anthropic.com/)
- [sentence-transformers](https://www.sbert.net/)
- [SQLite FTS5](https://www.sqlite.org/fts5.html)

---

## Status Updates

Update this section as phases complete:

| Phase | Status | Completed | Notes |
|-------|--------|-----------|-------|
| 0.1 Config | Not Started | — | |
| 0.2 Zero-token | Not Started | — | |
| 0.3 Embeddings | Not Started | — | |
| 0.4 Artifacts | Not Started | — | |
| 0.5 Context | Not Started | — | |
| 0.6 Deep Analysis | Not Started | — | |
| 0.7 Documentation | In Progress | — | Creating planning docs |
| 1 Database | Not Started | — | |
| 2 Semantic Search | Not Started | — | |
| 3 Enhanced UI | Not Started | — | |
| 4 Production | Not Started | — | |
| 5 Intelligence | Not Started | — | |

---

## Final Notes

This project is designed to be picked up and continued at any time. The key principles:

1. **Each phase is independent** — Complete one before starting the next
2. **Tests before changes** — Always run `pytest tests/ -v` first
3. **Backup before migration** — Keep JSONL until SQLite is proven
4. **Documentation is code** — Update docs with every change
5. **Graceful degradation** — Features should fail safely

Good luck, future developer/LLM!
