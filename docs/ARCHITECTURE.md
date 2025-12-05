# CogRepo v2 Architecture

> **Version:** 2.0.0
> **Last Updated:** December 2024

---

## Overview

CogRepo v2 transforms LLM conversation archives into a **knowledge system** with:

- **SQLite FTS5** for lightning-fast full-text search
- **Semantic search** via sentence-transformers embeddings
- **Intelligence layer** for insights, clustering, and recommendations
- **Multi-topic segmentation** for conversations spanning multiple subjects
- **1-100 scoring system** with 5 weighted dimensions
- **Production-ready infrastructure** with health checks and Docker support

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Web UI Layer                               │
│  cogrepo-ui/app.py (Flask + WebSocket + SocketIO)                   │
│  ├── Modern responsive interface with split-pane view               │
│  ├── Real-time import progress via WebSocket                        │
│  ├── Keyboard shortcuts (⌘K search, J/K navigation)                │
│  └── PWA with offline support (service worker)                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          API Layer                                   │
│  ├── api_v2.py          - Core v2 search/browse endpoints           │
│  ├── api_intelligence.py - Intelligence & insights endpoints        │
│  ├── health.py          - Health/readiness probes                   │
│  └── error_handlers.py  - Centralized error handling                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Intelligence Layer                              │
│  intelligence/                                                       │
│  ├── clustering.py       - HDBSCAN/K-Means conversation grouping    │
│  ├── recommendations.py  - Vector similarity suggestions            │
│  ├── insights.py         - Trend analysis and dashboards            │
│  ├── topic_segmentation.py - Multi-topic conversation splitting     │
│  └── scoring.py          - 5-dimension 1-100 scoring system         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Search Layer                                  │
│  ├── database/repository.py - SQLite + FTS5 full-text search        │
│  ├── search/hybrid_search.py - BM25 + semantic hybrid ranking       │
│  └── search/embeddings.py - sentence-transformers vectors           │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Enrichment Layer                                │
│  enrichment/                                                         │
│  ├── enrichment_pipeline.py - Claude API enrichment                 │
│  ├── zero_token.py       - Regex-based free analysis                │
│  └── artifact_extractor.py - Code/command extraction                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Context Layer                                 │
│  context/                                                            │
│  ├── project_inference.py - Auto-grouping by project                │
│  ├── chain_detection.py - Linked conversation chains                │
│  └── knowledge_graph.py - Entity relationships                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Parser Layer                                  │
│  parsers/                                                            │
│  ├── chatgpt_parser.py - ChatGPT conversations.json                 │
│  ├── claude_parser.py  - Claude JSON/JSONL exports                  │
│  ├── gemini_parser.py  - Gemini JSON/HTML exports                   │
│  └── smart_parser.py   - Auto-detect with incremental parsing       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### Intelligence Layer

#### `intelligence/scoring.py`

Advanced 1-100 scoring system with 5 weighted dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Technical Depth | 25% | Code complexity, advanced concepts, frameworks |
| Practical Value | 30% | Reusability, solutions, completeness |
| Completeness | 20% | Resolution, follow-through, examples |
| Clarity | 15% | Structure, organization, explanation quality |
| Uniqueness | 10% | Novelty, unconventional approaches |

**Letter Grades:** A+ (95+), A (90-94), B+ (85-89), B (75-84), C+ (65-74), C (50-64), D (35-49), F (<35)

```python
from intelligence.scoring import ConversationScorer

scorer = ConversationScorer(conversations)
score = scorer.score(conversation)

print(f"Overall: {score.overall}/100 ({score.grade})")
print(f"Technical: {score.technical_depth.score}")
print(f"Practical: {score.practical_value.score}")
```

#### `intelligence/topic_segmentation.py`

Handles conversations spanning multiple subjects:

```python
from intelligence.topic_segmentation import TopicSegmenter

segmenter = TopicSegmenter()
topics = segmenter.segment(conversation)

for segment in topics.segments:
    print(f"Topic: {segment.topic}")
    print(f"Turns: {segment.start_turn}-{segment.end_turn}")
    print(f"Technologies: {segment.technologies}")
```

**Detection Methods:**
- Explicit markers ("new question", "different topic")
- Terminology shifts between turns
- Code block language boundaries
- Temporal gaps in conversation

#### `intelligence/clustering.py`

Auto-groups similar conversations:

```python
from intelligence.clustering import ConversationClusterer

clusterer = ConversationClusterer()
clusters = clusterer.cluster(conversations, embeddings, ids)

for cluster in clusters:
    print(f"Cluster: {cluster.label}")
    print(f"Size: {cluster.size}")
    print(f"Keywords: {cluster.keywords}")
```

**Algorithms:**
- HDBSCAN (default): Discovers clusters automatically
- K-Means: Fixed cluster count
- Agglomerative: Hierarchical clustering

#### `intelligence/recommendations.py`

Vector-similarity recommendations:

```python
from intelligence.recommendations import RecommendationEngine

engine = RecommendationEngine(embeddings, ids, conversations)
similar = engine.find_similar("convo-123", limit=5)

for rec in similar:
    print(f"{rec.conversation_id}: {rec.score:.2f}")
```

#### `intelligence/insights.py`

Trend analysis and dashboard data:

```python
from intelligence.insights import InsightsEngine

engine = InsightsEngine(conversations)
dashboard = engine.export_dashboard_data()

print(f"Top trends: {dashboard['trends']}")
print(f"Activity: {dashboard['activity']}")
```

---

### Context Layer

#### `context/knowledge_graph.py`

Entity relationship mapping:

```python
from context.knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph()
kg.build_from_conversations(conversations)

# Find related entities
related = kg.get_related("React", limit=10)

# Export for visualization
viz_data = kg.export_for_visualization(max_nodes=100)
```

**Entity Types:**
- Technologies (Python, React, PostgreSQL)
- Concepts (machine learning, API design)
- Libraries/Frameworks (FastAPI, TensorFlow)
- Tools (Docker, git, npm)

---

### Production Infrastructure

#### `health.py`

Kubernetes-style health probes:

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness probe - is the app running? |
| `GET /ready` | Readiness probe - can it serve traffic? |
| `GET /health/detailed` | Component-level health status |
| `GET /metrics` | Prometheus-compatible metrics |

#### `error_handlers.py`

Centralized error handling:

```python
from error_handlers import NotFoundError, ValidationError, CircuitBreaker

# Custom exceptions
raise NotFoundError("Conversation not found", convo_id="xyz")

# Circuit breaker for external services
api_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

@api_breaker
def call_external_api():
    ...
```

#### `logging_config.py`

Structured logging with two modes:

| Mode | Format | Use Case |
|------|--------|----------|
| Development | Colorized console | Local debugging |
| Production | JSON | Log aggregation (ELK, CloudWatch) |

---

### Database Schema

```sql
-- Core conversations table
CREATE TABLE conversations (
    convo_id TEXT PRIMARY KEY,
    external_id TEXT,
    timestamp TEXT,
    source TEXT,           -- chatgpt, claude, gemini
    raw_text TEXT,

    -- Enriched fields
    generated_title TEXT,
    summary_abstractive TEXT,
    summary_extractive TEXT,
    primary_domain TEXT,
    score INTEGER,         -- 1-100 scale
    score_reasoning TEXT,

    -- Zero-token analysis
    has_code BOOLEAN,
    code_block_count INTEGER,
    turn_count INTEGER,
    question_count INTEGER,

    -- Semantic
    embedding BLOB,
    cluster_id TEXT,

    -- Context
    project_id TEXT,
    chain_id TEXT
);

-- Full-text search
CREATE VIRTUAL TABLE conversations_fts USING fts5(
    generated_title,
    summary_abstractive,
    raw_text,
    content=conversations
);

-- Tags
CREATE TABLE tags (name TEXT UNIQUE);
CREATE TABLE conversation_tags (
    conversation_id TEXT,
    tag_id INTEGER
);

-- Artifacts
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    artifact_type TEXT,    -- code_snippet, shell_command, etc.
    content TEXT,
    language TEXT
);
```

---

## Data Flow

### Import Pipeline

```
Export File → Parser → Normalizer → Enricher → Repository
    │           │          │           │           │
    │           │          │           │           └── SQLite + JSONL
    │           │          │           └── Claude API (optional)
    │           │          └── Unified format
    │           └── ChatGPT/Claude/Gemini
    └── .json/.jsonl/.html
```

### Search Pipeline

```
Query → FTS5 (BM25) ────┐
                        ├── Hybrid Ranker → Results
Query → Embeddings ─────┘
```

### Intelligence Pipeline

```
Conversations → Embeddings → Clustering ────┐
                    │                       │
                    └── Similarity ──────── ├── Insights
                                            │
Topics → Segmentation ──────────────────────┘
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | For enrichment | Claude API key |
| `FLASK_ENV` | No | `development` or `production` |
| `PORT` | No | Server port (default: 5001) |

### Config Files

```yaml
# ~/.cogrepo/config.yaml
anthropic:
  api_key: sk-ant-...

enrichment:
  model: claude-sonnet-4-20250514
  batch_size: 10

search:
  use_semantic: true
  default_limit: 50

embeddings:
  model: all-MiniLM-L6-v2
```

---

## Deployment

### Docker

```bash
# Build
docker build -t cogrepo:latest .

# Run
docker-compose up -d
```

### Production (Gunicorn)

```bash
cd cogrepo-ui
gunicorn -c gunicorn.conf.py app:app
```

### Backup/Restore

```bash
# Create backup
python cogrepo_backup.py backup

# Restore
python cogrepo_backup.py restore backup_20241201.tar.gz

# Verify integrity
python cogrepo_backup.py verify backup_20241201.tar.gz
```

---

## File Structure

```
cogrepo/
├── cogrepo-ui/                 # Web interface
│   ├── app.py                  # Main Flask application
│   ├── api_v2.py               # v2 API endpoints
│   ├── api_intelligence.py     # Intelligence endpoints
│   ├── health.py               # Health probes
│   ├── error_handlers.py       # Error handling
│   ├── logging_config.py       # Structured logging
│   ├── gunicorn.conf.py        # Production server config
│   ├── index.html              # Modern UI
│   └── static/                 # CSS, JS assets
│
├── intelligence/               # Intelligence layer
│   ├── clustering.py           # Auto-clustering
│   ├── recommendations.py      # Similarity engine
│   ├── insights.py             # Trend analysis
│   ├── topic_segmentation.py   # Multi-topic handling
│   └── scoring.py              # 1-100 scoring
│
├── context/                    # Context layer
│   ├── project_inference.py    # Project grouping
│   ├── chain_detection.py      # Conversation chains
│   └── knowledge_graph.py      # Entity graph
│
├── database/                   # Data layer
│   ├── schema.sql              # SQLite schema
│   └── repository.py           # Data access
│
├── search/                     # Search layer
│   ├── embeddings.py           # Vector embeddings
│   └── hybrid_search.py        # BM25 + semantic
│
├── enrichment/                 # Enrichment layer
│   ├── enrichment_pipeline.py  # Claude API
│   ├── zero_token.py           # Free analysis
│   └── artifact_extractor.py   # Code extraction
│
├── parsers/                    # Parser layer
│   ├── chatgpt_parser.py
│   ├── claude_parser.py
│   └── gemini_parser.py
│
├── cogrepo_backup.py           # Backup/restore
├── Dockerfile                  # Container build
├── docker-compose.yml          # Container orchestration
└── requirements.txt            # Dependencies
```

---

## Performance Targets

| Metric | Target | Actual |
|--------|--------|--------|
| Search latency | < 100ms | ~50ms |
| Import speed | > 100/sec | ~200/sec |
| Embedding generation | < 50ms/convo | ~30ms |
| Memory usage | < 500MB | ~300MB |
