# CogRepo v2 (Cognitive Repository)

**Transform LLM conversations into searchable knowledge**

Import, enrich, and search conversations from ChatGPT, Claude, and Gemini. Features AI-powered enrichment, semantic search, intelligent insights, and production-ready infrastructure.

## What's New in v2

- **Intelligence Layer** — Auto-clustering, recommendations, trend analysis
- **Multi-Topic Segmentation** — Handle conversations spanning multiple subjects
- **1-100 Scoring System** — 5-dimension scoring with letter grades (A+ to F)
- **Knowledge Graph** — Entity relationships across conversations
- **Production Ready** — Health probes, Docker support, backup/restore
- **SQLite FTS5** — Lightning-fast full-text search

## Features

### Core
- **Multi-platform**: ChatGPT, Claude, Gemini parsers with auto-detection
- **Incremental sync**: Track archives, only process new conversations
- **AI enrichment**: Titles, summaries, tags, quality scores via Claude API
- **Full-text search**: SQLite FTS5 with BM25 ranking

### Intelligence (v2)
- **Clustering**: Auto-group similar conversations (HDBSCAN/K-Means)
- **Recommendations**: Vector-similarity suggestions
- **Insights**: Trend analysis, technology adoption tracking
- **Topic Segmentation**: Split multi-subject conversations
- **Advanced Scoring**: 5 dimensions, 1-100 scale, percentiles

### UI
- Modern responsive interface with split-pane view
- Keyboard shortcuts (⌘K search, J/K navigation)
- Real-time import progress via WebSocket
- PWA with offline support

### Production
- Kubernetes-style health probes (`/health`, `/ready`)
- Docker containerization
- Structured logging (JSON for production)
- Backup/restore with integrity verification

---

## Quick Start

### macOS (Spotlight/Alfred/Raycast)

Double-click `CogRepo.command` or search "CogRepo" in Spotlight/Alfred/Raycast.

### Command Line

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key for enrichment (optional)
export ANTHROPIC_API_KEY="sk-..."

# Start web server
cd cogrepo-ui && python app.py
# Open http://localhost:5001
```

### Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f
```

---

## Usage

### Import Conversations

```bash
# Import with AI enrichment
python cogrepo_import.py --source chatgpt --file conversations.json --enrich

# Import without enrichment (faster)
python cogrepo_import.py --source claude --file export.json

# Auto-detect format
python cogrepo_import.py --file any_export.json
```

### Incremental Sync

```bash
# Register archive
python cogrepo_manage.py register ~/Downloads/chatgpt_export.json chatgpt

# Sync only new conversations
python quick_sync.py

# Check status
python cogrepo_manage.py status
```

### Backup & Restore

```bash
# Create backup
python cogrepo_backup.py backup

# List backups
python cogrepo_backup.py list

# Restore
python cogrepo_backup.py restore backup_20241201.tar.gz

# Verify integrity
python cogrepo_backup.py verify backup_20241201.tar.gz
```

---

## API Endpoints

### Core API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/conversations` | GET | List conversations |
| `/api/conversation/<id>` | GET | Get single conversation |
| `/api/search` | GET | Full-text search |
| `/api/stats` | GET | Repository statistics |
| `/api/tags` | GET | Tag cloud data |
| `/api/upload` | POST | Upload export file |

### Intelligence API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/intelligence/knowledge-graph` | GET | Entity relationship graph |
| `/api/intelligence/recommendations/<id>` | GET | Similar conversations |
| `/api/intelligence/insights` | GET | Repository insights |
| `/api/intelligence/topics/<id>` | GET | Topic segmentation |
| `/api/intelligence/score/<id>` | GET | Detailed scoring |
| `/api/intelligence/clusters` | GET | Conversation clusters |

### Health & Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness probe |
| `/health/detailed` | GET | Component health |
| `/metrics` | GET | Prometheus metrics |

See [API Reference](docs/API_REFERENCE.md) for complete documentation.

---

## Scoring System

CogRepo v2 uses a 5-dimension scoring system (1-100 scale):

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Technical Depth | 25% | Code complexity, advanced concepts |
| Practical Value | 30% | Reusability, complete solutions |
| Completeness | 20% | Resolution, follow-through |
| Clarity | 15% | Structure, organization |
| Uniqueness | 10% | Novelty, unconventional approaches |

**Letter Grades:** A+ (95+), A (90-94), A- (87-89), B+ (85-86), B (75-84), B- (70-74), C+ (65-69), C (50-64), D (35-49), F (<35)

---

## Multi-Topic Conversations

CogRepo automatically detects and segments conversations covering multiple topics:

```json
{
  "topic_count": 3,
  "topic_summary": "Docker setup → Python debugging → API design",
  "segments": [
    {"topic": "Docker Configuration", "turns": "0-5"},
    {"topic": "Python Debugging", "turns": "6-12"},
    {"topic": "API Design", "turns": "13-20"}
  ]
}
```

Detection methods:
- Explicit markers ("new question", "different topic")
- Terminology shifts between turns
- Code language boundaries
- Temporal gaps

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | For enrichment | Claude API key |
| `FLASK_ENV` | No | `development` or `production` |
| `PORT` | No | Server port (default: 5001) |

### Config File

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
```

---

## Project Structure

```
cogrepo/
├── cogrepo-ui/                 # Web interface
│   ├── app.py                  # Flask application
│   ├── api_v2.py               # v2 API endpoints
│   ├── api_intelligence.py     # Intelligence endpoints
│   ├── health.py               # Health probes
│   └── index.html              # Modern UI
│
├── intelligence/               # Intelligence layer
│   ├── clustering.py           # HDBSCAN/K-Means clustering
│   ├── recommendations.py      # Similarity engine
│   ├── insights.py             # Trend analysis
│   ├── topic_segmentation.py   # Multi-topic handling
│   └── scoring.py              # 1-100 scoring system
│
├── context/                    # Context layer
│   ├── knowledge_graph.py      # Entity relationships
│   ├── project_inference.py    # Project grouping
│   └── chain_detection.py      # Conversation chains
│
├── parsers/                    # Format parsers
│   ├── chatgpt_parser.py
│   ├── claude_parser.py
│   └── gemini_parser.py
│
├── enrichment/                 # AI enrichment
│   └── enrichment_pipeline.py
│
├── cogrepo_backup.py           # Backup/restore
├── Dockerfile                  # Container build
├── docker-compose.yml          # Container orchestration
└── requirements.txt            # Dependencies
```

---

## Keyboard Shortcuts (Web UI)

| Shortcut | Action |
|----------|--------|
| ⌘K / Ctrl+K | Focus search |
| ⌘S | Save search |
| ⌘E | Export results |
| J / K | Navigate results |
| Enter | Open selected |
| Esc | Close modal |
| ? | Show shortcuts |

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System design and components
- [API Reference](docs/API_REFERENCE.md) — Complete endpoint documentation
- [Master Plan](docs/v2-planning/COGREPO_V2_MASTER_PLAN.md) — Development roadmap

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT
