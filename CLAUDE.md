# CogRepo - LLM Navigation Guide

Quick reference for understanding and working with this codebase.

## What This Does

Imports, enriches, and searches LLM conversations from ChatGPT, Claude, and Gemini. Supports incremental updates (only processes new conversations).

---

## CRITICAL: v3 Implementation Status

**Read [docs/V3_IMPLEMENTATION_GUIDE.md](docs/V3_IMPLEMENTATION_GUIDE.md) before making changes.**

### Current Reality (December 2024)

The codebase is **60% functional**. There are **12 broken features** to fix:

| # | Feature | Status | Root Cause |
|---|---------|--------|------------|
| 1 | Dependencies | **BROKEN** | `sentence-transformers`, `scikit-learn`, `hdbscan` missing |
| 2 | Embeddings | **MISSING** | No `.npy` files exist - never generated |
| 3 | Hybrid Search | **UNUSED** | Code exists in `search/`, not wired in |
| 4 | Clustering | **BROKEN** | Missing deps + no embeddings |
| 5 | Recommendations | **BROKEN** | Missing deps + no embeddings |
| 6 | Knowledge Graph | **EMPTY** | Code exists, never called |
| 7 | Project Inference | **NOT INTEGRATED** | Code exists, never called |
| 8 | Chain Detection | **NOT INTEGRATED** | Code exists, never called |
| 9 | Intelligence APIs | **503 ERRORS** | All above issues |
| 10 | AI Enrichment | **DISABLED** | Opt-in instead of opt-out |
| 11 | UI Dashboard | **INCOMPLETE** | Doesn't use intelligence endpoints |
| 12 | Test Coverage | **MISSING** | Intelligence layer untested |

### Fix Order (Critical Path)

1. **Fix requirements.txt** - Add missing packages
2. **Generate embeddings** - Run embedding generation script
3. **Build knowledge graph** - Run graph builder script
4. **Wire hybrid search** - Connect `search/hybrid_search.py` to API
5. **Integrate context layer** - Wire `context/` into import pipeline
6. **Verify intelligence APIs** - All should return 200
7. **Update UI** - Add dashboard for intelligence features
8. **Add tests** - Cover intelligence layer
9. **Fix enrichment default** - Make opt-out, not opt-in

### Quick Verification

```bash
# Check dependencies
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
# If ImportError: Fix #1 needed

# Check embeddings
ls data/embeddings.npy
# If not found: Fix #2 needed

# Check knowledge graph
ls data/knowledge_graph.json
# If not found: Fix #6 needed

# Check intelligence APIs
curl http://localhost:5001/api/intelligence/health
# All features should show "available"
```

---

## Key Entry Points

| Task | File | Command |
|------|------|---------|
| Start web UI (macOS) | `CogRepo.command` | Double-click or search in Spotlight/Alfred/Raycast |
| Start web UI (CLI) | `cogrepo-ui/app.py` | `cd cogrepo-ui && python app.py` |
| Import conversations | `cogrepo_import.py` | `python cogrepo_import.py --file export.json --enrich` |
| Incremental sync | `quick_sync.py` | `python quick_sync.py` |
| CLI search | `cogrepo_search.py` | `python cogrepo_search.py "query"` |
| Manage archives | `cogrepo_manage.py` | `python cogrepo_manage.py status` |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Web UI                               │
│  cogrepo-ui/app.py (Flask + WebSocket)                      │
│  └── static/js/{api,ui,app}.js (ES6 modules)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Pipeline                           │
│  smart_parser.py → enrichment_pipeline.py → search_engine.py│
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ chatgpt_parser│    │ claude_parser │    │ gemini_parser │
│   (parsers/)  │    │   (parsers/)  │    │   (parsers/)  │
└───────────────┘    └───────────────┘    └───────────────┘
```

## File Map

### Parsers (`parsers/`)
- `base_parser.py` - Abstract base class, common utilities
- `chatgpt_parser.py` - ChatGPT conversations.json (tree structure)
- `claude_parser.py` - Claude JSON/JSONL exports
- `gemini_parser.py` - Gemini JSON/HTML exports

### Core (`core/`)
- `config.py` - Typed configuration with validation
- `exceptions.py` - Custom exception hierarchy
- `logging_config.py` - Structured JSON logging

### Enrichment (`enrichment/`)
- `enrichment_pipeline.py` - Claude API integration for generating titles, summaries, tags, scores

### Search
- `search_engine.py` - SQLite FTS5 with BM25 ranking, filters, pagination

### State Management
- `archive_registry.py` - Track archive files, cursors, sync state
- `smart_parser.py` - Auto-detect format, incremental parsing
- `state_manager.py` - Processing state persistence

### Web UI (`cogrepo-ui/`)
- `app.py` - Flask server with 9 API endpoints
- `index.html` - Modern search UI
- `static/css/design-system.css` - CSS custom properties
- `static/js/api.js` - API client with retry logic
- `static/js/ui.js` - Utilities (escapeHTML, modals, toasts)
- `static/js/app.js` - Main app with state management
- `sw.js` - Service worker for offline support

## API Endpoints

```
GET  /api/conversations      - List all (with ?q=, ?source=, ?limit=)
GET  /api/conversation/<id>  - Single conversation
GET  /api/search             - Full-text search with filters
GET  /api/stats              - Repository statistics
GET  /api/tags               - Tag cloud data
GET  /api/sources            - Source breakdown
GET  /api/suggestions        - Autocomplete
POST /api/export             - Export selected conversations
POST /api/upload             - Upload file for import
```

## Data Flow

1. **Import**: `cogrepo_import.py` → parser → enrichment → `data/enriched_repository.jsonl`
2. **Sync**: `archive_registry.py` tracks cursor → `smart_parser.py` extracts new only
3. **Search**: `search_engine.py` indexes JSONL → FTS5 queries
4. **Web**: `app.py` reads JSONL directly (or could use search_engine)

## Key Classes

```python
# smart_parser.py
SmartParser.parse_incremental(file_path, cursor) -> (conversations, new_cursor)

# archive_registry.py
ArchiveRegistry.register(path, source) -> archive_id
ArchiveRegistry.detect_changes() -> list of pending archives

# search_engine.py
SearchEngine.search(query, filters) -> SearchResults
SearchEngine.index_conversations(conversations)

# enrichment/enrichment_pipeline.py
EnrichmentPipeline.enrich_conversation(conv) -> enriched_conv
```

## Configuration

```yaml
# config/enrichment_config.yaml
enrichment:
  batch_size: 10
  model: claude-sonnet-4-20250514
```

```python
# core/config.py
config = get_config()
config.anthropic.api_key  # from ANTHROPIC_API_KEY env var
config.search.use_fts5    # True
```

## Tests

```bash
pytest tests/ -v  # 101 tests, ~1.5s
```

Test files mirror source structure:
- `test_parsers.py` - Parser unit tests
- `test_search_engine.py` - Search/FTS tests
- `test_integration.py` - End-to-end workflows
- `test_config.py` - Configuration validation

## Common Tasks

### Add a new parser
1. Create `parsers/new_parser.py` extending `BaseParser`
2. Implement `detect_format()` and `parse()`
3. Register in `smart_parser.py` PARSER_REGISTRY
4. Add tests in `tests/test_parsers.py`

### Add an API endpoint
1. Add route in `cogrepo-ui/app.py`
2. Update `static/js/api.js` with client method
3. Document in README.md API table

### Debug search issues
1. Check `data/enriched_repository.jsonl` exists
2. Verify JSON structure: `head -1 data/enriched_repository.jsonl | jq .`
3. Test search_engine directly: `python -c "from search_engine import SearchEngine; ..."`

## Environment

```bash
ANTHROPIC_API_KEY=sk-...  # Required for enrichment
COGREPO_PATH=/path/to/data  # Optional, auto-detected
```
