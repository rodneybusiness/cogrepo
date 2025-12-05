# CogRepo Web UI Guide

Modern web interface for uploading, searching, and browsing AI conversations.

## Quick Start

**macOS**: Double-click `CogRepo.command` or search "CogRepo" in Spotlight/Alfred/Raycast (auto-opens browser).

**Command line**:
```bash
cd cogrepo-ui
python app.py
# Open http://localhost:5000
```

## Features

- **Search**: Full-text search with filters (source, date, score)
- **Keyboard shortcuts**: ⌘K (search), J/K (navigate), ⌘S (save), ⌘E (export)
- **Offline support**: Service worker caches for offline use
- **Dark mode**: Respects system preference
- **Responsive**: Works on desktop and mobile

## Pages

| URL | Purpose |
|-----|---------|
| `/` | Search interface |
| `/upload.html` | File upload |
| `/api/status` | Server health check |

## Uploading Conversations

1. Open http://localhost:5000/upload.html
2. Drag & drop export file (or click to browse)
3. Select source (auto-detect, ChatGPT, Claude, Gemini)
4. Toggle AI enrichment (requires `ANTHROPIC_API_KEY`)
5. Click "Start Import"
6. Watch real-time progress via WebSocket

## API Endpoints

### Search & Browse

```bash
# List conversations
GET /api/conversations?q=query&source=OpenAI&limit=50

# Get single conversation
GET /api/conversation/{id}

# Full-text search with filters
GET /api/search?q=query&source=OpenAI&date_from=2024-01-01&min_score=7

# Semantic search (same params)
GET /api/semantic_search?q=query

# Autocomplete suggestions
GET /api/suggestions?q=mach
```

### Metadata

```bash
# Repository statistics
GET /api/stats
# Returns: total_conversations, sources, date_range, avg_score, top_tags

# All tags with counts
GET /api/tags

# All sources with counts
GET /api/sources
```

### Export & Import

```bash
# Export selected conversations
POST /api/export
Body: {"conversation_ids": ["id1", "id2"], "format": "json"}

# Upload file for import
POST /api/upload
Form: file, source (auto|chatgpt|claude|gemini), enrich (true|false)

# List imports
GET /api/imports

# Get import status
GET /api/imports/{import_id}
```

## WebSocket Events

Real-time updates during import:

| Event | Direction | Data |
|-------|-----------|------|
| `connected` | Server→Client | Connection confirmed |
| `import_progress` | Server→Client | `{current, total, percentage, status}` |
| `import_complete` | Server→Client | `{stats, message}` |
| `import_error` | Server→Client | `{error, message}` |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| ⌘K / Ctrl+K | Focus search |
| ⌘S / Ctrl+S | Save current search |
| ⌘E / Ctrl+E | Export results |
| J | Next result |
| K | Previous result |
| Enter | Open selected |
| Escape | Close modal |
| ? | Show shortcuts help |

## Configuration

Server runs on port 5000 by default. To change:

```python
# In cogrepo-ui/app.py, line ~407
socketio.run(app, host='0.0.0.0', port=5001)  # Change port
```

## Troubleshooting

### Server won't start

```bash
pip install flask flask-socketio flask-cors python-dotenv
```

### API key error on upload

Either:
- Set `ANTHROPIC_API_KEY` in `.env` file
- Uncheck "Enable AI enrichment" during upload

### Port in use

```bash
lsof -i :5000  # Find process
kill -9 <PID>  # Kill it
```

### No search results

1. Check data file exists: `ls data/enriched_repository.jsonl`
2. Verify it has content: `wc -l data/enriched_repository.jsonl`
3. Check server logs for errors

## Architecture

```
cogrepo-ui/
├── app.py              # Flask server (API + WebSocket)
├── index.html          # Search UI (modern redesign)
├── upload.html         # Upload interface
├── sw.js               # Service worker (offline)
├── static/
│   ├── css/
│   │   ├── design-system.css  # CSS tokens, components
│   │   └── components.css     # App-specific styles
│   └── js/
│       ├── api.js      # API client (retry, cancellation)
│       ├── ui.js       # Utilities (modals, toasts, escapeHTML)
│       └── app.js      # Main app (state, search, render)
└── uploads/            # Temporary upload storage
```
