# CogRepo Import Guide

How to export and import conversations from ChatGPT, Claude, and Gemini.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (optional, for AI enrichment)
export ANTHROPIC_API_KEY="sk-ant-..."

# Import conversations
python cogrepo_import.py --source chatgpt --file conversations.json --enrich
```

## Exporting Conversations

### ChatGPT

1. Go to https://chat.openai.com
2. Settings → Data controls → Export data
3. Wait for email, download ZIP
4. Extract `conversations.json`

### Claude

**Option A: Browser Extension**
1. Install [Claude Exporter](https://chromewebstore.google.com/detail/claude-exporter)
2. Open conversation, click extension
3. Export as JSON

**Option B: DevTools**
1. Open https://claude.ai, press F12
2. Network tab, filter: `chat_`
3. Open conversation, copy response

### Gemini

1. Install [Gemini Chat Exporter](https://chromewebstore.google.com/detail/gemini-chat-exporter)
2. Open conversation, click extension
3. Export as JSON

## Import Commands

### First-time import

```bash
# With AI enrichment (~$0.025/conversation)
python cogrepo_import.py --source chatgpt --file conversations.json --enrich

# Without enrichment (free, faster)
python cogrepo_import.py --source chatgpt --file conversations.json

# Auto-detect format
python cogrepo_import.py --file export.json --enrich
```

### Incremental updates

```bash
# Register archive for tracking
python cogrepo_manage.py register ~/Downloads/conversations.json chatgpt

# Sync only new conversations
python quick_sync.py

# Check status
python cogrepo_manage.py status
```

### Flags

| Flag | Description |
|------|-------------|
| `--source` | Platform: `chatgpt`, `claude`, `gemini`, `auto` |
| `--file` | Path to export file |
| `--enrich` | Enable AI enrichment |
| `--dry-run` | Preview without saving |
| `--config` | Custom config file |

## AI Enrichment

When `--enrich` is used, each conversation gets:

- Generated title (5-8 words)
- Summary (250-300 chars)
- Tags (3-10 keywords)
- Quality score (1-10)
- Key topics and insights

### Cost

- ~$0.025 per conversation (Claude Sonnet)
- 50 conversations: ~$1.25
- 100 conversations: ~$2.50

### Performance

- With enrichment: ~2 min per 50 conversations
- Without enrichment: ~5 sec per 1,000 conversations

## Configuration

Edit `config/enrichment_config.yaml`:

```yaml
api:
  use_haiku_for_simple_tasks: true  # Cheaper model for simple tasks

processing:
  batch_size: 50
  rate_limit_rpm: 50

enrichment:
  generate_titles: true
  generate_summaries: true
  extract_tags: true
  calculate_scores: true
```

## Troubleshooting

### "ANTHROPIC_API_KEY not set"

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Or skip enrichment (remove --enrich flag)
```

### "Could not detect format"

Specify source explicitly:
```bash
python cogrepo_import.py --source chatgpt --file export.json
```

### Rate limit errors

Reduce batch size in config:
```yaml
processing:
  batch_size: 25
  rate_limit_rpm: 30
```

### Import interrupted

Safe to re-run - state is saved:
```bash
python cogrepo_import.py --source chatgpt --file export.json --enrich
# Skips already-processed conversations
```

## Workflow

### Initial setup

1. Export from all platforms
2. Import each with `--enrich`
3. Start searching via web UI or CLI

### Weekly updates

```bash
# Export fresh conversations
# Then run:
python quick_sync.py  # Syncs all registered archives
```

Or manually:
```bash
python cogrepo_import.py --source chatgpt --file new_export.json --enrich
```

## File Locations

| File | Purpose |
|------|---------|
| `data/enriched_repository.jsonl` | Main conversation database |
| `data/archives/registry.json` | Tracked archives and cursors |
| `data/processing_state.json` | Import state (for resume) |
