# CogRepo v2 Planning Documentation

> **Status:** Plan approved, ready for implementation
> **Created:** December 2024

This directory contains the complete planning documentation for CogRepo v2, a major upgrade transforming CogRepo from a search tool into a comprehensive knowledge system.

## Documents

| Document | Purpose |
|----------|---------|
| [COGREPO_V2_MASTER_PLAN.md](./COGREPO_V2_MASTER_PLAN.md) | Complete overview of the v2 upgrade |
| [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) | Step-by-step implementation instructions |
| [CONTINUATION_GUIDE.md](./CONTINUATION_GUIDE.md) | How to resume this project later |
| [ENRICHMENT_SCHEMA.md](./ENRICHMENT_SCHEMA.md) | Complete data structure reference |

## Quick Start

### If You're Starting Fresh

1. Read [COGREPO_V2_MASTER_PLAN.md](./COGREPO_V2_MASTER_PLAN.md) for the big picture
2. Follow [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) Phase 0

### If You're Resuming

1. Read [CONTINUATION_GUIDE.md](./CONTINUATION_GUIDE.md)
2. Check the status table at the bottom of that file
3. Continue from where we left off

## Key Features of v2

1. **SQLite Database** — 10-100x faster search
2. **Semantic Search** — Local embeddings, zero API cost
3. **Artifact Extraction** — Code, commands, solutions directly usable
4. **Project Grouping** — Auto-detected project contexts
5. **Conversation Chains** — Linked problem-solving journeys
6. **Knowledge Graph** — Entity relationships
7. **Secure Config** — API keys in `~/.cogrepo/`, not plaintext

## Estimated Costs

| Phase | Token Cost |
|-------|------------|
| 0 (Foundation + Enrichment) | ~$9 total |
| 1-5 (All other phases) | $0 |

The $9 is a one-time cost for backfilling existing conversations with new enrichment.

## Timeline

Each phase is independent and can be completed separately. No time estimates provided — work at your own pace.

## Questions?

If something is unclear:

1. Check the relevant document in this directory
2. Check the main [CLAUDE.md](../../CLAUDE.md) for codebase navigation
3. Check [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) (to be created in Phase 0.7)
