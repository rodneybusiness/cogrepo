"""
Database Layer for CogRepo v2

SQLite-based storage with FTS5 full-text search.

Features:
- Fast indexed queries
- Full-text search with ranking
- Artifact storage
- Migration from JSONL

Usage:
    from database import ConversationRepository

    repo = ConversationRepository()
    repo.save(conversation)
    results = repo.search("python flask error", limit=10)
"""

from .repository import ConversationRepository

__all__ = [
    'ConversationRepository',
]
