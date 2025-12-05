"""
Unified Search Engine for CogRepo

High-performance search with:
- SQLite Full-Text Search (FTS5) for instant queries
- BM25 ranking for relevance
- Semantic search with embeddings (optional)
- Advanced query parsing (+required, -excluded, "phrases")
- Faceted filtering (source, date, score, tags)
- Result highlighting

This replaces the scattered search implementations with a single,
optimized, well-tested search system.

Usage:
    from search_engine import SearchEngine

    engine = SearchEngine()

    # Index conversations (one-time or on new data)
    engine.index_conversations(conversations)

    # Search with ranking
    results = engine.search("AI animation", filters={"source": "OpenAI", "min_score": 7})

    # Get suggestions
    suggestions = engine.suggest("anim")
"""

import sqlite3
import json
import re
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading

import sys
sys.path.append(str(Path(__file__).parent))

from core.logging_config import get_logger
from core.exceptions import SearchError, IndexNotAvailableError, DatabaseError

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SearchQuery:
    """Parsed search query with operators."""
    original: str
    required: List[str] = field(default_factory=list)      # +term
    excluded: List[str] = field(default_factory=list)      # -term
    phrases: List[str] = field(default_factory=list)       # "exact phrase"
    regular: List[str] = field(default_factory=list)       # normal terms

    @property
    def is_empty(self) -> bool:
        return not (self.required or self.excluded or self.phrases or self.regular)

    def to_fts_query(self) -> str:
        """Convert to FTS5 query syntax."""
        parts = []

        # Required terms (AND)
        for term in self.required:
            parts.append(f'"{term}"')

        # Exact phrases
        for phrase in self.phrases:
            parts.append(f'"{phrase}"')

        # Regular terms (OR by default, boost with *)
        for term in self.regular:
            # Use prefix matching for flexibility
            parts.append(f'{term}*')

        # Excluded terms (NOT)
        for term in self.excluded:
            parts.append(f'NOT "{term}"')

        return ' '.join(parts) if parts else '*'


@dataclass
class SearchResult:
    """Single search result with metadata."""
    convo_id: str
    external_id: str
    title: str
    summary: str
    source: str
    timestamp: str
    score: float                # Quality score (1-10)
    relevance: float            # Search relevance score
    tags: List[str]
    highlights: Dict[str, str] = field(default_factory=dict)  # Field -> highlighted text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "convo_id": self.convo_id,
            "external_id": self.external_id,
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "timestamp": self.timestamp,
            "score": self.score,
            "relevance": round(self.relevance, 4),
            "tags": self.tags,
            "highlights": self.highlights
        }


@dataclass
class SearchFilters:
    """Search filters for narrowing results."""
    source: Optional[str] = None           # OpenAI, Anthropic, Google
    date_from: Optional[str] = None        # YYYY-MM-DD
    date_to: Optional[str] = None          # YYYY-MM-DD
    min_score: Optional[int] = None        # Minimum quality score
    max_score: Optional[int] = None        # Maximum quality score
    tags: Optional[List[str]] = None       # Required tags (any)
    domain: Optional[str] = None           # Primary domain filter

    def to_sql_conditions(self) -> Tuple[str, List[Any]]:
        """Convert to SQL WHERE conditions."""
        conditions = []
        params = []

        if self.source:
            conditions.append("c.source = ?")
            params.append(self.source)

        if self.date_from:
            conditions.append("c.timestamp >= ?")
            params.append(self.date_from)

        if self.date_to:
            conditions.append("c.timestamp <= ?")
            params.append(self.date_to + "T23:59:59")

        if self.min_score is not None:
            conditions.append("c.score >= ?")
            params.append(self.min_score)

        if self.max_score is not None:
            conditions.append("c.score <= ?")
            params.append(self.max_score)

        if self.domain:
            conditions.append("c.primary_domain = ?")
            params.append(self.domain)

        # Tags require JSON checking
        if self.tags:
            tag_conditions = []
            for tag in self.tags:
                tag_conditions.append("c.tags LIKE ?")
                params.append(f'%"{tag}"%')
            conditions.append(f"({' OR '.join(tag_conditions)})")

        sql = " AND ".join(conditions) if conditions else "1=1"
        return sql, params


@dataclass
class SearchStats:
    """Search statistics and facets."""
    total_results: int
    query_time_ms: float
    sources: Dict[str, int] = field(default_factory=dict)       # Source -> count
    domains: Dict[str, int] = field(default_factory=dict)       # Domain -> count
    score_distribution: Dict[str, int] = field(default_factory=dict)  # Score range -> count
    top_tags: List[Tuple[str, int]] = field(default_factory=list)    # Tag -> count


# =============================================================================
# Query Parser
# =============================================================================

class QueryParser:
    """
    Parse search queries with advanced operators.

    Supported operators:
    - +term: Required (must appear)
    - -term: Excluded (must not appear)
    - "exact phrase": Exact phrase match
    - term: Regular term (ranked by relevance)
    """

    # Pattern for extracting quoted phrases
    PHRASE_PATTERN = re.compile(r'"([^"]*)"')

    def parse(self, query: str) -> SearchQuery:
        """Parse query string into SearchQuery object."""
        if not query or not query.strip():
            return SearchQuery(original="")

        original = query.strip()
        required = []
        excluded = []
        phrases = []
        regular = []

        # Extract exact phrases first
        phrases = [p.lower().strip() for p in self.PHRASE_PATTERN.findall(query) if p.strip()]

        # Remove phrases from query
        query_no_phrases = self.PHRASE_PATTERN.sub('', query)

        # Parse remaining terms
        for word in query_no_phrases.split():
            word = word.strip()
            if not word:
                continue

            if word.startswith('+') and len(word) > 1:
                required.append(word[1:].lower())
            elif word.startswith('-') and len(word) > 1:
                excluded.append(word[1:].lower())
            else:
                # Clean and add regular term
                cleaned = re.sub(r'[^\w\s]', '', word).lower()
                if cleaned:
                    regular.append(cleaned)

        return SearchQuery(
            original=original,
            required=required,
            excluded=excluded,
            phrases=phrases,
            regular=regular
        )


# =============================================================================
# Search Engine
# =============================================================================

class SearchEngine:
    """
    Unified search engine with SQLite FTS5.

    Provides:
    - Fast full-text search with BM25 ranking
    - Advanced query operators
    - Faceted filtering
    - Result highlighting
    - Automatic index management
    """

    # SQLite FTS5 configuration
    FTS_CONFIG = "tokenize='porter unicode61'"  # Stemming + unicode support

    def __init__(
        self,
        db_path: str = "data/search_index.db",
        auto_create: bool = True
    ):
        """
        Initialize search engine.

        Args:
            db_path: Path to SQLite database
            auto_create: Create database if not exists
        """
        self.db_path = Path(__file__).parent / db_path
        self.query_parser = QueryParser()
        self._local = threading.local()  # Thread-local storage for connections

        if auto_create:
            self._ensure_database()

    @contextmanager
    def _get_connection(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable FTS5
            self._local.conn.execute("PRAGMA journal_mode=WAL")

        try:
            yield self._local.conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise DatabaseError(str(e), cause=e)

    def _ensure_database(self):
        """Create database and tables if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            # Main conversations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    convo_id TEXT PRIMARY KEY,
                    external_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    title TEXT,
                    summary TEXT,
                    raw_text TEXT,
                    primary_domain TEXT,
                    tags TEXT,
                    score INTEGER DEFAULT 5,
                    indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON conversations(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON conversations(score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON conversations(primary_domain)")

            # FTS5 virtual table for full-text search
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
                    convo_id,
                    title,
                    summary,
                    raw_text,
                    tags,
                    content='conversations',
                    content_rowid='rowid',
                    {self.FTS_CONFIG}
                )
            """)

            # Triggers to keep FTS in sync
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
                    INSERT INTO conversations_fts(rowid, convo_id, title, summary, raw_text, tags)
                    VALUES (new.rowid, new.convo_id, new.title, new.summary, new.raw_text, new.tags);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations BEGIN
                    INSERT INTO conversations_fts(conversations_fts, rowid, convo_id, title, summary, raw_text, tags)
                    VALUES ('delete', old.rowid, old.convo_id, old.title, old.summary, old.raw_text, old.tags);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS conversations_au AFTER UPDATE ON conversations BEGIN
                    INSERT INTO conversations_fts(conversations_fts, rowid, convo_id, title, summary, raw_text, tags)
                    VALUES ('delete', old.rowid, old.convo_id, old.title, old.summary, old.raw_text, old.tags);
                    INSERT INTO conversations_fts(rowid, convo_id, title, summary, raw_text, tags)
                    VALUES (new.rowid, new.convo_id, new.title, new.summary, new.raw_text, new.tags);
                END
            """)

            conn.commit()
            logger.info("Search database initialized", extra={"db_path": str(self.db_path)})

    def index_conversation(self, conv: Dict[str, Any]) -> bool:
        """
        Index a single conversation.

        Args:
            conv: Conversation dictionary (EnrichedConversation format)

        Returns:
            True if indexed successfully
        """
        try:
            with self._get_connection() as conn:
                # Extract fields
                tags_json = json.dumps(conv.get('tags', []))

                conn.execute("""
                    INSERT OR REPLACE INTO conversations
                    (convo_id, external_id, source, timestamp, title, summary,
                     raw_text, primary_domain, tags, score, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    conv.get('convo_id'),
                    conv.get('external_id'),
                    conv.get('source'),
                    conv.get('timestamp'),
                    conv.get('generated_title', conv.get('title', '')),
                    conv.get('summary_abstractive', ''),
                    conv.get('raw_text', '')[:50000],  # Limit raw text size
                    conv.get('primary_domain', ''),
                    tags_json,
                    conv.get('score', 5),
                    datetime.now().isoformat()
                ))
                conn.commit()
                return True

        except Exception as e:
            logger.warning(f"Failed to index conversation {conv.get('convo_id')}: {e}")
            return False

    def index_conversations(
        self,
        conversations: Iterator[Dict[str, Any]],
        batch_size: int = 100,
        progress_callback: callable = None
    ) -> int:
        """
        Bulk index conversations.

        Args:
            conversations: Iterator of conversation dictionaries
            batch_size: Commit every N conversations
            progress_callback: Optional callback(indexed, failed)

        Returns:
            Number of successfully indexed conversations
        """
        indexed = 0
        failed = 0
        batch = []

        with self._get_connection() as conn:
            for conv in conversations:
                try:
                    tags_json = json.dumps(conv.get('tags', []))
                    batch.append((
                        conv.get('convo_id'),
                        conv.get('external_id'),
                        conv.get('source'),
                        conv.get('timestamp'),
                        conv.get('generated_title', conv.get('title', '')),
                        conv.get('summary_abstractive', ''),
                        conv.get('raw_text', '')[:50000],
                        conv.get('primary_domain', ''),
                        tags_json,
                        conv.get('score', 5),
                        datetime.now().isoformat()
                    ))

                    if len(batch) >= batch_size:
                        conn.executemany("""
                            INSERT OR REPLACE INTO conversations
                            (convo_id, external_id, source, timestamp, title, summary,
                             raw_text, primary_domain, tags, score, indexed_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, batch)
                        conn.commit()
                        indexed += len(batch)
                        batch = []

                        if progress_callback:
                            progress_callback(indexed, failed)

                except Exception as e:
                    failed += 1
                    logger.warning(f"Failed to index conversation: {e}")

            # Final batch
            if batch:
                conn.executemany("""
                    INSERT OR REPLACE INTO conversations
                    (convo_id, external_id, source, timestamp, title, summary,
                     raw_text, primary_domain, tags, score, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                conn.commit()
                indexed += len(batch)

        logger.info(
            f"Indexing complete",
            extra={"indexed": indexed, "failed": failed}
        )
        return indexed

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
        include_stats: bool = False
    ) -> Dict[str, Any]:
        """
        Search conversations with ranking.

        Args:
            query: Search query (supports +required, -excluded, "phrases")
            filters: Optional filters (source, date_from, date_to, min_score, tags)
            limit: Maximum results to return
            offset: Pagination offset
            include_stats: Include faceted statistics

        Returns:
            Dictionary with results, total_count, and optionally stats
        """
        import time
        start_time = time.perf_counter()

        # Parse query
        parsed = self.query_parser.parse(query)

        # Build filters
        search_filters = SearchFilters(
            source=filters.get('source') if filters else None,
            date_from=filters.get('date_from') or filters.get('from') if filters else None,
            date_to=filters.get('date_to') or filters.get('to') if filters else None,
            min_score=filters.get('min_score') if filters else None,
            max_score=filters.get('max_score') if filters else None,
            tags=filters.get('tags') if filters else None,
            domain=filters.get('domain') if filters else None
        )

        filter_sql, filter_params = search_filters.to_sql_conditions()

        with self._get_connection() as conn:
            if parsed.is_empty:
                # No query - return filtered results sorted by date
                sql = f"""
                    SELECT c.*, 0 as relevance
                    FROM conversations c
                    WHERE {filter_sql}
                    ORDER BY c.timestamp DESC
                    LIMIT ? OFFSET ?
                """
                params = filter_params + [limit, offset]

            else:
                # Full-text search with BM25 ranking
                fts_query = parsed.to_fts_query()

                sql = f"""
                    SELECT c.*, bm25(conversations_fts, 1, 5, 2, 1, 3) as relevance
                    FROM conversations c
                    INNER JOIN conversations_fts fts ON c.convo_id = fts.convo_id
                    WHERE conversations_fts MATCH ?
                    AND {filter_sql}
                    ORDER BY relevance
                    LIMIT ? OFFSET ?
                """
                params = [fts_query] + filter_params + [limit, offset]

            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as e:
                # FTS query syntax error - fallback to LIKE search
                logger.warning(f"FTS query failed, falling back to LIKE: {e}")
                rows = self._fallback_search(conn, parsed, filter_sql, filter_params, limit, offset)

            # Convert to SearchResult objects
            results = []
            for row in rows:
                tags = json.loads(row['tags']) if row['tags'] else []
                results.append(SearchResult(
                    convo_id=row['convo_id'],
                    external_id=row['external_id'],
                    title=row['title'] or 'Untitled',
                    summary=row['summary'] or '',
                    source=row['source'],
                    timestamp=row['timestamp'][:10] if row['timestamp'] else '',
                    score=row['score'] or 5,
                    relevance=abs(row['relevance']) if row['relevance'] else 0,
                    tags=tags[:5],
                    highlights=self._generate_highlights(row, parsed)
                ))

            # Get total count
            if parsed.is_empty:
                count_sql = f"SELECT COUNT(*) FROM conversations c WHERE {filter_sql}"
                count_params = filter_params
            else:
                count_sql = f"""
                    SELECT COUNT(*) FROM conversations c
                    INNER JOIN conversations_fts fts ON c.convo_id = fts.convo_id
                    WHERE conversations_fts MATCH ? AND {filter_sql}
                """
                count_params = [fts_query] + filter_params

            try:
                total_count = conn.execute(count_sql, count_params).fetchone()[0]
            except (sqlite3.Error, TypeError):
                total_count = len(results)

        query_time = (time.perf_counter() - start_time) * 1000

        response = {
            "results": [r.to_dict() for r in results],
            "total_count": total_count,
            "query": query,
            "filters": filters,
            "query_time_ms": round(query_time, 2)
        }

        if include_stats:
            response["stats"] = self._get_search_stats(conn, parsed, search_filters)

        logger.debug(
            f"Search completed",
            extra={
                "query": query,
                "results": len(results),
                "total": total_count,
                "time_ms": round(query_time, 2)
            }
        )

        return response

    def _fallback_search(
        self,
        conn: sqlite3.Connection,
        parsed: SearchQuery,
        filter_sql: str,
        filter_params: List,
        limit: int,
        offset: int
    ) -> List[sqlite3.Row]:
        """Fallback LIKE-based search when FTS fails."""
        conditions = [filter_sql]
        params = list(filter_params)

        # Build LIKE conditions
        all_terms = parsed.required + parsed.regular + parsed.phrases
        if all_terms:
            term_conditions = []
            for term in all_terms:
                term_conditions.append(
                    "(c.title LIKE ? OR c.summary LIKE ? OR c.raw_text LIKE ?)"
                )
                like_term = f"%{term}%"
                params.extend([like_term, like_term, like_term])
            conditions.append(f"({' AND '.join(term_conditions)})")

        # Excluded terms
        for term in parsed.excluded:
            conditions.append("c.raw_text NOT LIKE ?")
            params.append(f"%{term}%")

        where_clause = " AND ".join(conditions)

        sql = f"""
            SELECT c.*, 0 as relevance
            FROM conversations c
            WHERE {where_clause}
            ORDER BY c.timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        return conn.execute(sql, params).fetchall()

    def _generate_highlights(
        self,
        row: sqlite3.Row,
        parsed: SearchQuery
    ) -> Dict[str, str]:
        """Generate highlighted snippets for search terms."""
        highlights = {}
        terms = parsed.required + parsed.regular + parsed.phrases

        if not terms:
            return highlights

        # Highlight in title
        title = row['title'] or ''
        for term in terms:
            if term.lower() in title.lower():
                # Simple highlighting with **bold**
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlights['title'] = pattern.sub(f'**{term}**', title)
                break

        # Highlight in summary (find relevant snippet)
        summary = row['summary'] or ''
        for term in terms:
            pos = summary.lower().find(term.lower())
            if pos >= 0:
                start = max(0, pos - 50)
                end = min(len(summary), pos + len(term) + 100)
                snippet = summary[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(summary):
                    snippet = snippet + "..."
                highlights['summary'] = snippet
                break

        return highlights

    def _get_search_stats(
        self,
        conn: sqlite3.Connection,
        parsed: SearchQuery,
        filters: SearchFilters
    ) -> Dict[str, Any]:
        """Get faceted statistics for search results."""
        # This would be expensive for large result sets
        # Consider caching or pre-computing
        return {
            "sources": {},
            "domains": {},
            "score_distribution": {}
        }

    def suggest(self, prefix: str, limit: int = 10) -> List[str]:
        """
        Get autocomplete suggestions.

        Args:
            prefix: Search prefix
            limit: Maximum suggestions

        Returns:
            List of suggested terms/titles
        """
        with self._get_connection() as conn:
            # Suggest from titles
            rows = conn.execute("""
                SELECT DISTINCT title FROM conversations
                WHERE title LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (f"{prefix}%", limit)).fetchall()

            return [row['title'] for row in rows if row['title']]

    def get_conversation(self, convo_id: str) -> Optional[Dict[str, Any]]:
        """Get full conversation by ID."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM conversations WHERE convo_id = ?
            """, (convo_id,)).fetchone()

            if row:
                return {
                    "convo_id": row['convo_id'],
                    "external_id": row['external_id'],
                    "source": row['source'],
                    "timestamp": row['timestamp'],
                    "title": row['title'],
                    "summary": row['summary'],
                    "raw_text": row['raw_text'],
                    "primary_domain": row['primary_domain'],
                    "tags": json.loads(row['tags']) if row['tags'] else [],
                    "score": row['score']
                }
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]

            sources = dict(conn.execute("""
                SELECT source, COUNT(*) FROM conversations GROUP BY source
            """).fetchall())

            domains = dict(conn.execute("""
                SELECT primary_domain, COUNT(*) FROM conversations
                WHERE primary_domain IS NOT NULL AND primary_domain != ''
                GROUP BY primary_domain
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """).fetchall())

            avg_score = conn.execute(
                "SELECT AVG(score) FROM conversations"
            ).fetchone()[0]

            return {
                "total_conversations": total,
                "by_source": sources,
                "top_domains": domains,
                "average_score": round(avg_score, 2) if avg_score else 0,
                "index_path": str(self.db_path),
                "index_size_mb": round(self.db_path.stat().st_size / 1024 / 1024, 2) if self.db_path.exists() else 0
            }

    def rebuild_index(self) -> int:
        """
        Rebuild search index from JSONL repository.

        Returns:
            Number of indexed conversations
        """
        repo_path = self.db_path.parent / "enriched_repository.jsonl"

        if not repo_path.exists():
            logger.warning(f"Repository file not found: {repo_path}")
            return 0

        def conversation_iterator():
            with open(repo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

        return self.index_conversations(conversation_iterator())

    def optimize(self):
        """Optimize the search index."""
        with self._get_connection() as conn:
            conn.execute("INSERT INTO conversations_fts(conversations_fts) VALUES('optimize')")
            conn.execute("VACUUM")
            conn.commit()
        logger.info("Search index optimized")


# =============================================================================
# Convenience Functions
# =============================================================================

_default_engine: Optional[SearchEngine] = None


def get_search_engine() -> SearchEngine:
    """Get the default search engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = SearchEngine()
    return _default_engine


def quick_search(query: str, **filters) -> List[Dict[str, Any]]:
    """
    Quick search convenience function.

    Args:
        query: Search query
        **filters: Optional filters (source, date_from, date_to, min_score)

    Returns:
        List of result dictionaries
    """
    engine = get_search_engine()
    results = engine.search(query, filters=filters if filters else None)
    return results.get('results', [])


if __name__ == "__main__":
    # CLI for testing
    import argparse

    parser = argparse.ArgumentParser(description="CogRepo Search Engine")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from JSONL")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument("--source", help="Filter by source")
    parser.add_argument("--min-score", type=int, help="Minimum score filter")

    args = parser.parse_args()

    engine = SearchEngine()

    if args.rebuild:
        print("Rebuilding search index...")
        count = engine.rebuild_index()
        print(f"Indexed {count} conversations")

    elif args.stats:
        stats = engine.get_stats()
        print("\n=== Search Index Statistics ===")
        print(f"Total conversations: {stats['total_conversations']}")
        print(f"Average score: {stats['average_score']}")
        print(f"Index size: {stats['index_size_mb']} MB")
        print("\nBy source:")
        for source, count in stats['by_source'].items():
            print(f"  {source}: {count}")

    elif args.query:
        filters = {}
        if args.source:
            filters['source'] = args.source
        if args.min_score:
            filters['min_score'] = args.min_score

        results = engine.search(args.query, filters=filters if filters else None)

        print(f"\n=== Search Results for '{args.query}' ===")
        print(f"Found {results['total_count']} results in {results['query_time_ms']}ms\n")

        for i, r in enumerate(results['results'][:10], 1):
            print(f"{i}. [{r['source']}] {r['title']}")
            print(f"   Score: {r['score']} | Relevance: {r['relevance']}")
            print(f"   {r['summary'][:100]}...")
            print()

    else:
        parser.print_help()
