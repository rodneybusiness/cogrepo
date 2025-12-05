"""
Hybrid Search for CogRepo v2

Combines BM25 (SQLite FTS5) and semantic search (embeddings) for
optimal search results. Uses Reciprocal Rank Fusion (RRF) to merge.

Features:
- Parallel BM25 and semantic search
- Configurable weighting
- Result deduplication and merging
- Score normalization

Usage:
    from search.hybrid_search import HybridSearcher

    searcher = HybridSearcher(db_path, embedding_store)
    results = searcher.search("react hooks error", limit=10)

    for result in results:
        print(f"{result.score:.2f} - {result.title}")
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import sqlite3
import numpy as np

from .embeddings import EmbeddingEngine, EmbeddingStore


@dataclass
class SearchResult:
    """A search result with combined score."""
    convo_id: str
    title: str
    summary: str
    score: float
    bm25_score: float
    semantic_score: float
    matched_terms: List[str]
    source: str
    created_at: str
    tags: List[str]

    def to_dict(self) -> dict:
        return {
            'convo_id': self.convo_id,
            'title': self.title,
            'summary': self.summary,
            'score': self.score,
            'bm25_score': self.bm25_score,
            'semantic_score': self.semantic_score,
            'matched_terms': self.matched_terms,
            'source': self.source,
            'created_at': self.created_at,
            'tags': self.tags,
        }


class HybridSearcher:
    """
    Hybrid search combining BM25 and semantic search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from
    both search methods.

    RRF formula: score = sum(1 / (k + rank)) for each method
    where k is a constant (default 60) to prevent over-emphasis on top results.
    """

    def __init__(
        self,
        db_path: str,
        embedding_store: Optional[EmbeddingStore] = None,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid searcher.

        Args:
            db_path: Path to SQLite database
            embedding_store: Optional embedding store for semantic search
            bm25_weight: Weight for BM25 results (0-1)
            semantic_weight: Weight for semantic results (0-1)
            rrf_k: RRF constant (higher = more even distribution)
        """
        self.db_path = Path(db_path)
        self.embedding_store = embedding_store
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.rrf_k = rrf_k
        self._engine = None

    @property
    def engine(self) -> EmbeddingEngine:
        """Lazy load embedding engine."""
        if self._engine is None:
            self._engine = EmbeddingEngine()
        return self._engine

    def search(
        self,
        query: str,
        limit: int = 10,
        source_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        min_score: float = 0.0,
        semantic_only: bool = False,
        bm25_only: bool = False
    ) -> List[SearchResult]:
        """
        Search using hybrid BM25 + semantic approach.

        Args:
            query: Search query
            limit: Maximum results to return
            source_filter: Filter by source (OpenAI, Anthropic, Google)
            tag_filter: Filter by tags
            min_score: Minimum combined score
            semantic_only: Use only semantic search
            bm25_only: Use only BM25 search

        Returns:
            List of SearchResult objects sorted by combined score
        """
        if semantic_only:
            return self._semantic_search(query, limit * 2, source_filter, tag_filter)[:limit]

        if bm25_only:
            return self._bm25_search(query, limit * 2, source_filter, tag_filter)[:limit]

        # Run both searches
        bm25_results = self._bm25_search(query, limit * 3, source_filter, tag_filter)
        semantic_results = self._semantic_search(query, limit * 3, source_filter, tag_filter)

        # Merge with RRF
        merged = self._merge_rrf(bm25_results, semantic_results)

        # Apply score filter and limit
        filtered = [r for r in merged if r.score >= min_score]
        return filtered[:limit]

    def _bm25_search(
        self,
        query: str,
        limit: int,
        source_filter: Optional[str],
        tag_filter: Optional[List[str]]
    ) -> List[SearchResult]:
        """Run BM25 search using SQLite FTS5."""
        if not self.db_path.exists():
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Build FTS query
            fts_query = self._build_fts_query(query)

            sql = """
                SELECT
                    c.convo_id,
                    c.generated_title,
                    c.summary_abstractive,
                    c.source,
                    c.created_at,
                    c.tags,
                    bm25(conversations_fts) as rank
                FROM conversations_fts
                JOIN conversations c ON conversations_fts.rowid = c.id
                WHERE conversations_fts MATCH ?
            """

            params: List = [fts_query]

            if source_filter:
                sql += " AND c.source = ?"
                params.append(source_filter)

            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

            results = []
            for i, row in enumerate(rows):
                # Normalize BM25 score (lower is better, convert to 0-1 scale)
                raw_score = abs(row['rank'])
                normalized_score = 1.0 / (1.0 + raw_score / 10.0)

                result = SearchResult(
                    convo_id=row['convo_id'],
                    title=row['generated_title'] or '',
                    summary=row['summary_abstractive'] or '',
                    score=normalized_score,
                    bm25_score=normalized_score,
                    semantic_score=0.0,
                    matched_terms=self._extract_matched_terms(query, row),
                    source=row['source'] or '',
                    created_at=row['created_at'] or '',
                    tags=self._parse_tags(row['tags']),
                )

                # Apply tag filter
                if tag_filter:
                    if not any(t in result.tags for t in tag_filter):
                        continue

                results.append(result)

            return results

        except Exception as e:
            print(f"BM25 search error: {e}")
            return []

        finally:
            conn.close()

    def _semantic_search(
        self,
        query: str,
        limit: int,
        source_filter: Optional[str],
        tag_filter: Optional[List[str]]
    ) -> List[SearchResult]:
        """Run semantic search using embeddings."""
        if not self.embedding_store or self.embedding_store.embeddings is None:
            return []

        # Generate query embedding
        query_vec = self.engine.embed(query)

        # Find similar
        similar = self.engine.find_similar(
            query_vec,
            self.embedding_store.embeddings,
            top_k=limit * 2,
            threshold=0.3
        )

        if not similar:
            return []

        # Get conversation details from database
        if not self.db_path.exists():
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            results = []

            for idx, score in similar:
                convo_id = self.embedding_store.ids[idx]

                cursor = conn.execute("""
                    SELECT convo_id, generated_title, summary_abstractive,
                           source, created_at, tags
                    FROM conversations
                    WHERE convo_id = ?
                """, (convo_id,))

                row = cursor.fetchone()
                if not row:
                    continue

                # Apply source filter
                if source_filter and row['source'] != source_filter:
                    continue

                result = SearchResult(
                    convo_id=row['convo_id'],
                    title=row['generated_title'] or '',
                    summary=row['summary_abstractive'] or '',
                    score=score,
                    bm25_score=0.0,
                    semantic_score=score,
                    matched_terms=[],
                    source=row['source'] or '',
                    created_at=row['created_at'] or '',
                    tags=self._parse_tags(row['tags']),
                )

                # Apply tag filter
                if tag_filter:
                    if not any(t in result.tags for t in tag_filter):
                        continue

                results.append(result)

            return results

        finally:
            conn.close()

    def _merge_rrf(
        self,
        bm25_results: List[SearchResult],
        semantic_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Merge results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each ranking method
        """
        # Create ID to result mapping
        results_map: Dict[str, SearchResult] = {}
        rrf_scores: Dict[str, float] = {}

        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            rrf_score = self.bm25_weight / (self.rrf_k + rank)
            results_map[result.convo_id] = result
            rrf_scores[result.convo_id] = rrf_score

        # Process semantic results
        for rank, result in enumerate(semantic_results, 1):
            rrf_score = self.semantic_weight / (self.rrf_k + rank)

            if result.convo_id in results_map:
                # Combine scores
                rrf_scores[result.convo_id] += rrf_score
                existing = results_map[result.convo_id]
                # Update with semantic score
                results_map[result.convo_id] = SearchResult(
                    convo_id=existing.convo_id,
                    title=existing.title,
                    summary=existing.summary,
                    score=rrf_scores[result.convo_id],
                    bm25_score=existing.bm25_score,
                    semantic_score=result.semantic_score,
                    matched_terms=existing.matched_terms,
                    source=existing.source,
                    created_at=existing.created_at,
                    tags=existing.tags,
                )
            else:
                results_map[result.convo_id] = result
                rrf_scores[result.convo_id] = rrf_score

        # Update final scores and sort
        final_results = []
        for convo_id, result in results_map.items():
            final_result = SearchResult(
                convo_id=result.convo_id,
                title=result.title,
                summary=result.summary,
                score=rrf_scores[convo_id],
                bm25_score=result.bm25_score,
                semantic_score=result.semantic_score,
                matched_terms=result.matched_terms,
                source=result.source,
                created_at=result.created_at,
                tags=result.tags,
            )
            final_results.append(final_result)

        return sorted(final_results, key=lambda r: r.score, reverse=True)

    def _build_fts_query(self, query: str) -> str:
        """
        Build FTS5 query from user query.

        Handles:
        - Multi-word queries
        - Phrase matching with quotes
        - Boolean operators
        """
        # Simple approach: treat as OR query for each term
        terms = query.strip().split()

        if not terms:
            return '*'

        # Escape special FTS characters
        escaped = []
        for term in terms:
            # Remove problematic characters
            clean = ''.join(c for c in term if c.isalnum() or c in '-_')
            if clean:
                escaped.append(clean)

        if not escaped:
            return '*'

        # Use OR for better recall
        return ' OR '.join(escaped)

    def _extract_matched_terms(self, query: str, row: sqlite3.Row) -> List[str]:
        """Extract terms from query that appear in result."""
        query_terms = set(query.lower().split())
        matched = []

        text = f"{row['generated_title'] or ''} {row['summary_abstractive'] or ''}".lower()

        for term in query_terms:
            if term in text:
                matched.append(term)

        return matched

    def _parse_tags(self, tags_str: Optional[str]) -> List[str]:
        """Parse tags from JSON string."""
        if not tags_str:
            return []

        try:
            import json
            return json.loads(tags_str)
        except Exception:
            return []


class QuickSearcher:
    """
    Fast searcher for simple queries.

    Uses only BM25 when semantic search isn't needed.
    Good for exact term matching.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)

    def search(
        self,
        query: str,
        limit: int = 10,
        fields: List[str] = None
    ) -> List[Dict]:
        """
        Quick BM25 search.

        Args:
            query: Search query
            limit: Max results
            fields: Fields to return

        Returns:
            List of matching conversation dicts
        """
        if not self.db_path.exists():
            return []

        fields = fields or ['convo_id', 'generated_title', 'summary_abstractive']

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Escape query
            terms = query.strip().split()
            fts_query = ' OR '.join(
                ''.join(c for c in t if c.isalnum() or c in '-_')
                for t in terms if t
            ) or '*'

            sql = f"""
                SELECT {', '.join(f'c.{f}' for f in fields)}
                FROM conversations_fts
                JOIN conversations c ON conversations_fts.rowid = c.id
                WHERE conversations_fts MATCH ?
                ORDER BY bm25(conversations_fts)
                LIMIT ?
            """

            cursor = conn.execute(sql, (fts_query, limit))

            return [dict(row) for row in cursor.fetchall()]

        finally:
            conn.close()


# =============================================================================
# Search Utilities
# =============================================================================

def create_search_index(db_path: str, embedding_store_path: str = None) -> HybridSearcher:
    """
    Create a fully configured hybrid searcher.

    Args:
        db_path: Path to SQLite database
        embedding_store_path: Path to embedding store directory

    Returns:
        Configured HybridSearcher
    """
    embedding_store = None

    if embedding_store_path:
        store_path = Path(embedding_store_path)
        if store_path.exists():
            embedding_store = EmbeddingStore(store_path)
            if not embedding_store.load():
                print("Warning: Could not load embeddings")
                embedding_store = None

    return HybridSearcher(db_path, embedding_store)


def search_conversations(
    query: str,
    db_path: str,
    embedding_store_path: str = None,
    limit: int = 10,
    **kwargs
) -> List[SearchResult]:
    """
    Convenience function for one-off searches.

    Args:
        query: Search query
        db_path: Path to SQLite database
        embedding_store_path: Path to embedding store
        limit: Max results
        **kwargs: Additional search parameters

    Returns:
        List of SearchResult objects
    """
    searcher = create_search_index(db_path, embedding_store_path)
    return searcher.search(query, limit=limit, **kwargs)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Hybrid search for conversations")
    parser.add_argument('query', help="Search query")
    parser.add_argument('--db', '-d', required=True, help="Database path")
    parser.add_argument('--embeddings', '-e', help="Embeddings directory")
    parser.add_argument('--limit', '-l', type=int, default=10, help="Max results")
    parser.add_argument('--bm25-only', action='store_true', help="BM25 only")
    parser.add_argument('--semantic-only', action='store_true', help="Semantic only")
    parser.add_argument('--json', action='store_true', help="Output as JSON")

    args = parser.parse_args()

    results = search_conversations(
        args.query,
        args.db,
        args.embeddings,
        limit=args.limit,
        bm25_only=args.bm25_only,
        semantic_only=args.semantic_only
    )

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print(f"\nFound {len(results)} results for '{args.query}':\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result.score:.3f}] {result.title}")
            print(f"   Source: {result.source} | Tags: {', '.join(result.tags[:3])}")
            if result.summary:
                print(f"   {result.summary[:100]}...")
            print()
