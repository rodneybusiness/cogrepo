"""
Tests for Search Engine

Tests:
- Index creation and management
- Full-text search with FTS5
- Query parsing (operators, phrases)
- Filtering (source, date, score)
- Result ranking
- Error handling
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from search_engine import SearchEngine, QueryParser, SearchQuery, SearchFilters


class TestQueryParser:
    """Tests for search query parsing."""

    def test_parse_simple_query(self):
        """Test parsing simple search query."""
        parser = QueryParser()
        result = parser.parse("hello world")

        assert result.original == "hello world"
        assert "hello" in result.regular
        assert "world" in result.regular
        assert len(result.required) == 0
        assert len(result.excluded) == 0

    def test_parse_required_terms(self):
        """Test parsing +required terms."""
        parser = QueryParser()
        result = parser.parse("+python +programming")

        assert "python" in result.required
        assert "programming" in result.required
        assert len(result.regular) == 0

    def test_parse_excluded_terms(self):
        """Test parsing -excluded terms."""
        parser = QueryParser()
        result = parser.parse("python -java")

        assert "python" in result.regular
        assert "java" in result.excluded

    def test_parse_exact_phrases(self):
        """Test parsing "exact phrases"."""
        parser = QueryParser()
        result = parser.parse('"machine learning" tutorial')

        assert "machine learning" in result.phrases
        assert "tutorial" in result.regular

    def test_parse_mixed_query(self):
        """Test parsing query with all operator types."""
        parser = QueryParser()
        result = parser.parse('+required "exact phrase" regular -excluded')

        assert "required" in result.required
        assert "exact phrase" in result.phrases
        assert "regular" in result.regular
        assert "excluded" in result.excluded

    def test_parse_empty_query(self):
        """Test parsing empty query."""
        parser = QueryParser()
        result = parser.parse("")

        assert result.is_empty is True

    def test_to_fts_query(self):
        """Test FTS query generation."""
        parser = QueryParser()
        result = parser.parse("python tutorial")
        fts_query = result.to_fts_query()

        assert "python" in fts_query
        assert "tutorial" in fts_query


class TestSearchEngine:
    """Tests for SearchEngine class."""

    def test_create_database(self, temp_dir):
        """Test database creation."""
        db_path = temp_dir / "test.db"
        engine = SearchEngine(db_path=str(db_path))

        assert db_path.exists()

    def test_index_conversation(self, search_db, sample_enriched_conversation):
        """Test indexing a single conversation."""
        result = search_db.index_conversation(sample_enriched_conversation)
        assert result is True

        # Verify it's in the database
        stats = search_db.get_stats()
        assert stats['total_conversations'] == 1

    def test_index_batch(self, search_db):
        """Test batch indexing."""
        conversations = [
            {
                "convo_id": f"conv-{i}",
                "external_id": f"ext-{i}",
                "source": "OpenAI",
                "timestamp": "2025-10-15T14:30:00Z",
                "generated_title": f"Test Conversation {i}",
                "summary_abstractive": f"Summary {i}",
                "raw_text": f"Test content {i}",
                "primary_domain": "Technical",
                "tags": ["test"],
                "score": 5
            }
            for i in range(10)
        ]

        count = search_db.index_conversations(iter(conversations))
        assert count == 10

        stats = search_db.get_stats()
        assert stats['total_conversations'] == 10

    def test_search_basic(self, populated_search_db):
        """Test basic search."""
        results = populated_search_db.search("Python")

        assert results['total_count'] >= 1
        assert len(results['results']) >= 1

    def test_search_no_results(self, populated_search_db):
        """Test search with no matches."""
        results = populated_search_db.search("xyznonexistent123")

        assert results['total_count'] == 0
        assert len(results['results']) == 0

    def test_search_with_source_filter(self, search_db):
        """Test search with source filter."""
        # Index conversations from different sources
        search_db.index_conversation({
            "convo_id": "openai-1",
            "external_id": "ext-1",
            "source": "OpenAI",
            "timestamp": "2025-10-15T14:30:00Z",
            "generated_title": "OpenAI Conversation",
            "raw_text": "Test content",
            "score": 5
        })
        search_db.index_conversation({
            "convo_id": "anthropic-1",
            "external_id": "ext-2",
            "source": "Anthropic",
            "timestamp": "2025-10-15T14:30:00Z",
            "generated_title": "Anthropic Conversation",
            "raw_text": "Test content",
            "score": 5
        })

        results = search_db.search("content", filters={"source": "OpenAI"})

        assert results['total_count'] == 1
        assert results['results'][0]['source'] == "OpenAI"

    def test_search_with_score_filter(self, search_db):
        """Test search with minimum score filter."""
        search_db.index_conversation({
            "convo_id": "high-score",
            "external_id": "ext-1",
            "source": "OpenAI",
            "timestamp": "2025-10-15T14:30:00Z",
            "generated_title": "High Score",
            "raw_text": "Quality content",
            "score": 9
        })
        search_db.index_conversation({
            "convo_id": "low-score",
            "external_id": "ext-2",
            "source": "OpenAI",
            "timestamp": "2025-10-15T14:30:00Z",
            "generated_title": "Low Score",
            "raw_text": "Quality content",
            "score": 3
        })

        results = search_db.search("content", filters={"min_score": 7})

        assert results['total_count'] == 1
        assert results['results'][0]['score'] >= 7

    def test_search_with_date_filter(self, search_db):
        """Test search with date range filter."""
        search_db.index_conversation({
            "convo_id": "old",
            "external_id": "ext-1",
            "source": "OpenAI",
            "timestamp": "2024-01-15T14:30:00Z",
            "generated_title": "Old Conversation",
            "raw_text": "Test",
            "score": 5
        })
        search_db.index_conversation({
            "convo_id": "new",
            "external_id": "ext-2",
            "source": "OpenAI",
            "timestamp": "2025-11-15T14:30:00Z",
            "generated_title": "New Conversation",
            "raw_text": "Test",
            "score": 5
        })

        results = search_db.search("", filters={"date_from": "2025-01-01"})

        assert results['total_count'] == 1
        assert results['results'][0]['convo_id'] == "new"

    def test_search_pagination(self, search_db):
        """Test search pagination."""
        # Index 20 conversations
        for i in range(20):
            search_db.index_conversation({
                "convo_id": f"conv-{i}",
                "external_id": f"ext-{i}",
                "source": "OpenAI",
                "timestamp": "2025-10-15T14:30:00Z",
                "generated_title": f"Test {i}",
                "raw_text": "Searchable content",
                "score": 5
            })

        # Get first page
        page1 = search_db.search("content", limit=10, offset=0)
        assert len(page1['results']) == 10

        # Get second page
        page2 = search_db.search("content", limit=10, offset=10)
        assert len(page2['results']) == 10

        # Verify no overlap
        page1_ids = {r['convo_id'] for r in page1['results']}
        page2_ids = {r['convo_id'] for r in page2['results']}
        assert len(page1_ids & page2_ids) == 0

    def test_get_conversation(self, populated_search_db, sample_enriched_conversation):
        """Test retrieving single conversation."""
        convo_id = sample_enriched_conversation['convo_id']
        result = populated_search_db.get_conversation(convo_id)

        assert result is not None
        assert result['convo_id'] == convo_id

    def test_get_conversation_not_found(self, populated_search_db):
        """Test retrieving non-existent conversation."""
        result = populated_search_db.get_conversation("nonexistent")
        assert result is None

    def test_get_stats(self, populated_search_db):
        """Test getting index statistics."""
        stats = populated_search_db.get_stats()

        assert 'total_conversations' in stats
        assert 'by_source' in stats
        assert 'average_score' in stats
        assert stats['total_conversations'] >= 1

    def test_suggest(self, search_db):
        """Test autocomplete suggestions."""
        search_db.index_conversation({
            "convo_id": "conv-1",
            "external_id": "ext-1",
            "source": "OpenAI",
            "timestamp": "2025-10-15T14:30:00Z",
            "generated_title": "Python Programming Basics",
            "raw_text": "Content",
            "score": 5
        })

        suggestions = search_db.suggest("Python")
        assert len(suggestions) >= 1
        assert any("Python" in s for s in suggestions)


class TestSearchFilters:
    """Tests for SearchFilters class."""

    def test_empty_filters(self):
        """Test empty filters generate valid SQL."""
        filters = SearchFilters()
        sql, params = filters.to_sql_conditions()

        assert sql == "1=1"
        assert len(params) == 0

    def test_source_filter(self):
        """Test source filter."""
        filters = SearchFilters(source="OpenAI")
        sql, params = filters.to_sql_conditions()

        assert "source = ?" in sql
        assert "OpenAI" in params

    def test_date_range_filter(self):
        """Test date range filter."""
        filters = SearchFilters(date_from="2025-01-01", date_to="2025-12-31")
        sql, params = filters.to_sql_conditions()

        assert "timestamp >=" in sql
        assert "timestamp <=" in sql
        assert "2025-01-01" in params

    def test_score_range_filter(self):
        """Test score range filter."""
        filters = SearchFilters(min_score=7, max_score=10)
        sql, params = filters.to_sql_conditions()

        assert "score >=" in sql
        assert "score <=" in sql
        assert 7 in params
        assert 10 in params

    def test_combined_filters(self):
        """Test multiple filters combined."""
        filters = SearchFilters(
            source="Anthropic",
            min_score=5,
            date_from="2025-01-01"
        )
        sql, params = filters.to_sql_conditions()

        assert "source = ?" in sql
        assert "score >=" in sql
        assert "timestamp >=" in sql
        assert len(params) == 3


class TestSearchQuery:
    """Tests for SearchQuery class."""

    def test_is_empty(self):
        """Test is_empty property."""
        empty = SearchQuery(original="")
        assert empty.is_empty is True

        not_empty = SearchQuery(original="test", regular=["test"])
        assert not_empty.is_empty is False

    def test_to_fts_query_with_required(self):
        """Test FTS query with required terms."""
        query = SearchQuery(original="test", required=["python"])
        fts = query.to_fts_query()

        assert '"python"' in fts

    def test_to_fts_query_with_phrases(self):
        """Test FTS query with phrases."""
        query = SearchQuery(original="test", phrases=["machine learning"])
        fts = query.to_fts_query()

        assert '"machine learning"' in fts

    def test_to_fts_query_with_excluded(self):
        """Test FTS query with excluded terms."""
        query = SearchQuery(original="test", excluded=["java"])
        fts = query.to_fts_query()

        assert 'NOT "java"' in fts
