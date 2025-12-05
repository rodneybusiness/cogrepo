"""
Pytest Configuration and Shared Fixtures

Provides:
- Sample conversation data
- Mock API responses
- Temporary directories
- Database fixtures
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_chatgpt_conversation():
    """Sample ChatGPT export format."""
    return {
        "id": "conv-chatgpt-123",
        "title": "Test ChatGPT Conversation",
        "create_time": 1678015311.655875,
        "update_time": 1678015500.123456,
        "mapping": {
            "root-node": {
                "id": "root-node",
                "message": None,
                "parent": None,
                "children": ["msg-1"]
            },
            "msg-1": {
                "id": "msg-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "user"},
                    "content": {"parts": ["What is Python?"]},
                    "create_time": 1678015311.655875
                },
                "parent": "root-node",
                "children": ["msg-2"]
            },
            "msg-2": {
                "id": "msg-2",
                "message": {
                    "id": "msg-2",
                    "author": {"role": "assistant"},
                    "content": {"parts": ["Python is a high-level programming language known for its readability and versatility."]},
                    "create_time": 1678015320.123456
                },
                "parent": "msg-1",
                "children": []
            }
        }
    }


@pytest.fixture
def sample_claude_conversation():
    """Sample Claude export format."""
    return {
        "uuid": "conv-claude-456",
        "name": "Test Claude Conversation",
        "created_at": "2025-10-15T14:30:00.000Z",
        "updated_at": "2025-10-15T15:45:00.000Z",
        "chat_messages": [
            {
                "uuid": "msg-claude-1",
                "text": "How do I implement a binary search?",
                "sender": "human",
                "created_at": "2025-10-15T14:30:00.000Z"
            },
            {
                "uuid": "msg-claude-2",
                "text": "Binary search works by repeatedly dividing the search interval in half. Here's a Python implementation...",
                "sender": "assistant",
                "created_at": "2025-10-15T14:31:00.000Z"
            }
        ]
    }


@pytest.fixture
def sample_gemini_conversation():
    """Sample Gemini export format."""
    return {
        "id": "conv-gemini-789",
        "title": "Test Gemini Conversation",
        "timestamp": "2025-10-20T10:00:00.000Z",
        "messages": [
            {
                "role": "user",
                "content": "Explain machine learning in simple terms",
                "timestamp": "2025-10-20T10:00:00.000Z"
            },
            {
                "role": "model",
                "content": "Machine learning is like teaching a computer to learn from examples instead of giving it explicit instructions.",
                "timestamp": "2025-10-20T10:01:00.000Z"
            }
        ]
    }


@pytest.fixture
def sample_enriched_conversation():
    """Sample enriched conversation."""
    return {
        "convo_id": "enriched-123",
        "external_id": "ext-123",
        "timestamp": "2025-10-15T14:30:00.000Z",
        "source": "Anthropic",
        "raw_text": "USER: What is Python?\n\nASSISTANT: Python is a programming language...",
        "generated_title": "Introduction to Python Programming",
        "summary_abstractive": "A conversation about the basics of Python programming language.",
        "summary_extractive": "Python is a programming language.",
        "primary_domain": "Technical",
        "tags": ["python", "programming", "beginners"],
        "key_topics": ["Python", "Programming Languages"],
        "brilliance_score": {"score": 7, "reasoning": "Good introductory explanation"},
        "key_insights": ["Python is beginner-friendly", "Used for many applications"],
        "status": "Completed",
        "future_potential": {"value_proposition": "Reference material", "next_steps": ""},
        "score": 7,
        "score_reasoning": "Clear and informative",
        "metadata": {"message_count": 2}
    }


# =============================================================================
# File Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def chatgpt_export_file(temp_dir, sample_chatgpt_conversation):
    """Create temporary ChatGPT export file."""
    file_path = temp_dir / "chatgpt_export.json"
    with open(file_path, 'w') as f:
        json.dump([sample_chatgpt_conversation], f)
    return file_path


@pytest.fixture
def claude_export_file(temp_dir, sample_claude_conversation):
    """Create temporary Claude export file."""
    file_path = temp_dir / "claude_export.json"
    with open(file_path, 'w') as f:
        json.dump([sample_claude_conversation], f)
    return file_path


@pytest.fixture
def enriched_repository_file(temp_dir, sample_enriched_conversation):
    """Create temporary enriched repository."""
    file_path = temp_dir / "enriched_repository.jsonl"
    with open(file_path, 'w') as f:
        f.write(json.dumps(sample_enriched_conversation) + '\n')
    return file_path


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing without API calls."""
    with patch('anthropic.Anthropic') as mock:
        client = MagicMock()
        mock.return_value = client

        # Configure mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"title": "Test Title", "confidence": 0.9}')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        client.messages.create.return_value = mock_response

        yield client


@pytest.fixture
def mock_enrichment_response():
    """Standard mock enrichment response."""
    return {
        "title": "Generated Test Title",
        "confidence": 0.9,
        "abstractive": "This is a test summary.",
        "extractive": ["Key point 1", "Key point 2"],
        "primary_domain": "Technical",
        "tags": ["test", "sample"],
        "key_topics": ["Testing"],
        "score": 7,
        "reasoning": "Test score reasoning"
    }


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def search_db(temp_dir):
    """Create temporary search database."""
    from search_engine import SearchEngine
    db_path = temp_dir / "test_search.db"
    engine = SearchEngine(db_path=str(db_path))
    return engine


@pytest.fixture
def populated_search_db(search_db, sample_enriched_conversation):
    """Search database with sample data."""
    search_db.index_conversation(sample_enriched_conversation)
    return search_db


# =============================================================================
# Utility Functions
# =============================================================================

def assert_valid_conversation(conv):
    """Assert conversation has required fields."""
    assert conv.external_id, "Missing external_id"
    assert conv.source in ["OpenAI", "Anthropic", "Google"], f"Invalid source: {conv.source}"
    assert conv.title, "Missing title"
    assert conv.create_time, "Missing create_time"
    assert len(conv.messages) > 0, "No messages"


def assert_valid_enrichment(enrichments):
    """Assert enrichments have required fields."""
    assert enrichments.get('generated_title'), "Missing generated_title"
    assert enrichments.get('summary_abstractive'), "Missing summary"
    assert enrichments.get('primary_domain'), "Missing domain"
    assert isinstance(enrichments.get('tags'), list), "Tags should be a list"
    assert isinstance(enrichments.get('score'), (int, float)), "Score should be numeric"
