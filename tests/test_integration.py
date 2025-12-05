"""
Integration Tests for CogRepo

End-to-end tests verifying all components work together:
- Parsing → Validation → Storage → Search
- Archive registration → Incremental sync
- Configuration → Logging → Exception handling
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    get_config,
    init_config,
    reset_config,
    Message,
    RawConversation,
    EnrichedConversation,
    MessageRole,
    ConversationSource,
    CogRepoException,
    ParsingError,
    get_logger,
    setup_logging,
)
from parsers import ChatGPTParser, ClaudeParser, GeminiParser
from search_engine import SearchEngine, QueryParser
from archive_registry import ArchiveRegistry
from smart_parser import SmartParser


class TestParsingToValidationPipeline:
    """Test parsing and validation integration."""

    def test_chatgpt_parsing_to_validated_model(self, chatgpt_export_file):
        """Test ChatGPT parsing produces valid model."""
        parser = ChatGPTParser(str(chatgpt_export_file))
        conversations = parser.parse()

        assert len(conversations) == 1
        conv = conversations[0]

        # Create validated RawConversation from parsed data
        raw = RawConversation(
            external_id=conv.external_id,
            source=conv.source,
            title=conv.title,
            create_time=conv.create_time,
            messages=[
                Message(role=msg.role, content=msg.content)
                for msg in conv.messages
            ]
        )

        assert raw.source == ConversationSource.OPENAI
        assert raw.message_count == 2
        assert raw.user_message_count == 1
        assert raw.assistant_message_count == 1

    def test_claude_parsing_to_validated_model(self, claude_export_file):
        """Test Claude parsing produces valid model."""
        parser = ClaudeParser(str(claude_export_file))
        conversations = parser.parse()

        assert len(conversations) == 1
        conv = conversations[0]

        raw = RawConversation(
            external_id=conv.external_id,
            source=conv.source,
            title=conv.title,
            create_time=conv.create_time,
            messages=[
                Message(role=msg.role, content=msg.content)
                for msg in conv.messages
            ]
        )

        assert raw.source == ConversationSource.ANTHROPIC
        assert raw.message_count == 2

    def test_raw_to_enriched_conversion(self, chatgpt_export_file):
        """Test RawConversation to EnrichedConversation conversion."""
        parser = ChatGPTParser(str(chatgpt_export_file))
        conversations = parser.parse()
        conv = conversations[0]

        raw = RawConversation(
            external_id=conv.external_id,
            source=conv.source,
            title=conv.title,
            create_time=conv.create_time,
            messages=[
                Message(role=msg.role, content=msg.content)
                for msg in conv.messages
            ]
        )

        # Convert to enriched (without AI enrichment)
        enriched = EnrichedConversation.from_raw(raw)

        assert enriched.external_id == raw.external_id
        assert enriched.source == raw.source
        assert len(enriched.raw_text) > 0
        assert enriched.metadata['message_count'] == 2


class TestSearchEngineIntegration:
    """Test search engine with parsed data."""

    @pytest.fixture
    def search_db_with_data(self, temp_dir, sample_enriched_conversation):
        """Create search DB with sample data."""
        db_path = temp_dir / "integration_test.db"
        engine = SearchEngine(db_path=str(db_path))
        engine.index_conversation(sample_enriched_conversation)

        # Add another conversation
        conv2 = sample_enriched_conversation.copy()
        conv2['convo_id'] = 'test-conv-2'
        conv2['external_id'] = 'ext-456'
        conv2['generated_title'] = 'Machine Learning Basics'
        conv2['summary_abstractive'] = 'Discussion about neural networks and deep learning.'
        conv2['tags'] = ['machine-learning', 'neural-networks', 'AI']
        conv2['score'] = 9
        engine.index_conversation(conv2)

        return engine

    def test_search_returns_relevant_results(self, search_db_with_data):
        """Test search finds relevant conversations."""
        response = search_db_with_data.search("Python programming")

        assert len(response['results']) >= 1
        # First result should be about Python
        assert 'Python' in response['results'][0]['title']

    def test_search_multiple_results(self, search_db_with_data):
        """Test search can return multiple results."""
        response = search_db_with_data.search("test")

        # Both conversations have "test" somewhere
        assert len(response['results']) >= 1

    def test_search_with_filters(self, search_db_with_data):
        """Test search with score filter."""
        # Only the ML conversation has score 9
        response = search_db_with_data.search("", filters={"min_score": 9})

        assert len(response['results']) == 1
        assert 'Machine Learning' in response['results'][0]['title']

    def test_search_stats(self, search_db_with_data):
        """Test search stats."""
        stats = search_db_with_data.get_stats()

        assert stats['total_conversations'] == 2


class TestArchiveRegistryIntegration:
    """Test archive registry with parsers."""

    def test_register_and_check_archive(self, temp_dir, chatgpt_export_file):
        """Test archive registration and change detection."""
        registry_file = temp_dir / "registry.json"
        registry = ArchiveRegistry(str(registry_file))

        # Register archive (uses 'register' method)
        archive = registry.register(
            name="Test ChatGPT",
            source="chatgpt",
            file_path=str(chatgpt_export_file)
        )

        assert archive is not None
        assert archive.name == "Test ChatGPT"

        # Check for changes using archive name
        changes = registry.detect_changes("Test ChatGPT")
        # Changes object should exist
        assert changes is not None

        # Get archive by name
        archive_retrieved = registry.get_archive("Test ChatGPT")
        assert archive_retrieved is not None
        assert archive_retrieved.name == "Test ChatGPT"
        assert archive_retrieved.source == "OpenAI"  # Normalized to OpenAI

    def test_list_archives(self, temp_dir, chatgpt_export_file, claude_export_file):
        """Test listing multiple archives."""
        registry_file = temp_dir / "registry.json"
        registry = ArchiveRegistry(str(registry_file))

        # Register multiple archives
        registry.register("ChatGPT", "chatgpt", str(chatgpt_export_file))
        registry.register("Claude", "claude", str(claude_export_file))

        archives = registry.list_archives()
        assert len(archives) == 2

        names = [a.name for a in archives]
        assert "ChatGPT" in names
        assert "Claude" in names


class TestConfigurationIntegration:
    """Test configuration with other components."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_config_with_custom_paths(self, temp_dir):
        """Test config with custom paths works with search engine."""
        config = init_config(base_dir=temp_dir)
        config.paths.ensure_dirs()

        # Create search engine with config path
        engine = SearchEngine(db_path=str(config.paths.search_database))

        assert Path(config.paths.search_database).exists()

    def test_config_validation_for_pipeline(self):
        """Test config validation identifies missing requirements."""
        reset_config()
        config = get_config()

        # Without API key, enrichment should not be ready
        issues = config.validate_for_enrichment()

        assert len(issues) > 0
        assert any('API_KEY' in issue.upper() for issue in issues)


class TestLoggingIntegration:
    """Test logging integration."""

    def test_logger_works_with_components(self, temp_dir):
        """Test logger integrates with components."""
        setup_logging(level='DEBUG')
        logger = get_logger('test.integration')

        # Logger should work
        logger.info("Test log message")
        logger.debug("Debug message")

        # Should not raise
        assert logger is not None


class TestExceptionIntegration:
    """Test exception handling across components."""

    def test_parsing_exception_has_context(self, temp_dir):
        """Test parsing errors have proper context."""
        # Create invalid file
        invalid_file = temp_dir / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("{ invalid json }")

        parser = ChatGPTParser(str(invalid_file))

        with pytest.raises(ValueError):
            parser.parse()

    def test_file_not_found_exception(self):
        """Test proper exception for missing files."""
        with pytest.raises(FileNotFoundError):
            ChatGPTParser("/nonexistent/path/file.json")


class TestSmartParserIntegration:
    """Test smart parser with archive registry."""

    def test_smart_parser_parses_chatgpt(self, chatgpt_export_file):
        """Test smart parser with ChatGPT format."""
        from archive_registry import ProcessingCursor

        parser = SmartParser(source='chatgpt', file_path=str(chatgpt_export_file))
        cursor = ProcessingCursor()  # Empty cursor = parse all

        convs = list(parser.parse_incremental(cursor))
        assert len(convs) == 1
        assert convs[0].source == 'OpenAI'

    def test_smart_parser_parses_all_formats(
        self,
        chatgpt_export_file,
        claude_export_file,
        temp_dir,
        sample_gemini_conversation
    ):
        """Test smart parser handles all formats."""
        from archive_registry import ProcessingCursor

        # ChatGPT
        parser = SmartParser(source='chatgpt', file_path=str(chatgpt_export_file))
        convs = list(parser.parse_incremental(ProcessingCursor()))
        assert len(convs) == 1

        # Claude
        parser = SmartParser(source='claude', file_path=str(claude_export_file))
        convs = list(parser.parse_incremental(ProcessingCursor()))
        assert len(convs) == 1

        # Gemini
        gemini_file = temp_dir / "gemini.json"
        with open(gemini_file, 'w') as f:
            json.dump([sample_gemini_conversation], f)

        parser = SmartParser(source='gemini', file_path=str(gemini_file))
        convs = list(parser.parse_incremental(ProcessingCursor()))
        assert len(convs) == 1


class TestEndToEndWorkflow:
    """Test complete workflow from parsing to search."""

    def test_parse_validate_store_search(
        self,
        temp_dir,
        chatgpt_export_file,
        claude_export_file
    ):
        """Test full workflow: parse → validate → store → search."""
        # Step 1: Parse conversations from both sources
        chatgpt_parser = ChatGPTParser(str(chatgpt_export_file))
        claude_parser = ClaudeParser(str(claude_export_file))

        chatgpt_convs = chatgpt_parser.parse()
        claude_convs = claude_parser.parse()

        assert len(chatgpt_convs) == 1
        assert len(claude_convs) == 1

        # Step 2: Validate and convert to enriched format
        enriched_convs = []

        for conv in chatgpt_convs + claude_convs:
            raw = RawConversation(
                external_id=conv.external_id,
                source=conv.source,
                title=conv.title,
                create_time=conv.create_time,
                messages=[
                    Message(role=msg.role, content=msg.content)
                    for msg in conv.messages
                ]
            )
            enriched = EnrichedConversation.from_raw(raw, {
                'generated_title': conv.title,
                'summary_abstractive': f"Conversation about {conv.title}",
                'tags': ['test'],
                'score': 7
            })
            enriched_convs.append(enriched)

        assert len(enriched_convs) == 2

        # Step 3: Store in search engine
        db_path = temp_dir / "workflow_test.db"
        engine = SearchEngine(db_path=str(db_path))

        for conv in enriched_convs:
            engine.index_conversation(conv.model_dump(mode='json'))

        # Step 4: Search and verify
        stats = engine.get_stats()
        assert stats['total_conversations'] == 2

        # Search for ChatGPT conversation
        response = engine.search("Python")
        assert len(response['results']) >= 1
        # Verify we found the OpenAI conversation
        openai_results = [r for r in response['results'] if r['source'] == 'OpenAI']
        assert len(openai_results) >= 1

        # Search for Claude conversation
        response = engine.search("binary search")
        assert len(response['results']) >= 1
        # Verify we found the Anthropic conversation
        anthropic_results = [r for r in response['results'] if r['source'] == 'Anthropic']
        assert len(anthropic_results) >= 1

    def test_archive_registration_to_sync_preview(
        self,
        temp_dir,
        chatgpt_export_file
    ):
        """Test archive registration through sync preview."""
        from archive_registry import ProcessingCursor

        # Set up registry
        registry_file = temp_dir / "e2e_registry.json"
        registry = ArchiveRegistry(str(registry_file))

        # Register archive
        archive = registry.register(
            name="E2E Test",
            source="chatgpt",
            file_path=str(chatgpt_export_file)
        )

        assert archive is not None

        # Get archive info by name
        archive_retrieved = registry.get_archive("E2E Test")
        assert archive_retrieved is not None
        assert archive_retrieved.name == "E2E Test"

        # Smart parse the file using parse_incremental
        parser = SmartParser(source='chatgpt', file_path=str(chatgpt_export_file))
        convs = list(parser.parse_incremental(ProcessingCursor()))

        assert len(convs) == 1
        assert convs[0].source == 'OpenAI'
