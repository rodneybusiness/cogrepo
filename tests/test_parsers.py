"""
Tests for Conversation Parsers

Tests:
- ChatGPT parser (tree structure)
- Claude parser (linear structure)
- Gemini parser (variable formats)
- Format detection
- Error handling
"""

import pytest
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers import ChatGPTParser, ClaudeParser, GeminiParser
from parsers.base_parser import ConversationParser


class TestChatGPTParser:
    """Tests for ChatGPT conversation parser."""

    def test_parse_valid_export(self, chatgpt_export_file):
        """Test parsing valid ChatGPT export."""
        parser = ChatGPTParser(str(chatgpt_export_file))
        conversations = parser.parse()

        assert len(conversations) == 1
        conv = conversations[0]

        assert conv.external_id == "conv-chatgpt-123"
        assert conv.source == "OpenAI"
        assert conv.title == "Test ChatGPT Conversation"
        assert len(conv.messages) == 2

    def test_parse_message_roles(self, chatgpt_export_file):
        """Test that message roles are correctly parsed."""
        parser = ChatGPTParser(str(chatgpt_export_file))
        conversations = parser.parse()
        messages = conversations[0].messages

        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_parse_message_content(self, chatgpt_export_file):
        """Test that message content is correctly extracted."""
        parser = ChatGPTParser(str(chatgpt_export_file))
        conversations = parser.parse()
        messages = conversations[0].messages

        assert "Python" in messages[0].content
        assert "programming language" in messages[1].content

    def test_detect_format(self, chatgpt_export_file):
        """Test format detection."""
        parser = ChatGPTParser(str(chatgpt_export_file))
        assert parser.detect_format() is True

    def test_detect_format_wrong_file(self, claude_export_file):
        """Test format detection rejects wrong format."""
        parser = ChatGPTParser(str(claude_export_file))
        assert parser.detect_format() is False

    def test_parse_empty_mapping(self, temp_dir):
        """Test handling of conversation with empty mapping."""
        empty_conv = {
            "id": "empty-conv",
            "title": "Empty",
            "create_time": 1678015311.0,
            "mapping": {}
        }
        file_path = temp_dir / "empty.json"
        with open(file_path, 'w') as f:
            json.dump([empty_conv], f)

        parser = ChatGPTParser(str(file_path))
        conversations = parser.parse()

        # Should return empty list (no valid messages)
        assert len(conversations) == 0

    def test_parse_branched_conversation(self, temp_dir):
        """Test parsing conversation with branches (only follows primary path)."""
        branched_conv = {
            "id": "branched-conv",
            "title": "Branched",
            "create_time": 1678015311.0,
            "mapping": {
                "root": {
                    "id": "root",
                    "message": None,
                    "parent": None,
                    "children": ["msg-1"]
                },
                "msg-1": {
                    "id": "msg-1",
                    "message": {
                        "id": "msg-1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Question"]}
                    },
                    "parent": "root",
                    "children": ["msg-2a", "msg-2b"]  # Branched responses
                },
                "msg-2a": {
                    "id": "msg-2a",
                    "message": {
                        "id": "msg-2a",
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Response A (primary)"]}
                    },
                    "parent": "msg-1",
                    "children": []
                },
                "msg-2b": {
                    "id": "msg-2b",
                    "message": {
                        "id": "msg-2b",
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Response B (alternate)"]}
                    },
                    "parent": "msg-1",
                    "children": []
                }
            }
        }
        file_path = temp_dir / "branched.json"
        with open(file_path, 'w') as f:
            json.dump([branched_conv], f)

        parser = ChatGPTParser(str(file_path))
        conversations = parser.parse()

        assert len(conversations) == 1
        # Should follow primary path (first child)
        assert len(conversations[0].messages) == 2
        assert "Response A" in conversations[0].messages[1].content


class TestClaudeParser:
    """Tests for Claude conversation parser."""

    def test_parse_valid_export(self, claude_export_file):
        """Test parsing valid Claude export."""
        parser = ClaudeParser(str(claude_export_file))
        conversations = parser.parse()

        assert len(conversations) == 1
        conv = conversations[0]

        assert conv.external_id == "conv-claude-456"
        assert conv.source == "Anthropic"
        assert conv.title == "Test Claude Conversation"
        assert len(conv.messages) == 2

    def test_parse_human_sender(self, claude_export_file):
        """Test that 'human' sender is normalized to 'user'."""
        parser = ClaudeParser(str(claude_export_file))
        conversations = parser.parse()
        messages = conversations[0].messages

        assert messages[0].role == "user"  # Was 'human' in source
        assert messages[1].role == "assistant"

    def test_detect_format(self, claude_export_file):
        """Test format detection."""
        parser = ClaudeParser(str(claude_export_file))
        assert parser.detect_format() is True

    def test_parse_jsonl_format(self, temp_dir, sample_claude_conversation):
        """Test parsing JSONL format."""
        file_path = temp_dir / "claude.jsonl"
        with open(file_path, 'w') as f:
            f.write(json.dumps(sample_claude_conversation) + '\n')

        parser = ClaudeParser(str(file_path))
        conversations = parser.parse()

        assert len(conversations) == 1

    def test_parse_alternative_field_names(self, temp_dir):
        """Test parsing with alternative field names."""
        alt_format = {
            "id": "alt-conv",  # Instead of 'uuid'
            "title": "Alternative Format",  # Instead of 'name'
            "timestamp": "2025-10-15T14:30:00.000Z",  # Instead of 'created_at'
            "messages": [  # Instead of 'chat_messages'
                {
                    "role": "user",  # Instead of 'sender: human'
                    "content": "Test message"  # Instead of 'text'
                }
            ]
        }
        file_path = temp_dir / "alt_claude.json"
        with open(file_path, 'w') as f:
            json.dump([alt_format], f)

        parser = ClaudeParser(str(file_path))
        conversations = parser.parse()

        assert len(conversations) == 1
        assert conversations[0].external_id == "alt-conv"


class TestGeminiParser:
    """Tests for Gemini conversation parser."""

    def test_parse_valid_export(self, temp_dir, sample_gemini_conversation):
        """Test parsing valid Gemini export."""
        file_path = temp_dir / "gemini.json"
        with open(file_path, 'w') as f:
            json.dump([sample_gemini_conversation], f)

        parser = GeminiParser(str(file_path))
        conversations = parser.parse()

        assert len(conversations) == 1
        conv = conversations[0]

        assert conv.external_id == "conv-gemini-789"
        assert conv.source == "Google"
        assert len(conv.messages) == 2

    def test_parse_model_role(self, temp_dir, sample_gemini_conversation):
        """Test that 'model' role is normalized to 'assistant'."""
        file_path = temp_dir / "gemini.json"
        with open(file_path, 'w') as f:
            json.dump([sample_gemini_conversation], f)

        parser = GeminiParser(str(file_path))
        conversations = parser.parse()
        messages = conversations[0].messages

        assert messages[0].role == "user"
        assert messages[1].role == "assistant"  # Was 'model' in source


class TestParserErrorHandling:
    """Tests for parser error handling."""

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ChatGPTParser("/nonexistent/path.json")

    def test_invalid_json(self, temp_dir):
        """Test error handling for invalid JSON."""
        file_path = temp_dir / "invalid.json"
        with open(file_path, 'w') as f:
            f.write("{ invalid json }")

        parser = ChatGPTParser(str(file_path))
        with pytest.raises(ValueError):
            parser.parse()

    def test_malformed_conversation(self, temp_dir):
        """Test handling of malformed conversation data."""
        malformed = {
            "id": "malformed",
            # Missing 'mapping' field
            "create_time": 1678015311.0
        }
        file_path = temp_dir / "malformed.json"
        with open(file_path, 'w') as f:
            json.dump([malformed], f)

        parser = ChatGPTParser(str(file_path))
        # Should not raise, but return empty list
        conversations = parser.parse()
        assert len(conversations) == 0


class TestBaseParser:
    """Tests for base parser functionality."""

    def test_platform_name(self, chatgpt_export_file):
        """Test platform name extraction."""
        parser = ChatGPTParser(str(chatgpt_export_file))
        assert parser.platform_name == "ChatGPT"

    def test_truncate_title(self, chatgpt_export_file):
        """Test title truncation."""
        parser = ChatGPTParser(str(chatgpt_export_file))

        short_title = "Short"
        assert parser._truncate_title(short_title, 100) == short_title

        long_title = "A" * 150
        truncated = parser._truncate_title(long_title, 100)
        assert len(truncated) == 100
        assert truncated.endswith("...")
