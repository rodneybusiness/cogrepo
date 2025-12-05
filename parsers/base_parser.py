"""
Abstract base class for conversation parsers.

All platform-specific parsers must inherit from ConversationParser
and implement the required abstract methods.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
import json
import hashlib

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models import RawConversation


class ConversationParser(ABC):
    """
    Abstract base class for parsing conversation exports.

    Each platform (ChatGPT, Claude, Gemini) has its own parser
    that inherits from this class.
    """

    def __init__(self, file_path: str):
        """
        Initialize parser with file path.

        Args:
            file_path: Path to the conversation export file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    @abstractmethod
    def parse(self) -> List[RawConversation]:
        """
        Parse the conversation file and return list of RawConversation objects.

        Returns:
            List of RawConversation objects

        Raises:
            ValueError: If file format is invalid
            JSONDecodeError: If JSON parsing fails
        """
        pass

    @abstractmethod
    def detect_format(self) -> bool:
        """
        Check if this parser can handle the given file format.

        Returns:
            True if this parser can handle the file, False otherwise
        """
        pass

    @property
    def platform_name(self) -> str:
        """Return the platform name for this parser."""
        return self.__class__.__name__.replace("Parser", "")

    def _load_json(self) -> dict:
        """
        Load JSON file with error handling.

        Returns:
            Parsed JSON data

        Raises:
            JSONDecodeError: If file is not valid JSON
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.file_path}: {e}")

    def _load_jsonl(self) -> List[dict]:
        """
        Load JSONL file (one JSON object per line).

        Returns:
            List of parsed JSON objects

        Raises:
            JSONDecodeError: If any line is not valid JSON
        """
        conversations = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    conversations.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")
        return conversations

    def _generate_content_hash(self, conversation: RawConversation) -> str:
        """
        Generate SHA256 hash of conversation content for deduplication.

        Hash includes:
        - Source platform
        - Title
        - First message content
        - Last message content
        - Create time

        Args:
            conversation: RawConversation object

        Returns:
            SHA256 hash as hex string
        """
        components = [
            conversation.source,
            conversation.title,
            conversation.messages[0].content if conversation.messages else "",
            conversation.messages[-1].content if conversation.messages else "",
            conversation.create_time.isoformat()
        ]
        content = "|".join(components)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _truncate_title(self, title: str, max_length: int = 100) -> str:
        """
        Truncate title to max length, adding ellipsis if needed.

        Args:
            title: Original title
            max_length: Maximum length

        Returns:
            Truncated title
        """
        if len(title) <= max_length:
            return title
        return title[:max_length-3] + "..."

    def _extract_text_preview(self, text: str, max_chars: int = 60) -> str:
        """
        Extract a preview from text (first line or max_chars).

        Args:
            text: Full text
            max_chars: Maximum characters

        Returns:
            Preview text
        """
        # Take first line or first max_chars characters
        first_line = text.split('\n')[0]
        if len(first_line) <= max_chars:
            return first_line
        return first_line[:max_chars-3] + "..."
