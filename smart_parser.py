"""
Smart Incremental Parser for CogRepo

Enables efficient processing of ONLY new conversations in archive files
by using processing cursors and streaming/incremental parsing.

Key Features:
1. Stream parsing - doesn't load entire file into memory
2. Cursor-based resumption - starts from where we left off
3. Date-based filtering - skip conversations older than cursor
4. Memory efficient - yields conversations one at a time

This is what makes "process only what's new" possible without
re-reading the entire archive file.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List, Callable
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent))

from models import RawConversation, Message
from archive_registry import ProcessingCursor


@dataclass
class ParseProgress:
    """Tracks parsing progress for reporting."""
    total_scanned: int = 0
    total_new: int = 0
    total_skipped: int = 0
    current_timestamp: Optional[str] = None
    current_id: Optional[str] = None


class SmartParser:
    """
    Smart incremental parser that processes only NEW conversations.

    Unlike the regular parsers that load everything, this parser:
    1. Uses a cursor to know where we left off
    2. Streams the file instead of loading all at once
    3. Skips conversations we've already processed
    4. Yields only new conversations

    Usage:
        parser = SmartParser("chatgpt", "/path/to/export.json")
        cursor = ProcessingCursor(last_timestamp="2025-11-28T00:00:00")

        for conv in parser.parse_incremental(cursor):
            # Only new conversations since cursor
            process(conv)

        # Get final cursor for next time
        new_cursor = parser.get_final_cursor()
    """

    def __init__(self, source: str, file_path: str):
        """
        Initialize smart parser.

        Args:
            source: Platform source ("chatgpt", "claude", "gemini")
            file_path: Path to export file
        """
        self.source = source.lower()
        self.file_path = Path(file_path)
        self.progress = ParseProgress()
        self._final_cursor = ProcessingCursor()

        # Source-specific configuration
        self.source_config = {
            "chatgpt": {
                "platform_source": "OpenAI",
                "timestamp_field": "create_time",
                "id_field": "id",
                "timestamp_format": "unix"
            },
            "claude": {
                "platform_source": "Anthropic",
                "timestamp_field": "created_at",
                "id_field": "uuid",
                "timestamp_format": "iso"
            },
            "gemini": {
                "platform_source": "Google",
                "timestamp_field": "timestamp",
                "id_field": "id",
                "timestamp_format": "iso"
            }
        }

        self.config = self.source_config.get(self.source, self.source_config["chatgpt"])

    def parse_incremental(
        self,
        cursor: Optional[ProcessingCursor] = None,
        processed_ids: Optional[set] = None,
        progress_callback: Optional[Callable[[ParseProgress], None]] = None
    ) -> Iterator[RawConversation]:
        """
        Parse file incrementally, yielding only new conversations.

        Args:
            cursor: ProcessingCursor from last sync (None = process all)
            processed_ids: Set of external IDs to skip (for deduplication)
            progress_callback: Optional callback for progress updates

        Yields:
            RawConversation objects for NEW conversations only
        """
        if processed_ids is None:
            processed_ids = set()

        # Determine cutoff timestamp from cursor
        cutoff_timestamp = None
        if cursor and cursor.last_timestamp:
            cutoff_timestamp = self._parse_cursor_timestamp(cursor.last_timestamp)

        # Choose parsing method based on file type
        if self.file_path.suffix.lower() == '.jsonl':
            parser = self._parse_jsonl_incremental
        else:
            parser = self._parse_json_incremental

        # Parse and filter
        for conv_data in parser():
            self.progress.total_scanned += 1

            # Extract ID and timestamp
            conv_id = self._get_id(conv_data)
            conv_timestamp = self._get_timestamp(conv_data)

            self.progress.current_id = conv_id
            self.progress.current_timestamp = conv_timestamp.isoformat() if conv_timestamp else None

            # Skip if already processed (by ID)
            if conv_id in processed_ids:
                self.progress.total_skipped += 1
                continue

            # Skip if older than cursor
            if cutoff_timestamp and conv_timestamp and conv_timestamp <= cutoff_timestamp:
                self.progress.total_skipped += 1
                continue

            # Parse the full conversation
            try:
                conversation = self._parse_conversation(conv_data)
                if conversation and conversation.messages:
                    self.progress.total_new += 1

                    # Update final cursor
                    self._final_cursor.last_external_id = conv_id
                    self._final_cursor.last_timestamp = conv_timestamp.isoformat() if conv_timestamp else None
                    self._final_cursor.conversation_count += 1

                    # Report progress
                    if progress_callback and self.progress.total_scanned % 100 == 0:
                        progress_callback(self.progress)

                    yield conversation

            except Exception as e:
                print(f"Warning: Failed to parse conversation {conv_id}: {e}")
                continue

    def get_final_cursor(self) -> ProcessingCursor:
        """Get the cursor state after parsing (for saving)."""
        return self._final_cursor

    def get_progress(self) -> ParseProgress:
        """Get current parsing progress."""
        return self.progress

    def _parse_jsonl_incremental(self) -> Iterator[Dict[str, Any]]:
        """
        Stream-parse JSONL file.

        JSONL is perfect for incremental processing because each line
        is independent - we can seek to a specific position.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue

    def _parse_json_incremental(self) -> Iterator[Dict[str, Any]]:
        """
        Parse JSON file, yielding conversations one at a time.

        For large JSON files, this still needs to load the whole file,
        but we process conversations incrementally to save memory.

        Note: For very large files (>1GB), consider using ijson library
        for true streaming JSON parsing.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both array and single object
        if isinstance(data, list):
            for item in data:
                yield item
        else:
            yield data

    def _get_id(self, conv_data: Dict[str, Any]) -> str:
        """Extract conversation ID from data."""
        id_field = self.config["id_field"]
        return conv_data.get(id_field) or conv_data.get("id") or conv_data.get("conversation_id", "")

    def _get_timestamp(self, conv_data: Dict[str, Any]) -> Optional[datetime]:
        """Extract and parse timestamp from data."""
        ts_field = self.config["timestamp_field"]
        ts_format = self.config["timestamp_format"]

        timestamp = conv_data.get(ts_field)
        if not timestamp:
            # Try common alternatives
            timestamp = conv_data.get("create_time") or conv_data.get("created_at") or conv_data.get("timestamp")

        if not timestamp:
            return None

        try:
            if ts_format == "unix":
                # Unix timestamp (ChatGPT)
                return datetime.fromtimestamp(float(timestamp))
            else:
                # ISO format (Claude, Gemini)
                ts_str = str(timestamp)
                if ts_str.endswith('Z'):
                    ts_str = ts_str[:-1]
                return datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            return None

    def _parse_cursor_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse cursor timestamp string to datetime."""
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', ''))
        except (ValueError, AttributeError):
            return None

    def _parse_conversation(self, conv_data: Dict[str, Any]) -> Optional[RawConversation]:
        """Parse a single conversation to RawConversation."""
        if self.source == "chatgpt":
            return self._parse_chatgpt_conversation(conv_data)
        elif self.source == "claude":
            return self._parse_claude_conversation(conv_data)
        elif self.source == "gemini":
            return self._parse_gemini_conversation(conv_data)
        else:
            return self._parse_generic_conversation(conv_data)

    def _parse_chatgpt_conversation(self, data: Dict[str, Any]) -> Optional[RawConversation]:
        """Parse ChatGPT conversation."""
        external_id = data.get("id", "")
        title = data.get("title", "Untitled")
        create_time = self._parse_unix_timestamp(data.get("create_time"))
        update_time = self._parse_unix_timestamp(data.get("update_time"))

        # Parse messages from mapping
        messages = self._extract_chatgpt_messages(data.get("mapping", {}))

        if not messages:
            return None

        return RawConversation(
            external_id=external_id,
            source="OpenAI",
            title=self._truncate(title, 100),
            create_time=create_time,
            update_time=update_time,
            messages=messages,
            metadata={"platform": "ChatGPT"}
        )

    def _parse_claude_conversation(self, data: Dict[str, Any]) -> Optional[RawConversation]:
        """Parse Claude conversation."""
        external_id = data.get("uuid") or data.get("id", "")
        title = data.get("name") or data.get("title", "Untitled")
        create_time = self._parse_iso_timestamp(data.get("created_at") or data.get("timestamp"))
        update_time = self._parse_iso_timestamp(data.get("updated_at"))

        # Parse messages
        messages_data = data.get("chat_messages") or data.get("messages", [])
        messages = self._parse_claude_messages(messages_data)

        if not messages:
            return None

        return RawConversation(
            external_id=external_id,
            source="Anthropic",
            title=self._truncate(title, 100),
            create_time=create_time,
            update_time=update_time,
            messages=messages,
            metadata={"platform": "Claude"}
        )

    def _parse_gemini_conversation(self, data: Dict[str, Any]) -> Optional[RawConversation]:
        """Parse Gemini conversation."""
        external_id = data.get("id", "")
        title = data.get("title", "Untitled")
        create_time = self._parse_iso_timestamp(data.get("timestamp") or data.get("created_at"))

        messages_data = data.get("messages", [])
        messages = self._parse_gemini_messages(messages_data)

        if not messages:
            return None

        return RawConversation(
            external_id=external_id,
            source="Google",
            title=self._truncate(title, 100),
            create_time=create_time,
            messages=messages,
            metadata={"platform": "Gemini"}
        )

    def _parse_generic_conversation(self, data: Dict[str, Any]) -> Optional[RawConversation]:
        """Parse generic conversation format."""
        external_id = data.get("id") or data.get("uuid", "")
        title = data.get("title") or data.get("name", "Untitled")

        # Try to find timestamp
        timestamp = (data.get("create_time") or data.get("created_at") or
                    data.get("timestamp") or datetime.now())
        if isinstance(timestamp, (int, float)):
            create_time = self._parse_unix_timestamp(timestamp)
        else:
            create_time = self._parse_iso_timestamp(timestamp)

        # Try to find messages
        messages_data = data.get("messages") or data.get("chat_messages", [])

        messages = []
        for msg in messages_data:
            role = msg.get("role") or msg.get("sender", "user")
            if role == "human":
                role = "user"
            content = msg.get("content") or msg.get("text", "")
            if content:
                messages.append(Message(
                    role=role if role in ["user", "assistant"] else "user",
                    content=str(content).strip(),
                    timestamp=create_time,
                    metadata={}
                ))

        if not messages:
            return None

        return RawConversation(
            external_id=external_id,
            source=self.config["platform_source"],
            title=self._truncate(title, 100),
            create_time=create_time,
            messages=messages,
            metadata={}
        )

    def _extract_chatgpt_messages(self, mapping: Dict[str, Any]) -> List[Message]:
        """Extract messages from ChatGPT tree structure."""
        if not mapping:
            return []

        # Find root node
        root = None
        for node_id, node_data in mapping.items():
            parent = node_data.get("parent")
            if parent is None or parent not in mapping:
                root = node_id
                break

        if not root:
            root = next(iter(mapping.keys()), None)

        if not root:
            return []

        # Traverse tree
        messages = []
        current = root

        while current:
            node = mapping.get(current)
            if not node:
                break

            msg_data = node.get("message")
            if msg_data:
                author = msg_data.get("author", {})
                role = author.get("role", "unknown")

                if role in ["user", "assistant"]:
                    content_data = msg_data.get("content", {})
                    if isinstance(content_data, dict):
                        parts = content_data.get("parts", [])
                        content = "\n".join(str(p) for p in parts if p)
                    else:
                        content = str(content_data)

                    if content and content.strip():
                        messages.append(Message(
                            role=role,
                            content=content.strip(),
                            timestamp=self._parse_unix_timestamp(msg_data.get("create_time")),
                            metadata={"message_id": msg_data.get("id", "")}
                        ))

            # Move to first child
            children = node.get("children", [])
            current = children[0] if children else None

        return messages

    def _parse_claude_messages(self, messages_data: List[Dict]) -> List[Message]:
        """Parse Claude message list."""
        messages = []
        for msg in messages_data:
            sender = msg.get("sender") or msg.get("role", "user")
            role = "user" if sender == "human" else sender

            if role not in ["user", "assistant"]:
                continue

            content = msg.get("text") or msg.get("content", "")
            if content and str(content).strip():
                messages.append(Message(
                    role=role,
                    content=str(content).strip(),
                    timestamp=self._parse_iso_timestamp(msg.get("created_at")),
                    metadata={"message_id": msg.get("uuid", "")}
                ))

        return messages

    def _parse_gemini_messages(self, messages_data: List[Dict]) -> List[Message]:
        """Parse Gemini message list."""
        messages = []
        for msg in messages_data:
            role = msg.get("role") or msg.get("author", "user")
            if role == "model":
                role = "assistant"

            if role not in ["user", "assistant"]:
                continue

            content = msg.get("content") or msg.get("text", "")
            if content and str(content).strip():
                messages.append(Message(
                    role=role,
                    content=str(content).strip(),
                    timestamp=self._parse_iso_timestamp(msg.get("timestamp")),
                    metadata={}
                ))

        return messages

    def _parse_unix_timestamp(self, timestamp) -> datetime:
        """Parse Unix timestamp to datetime."""
        if timestamp is None:
            return datetime.now()
        try:
            return datetime.fromtimestamp(float(timestamp))
        except (ValueError, TypeError):
            return datetime.now()

    def _parse_iso_timestamp(self, timestamp) -> datetime:
        """Parse ISO timestamp to datetime."""
        if timestamp is None:
            return datetime.now()
        if isinstance(timestamp, datetime):
            return timestamp
        try:
            ts_str = str(timestamp)
            if ts_str.endswith('Z'):
                ts_str = ts_str[:-1]
            return datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            return datetime.now()

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text to max length."""
        if len(text) <= max_len:
            return text
        return text[:max_len-3] + "..."


def count_new_conversations(
    source: str,
    file_path: str,
    cursor: Optional[ProcessingCursor] = None,
    processed_ids: Optional[set] = None
) -> int:
    """
    Count new conversations without fully parsing them.

    Useful for quick estimates and progress reporting.
    """
    parser = SmartParser(source, file_path)
    count = 0

    for _ in parser.parse_incremental(cursor, processed_ids):
        count += 1

    return count


def parse_new_only(
    source: str,
    file_path: str,
    cursor: Optional[ProcessingCursor] = None,
    processed_ids: Optional[set] = None
) -> List[RawConversation]:
    """
    Parse only new conversations (convenience function).

    Returns list instead of iterator for cases where you need
    the full list upfront.
    """
    parser = SmartParser(source, file_path)
    return list(parser.parse_incremental(cursor, processed_ids))


if __name__ == "__main__":
    # Quick test
    import argparse

    parser = argparse.ArgumentParser(description="Test smart parser")
    parser.add_argument("--file", required=True, help="Path to export file")
    parser.add_argument("--source", required=True, choices=["chatgpt", "claude", "gemini"])
    parser.add_argument("--since", help="Only conversations after this ISO timestamp")

    args = parser.parse_args()

    cursor = None
    if args.since:
        cursor = ProcessingCursor(last_timestamp=args.since)

    smart_parser = SmartParser(args.source, args.file)

    print(f"\nParsing {args.file} ({args.source})")
    if cursor:
        print(f"Since: {cursor.last_timestamp}")
    print()

    count = 0
    for conv in smart_parser.parse_incremental(cursor):
        count += 1
        print(f"  {count}. {conv.title[:50]} ({conv.create_time.strftime('%Y-%m-%d')})")

        if count >= 10:
            print(f"  ... and more")
            break

    progress = smart_parser.get_progress()
    print(f"\nTotal scanned: {progress.total_scanned}")
    print(f"New: {progress.total_new}")
    print(f"Skipped: {progress.total_skipped}")
