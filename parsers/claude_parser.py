"""
Parser for Claude conversation exports.

Handles multiple formats:
- JSON from browser DevTools (Network tab → chat_ responses)
- JSON/JSONL from Chrome extensions (Claude Exporter)
- JSONL from Claude Code (~/.claude/projects)

Structure: Linear message array (simpler than ChatGPT)
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models import RawConversation, Message
from parsers.base_parser import ConversationParser


class ClaudeParser(ConversationParser):
    """
    Parser for Claude conversation exports.

    Supports multiple formats:

    Format 1: Browser DevTools JSON
    {
      "uuid": "chat-uuid-123",
      "name": "Conversation Title",
      "created_at": "2025-10-15T14:30:00.000Z",
      "updated_at": "2025-10-15T15:45:00.000Z",
      "chat_messages": [
        {
          "uuid": "msg-uuid-1",
          "text": "Message content",
          "sender": "human" or "assistant",
          "created_at": "2025-10-15T14:30:00.000Z"
        }
      ]
    }

    Format 2: Extension Export
    Similar structure with possible variations in field names
    (e.g., "messages" instead of "chat_messages", "content" instead of "text")
    """

    def detect_format(self) -> bool:
        """
        Detect if file is a Claude export.

        Checks for:
        - JSON with 'uuid' and ('chat_messages' or 'messages')
        - ISO format timestamps
        - 'sender' field with 'human'/'assistant' values
        """
        try:
            # Try JSON first
            try:
                data = self._load_json()
            except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
                # Try JSONL
                data = self._load_jsonl()
                if not data:
                    return False
                data = data[0] if isinstance(data, list) else data

            # Handle array of conversations
            if isinstance(data, list):
                if not data:
                    return False
                sample = data[0]
            else:
                sample = data

            # Check for Claude-specific structure
            has_uuid = "uuid" in sample or "id" in sample
            has_messages = "chat_messages" in sample or "messages" in sample
            has_created_at = "created_at" in sample or "timestamp" in sample

            # Check if timestamps are ISO format (Claude uses ISO, ChatGPT uses Unix)
            if has_created_at:
                timestamp_field = "created_at" if "created_at" in sample else "timestamp"
                timestamp = sample.get(timestamp_field, "")
                is_iso_format = isinstance(timestamp, str) and ("T" in timestamp or "-" in timestamp)
                return has_uuid and has_messages and is_iso_format

            return has_uuid and has_messages

        except Exception:
            return False

    def parse(self) -> List[RawConversation]:
        """
        Parse Claude conversation export file.

        Returns:
            List of RawConversation objects
        """
        # Try JSON first, then JSONL
        try:
            data = self._load_json()
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            try:
                data = self._load_jsonl()
            except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
                raise ValueError("File is neither valid JSON nor JSONL")

        # Handle both single conversation and array
        if isinstance(data, dict):
            conversations_data = [data]
        else:
            conversations_data = data

        conversations = []
        for conv_data in conversations_data:
            try:
                conversation = self._parse_single_conversation(conv_data)
                if conversation and conversation.messages:
                    conversations.append(conversation)
            except Exception as e:
                print(f"Warning: Failed to parse conversation {conv_data.get('uuid', conv_data.get('id', 'unknown'))}: {e}")
                continue

        return conversations

    def _parse_single_conversation(self, data: Dict[str, Any]) -> Optional[RawConversation]:
        """
        Parse a single Claude conversation from JSON data.

        Args:
            data: Conversation JSON object

        Returns:
            RawConversation object or None if parsing fails
        """
        # Extract ID (try multiple possible fields)
        external_id = data.get("uuid") or data.get("id") or data.get("conversation_id", "")

        # Extract title (try multiple possible fields)
        title = data.get("name") or data.get("title") or "Untitled Conversation"

        # Extract timestamps
        create_time = self._parse_iso_timestamp(
            data.get("created_at") or data.get("timestamp") or data.get("create_time")
        )
        update_time = self._parse_iso_timestamp(
            data.get("updated_at") or data.get("update_time")
        )

        # Extract messages (try multiple possible fields)
        messages_data = data.get("chat_messages") or data.get("messages") or []
        messages = self._parse_messages(messages_data)

        if not messages:
            return None

        # Truncate title if too long
        title = self._truncate_title(title)

        return RawConversation(
            external_id=external_id,
            source="Anthropic",
            title=title,
            create_time=create_time,
            update_time=update_time,
            messages=messages,
            metadata={
                "platform": "Claude",
                "original_title": data.get("name") or data.get("title", ""),
                "message_count": len(messages)
            }
        )

    def _parse_messages(self, messages_data: List[Dict[str, Any]]) -> List[Message]:
        """
        Parse list of messages from Claude format.

        Args:
            messages_data: List of message objects

        Returns:
            List of Message objects
        """
        messages = []
        for msg_data in messages_data:
            message = self._parse_single_message(msg_data)
            if message:
                messages.append(message)
        return messages

    def _parse_single_message(self, msg_data: Dict[str, Any]) -> Optional[Message]:
        """
        Parse a single message from Claude format.

        Args:
            msg_data: Message object

        Returns:
            Message object or None if not valid
        """
        # Extract role (try multiple possible fields)
        sender = msg_data.get("sender") or msg_data.get("role") or msg_data.get("author")

        # Normalize role
        if sender == "human":
            role = "user"
        elif sender == "assistant":
            role = "assistant"
        elif sender in ["user", "assistant", "system"]:
            role = sender
        else:
            role = "user"  # Default

        # Skip system messages
        if role == "system":
            return None

        # Extract content (try multiple possible fields)
        content = msg_data.get("text") or msg_data.get("content") or msg_data.get("message", "")

        if not content or not str(content).strip():
            return None

        # Extract timestamp
        timestamp = self._parse_iso_timestamp(
            msg_data.get("created_at") or msg_data.get("timestamp") or msg_data.get("create_time")
        )

        # Extract metadata
        metadata = {
            "message_id": msg_data.get("uuid") or msg_data.get("id", ""),
            "sender": sender
        }

        return Message(
            role=role,
            content=str(content).strip(),
            timestamp=timestamp,
            metadata=metadata
        )

    def _parse_iso_timestamp(self, timestamp: Any) -> datetime:
        """
        Parse ISO format timestamp to datetime.

        Claude uses ISO 8601 format: "2025-10-15T14:30:00.000Z"

        Args:
            timestamp: ISO timestamp string or None

        Returns:
            datetime object (current time if timestamp is None)
        """
        if timestamp is None:
            return datetime.now()

        if isinstance(timestamp, datetime):
            return timestamp

        try:
            # Handle ISO format with timezone
            timestamp_str = str(timestamp)

            # Remove timezone suffix for parsing
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1]

            # Try parsing with microseconds
            try:
                return datetime.fromisoformat(timestamp_str)
            except ValueError:
                # Try without microseconds
                timestamp_str = timestamp_str.split('.')[0]
                return datetime.fromisoformat(timestamp_str)

        except (ValueError, TypeError):
            return datetime.now()
