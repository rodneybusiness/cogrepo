"""
Parser for ChatGPT conversation exports.

Handles the conversations.json format from OpenAI exports.
Structure: Tree-based with node mapping and parent-child relationships.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models import RawConversation, Message
from parsers.base_parser import ConversationParser


class ChatGPTParser(ConversationParser):
    """
    Parser for ChatGPT conversations.json exports.

    Format:
    {
      "id": "conversation-id",
      "title": "Conversation Title",
      "create_time": 1678015311.655875,
      "update_time": 1678015500.123456,
      "mapping": {
        "node-id": {
          "id": "node-id",
          "message": {...},
          "parent": "parent-node-id",
          "children": ["child-node-id"]
        }
      }
    }
    """

    def detect_format(self) -> bool:
        """
        Detect if file is a ChatGPT export.

        Checks for:
        - JSON array or single object
        - Contains 'mapping' field
        - Contains 'create_time' as Unix timestamp
        """
        try:
            data = self._load_json()

            # Handle both single conversation and array of conversations
            if isinstance(data, list):
                if not data:
                    return False
                sample = data[0]
            else:
                sample = data

            # Check for ChatGPT-specific structure
            return (
                isinstance(sample, dict) and
                "mapping" in sample and
                "create_time" in sample and
                isinstance(sample.get("create_time"), (int, float))
            )
        except Exception:
            return False

    def parse(self) -> List[RawConversation]:
        """
        Parse ChatGPT conversations.json file.

        Returns:
            List of RawConversation objects
        """
        data = self._load_json()

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
                print(f"Warning: Failed to parse conversation {conv_data.get('id', 'unknown')}: {e}")
                continue

        return conversations

    def _parse_single_conversation(self, data: Dict[str, Any]) -> Optional[RawConversation]:
        """
        Parse a single ChatGPT conversation from JSON data.

        Args:
            data: Conversation JSON object

        Returns:
            RawConversation object or None if parsing fails
        """
        # Extract basic metadata
        external_id = data.get("id", "")
        title = data.get("title", "Untitled Conversation")
        create_time = self._parse_timestamp(data.get("create_time"))
        update_time = self._parse_timestamp(data.get("update_time"))

        # Parse the conversation tree to extract messages
        mapping = data.get("mapping", {})
        messages = self._extract_messages_from_mapping(mapping)

        if not messages:
            return None

        # Truncate title if too long
        title = self._truncate_title(title)

        return RawConversation(
            external_id=external_id,
            source="OpenAI",
            title=title,
            create_time=create_time,
            update_time=update_time,
            messages=messages,
            metadata={
                "platform": "ChatGPT",
                "original_title": data.get("title", ""),
                "node_count": len(mapping)
            }
        )

    def _extract_messages_from_mapping(self, mapping: Dict[str, Any]) -> List[Message]:
        """
        Extract linear message flow from ChatGPT's tree structure.

        ChatGPT stores conversations as a tree (for branching).
        We reconstruct the main conversation path.

        Strategy:
        1. Find the root node (no parent or parent is null)
        2. Follow the primary path (first child at each level)
        3. Extract user/assistant messages

        Args:
            mapping: The mapping dict from ChatGPT export

        Returns:
            List of Message objects in chronological order
        """
        if not mapping:
            return []

        # Find root node (parent is None or not in mapping)
        root = None
        for node_id, node_data in mapping.items():
            parent = node_data.get("parent")
            if parent is None or parent not in mapping:
                root = node_id
                break

        if not root:
            # Fallback: use first node
            root = next(iter(mapping.keys()))

        # Traverse from root to leaf, collecting messages
        messages = []
        current_node = root

        while current_node:
            node_data = mapping.get(current_node)
            if not node_data:
                break

            # Extract message if present
            message_data = node_data.get("message")
            if message_data:
                message = self._parse_message(message_data)
                if message:
                    messages.append(message)

            # Move to first child (primary path)
            children = node_data.get("children", [])
            if children:
                current_node = children[0]  # Take primary path
            else:
                current_node = None  # Reached leaf

        return messages

    def _parse_message(self, message_data: Dict[str, Any]) -> Optional[Message]:
        """
        Parse a single message from ChatGPT format.

        Args:
            message_data: Message object from mapping node

        Returns:
            Message object or None if not a valid message
        """
        if not message_data:
            return None

        # Extract author role
        author = message_data.get("author", {})
        role = author.get("role", "unknown")

        # Skip system messages and non-text messages
        if role == "system":
            return None

        # Normalize role
        if role not in ["user", "assistant"]:
            role = "user"  # Default to user for unknown roles

        # Extract content
        content_data = message_data.get("content", {})
        if isinstance(content_data, dict):
            parts = content_data.get("parts", [])
            if not parts:
                return None
            # Join all parts (usually just one)
            content = "\n".join(str(part) for part in parts if part)
        elif isinstance(content_data, str):
            content = content_data
        else:
            return None

        if not content or not content.strip():
            return None

        # Extract timestamp
        create_time = self._parse_timestamp(message_data.get("create_time"))

        # Extract metadata
        metadata = {
            "message_id": message_data.get("id", ""),
            "author_metadata": author.get("metadata", {}),
            "content_type": content_data.get("content_type", "text") if isinstance(content_data, dict) else "text"
        }

        return Message(
            role=role,
            content=content.strip(),
            timestamp=create_time,
            metadata=metadata
        )

    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """
        Parse ChatGPT Unix timestamp to datetime.

        Args:
            timestamp: Unix timestamp (float or int) or None

        Returns:
            datetime object (current time if timestamp is None)
        """
        if timestamp is None:
            return datetime.now()

        try:
            # ChatGPT uses Unix timestamps (seconds since epoch)
            return datetime.fromtimestamp(float(timestamp))
        except (ValueError, TypeError):
            return datetime.now()
