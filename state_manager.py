"""
State management for tracking processed conversations.

Handles:
- Tracking which conversations have been processed
- Deduplication using external IDs and content hashes
- Processing statistics
- Failed conversation tracking for retry
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, asdict

import sys
sys.path.append(str(Path(__file__).parent))
from models import RawConversation, ProcessingStats


@dataclass
class FailedConversation:
    """Record of a failed conversation processing attempt."""
    source: Literal["OpenAI", "Anthropic", "Google"]
    external_id: str
    error: str
    attempts: int
    last_attempted: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "external_id": self.external_id,
            "error": self.error,
            "attempts": self.attempts,
            "last_attempted": self.last_attempted.isoformat()
        }


class ProcessingStateManager:
    """
    Manages the state of conversation processing for incremental updates.

    State file structure:
    {
      "last_updated": "2025-10-31T12:00:00Z",
      "sources": {
        "OpenAI": {"total": 2100, "last_import": "...", "last_conversation": "..."},
        "Anthropic": {...},
        "Google": {...}
      },
      "processed_conversations": {
        "OpenAI": {
          "conv-123": {
            "internal_uuid": "uuid-xyz",
            "processed_at": "2025-10-15T12:00:00Z",
            "content_hash": "sha256:abc...",
            "conversation_date": "2025-10-15T10:30:00Z"
          }
        }
      },
      "processing_stats": {
        "total_processed": 3748,
        "total_enriched": 3748,
        "total_failed": 0,
        ...
      },
      "failed_conversations": [...]
    }
    """

    def __init__(self, state_file: str = "data/processing_state.json"):
        """
        Initialize state manager.

        Args:
            state_file: Path to the state file (relative to cogrepo root)
        """
        self.state_file = Path(__file__).parent / state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load state from file, or create new state if file doesn't exist."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load state file: {e}")
                print("Creating new state...")
                return self._create_new_state()
        else:
            return self._create_new_state()

    def _create_new_state(self) -> Dict[str, Any]:
        """Create a new empty state structure."""
        return {
            "last_updated": None,
            "sources": {
                "OpenAI": {
                    "total_conversations": 0,
                    "last_import_date": None,
                    "last_conversation_date": None
                },
                "Anthropic": {
                    "total_conversations": 0,
                    "last_import_date": None,
                    "last_conversation_date": None
                },
                "Google": {
                    "total_conversations": 0,
                    "last_import_date": None,
                    "last_conversation_date": None
                }
            },
            "processed_conversations": {
                "OpenAI": {},
                "Anthropic": {},
                "Google": {}
            },
            "content_hashes": {},  # hash -> source/external_id for reverse lookup
            "processing_stats": {
                "total_processed": 0,
                "total_enriched": 0,
                "total_failed": 0,
                "average_processing_time_ms": 0,
                "last_batch_size": 0,
                "last_batch_duration_seconds": 0
            },
            "failed_conversations": []
        }

    def save(self):
        """Save current state to file."""
        # Ensure directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Update last_updated timestamp
        self.state["last_updated"] = datetime.now().isoformat()

        # Write to file
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def is_processed(self, external_id: str, source: Literal["OpenAI", "Anthropic", "Google"]) -> bool:
        """
        Check if a conversation has already been processed.

        Args:
            external_id: External conversation ID from source platform
            source: Source platform

        Returns:
            True if conversation has been processed
        """
        processed = self.state["processed_conversations"].get(source, {})
        return external_id in processed

    def is_duplicate_content(self, content_hash: str) -> bool:
        """
        Check if content with this hash has already been processed.

        Args:
            content_hash: SHA256 hash of conversation content

        Returns:
            True if content hash exists
        """
        return content_hash in self.state.get("content_hashes", {})

    def get_content_hash(self, conversation: RawConversation) -> str:
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
            conversation.messages[-1].content if len(conversation.messages) > 1 else "",
            conversation.create_time.isoformat()
        ]
        content = "|".join(components)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def add_processed(
        self,
        external_id: str,
        internal_uuid: str,
        source: Literal["OpenAI", "Anthropic", "Google"],
        content_hash: str,
        conversation_date: datetime
    ):
        """
        Mark a conversation as processed.

        Args:
            external_id: External ID from source platform
            internal_uuid: Internal UUID assigned to conversation
            source: Source platform
            content_hash: SHA256 hash of content
            conversation_date: Date of conversation creation
        """
        # Add to processed conversations
        if source not in self.state["processed_conversations"]:
            self.state["processed_conversations"][source] = {}

        self.state["processed_conversations"][source][external_id] = {
            "internal_uuid": internal_uuid,
            "processed_at": datetime.now().isoformat(),
            "content_hash": content_hash,
            "conversation_date": conversation_date.isoformat() if isinstance(conversation_date, datetime) else conversation_date
        }

        # Add to content hashes (for reverse lookup)
        if "content_hashes" not in self.state:
            self.state["content_hashes"] = {}

        self.state["content_hashes"][content_hash] = {
            "source": source,
            "external_id": external_id,
            "internal_uuid": internal_uuid
        }

        # Update source stats
        self.state["sources"][source]["total_conversations"] = len(
            self.state["processed_conversations"][source]
        )

        # Update last conversation date if this is more recent
        last_date = self.state["sources"][source]["last_conversation_date"]
        conv_date_str = conversation_date.isoformat() if isinstance(conversation_date, datetime) else conversation_date

        if last_date is None or conv_date_str > last_date:
            self.state["sources"][source]["last_conversation_date"] = conv_date_str

    def add_failed(
        self,
        external_id: str,
        source: Literal["OpenAI", "Anthropic", "Google"],
        error: str
    ):
        """
        Record a failed conversation processing attempt.

        Args:
            external_id: External ID from source platform
            source: Source platform
            error: Error message
        """
        failed_list = self.state.get("failed_conversations", [])

        # Check if this conversation already has failed attempts
        existing = None
        for i, failed in enumerate(failed_list):
            if failed["external_id"] == external_id and failed["source"] == source:
                existing = i
                break

        if existing is not None:
            # Increment attempts
            failed_list[existing]["attempts"] += 1
            failed_list[existing]["error"] = error
            failed_list[existing]["last_attempted"] = datetime.now().isoformat()
        else:
            # Add new failed record
            failed_list.append({
                "source": source,
                "external_id": external_id,
                "error": error,
                "attempts": 1,
                "last_attempted": datetime.now().isoformat()
            })

        self.state["failed_conversations"] = failed_list
        self.state["processing_stats"]["total_failed"] = len(failed_list)

    def update_batch_stats(self, stats: ProcessingStats):
        """
        Update processing statistics after a batch.

        Args:
            stats: ProcessingStats object
        """
        current_stats = self.state["processing_stats"]

        # Update cumulative totals
        current_stats["total_processed"] += stats.total_processed

        # Update batch-specific stats
        current_stats["last_batch_size"] = stats.total_processed
        current_stats["last_batch_duration_seconds"] = stats.duration_seconds

        # Update average processing time (rolling average)
        if stats.duration_seconds > 0 and stats.total_processed > 0:
            batch_avg_ms = (stats.duration_seconds / stats.total_processed) * 1000
            current_avg = current_stats.get("average_processing_time_ms", 0)

            # Weighted average (give more weight to recent batches)
            if current_avg == 0:
                current_stats["average_processing_time_ms"] = batch_avg_ms
            else:
                current_stats["average_processing_time_ms"] = (current_avg * 0.7 + batch_avg_ms * 0.3)

    def update_source_import_date(self, source: Literal["OpenAI", "Anthropic", "Google"]):
        """
        Update the last import date for a source.

        Args:
            source: Source platform
        """
        self.state["sources"][source]["last_import_date"] = datetime.now().isoformat()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_processed": self.state["processing_stats"]["total_processed"],
            "total_failed": self.state["processing_stats"]["total_failed"],
            "by_source": {
                source: data["total_conversations"]
                for source, data in self.state["sources"].items()
            },
            "last_updated": self.state.get("last_updated"),
            "failed_conversations_count": len(self.state.get("failed_conversations", []))
        }

    def get_failed_conversations(self) -> List[Dict[str, Any]]:
        """
        Get list of failed conversations.

        Returns:
            List of failed conversation records
        """
        return self.state.get("failed_conversations", [])

    def clear_failed_conversation(self, external_id: str, source: str):
        """
        Remove a conversation from the failed list (after successful retry).

        Args:
            external_id: External ID from source platform
            source: Source platform
        """
        failed_list = self.state.get("failed_conversations", [])
        self.state["failed_conversations"] = [
            f for f in failed_list
            if not (f["external_id"] == external_id and f["source"] == source)
        ]
        self.state["processing_stats"]["total_failed"] = len(self.state["failed_conversations"])

    def get_processed_count(self, source: Optional[Literal["OpenAI", "Anthropic", "Google"]] = None) -> int:
        """
        Get count of processed conversations.

        Args:
            source: Optional source to filter by

        Returns:
            Count of processed conversations
        """
        if source:
            return len(self.state["processed_conversations"].get(source, {}))
        else:
            return sum(
                len(convs)
                for convs in self.state["processed_conversations"].values()
            )
