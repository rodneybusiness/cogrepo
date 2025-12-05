"""
Data models for CogRepo conversation processing.

Defines the structure for raw conversations, messages, and enriched conversations.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Literal, Optional
from uuid import uuid4
import json


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO format timestamp."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class RawConversation:
    """
    Raw conversation data after parsing but before enrichment.

    This is the intermediate format that all parsers output,
    regardless of the source platform.
    """
    external_id: str  # Original ID from source platform
    source: Literal["OpenAI", "Anthropic", "Google"]  # ChatGPT, Claude, Gemini
    title: str  # Original title or generated fallback
    create_time: datetime
    update_time: Optional[datetime] = None
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def raw_text(self) -> str:
        """Generate raw text representation of the conversation."""
        lines = [f"Title: {self.title}", ""]
        for msg in self.messages:
            role = msg.role.upper()
            lines.append(f"{role}: {msg.content}")
            lines.append("")
        return "\n".join(lines)

    @property
    def message_count(self) -> int:
        """Total number of messages in the conversation."""
        return len(self.messages)

    @property
    def user_message_count(self) -> int:
        """Count of user messages."""
        return sum(1 for msg in self.messages if msg.role == "user")

    @property
    def assistant_message_count(self) -> int:
        """Count of assistant messages."""
        return sum(1 for msg in self.messages if msg.role == "assistant")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "external_id": self.external_id,
            "source": self.source,
            "title": self.title,
            "create_time": self.create_time.isoformat(),
            "update_time": self.update_time.isoformat() if self.update_time else None,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata
        }


@dataclass
class EnrichedConversation:
    """
    Fully enriched conversation with AI-generated metadata.

    This is the final format stored in enriched_repository.jsonl
    and used by the search tools.
    """
    # Core identification
    convo_id: str  # Internal UUID
    external_id: str  # Original ID from source
    timestamp: datetime  # Create time
    source: Literal["OpenAI", "Anthropic", "Google"]

    # Raw content
    raw_text: str

    # AI-generated enrichments
    generated_title: str
    summary_abstractive: str  # AI-generated summary (250-300 chars)
    summary_extractive: str  # Key sentences extracted

    # Classification and tagging
    primary_domain: str  # e.g., "Business Strategy", "Technical", "Creative"
    tags: List[str] = field(default_factory=list)
    key_topics: List[str] = field(default_factory=list)

    # Scoring and insights
    brilliance_score: Dict[str, Any] = field(default_factory=dict)  # {score: 1-10, reasoning: "..."}
    key_insights: List[str] = field(default_factory=list)

    # Status and potential
    status: str = "Completed"  # Resolved, Ongoing, Reference, etc.
    future_potential: Dict[str, Any] = field(default_factory=dict)  # {value_proposition: "...", next_steps: "..."}

    # Overall quality score
    score: int = 5  # 1-10 overall quality/importance
    score_reasoning: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: RawConversation, enrichments: Optional[Dict[str, Any]] = None) -> 'EnrichedConversation':
        """
        Create an EnrichedConversation from a RawConversation.

        If enrichments is None, creates a minimal version with only raw data.
        Otherwise, applies the AI-generated enrichments.
        """
        convo_id = str(uuid4())

        if enrichments is None:
            # Minimal enrichment (no AI)
            return cls(
                convo_id=convo_id,
                external_id=raw.external_id,
                timestamp=raw.create_time,
                source=raw.source,
                raw_text=raw.raw_text,
                generated_title=raw.title,
                summary_abstractive="",
                summary_extractive="",
                primary_domain="Uncategorized",
                tags=[],
                key_topics=[],
                brilliance_score={"score": 5, "reasoning": "Not yet scored"},
                key_insights=[],
                status="Imported",
                future_potential={},
                score=5,
                score_reasoning="Not yet evaluated",
                metadata={
                    "message_count": raw.message_count,
                    "user_messages": raw.user_message_count,
                    "assistant_messages": raw.assistant_message_count,
                    **raw.metadata
                }
            )
        else:
            # Full enrichment with AI-generated data
            return cls(
                convo_id=convo_id,
                external_id=raw.external_id,
                timestamp=raw.create_time,
                source=raw.source,
                raw_text=raw.raw_text,
                generated_title=enrichments.get("generated_title", raw.title),
                summary_abstractive=enrichments.get("summary_abstractive", ""),
                summary_extractive=enrichments.get("summary_extractive", ""),
                primary_domain=enrichments.get("primary_domain", "Uncategorized"),
                tags=enrichments.get("tags", []),
                key_topics=enrichments.get("key_topics", []),
                brilliance_score=enrichments.get("brilliance_score", {"score": 5, "reasoning": ""}),
                key_insights=enrichments.get("key_insights", []),
                status=enrichments.get("status", "Completed"),
                future_potential=enrichments.get("future_potential", {}),
                score=enrichments.get("score", 5),
                score_reasoning=enrichments.get("score_reasoning", ""),
                metadata={
                    "message_count": raw.message_count,
                    "user_messages": raw.user_message_count,
                    "assistant_messages": raw.assistant_message_count,
                    **raw.metadata
                }
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "convo_id": self.convo_id,
            "external_id": self.external_id,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "source": self.source,
            "raw_text": self.raw_text,
            "generated_title": self.generated_title,
            "summary_abstractive": self.summary_abstractive,
            "summary_extractive": self.summary_extractive,
            "primary_domain": self.primary_domain,
            "tags": self.tags,
            "key_topics": self.key_topics,
            "brilliance_score": self.brilliance_score,
            "key_insights": self.key_insights,
            "status": self.status,
            "future_potential": self.future_potential,
            "score": self.score,
            "score_reasoning": self.score_reasoning,
            "metadata": self.metadata
        }

    def to_jsonl(self) -> str:
        """Convert to JSONL format (single line JSON)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnrichedConversation':
        """Create from dictionary (for loading from JSON)."""
        # Convert timestamp string back to datetime if needed
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        return cls(**data)


@dataclass
class ProcessingStats:
    """Statistics about the processing run."""
    total_found: int = 0
    total_new: int = 0
    total_duplicates: int = 0
    total_processed: int = 0
    total_failed: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        """Calculate duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def processing_rate(self) -> float:
        """Conversations processed per second."""
        if self.duration_seconds > 0:
            return self.total_processed / self.duration_seconds
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_found": self.total_found,
            "total_new": self.total_new,
            "total_duplicates": self.total_duplicates,
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "processing_rate": self.processing_rate
        }
