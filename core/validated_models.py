"""
Validated Data Models for CogRepo

Pydantic models with comprehensive validation for:
- Messages and conversations
- Enrichment results
- Configuration settings
- API requests/responses

Benefits over dataclasses:
- Runtime validation
- Type coercion
- JSON schema generation
- Better error messages
- Serialization/deserialization

Usage:
    from core.validated_models import Message, Conversation, EnrichedConversation

    # Validation happens automatically
    msg = Message(role="user", content="Hello")

    # Invalid data raises ValidationError
    msg = Message(role="invalid", content="")  # Raises error
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum
import json


# =============================================================================
# Enums for Type Safety
# =============================================================================

class MessageRole(str, Enum):
    """Valid message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationSource(str, Enum):
    """Valid conversation sources."""
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    GOOGLE = "Google"


class ConversationStatus(str, Enum):
    """Valid conversation statuses."""
    COMPLETED = "Completed"
    ONGOING = "Ongoing"
    REFERENCE = "Reference"
    PLANNING = "Planning"
    RESOLVED = "Resolved"
    ARCHIVED = "Archived"
    IMPORTED = "Imported"


class PrimaryDomain(str, Enum):
    """Common primary domains."""
    BUSINESS = "Business"
    TECHNICAL = "Technical"
    CREATIVE = "Creative"
    PERSONAL = "Personal"
    SCIENCE = "Science"
    EDUCATION = "Education"
    UNCATEGORIZED = "Uncategorized"


# =============================================================================
# Message Model
# =============================================================================

class Message(BaseModel):
    """
    Validated message within a conversation.

    Validates:
    - Role is valid enum value
    - Content is non-empty
    - Timestamp is valid datetime
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    role: MessageRole
    content: str = Field(min_length=1, max_length=500000)
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Message content cannot be empty')
        return v.strip()

    @field_validator('role', mode='before')
    @classmethod
    def normalize_role(cls, v: str) -> str:
        """Normalize role names from different platforms."""
        role_map = {
            'human': 'user',
            'model': 'assistant',
            'ai': 'assistant',
            'bot': 'assistant'
        }
        if isinstance(v, str):
            return role_map.get(v.lower(), v.lower())
        return v


# =============================================================================
# Conversation Models
# =============================================================================

class RawConversation(BaseModel):
    """
    Raw conversation after parsing, before enrichment.

    Validates:
    - Required fields present
    - Source is valid
    - At least one message
    - Timestamps are valid
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    external_id: str = Field(min_length=1)
    source: ConversationSource
    title: str = Field(default="Untitled", max_length=500)
    create_time: datetime
    update_time: Optional[datetime] = None
    messages: List[Message] = Field(min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def raw_text(self) -> str:
        """Generate raw text representation."""
        lines = [f"Title: {self.title}", ""]
        for msg in self.messages:
            lines.append(f"{msg.role.value.upper()}: {msg.content}")
            lines.append("")
        return "\n".join(lines)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def user_message_count(self) -> int:
        return sum(1 for m in self.messages if m.role == MessageRole.USER)

    @property
    def assistant_message_count(self) -> int:
        return sum(1 for m in self.messages if m.role == MessageRole.ASSISTANT)

    @field_validator('source', mode='before')
    @classmethod
    def normalize_source(cls, v: str) -> str:
        """Normalize source names."""
        source_map = {
            'chatgpt': 'OpenAI',
            'openai': 'OpenAI',
            'claude': 'Anthropic',
            'anthropic': 'Anthropic',
            'gemini': 'Google',
            'google': 'Google'
        }
        if isinstance(v, str):
            return source_map.get(v.lower(), v)
        return v


class BrillianceScore(BaseModel):
    """Validated brilliance score."""
    score: int = Field(ge=1, le=10, default=5)
    reasoning: str = Field(default="")
    factors: Dict[str, int] = Field(default_factory=dict)

    @field_validator('score', mode='before')
    @classmethod
    def ensure_int(cls, v):
        if isinstance(v, float):
            return int(round(v))
        return v


class FuturePotential(BaseModel):
    """Validated future potential assessment."""
    value_proposition: str = Field(default="")
    next_steps: str = Field(default="")


class EnrichmentMetadata(BaseModel):
    """Metadata about the enrichment process."""
    total_tokens: int = Field(ge=0, default=0)
    total_cost_usd: float = Field(ge=0, default=0.0)
    processing_time_ms: float = Field(ge=0, default=0.0)
    overall_confidence: float = Field(ge=0, le=1, default=0.0)
    enriched_at: Optional[datetime] = None
    model_used: Optional[str] = None
    schema_version: str = Field(default="2.0")


class EnrichedConversation(BaseModel):
    """
    Fully enriched conversation with AI-generated metadata.

    Validates all enrichment fields with appropriate constraints.
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    # Core identification
    convo_id: str = Field(min_length=1)
    external_id: str = Field(min_length=1)
    timestamp: datetime
    source: ConversationSource

    # Raw content
    raw_text: str = Field(min_length=1)

    # AI-generated enrichments
    generated_title: str = Field(default="Untitled", max_length=500)
    summary_abstractive: str = Field(default="", max_length=2000)
    summary_extractive: str = Field(default="", max_length=5000)

    # Classification
    primary_domain: str = Field(default="Uncategorized")
    tags: List[str] = Field(default_factory=list, max_length=20)
    key_topics: List[str] = Field(default_factory=list, max_length=10)

    # Scoring
    brilliance_score: BrillianceScore = Field(default_factory=BrillianceScore)
    score: int = Field(ge=1, le=10, default=5)
    score_reasoning: str = Field(default="")

    # Insights
    key_insights: List[str] = Field(default_factory=list, max_length=10)
    status: ConversationStatus = Field(default=ConversationStatus.COMPLETED)
    future_potential: FuturePotential = Field(default_factory=FuturePotential)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    enrichment_metadata: Optional[EnrichmentMetadata] = None

    @field_validator('source', mode='before')
    @classmethod
    def normalize_source(cls, v: str) -> str:
        source_map = {
            'chatgpt': 'OpenAI',
            'openai': 'OpenAI',
            'claude': 'Anthropic',
            'anthropic': 'Anthropic',
            'gemini': 'Google',
            'google': 'Google'
        }
        if isinstance(v, str):
            return source_map.get(v.lower(), v)
        return v

    @field_validator('status', mode='before')
    @classmethod
    def normalize_status(cls, v):
        if isinstance(v, str):
            # Try to match case-insensitively
            for status in ConversationStatus:
                if status.value.lower() == v.lower():
                    return status
            return ConversationStatus.COMPLETED
        return v

    @field_validator('score', mode='before')
    @classmethod
    def ensure_score_int(cls, v):
        if isinstance(v, float):
            return int(round(v))
        if v is None:
            return 5
        return v

    @field_validator('tags', mode='before')
    @classmethod
    def ensure_tags_list(cls, v):
        if isinstance(v, str):
            return [t.strip() for t in v.split(',') if t.strip()]
        return v or []

    @field_validator('brilliance_score', mode='before')
    @classmethod
    def parse_brilliance_score(cls, v):
        if isinstance(v, dict):
            return BrillianceScore(**v)
        return v

    @field_validator('future_potential', mode='before')
    @classmethod
    def parse_future_potential(cls, v):
        if isinstance(v, dict):
            return FuturePotential(**v)
        return v

    @classmethod
    def from_raw(cls, raw: RawConversation, enrichments: Optional[Dict[str, Any]] = None) -> 'EnrichedConversation':
        """Create from RawConversation with optional enrichments."""
        import uuid

        base = {
            'convo_id': str(uuid.uuid4()),
            'external_id': raw.external_id,
            'timestamp': raw.create_time,
            'source': raw.source,
            'raw_text': raw.raw_text,
            'metadata': {
                'message_count': raw.message_count,
                'user_messages': raw.user_message_count,
                'assistant_messages': raw.assistant_message_count,
                **raw.metadata
            }
        }

        if enrichments:
            base.update(enrichments)
        else:
            base['generated_title'] = raw.title
            base['status'] = ConversationStatus.IMPORTED

        return cls(**base)

    def to_jsonl(self) -> str:
        """Convert to JSONL format."""
        return self.model_dump_json()


# =============================================================================
# Processing Models
# =============================================================================

class ProcessingStats(BaseModel):
    """Statistics about a processing run."""
    total_found: int = Field(ge=0, default=0)
    total_new: int = Field(ge=0, default=0)
    total_duplicates: int = Field(ge=0, default=0)
    total_processed: int = Field(ge=0, default=0)
    total_failed: int = Field(ge=0, default=0)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def processing_rate(self) -> float:
        if self.duration_seconds > 0:
            return self.total_processed / self.duration_seconds
        return 0.0


class ArchiveInfo(BaseModel):
    """Information about a registered archive."""
    id: str
    name: str
    source: ConversationSource
    file_path: str
    file_hash: Optional[str] = None
    file_size: int = Field(ge=0, default=0)
    last_modified: Optional[datetime] = None
    total_conversations: int = Field(ge=0, default=0)
    processed_conversations: int = Field(ge=0, default=0)
    pending_conversations: int = Field(ge=0, default=0)
    last_sync_at: Optional[datetime] = None
    auto_sync: bool = True
    enabled: bool = True


# =============================================================================
# API Response Models
# =============================================================================

class SearchResult(BaseModel):
    """Single search result."""
    convo_id: str
    external_id: str
    title: str
    summary: str
    source: str
    timestamp: str
    score: int = Field(ge=1, le=10)
    relevance: float = Field(ge=0)
    tags: List[str] = Field(default_factory=list)
    highlights: Dict[str, str] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search API response."""
    results: List[SearchResult]
    total_count: int = Field(ge=0)
    query: str
    filters: Optional[Dict[str, Any]] = None
    query_time_ms: float = Field(ge=0)


class SyncResult(BaseModel):
    """Result of a sync operation."""
    archive_name: str
    processed: int = Field(ge=0)
    failed: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
    success: bool


# =============================================================================
# Utility Functions
# =============================================================================

def validate_conversation(data: Dict[str, Any]) -> EnrichedConversation:
    """
    Validate and parse a conversation dictionary.

    Raises ValidationError if invalid.
    """
    return EnrichedConversation.model_validate(data)


def validate_message(data: Dict[str, Any]) -> Message:
    """Validate and parse a message dictionary."""
    return Message.model_validate(data)


def safe_parse_conversation(data: Dict[str, Any]) -> Optional[EnrichedConversation]:
    """
    Safely parse a conversation, returning None on validation error.

    Use this when processing untrusted data.
    """
    try:
        return EnrichedConversation.model_validate(data)
    except Exception:
        return None


def conversation_to_dict(conv: EnrichedConversation) -> Dict[str, Any]:
    """Convert conversation to plain dictionary."""
    return conv.model_dump(mode='json')


# =============================================================================
# Schema Export
# =============================================================================

def get_json_schema() -> Dict[str, Any]:
    """Get JSON schema for all models."""
    return {
        "Message": Message.model_json_schema(),
        "RawConversation": RawConversation.model_json_schema(),
        "EnrichedConversation": EnrichedConversation.model_json_schema(),
        "ProcessingStats": ProcessingStats.model_json_schema(),
        "SearchResult": SearchResult.model_json_schema(),
        "SearchResponse": SearchResponse.model_json_schema()
    }


if __name__ == "__main__":
    # Test validation
    print("Testing Pydantic models...\n")

    # Test Message
    msg = Message(role="human", content="Hello, world!")
    print(f"Message role normalized: {msg.role}")

    # Test validation error
    try:
        bad_msg = Message(role="user", content="")
    except Exception as e:
        print(f"Empty content rejected: {type(e).__name__}")

    # Test RawConversation
    raw = RawConversation(
        external_id="test-123",
        source="chatgpt",
        title="Test",
        create_time=datetime.now(),
        messages=[msg]
    )
    print(f"Source normalized: {raw.source}")
    print(f"Message count: {raw.message_count}")

    # Test EnrichedConversation
    enriched = EnrichedConversation.from_raw(raw, {
        'generated_title': "Generated Title",
        'summary_abstractive': "Test summary",
        'tags': ['test', 'validation'],
        'score': 7.5  # Should be converted to int
    })
    print(f"Score as int: {enriched.score}")
    print(f"Tags: {enriched.tags}")

    # Export schema
    schema = get_json_schema()
    print(f"\nJSON schemas generated for {len(schema)} models")

    print("\n All validations passed!")
