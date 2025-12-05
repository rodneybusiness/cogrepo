# CogRepo v2: Enrichment Schema Reference

> Complete data structures for all enrichment tiers.

---

## Overview

CogRepo v2 uses a 6-tier enrichment system:

| Tier | Name | Cost | Purpose |
|------|------|------|---------|
| 1 | Metadata | Already paid | Core AI enrichment (existing) |
| 2 | Zero-Token | $0 | Extract from raw text |
| 3 | Semantic | $0 | Embeddings for similarity |
| 4 | Artifacts | ~$0.80/1K | Extract reusable code/commands |
| 5 | Context | ~$0.60/1K | Projects, chains, graph |
| 6 | Deep Analysis | ~$7/1K | Detailed analysis (optional) |

---

## Complete EnrichedConversation Schema

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

@dataclass
class EnrichedConversation:
    """Complete enrichment schema for CogRepo v2."""

    # ═══════════════════════════════════════════════════════════════
    # IDENTITY
    # ═══════════════════════════════════════════════════════════════

    convo_id: str
    """Internal UUID for this conversation."""

    external_id: str
    """Original ID from source platform (ChatGPT, Claude, Gemini)."""

    timestamp: datetime
    """When the conversation was created."""

    source: Literal["OpenAI", "Anthropic", "Google"]
    """Source platform."""

    raw_text: str
    """Full conversation text with role prefixes."""

    # ═══════════════════════════════════════════════════════════════
    # TIER 1: METADATA (Existing AI Enrichment)
    # ═══════════════════════════════════════════════════════════════

    generated_title: str
    """AI-generated descriptive title (5-8 words)."""

    summary_abstractive: str
    """AI-written summary capturing essence (250-300 chars)."""

    summary_extractive: str
    """Key sentences extracted from conversation."""

    primary_domain: str
    """Main category: Business, Technical, Creative, Personal, Science, etc."""

    tags: List[str] = field(default_factory=list)
    """5-10 relevant keywords/tags."""

    key_topics: List[str] = field(default_factory=list)
    """3-5 main topics discussed."""

    brilliance_score: Dict[str, Any] = field(default_factory=dict)
    """
    Quality evaluation with breakdown.
    Structure:
    {
        "score": 7,           # 1-10 overall
        "reasoning": "...",   # Brief explanation
        "factors": {
            "depth": 8,        # How deep/insightful
            "actionability": 6, # How practical
            "creativity": 7,   # How novel
            "problem_solving": 7  # How effective
        }
    }
    """

    key_insights: List[str] = field(default_factory=list)
    """3-5 most important takeaways."""

    status: str = "Completed"
    """Current state: Completed, Ongoing, Reference, Planning, Resolved, Archived."""

    future_potential: Dict[str, Any] = field(default_factory=dict)
    """
    Value proposition and next steps.
    Structure:
    {
        "value_proposition": "What makes this valuable",
        "next_steps": "Suggested follow-up actions"
    }
    """

    score: int = 5
    """Overall quality score 1-10 (same as brilliance_score.score)."""

    score_reasoning: str = ""
    """Brief explanation of the score."""

    # ═══════════════════════════════════════════════════════════════
    # TIER 2: ZERO-TOKEN METRICS (Extracted from raw text)
    # ═══════════════════════════════════════════════════════════════

    has_code: bool = False
    """Whether conversation contains code blocks."""

    code_languages: List[str] = field(default_factory=list)
    """Programming languages detected in code blocks."""

    code_block_count: int = 0
    """Number of code blocks in conversation."""

    has_error_traces: bool = False
    """Whether conversation contains stack traces/error messages."""

    turn_count: int = 0
    """Number of user-assistant exchanges."""

    question_count: int = 0
    """Number of questions asked (count of '?')."""

    question_types: List[str] = field(default_factory=list)
    """Types of questions: how-to, why, what, debug, explain."""

    avg_user_length: int = 0
    """Average character count of user messages."""

    avg_assistant_length: int = 0
    """Average character count of assistant messages."""

    has_links: bool = False
    """Whether conversation contains URLs."""

    link_domains: List[str] = field(default_factory=list)
    """Domains of URLs mentioned (e.g., github.com, stackoverflow.com)."""

    link_urls: List[str] = field(default_factory=list)
    """Full URLs mentioned (limited to 20)."""

    technical_terms: List[str] = field(default_factory=list)
    """Technical terms detected (api, database, docker, etc.)."""

    mentioned_files: List[str] = field(default_factory=list)
    """File paths mentioned in conversation."""

    duration_minutes: Optional[int] = None
    """Duration from first to last message (if timestamps available)."""

    # ═══════════════════════════════════════════════════════════════
    # TIER 3: SEMANTIC (Embeddings)
    # ═══════════════════════════════════════════════════════════════

    embedding: Optional[List[float]] = None
    """
    384-dimensional semantic vector from sentence-transformers.
    Used for similarity search and clustering.
    Model: all-MiniLM-L6-v2
    """

    cluster_id: Optional[str] = None
    """Cluster assignment from HDBSCAN (if clustered)."""

    # ═══════════════════════════════════════════════════════════════
    # TIER 4: ARTIFACTS (Extracted reusable items)
    # ═══════════════════════════════════════════════════════════════

    artifacts: List[Dict] = field(default_factory=list)
    """
    Reusable items extracted from conversation.
    Each artifact:
    {
        "artifact_id": "conv123_art_0",
        "artifact_type": "code_snippet|shell_command|configuration|...",
        "content": "actual code or command",
        "language": "python|bash|sql|...",
        "description": "what this does",
        "use_case": "when to use this",
        "tags": ["docker", "redis"],
        "verified_working": true
    }
    """

    artifact_count: int = 0
    """Number of artifacts extracted."""

    has_reusable_code: bool = False
    """Whether artifacts include code_snippet type."""

    has_shell_commands: bool = False
    """Whether artifacts include shell_command type."""

    has_error_solutions: bool = False
    """Whether artifacts include error_solution type."""

    # ═══════════════════════════════════════════════════════════════
    # TIER 5: CONTEXT (Projects, Chains, Graph)
    # ═══════════════════════════════════════════════════════════════

    project_id: Optional[str] = None
    """Auto-inferred project this conversation belongs to."""

    project_name: Optional[str] = None
    """Human-readable project name."""

    chain_id: Optional[str] = None
    """If part of a conversation chain, the chain ID."""

    chain_position: Optional[int] = None
    """Position in chain (1 = first, 2 = second, etc.)."""

    entities: List[Dict] = field(default_factory=list)
    """
    Knowledge graph entities referenced.
    Each entity:
    {
        "entity_type": "technology|concept|library|pattern|problem",
        "name": "Docker",
        "relationship": "uses|discusses|solves|implements"
    }
    """

    # ═══════════════════════════════════════════════════════════════
    # TIER 6: DEEP ANALYSIS (Optional, high-value only)
    # ═══════════════════════════════════════════════════════════════

    detailed_breakdown: Optional[str] = None
    """Multi-paragraph analysis of the conversation (Sonnet-generated)."""

    related_concepts: List[str] = field(default_factory=list)
    """Concepts this conversation relates to (for knowledge graph)."""

    suggested_followups: List[str] = field(default_factory=list)
    """Suggested next steps or explorations."""

    # ═══════════════════════════════════════════════════════════════
    # METADATA
    # ═══════════════════════════════════════════════════════════════

    metadata: Dict = field(default_factory=dict)
    """
    Additional metadata:
    {
        "message_count": 12,
        "user_messages": 6,
        "assistant_messages": 6,
        "import_source": "chatgpt_export.json",
        "enrichment_tokens": 1500,
        "enrichment_cost": 0.02
    }
    """

    enrichment_version: str = "2.0"
    """Version of enrichment schema used."""

    enriched_at: Optional[datetime] = None
    """When enrichment was last run."""
```

---

## Artifact Schema

```python
@dataclass
class Artifact:
    """A reusable piece extracted from a conversation."""

    artifact_id: str
    """Unique ID: {conversation_id}_art_{index}."""

    conversation_id: str
    """Parent conversation ID."""

    artifact_type: Literal[
        "code_snippet",      # Reusable code (functions, classes)
        "shell_command",     # Terminal commands
        "configuration",     # Config files, env vars
        "api_call",          # Curl commands, API examples
        "sql_query",         # Database queries
        "regex_pattern",     # Regular expressions
        "algorithm",         # Logic/pseudocode
        "data_structure",    # JSON schemas, data models
        "error_solution",    # Problem + fix pair
        "best_practice",     # Advice/pattern to follow
    ]

    content: str
    """The actual artifact content."""

    language: str = ""
    """Programming language if applicable."""

    description: str = ""
    """What this artifact does (1 sentence)."""

    use_case: str = ""
    """When to use this artifact."""

    prerequisites: List[str] = field(default_factory=list)
    """What you need before using this."""

    tags: List[str] = field(default_factory=list)
    """Relevant keywords."""

    technologies: List[str] = field(default_factory=list)
    """Technologies involved (docker, python, redis)."""

    verified_working: bool = False
    """Whether it worked in the original conversation."""

    complexity: Literal["simple", "moderate", "complex"] = "moderate"
    """How complex this artifact is."""
```

---

## Project Schema

```python
@dataclass
class Project:
    """Auto-inferred project grouping."""

    project_id: str
    """Unique project identifier."""

    project_name: str
    """Human-readable name (inferred or user-provided)."""

    technologies: List[str]
    """Technologies used in this project."""

    keywords: List[str]
    """Keywords that identify this project."""

    file_patterns: List[str]
    """File patterns mentioned (*.py, app.py, etc.)."""

    conversation_ids: List[str]
    """Conversations belonging to this project."""

    conversation_count: int
    """Number of conversations."""

    first_conversation: datetime
    """When first conversation occurred."""

    last_conversation: datetime
    """When most recent conversation occurred."""

    active_period_days: int
    """Days between first and last conversation."""

    project_summary: str
    """AI-generated overview of the project."""

    key_decisions: List[str]
    """Major choices made during the project."""

    current_status: Literal["active", "completed", "abandoned", "paused"]
    """Current project status."""
```

---

## Chain Schema

```python
@dataclass
class ConversationChain:
    """A sequence of related conversations."""

    chain_id: str
    """Unique chain identifier."""

    title: str
    """Descriptive title for the chain."""

    conversation_ids: List[str]
    """Conversations in order."""

    starting_point: str
    """Initial question or problem."""

    resolution: str
    """Final outcome or solution."""

    key_learnings: List[str]
    """What was learned across the chain."""

    mistakes_made: List[str]
    """Errors or wrong turns taken."""

    final_solution: str
    """The solution that worked."""

    total_duration_days: int
    """Days from first to last conversation."""
```

---

## Entity Schema (Knowledge Graph)

```python
@dataclass
class Entity:
    """A node in the knowledge graph."""

    entity_id: str
    """Unique entity identifier."""

    entity_type: Literal[
        "technology",  # Python, Docker, Redis
        "concept",     # async, REST API, caching
        "library",     # Flask, aiohttp, pandas
        "pattern",     # singleton, factory, pub-sub
        "problem",     # CORS errors, memory leaks
    ]

    name: str
    """Entity name."""

    aliases: List[str] = field(default_factory=list)
    """Alternative names (JS, JavaScript)."""

    occurrence_count: int = 0
    """How many conversations mention this entity."""

    related_entities: List[str] = field(default_factory=list)
    """Other entities frequently co-occurring."""


@dataclass
class EntityRelationship:
    """An edge in the knowledge graph."""

    conversation_id: str
    """Conversation where relationship was observed."""

    entity_id: str
    """Entity being referenced."""

    relationship: Literal[
        "uses",        # Conversation uses this technology
        "discusses",   # Conversation discusses this concept
        "solves",      # Conversation solves this problem
        "implements",  # Conversation implements this pattern
        "mentions",    # Conversation mentions this entity
    ]
```

---

## JSON Examples

### Complete Enriched Conversation

```json
{
  "convo_id": "550e8400-e29b-41d4-a716-446655440000",
  "external_id": "chatgpt_abc123",
  "timestamp": "2024-06-15T10:30:00",
  "source": "OpenAI",
  "raw_text": "USER: How do I set up async in Python?\n\nASSISTANT: Here's how to use async...",

  "generated_title": "Setting Up Async Python with aiohttp",
  "summary_abstractive": "Comprehensive guide to async Python programming covering asyncio basics, aiohttp for HTTP requests, and common patterns for concurrent operations.",
  "summary_extractive": "Use async def to define coroutines. aiohttp provides async HTTP client. asyncio.gather runs multiple coroutines concurrently.",
  "primary_domain": "Technical",
  "tags": ["python", "async", "aiohttp", "asyncio", "concurrency"],
  "key_topics": ["async programming", "HTTP clients", "concurrency patterns"],

  "brilliance_score": {
    "score": 8,
    "reasoning": "Thorough explanation with practical examples",
    "factors": {
      "depth": 9,
      "actionability": 8,
      "creativity": 6,
      "problem_solving": 8
    }
  },
  "key_insights": [
    "async/await is syntactic sugar for coroutines",
    "aiohttp is preferred over requests for async code",
    "asyncio.gather is the key to concurrent execution"
  ],
  "status": "Completed",
  "future_potential": {
    "value_proposition": "Reference for async Python patterns",
    "next_steps": "Apply to web scraping project"
  },
  "score": 8,
  "score_reasoning": "Comprehensive coverage with working examples",

  "has_code": true,
  "code_languages": ["python"],
  "code_block_count": 4,
  "has_error_traces": false,
  "turn_count": 6,
  "question_count": 3,
  "question_types": ["how-to", "explain"],
  "avg_user_length": 150,
  "avg_assistant_length": 800,
  "has_links": true,
  "link_domains": ["docs.python.org", "aiohttp.readthedocs.io"],
  "technical_terms": ["async", "await", "coroutine", "event loop"],

  "embedding": [0.123, -0.456, 0.789, ...],
  "cluster_id": "async-programming",

  "artifacts": [
    {
      "artifact_id": "550e8400_art_0",
      "artifact_type": "code_snippet",
      "content": "async def fetch_all(urls):\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch(session, url) for url in urls]\n        return await asyncio.gather(*tasks)",
      "language": "python",
      "description": "Concurrent HTTP fetching with aiohttp",
      "use_case": "Batch API calls, web scraping",
      "tags": ["async", "http", "concurrent"],
      "verified_working": true
    }
  ],
  "artifact_count": 1,
  "has_reusable_code": true,
  "has_shell_commands": false,
  "has_error_solutions": false,

  "project_id": "proj_async_learning",
  "project_name": "Async Python Learning",
  "chain_id": "chain_async_journey",
  "chain_position": 1,
  "entities": [
    {"entity_type": "technology", "name": "Python", "relationship": "uses"},
    {"entity_type": "library", "name": "aiohttp", "relationship": "uses"},
    {"entity_type": "concept", "name": "async programming", "relationship": "discusses"}
  ],

  "metadata": {
    "message_count": 12,
    "user_messages": 6,
    "assistant_messages": 6
  },
  "enrichment_version": "2.0",
  "enriched_at": "2024-12-01T15:00:00"
}
```

### Artifact Example

```json
{
  "artifact_id": "550e8400_art_0",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "artifact_type": "shell_command",
  "content": "docker run -d --name redis -p 6379:6379 -v redis_data:/data redis:alpine --appendonly yes",
  "language": "bash",
  "description": "Start persistent Redis container with Alpine image",
  "use_case": "Local development, caching layer, session storage",
  "prerequisites": ["Docker installed", "Port 6379 available"],
  "tags": ["docker", "redis", "cache", "persistence"],
  "technologies": ["docker", "redis"],
  "verified_working": true,
  "complexity": "simple"
}
```

---

## Migration Notes

### From v1 to v2

All v1 fields are preserved. New fields are added with defaults:

```python
# Fields added in v2 (defaults shown)
has_code = False
code_languages = []
code_block_count = 0
# ... all Tier 2-6 fields
enrichment_version = "2.0"
```

### Backward Compatibility

- JSONL format unchanged (just more fields)
- All existing queries still work
- New fields are optional (can be None or empty)
- Old data loads fine into new schema
