# CogRepo: Vendor Lock-in & Operational Gaps Remediation Plan

> **Status**: Implementation Ready
> **Created**: 2024-12-06
> **Priority**: High

This document addresses four key gaps identified in the CogRepo architecture:
1. LLM Vendor Lock-in (Anthropic-only)
2. PII Handling (none)
3. Evaluation & Feedback Loops (missing)
4. Observability Dashboard (infrastructure exists, UI missing)

---

## 1. Provider-Agnostic LLM Abstraction Layer

### Problem
Direct coupling to Anthropic API in `enrichment/enrichment_pipeline.py`:
```python
from anthropic import Anthropic
self.client = Anthropic(api_key=self.api_key)
```

No fallback, no alternative providers, no local model support.

### Solution: `core/llm_provider.py`

Create an abstraction layer with:
- **Interface**: `LLMProvider` abstract base class
- **Implementations**: `AnthropicProvider`, `OpenAIProvider`, `OllamaProvider` (local)
- **Factory**: Config-driven provider selection with fallback chain
- **Cost tracking**: Per-provider token/cost accounting

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   EnrichmentPipeline                     │
│                          │                               │
│                    uses LLMProvider                      │
│                          │                               │
│         ┌────────────────┼────────────────┐              │
│         ▼                ▼                ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  Anthropic  │  │   OpenAI    │  │   Ollama    │       │
│  │  Provider   │  │  Provider   │  │  Provider   │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
│         │                │                │              │
│         └────────────────┼────────────────┘              │
│                          ▼                               │
│                  Fallback Chain                          │
│           (primary → secondary → local)                  │
└─────────────────────────────────────────────────────────┘
```

### Implementation

**File: `core/llm_provider.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class ModelTier(Enum):
    """Model capability tiers for task routing."""
    FAST = "fast"      # Quick, cheap (Haiku, GPT-4o-mini, Llama-3-8B)
    STANDARD = "standard"  # Balanced (Sonnet, GPT-4o)
    ADVANCED = "advanced"  # Best quality (Opus, GPT-4, Claude-3-Opus)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class LLMRequest:
    """Standardized request to any LLM provider."""
    messages: List[Dict[str, str]]
    max_tokens: int = 1024
    temperature: float = 0.3
    tier: ModelTier = ModelTier.STANDARD
    system_prompt: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/metrics."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and reachable."""
        pass

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a completion request."""
        pass

    @abstractmethod
    def get_model_for_tier(self, tier: ModelTier) -> str:
        """Get the model name for a given capability tier."""
        pass

    def estimate_cost(self, input_tokens: int, output_tokens: int, tier: ModelTier) -> float:
        """Estimate cost for a request (override per provider)."""
        return 0.0


class ProviderChain:
    """Manages multiple providers with fallback logic."""

    def __init__(self, providers: List[LLMProvider]):
        self.providers = providers
        self._metrics = ProviderMetrics()

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Try providers in order until one succeeds."""
        last_error = None

        for provider in self.providers:
            if not provider.is_available:
                continue
            try:
                response = provider.complete(request)
                self._metrics.record_success(provider.name, response)
                return response
            except Exception as e:
                self._metrics.record_failure(provider.name, e)
                last_error = e
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")
```

### Config Extension (`core/config.py`)

```python
class LLMProviderConfig(BaseModel):
    """Multi-provider LLM configuration."""

    # Primary provider
    primary_provider: str = Field(
        default="anthropic",
        description="Primary LLM provider (anthropic, openai, ollama)"
    )

    # Fallback chain
    fallback_providers: List[str] = Field(
        default_factory=lambda: ["openai", "ollama"],
        description="Ordered list of fallback providers"
    )

    # Provider-specific configs
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    openai: Optional[OpenAIConfig] = None
    ollama: Optional[OllamaConfig] = None

    # Cost controls
    max_cost_per_conversation_usd: float = Field(
        default=0.10,
        description="Max cost per conversation enrichment"
    )
    prefer_local_for_pii: bool = Field(
        default=True,
        description="Route PII-containing content to local models"
    )
```

### Migration Path

1. **Phase 1**: Create abstraction layer, wrap existing Anthropic calls
2. **Phase 2**: Add OpenAI provider
3. **Phase 3**: Add Ollama/local model support
4. **Phase 4**: Implement smart routing (cost, PII, task complexity)

---

## 2. PII Detection & Scrubbing

### Problem
Conversations stored verbatim with no PII handling:
- Email addresses
- Phone numbers
- API keys / secrets
- Names / addresses
- Credit card numbers

### Solution: `core/pii_handler.py`

Multi-layer PII protection:

```
┌─────────────────────────────────────────────────────────┐
│                    PII Handler                           │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Detector   │→ │  Classifier  │→ │   Scrubber   │   │
│  │  (patterns)  │  │ (PII types)  │  │  (redact)    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│          │                                    │          │
│          ▼                                    ▼          │
│  ┌──────────────┐                   ┌──────────────┐    │
│  │  Audit Log   │                   │  PII Vault   │    │
│  │  (what/when) │                   │ (reversible) │    │
│  └──────────────┘                   └──────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Implementation

**File: `core/pii_handler.py`**

```python
import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
from pathlib import Path


class PIIType(Enum):
    """Types of PII we detect and handle."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    API_KEY = "api_key"
    IP_ADDRESS = "ip_address"
    NAME = "name"  # Requires NER
    ADDRESS = "address"  # Requires NER
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """A detected PII instance."""
    pii_type: PIIType
    original: str
    start: int
    end: int
    confidence: float
    replacement: str = ""


@dataclass
class PIIConfig:
    """PII handling configuration."""
    enabled: bool = True

    # Detection settings
    detect_emails: bool = True
    detect_phones: bool = True
    detect_api_keys: bool = True
    detect_credit_cards: bool = True
    detect_ssn: bool = True
    detect_ip_addresses: bool = True

    # Scrubbing settings
    scrub_mode: str = "redact"  # "redact", "hash", "mask", "remove"
    redact_placeholder: str = "[REDACTED-{type}]"

    # Vault settings (for reversible scrubbing)
    enable_vault: bool = False
    vault_path: Optional[Path] = None

    # Custom patterns
    custom_patterns: Dict[str, str] = field(default_factory=dict)


class PIIDetector:
    """Detects PII in text using pattern matching."""

    # Compiled regex patterns
    PATTERNS = {
        PIIType.EMAIL: re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ),
        PIIType.PHONE: re.compile(
            r'(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
        ),
        PIIType.SSN: re.compile(
            r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
        ),
        PIIType.CREDIT_CARD: re.compile(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        ),
        PIIType.API_KEY: re.compile(
            r'\b(sk-[a-zA-Z0-9]{32,}|api[_-]?key["\s:=]+["\']?[a-zA-Z0-9_-]{20,})\b',
            re.IGNORECASE
        ),
        PIIType.IP_ADDRESS: re.compile(
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        ),
    }

    def __init__(self, config: PIIConfig):
        self.config = config
        self._compile_custom_patterns()

    def detect(self, text: str) -> List[PIIMatch]:
        """Detect all PII in text."""
        matches = []

        for pii_type, pattern in self.PATTERNS.items():
            if not self._should_detect(pii_type):
                continue

            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    original=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95  # Pattern matches are high confidence
                ))

        return sorted(matches, key=lambda m: m.start)

    def _should_detect(self, pii_type: PIIType) -> bool:
        """Check if we should detect this PII type."""
        return getattr(self.config, f'detect_{pii_type.value}s', True)


class PIIScrubber:
    """Scrubs detected PII from text."""

    def __init__(self, config: PIIConfig):
        self.config = config
        self.vault = PIIVault(config) if config.enable_vault else None

    def scrub(self, text: str, matches: List[PIIMatch]) -> Tuple[str, Dict]:
        """Scrub PII from text, return scrubbed text and audit record."""
        if not matches:
            return text, {"pii_found": False, "count": 0}

        audit = {
            "pii_found": True,
            "count": len(matches),
            "types": {},
            "scrub_mode": self.config.scrub_mode
        }

        # Process matches in reverse order to preserve positions
        result = text
        for match in reversed(matches):
            replacement = self._get_replacement(match)
            match.replacement = replacement
            result = result[:match.start] + replacement + result[match.end:]

            # Track in audit
            type_name = match.pii_type.value
            audit["types"][type_name] = audit["types"].get(type_name, 0) + 1

            # Store in vault if enabled
            if self.vault:
                self.vault.store(match)

        return result, audit

    def _get_replacement(self, match: PIIMatch) -> str:
        """Get replacement text based on scrub mode."""
        mode = self.config.scrub_mode

        if mode == "redact":
            return self.config.redact_placeholder.format(type=match.pii_type.value.upper())
        elif mode == "hash":
            hash_val = hashlib.sha256(match.original.encode()).hexdigest()[:12]
            return f"[HASH:{hash_val}]"
        elif mode == "mask":
            return self._mask_value(match)
        elif mode == "remove":
            return ""
        else:
            return "[REDACTED]"

    def _mask_value(self, match: PIIMatch) -> str:
        """Partially mask a value (e.g., email@***.com)."""
        orig = match.original
        if match.pii_type == PIIType.EMAIL:
            parts = orig.split('@')
            if len(parts) == 2:
                return f"{parts[0][:2]}***@{parts[1]}"
        elif match.pii_type == PIIType.PHONE:
            return orig[:3] + "***" + orig[-4:]
        return "***"


class PIIHandler:
    """Main interface for PII detection and scrubbing."""

    def __init__(self, config: Optional[PIIConfig] = None):
        self.config = config or PIIConfig()
        self.detector = PIIDetector(self.config)
        self.scrubber = PIIScrubber(self.config)

    def process(self, text: str) -> Tuple[str, Dict]:
        """Detect and scrub PII, returning clean text and audit log."""
        if not self.config.enabled:
            return text, {"pii_enabled": False}

        matches = self.detector.detect(text)
        return self.scrubber.scrub(text, matches)

    def has_pii(self, text: str) -> bool:
        """Quick check if text contains PII."""
        return len(self.detector.detect(text)) > 0
```

### Integration Point

Modify `enrichment_pipeline.py`:

```python
def enrich_conversation(self, raw: RawConversation) -> EnrichedConversation:
    # NEW: Scrub PII before sending to external API
    if self.pii_handler.config.enabled:
        clean_text, pii_audit = self.pii_handler.process(raw.raw_text)
        # Use clean_text for enrichment
        # Store pii_audit in enrichment metadata

    # ... rest of enrichment
```

---

## 3. Evaluation & Feedback System

### Problem
No way to measure quality of:
- Clustering coherence
- Recommendation relevance
- Tag accuracy
- Summary quality

### Solution: `intelligence/evaluation.py`

```
┌─────────────────────────────────────────────────────────┐
│                  Evaluation System                       │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Metrics    │  │   Feedback   │  │   Reports    │   │
│  │  Collectors  │  │   Ingestion  │  │  Generator   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│          │                │                │             │
│          ▼                ▼                ▼             │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Evaluation Store                     │   │
│  │   (SQLite: ratings, metrics, experiments)         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Implementation

**File: `intelligence/evaluation.py`**

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import sqlite3
import json
from pathlib import Path


class FeedbackType(Enum):
    """Types of user feedback."""
    RECOMMENDATION_HELPFUL = "rec_helpful"
    RECOMMENDATION_IRRELEVANT = "rec_irrelevant"
    TAG_CORRECT = "tag_correct"
    TAG_INCORRECT = "tag_incorrect"
    TAG_MISSING = "tag_missing"
    CLUSTER_COHERENT = "cluster_coherent"
    CLUSTER_WRONG = "cluster_wrong"
    SUMMARY_GOOD = "summary_good"
    SUMMARY_BAD = "summary_bad"
    GENERAL_THUMBS_UP = "thumbs_up"
    GENERAL_THUMBS_DOWN = "thumbs_down"


@dataclass
class UserFeedback:
    """A piece of user feedback."""
    feedback_type: FeedbackType
    entity_type: str  # "conversation", "cluster", "recommendation", "tag"
    entity_id: str
    user_id: Optional[str] = None
    value: int = 1  # 1 = positive, -1 = negative, 0 = neutral
    comment: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityMetrics:
    """Quality metrics for a feature."""
    feature: str  # "clustering", "recommendations", "tags", "summaries"
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    user_satisfaction: Optional[float] = None  # % positive feedback
    sample_size: int = 0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvaluationStore:
    """Persistent store for evaluation data."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize evaluation database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY,
                    feedback_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    user_id TEXT,
                    value INTEGER DEFAULT 1,
                    comment TEXT,
                    context JSON,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY,
                    feature TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    sample_size INTEGER,
                    period_start DATETIME,
                    period_end DATETIME,
                    metadata JSON,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_entity
                ON feedback(entity_type, entity_id);

                CREATE INDEX IF NOT EXISTS idx_metrics_feature
                ON metrics(feature, timestamp);
            """)

    def record_feedback(self, feedback: UserFeedback):
        """Store user feedback."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO feedback
                (feedback_type, entity_type, entity_id, user_id, value, comment, context, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.feedback_type.value,
                feedback.entity_type,
                feedback.entity_id,
                feedback.user_id,
                feedback.value,
                feedback.comment,
                json.dumps(feedback.context),
                feedback.timestamp.isoformat()
            ))

    def get_satisfaction_rate(
        self,
        feature: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Calculate user satisfaction rate for a feature."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            result = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN value > 0 THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN value < 0 THEN 1 ELSE 0 END) as negative
                FROM feedback
                WHERE entity_type = ?
                AND timestamp > datetime('now', ?)
            """, (feature, f'-{days} days')).fetchone()

            total = result['total'] or 0
            positive = result['positive'] or 0

            return {
                "feature": feature,
                "total_feedback": total,
                "positive": positive,
                "negative": result['negative'] or 0,
                "satisfaction_rate": positive / total if total > 0 else None,
                "period_days": days
            }


class RecommendationEvaluator:
    """Evaluates recommendation quality."""

    def __init__(self, store: EvaluationStore):
        self.store = store

    def evaluate_recommendations(
        self,
        recommendations: List[Dict],
        clicked: List[str],
        ignored: List[str]
    ) -> Dict[str, float]:
        """Evaluate recommendations based on user behavior."""
        if not recommendations:
            return {"precision": 0, "recall": 0, "ndcg": 0}

        rec_ids = [r['conversation_id'] for r in recommendations]

        # Precision: what % of shown recs were clicked?
        precision = len(set(rec_ids) & set(clicked)) / len(rec_ids)

        # Calculate NDCG (Normalized Discounted Cumulative Gain)
        dcg = sum(
            1 / (i + 2)  # log2(i+2) simplified
            for i, rec_id in enumerate(rec_ids)
            if rec_id in clicked
        )
        ideal_dcg = sum(1 / (i + 2) for i in range(min(len(clicked), len(rec_ids))))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

        return {
            "precision": precision,
            "ndcg": ndcg,
            "click_rate": len(clicked) / len(rec_ids) if rec_ids else 0
        }


class ClusteringEvaluator:
    """Evaluates clustering quality."""

    def evaluate_clusters(
        self,
        clusters: List[Dict],
        embeddings: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Evaluate clustering coherence using silhouette-like metrics."""
        import numpy as np

        coherence_scores = []

        for cluster in clusters:
            if len(cluster['conversation_ids']) < 2:
                continue

            # Get embeddings for cluster members
            member_embeddings = [
                embeddings.get(cid)
                for cid in cluster['conversation_ids']
                if cid in embeddings
            ]

            if len(member_embeddings) < 2:
                continue

            # Calculate intra-cluster similarity
            member_embeddings = np.array(member_embeddings)
            centroid = np.mean(member_embeddings, axis=0)

            distances = np.linalg.norm(member_embeddings - centroid, axis=1)
            coherence = 1 - np.mean(distances) / (np.std(distances) + 1e-6)
            coherence_scores.append(coherence)

        return {
            "avg_coherence": np.mean(coherence_scores) if coherence_scores else 0,
            "min_coherence": np.min(coherence_scores) if coherence_scores else 0,
            "num_clusters": len(clusters),
            "clusters_evaluated": len(coherence_scores)
        }


class EvaluationDashboard:
    """Generates evaluation reports and KPIs."""

    def __init__(self, store: EvaluationStore):
        self.store = store

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        return {
            "generated_at": datetime.now().isoformat(),
            "features": {
                "recommendations": self.store.get_satisfaction_rate("recommendation"),
                "clustering": self.store.get_satisfaction_rate("cluster"),
                "tags": self.store.get_satisfaction_rate("tag"),
                "summaries": self.store.get_satisfaction_rate("summary")
            },
            "kpis": self._calculate_kpis()
        }

    def _calculate_kpis(self) -> Dict[str, Any]:
        """Calculate key performance indicators."""
        return {
            "recommendation_click_rate": "TBD",  # Requires click tracking
            "tag_accuracy": "TBD",  # Requires ground truth
            "cluster_coherence": "TBD",  # Requires embeddings
            "user_engagement": "TBD"  # Requires session tracking
        }
```

### API Endpoints (add to `cogrepo-ui/app.py`)

```python
@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback on recommendations/clusters/tags."""
    data = request.json
    feedback = UserFeedback(
        feedback_type=FeedbackType(data['type']),
        entity_type=data['entity_type'],
        entity_id=data['entity_id'],
        value=data.get('value', 1),
        comment=data.get('comment')
    )
    evaluation_store.record_feedback(feedback)
    return jsonify({"status": "recorded"})

@app.route('/api/evaluation/report')
def evaluation_report():
    """Get evaluation metrics report."""
    dashboard = EvaluationDashboard(evaluation_store)
    return jsonify(dashboard.generate_report())
```

---

## 4. Enhanced Observability Dashboard

### Problem
Infrastructure exists (`APICallLogger`, `ProgressLogger`) but no aggregated view.

### Solution: `core/metrics.py` + Dashboard UI

### Implementation

**File: `core/metrics.py`**

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import json
from pathlib import Path


@dataclass
class EnrichmentMetrics:
    """Metrics for enrichment pipeline runs."""
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    conversations_processed: int = 0
    conversations_failed: int = 0
    total_api_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    errors: List[Dict] = field(default_factory=list)
    provider_breakdown: Dict[str, Dict] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if not self.completed_at:
            return (datetime.now() - self.started_at).total_seconds()
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def success_rate(self) -> float:
        total = self.conversations_processed + self.conversations_failed
        return self.conversations_processed / total if total > 0 else 0.0


class MetricsCollector:
    """Thread-safe metrics collection."""

    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path
        self._lock = threading.Lock()
        self._current_run: Optional[EnrichmentMetrics] = None
        self._historical: List[EnrichmentMetrics] = []
        self._api_latencies: List[float] = []
        self._error_counts: Dict[str, int] = defaultdict(int)

    def start_run(self, run_id: str) -> EnrichmentMetrics:
        """Start tracking a new enrichment run."""
        with self._lock:
            self._current_run = EnrichmentMetrics(
                run_id=run_id,
                started_at=datetime.now()
            )
            return self._current_run

    def record_api_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_usd: float,
        success: bool
    ):
        """Record an API call."""
        with self._lock:
            if not self._current_run:
                return

            self._current_run.total_api_calls += 1
            self._current_run.total_tokens += input_tokens + output_tokens
            self._current_run.total_cost_usd += cost_usd
            self._api_latencies.append(latency_ms)

            # Provider breakdown
            if provider not in self._current_run.provider_breakdown:
                self._current_run.provider_breakdown[provider] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "failures": 0
                }

            breakdown = self._current_run.provider_breakdown[provider]
            breakdown["calls"] += 1
            breakdown["tokens"] += input_tokens + output_tokens
            breakdown["cost_usd"] += cost_usd
            if not success:
                breakdown["failures"] += 1

    def record_error(self, error_type: str, details: Dict):
        """Record an error."""
        with self._lock:
            self._error_counts[error_type] += 1
            if self._current_run:
                self._current_run.errors.append({
                    "type": error_type,
                    "details": details,
                    "timestamp": datetime.now().isoformat()
                })

    def complete_run(self, success_count: int, failure_count: int):
        """Mark current run as complete."""
        with self._lock:
            if not self._current_run:
                return

            self._current_run.completed_at = datetime.now()
            self._current_run.conversations_processed = success_count
            self._current_run.conversations_failed = failure_count
            self._historical.append(self._current_run)

            # Persist if configured
            if self.persist_path:
                self._persist()

    def get_dashboard_data(self) -> Dict:
        """Get data for dashboard display."""
        with self._lock:
            recent_runs = self._historical[-10:]  # Last 10 runs

            return {
                "current_run": self._format_run(self._current_run) if self._current_run else None,
                "recent_runs": [self._format_run(r) for r in recent_runs],
                "aggregate": {
                    "total_runs": len(self._historical),
                    "total_conversations": sum(r.conversations_processed for r in self._historical),
                    "total_cost_usd": sum(r.total_cost_usd for r in self._historical),
                    "avg_success_rate": (
                        sum(r.success_rate for r in self._historical) / len(self._historical)
                        if self._historical else 0
                    ),
                    "error_breakdown": dict(self._error_counts)
                },
                "latency": {
                    "avg_ms": sum(self._api_latencies) / len(self._api_latencies) if self._api_latencies else 0,
                    "p95_ms": sorted(self._api_latencies)[int(len(self._api_latencies) * 0.95)] if self._api_latencies else 0,
                    "max_ms": max(self._api_latencies) if self._api_latencies else 0
                }
            }

    def _format_run(self, run: EnrichmentMetrics) -> Dict:
        """Format a run for JSON output."""
        return {
            "run_id": run.run_id,
            "started_at": run.started_at.isoformat(),
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "duration_seconds": run.duration_seconds,
            "conversations_processed": run.conversations_processed,
            "conversations_failed": run.conversations_failed,
            "success_rate": run.success_rate,
            "api_calls": run.total_api_calls,
            "tokens": run.total_tokens,
            "cost_usd": run.total_cost_usd,
            "provider_breakdown": run.provider_breakdown,
            "error_count": len(run.errors)
        }

    def _persist(self):
        """Persist metrics to disk."""
        data = {
            "historical": [self._format_run(r) for r in self._historical],
            "error_counts": dict(self._error_counts)
        }
        with open(self.persist_path, 'w') as f:
            json.dump(data, f, indent=2)
```

### Dashboard API Endpoints

```python
@app.route('/api/metrics/dashboard')
def metrics_dashboard():
    """Get enrichment metrics for dashboard."""
    return jsonify(metrics_collector.get_dashboard_data())

@app.route('/api/metrics/health')
def health_check():
    """Health check with key metrics."""
    data = metrics_collector.get_dashboard_data()
    return jsonify({
        "status": "healthy",
        "current_run": data["current_run"] is not None,
        "avg_success_rate": data["aggregate"]["avg_success_rate"],
        "recent_errors": sum(1 for r in data["recent_runs"] if r["conversations_failed"] > 0)
    })
```

---

## Implementation Phases

### Phase 1: Foundation (Critical)
- [ ] Create `core/llm_provider.py` with abstraction layer
- [ ] Implement `AnthropicProvider` wrapping current code
- [ ] Create `core/pii_handler.py` with detection/scrubbing
- [ ] Add PII config to `core/config.py`

### Phase 2: Provider Expansion
- [ ] Implement `OpenAIProvider`
- [ ] Implement `OllamaProvider` for local models
- [ ] Add `ProviderChain` with fallback logic
- [ ] Add smart routing (cost/PII-based)

### Phase 3: Evaluation & Feedback
- [ ] Create `intelligence/evaluation.py`
- [ ] Add feedback API endpoints
- [ ] Implement satisfaction tracking
- [ ] Add evaluation dashboard

### Phase 4: Observability
- [ ] Create `core/metrics.py`
- [ ] Add metrics API endpoints
- [ ] Build dashboard UI component
- [ ] Add alerting hooks

---

## Config Changes Summary

Add to `config/cogrepo_config.yaml`:

```yaml
llm:
  primary_provider: anthropic
  fallback_providers:
    - openai
    - ollama
  max_cost_per_conversation_usd: 0.10
  prefer_local_for_pii: true

pii:
  enabled: true
  scrub_mode: redact  # redact | hash | mask | remove
  detect_emails: true
  detect_phones: true
  detect_api_keys: true
  enable_vault: false

evaluation:
  enabled: true
  db_path: data/evaluation.db
  track_clicks: true
  track_feedback: true

metrics:
  enabled: true
  persist_path: data/metrics.json
  dashboard_enabled: true
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `core/llm_provider.py` | Create | LLM abstraction layer |
| `core/pii_handler.py` | Create | PII detection & scrubbing |
| `core/metrics.py` | Create | Metrics collection |
| `intelligence/evaluation.py` | Create | Evaluation & feedback |
| `core/config.py` | Modify | Add new config sections |
| `enrichment/enrichment_pipeline.py` | Modify | Use LLM abstraction |
| `cogrepo-ui/app.py` | Modify | Add new API endpoints |
| `config/cogrepo_config.yaml` | Modify | Add new config options |

---

## Success Criteria

1. **LLM Abstraction**: Can switch providers via config without code changes
2. **PII Handling**: Emails/phones/API keys scrubbed before external API calls
3. **Evaluation**: User feedback captured, satisfaction rates visible
4. **Observability**: Dashboard shows enrichment success rates, costs, latencies
