"""
Observability and Metrics System for CogRepo

Provides comprehensive metrics collection and monitoring for:
- Enrichment pipeline runs
- API call tracking (tokens, costs, latency)
- Error rates and patterns
- Provider performance comparison
- Cost analysis and budgeting

Features:
- Thread-safe metrics collection
- Persistent storage to disk
- Real-time dashboards
- Alerting capabilities
- Historical analysis

Usage:
    from core.metrics import MetricsCollector, get_metrics_collector

    metrics = get_metrics_collector()

    # Start tracking a run
    run = metrics.start_run("enrichment_batch_001")

    # Record API calls
    metrics.record_api_call(
        provider="anthropic",
        model="claude-3-5-sonnet",
        input_tokens=1000,
        output_tokens=500,
        latency_ms=1234.5,
        cost_usd=0.015,
        success=True
    )

    # Complete the run
    metrics.complete_run(success_count=50, failure_count=2)

    # Get dashboard data
    dashboard = metrics.get_dashboard_data()
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from collections import defaultdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class RunStatus(Enum):
    """Status of an enrichment run."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class APICallMetric:
    """Metrics for a single API call."""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    error_type: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "success": self.success,
            "error_type": self.error_type,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EnrichmentRunMetrics:
    """Metrics for an enrichment pipeline run."""
    run_id: str
    started_at: datetime
    status: RunStatus = RunStatus.RUNNING
    completed_at: Optional[datetime] = None
    conversations_processed: int = 0
    conversations_failed: int = 0
    conversations_skipped: int = 0
    total_api_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    errors: List[Dict] = field(default_factory=list)
    provider_breakdown: Dict[str, Dict] = field(default_factory=dict)
    phase_timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Get run duration in seconds."""
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.conversations_processed + self.conversations_failed
        return self.conversations_processed / total if total > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate for API calls."""
        failed_calls = sum(
            p.get("failures", 0)
            for p in self.provider_breakdown.values()
        )
        return failed_calls / self.total_api_calls if self.total_api_calls > 0 else 0.0

    @property
    def avg_cost_per_conversation(self) -> float:
        """Average cost per processed conversation."""
        if self.conversations_processed == 0:
            return 0.0
        return self.total_cost_usd / self.conversations_processed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "conversations_processed": self.conversations_processed,
            "conversations_failed": self.conversations_failed,
            "conversations_skipped": self.conversations_skipped,
            "success_rate": self.success_rate,
            "total_api_calls": self.total_api_calls,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_per_conversation": self.avg_cost_per_conversation,
            "error_rate": self.error_rate,
            "error_count": len(self.errors),
            "provider_breakdown": self.provider_breakdown,
            "phase_timings": self.phase_timings,
            "metadata": self.metadata
        }


@dataclass
class Alert:
    """An alert triggered by metrics thresholds."""
    level: AlertLevel
    title: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged
        }


@dataclass
class DashboardData:
    """Data for the metrics dashboard."""
    generated_at: datetime
    current_run: Optional[Dict]
    recent_runs: List[Dict]
    aggregate: Dict[str, Any]
    latency: Dict[str, float]
    costs: Dict[str, float]
    providers: Dict[str, Dict]
    errors: Dict[str, int]
    alerts: List[Dict]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "current_run": self.current_run,
            "recent_runs": self.recent_runs,
            "aggregate": self.aggregate,
            "latency": self.latency,
            "costs": self.costs,
            "providers": self.providers,
            "errors": self.errors,
            "alerts": self.alerts
        }


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """
    Thread-safe metrics collection system.

    Collects and aggregates metrics from enrichment runs,
    API calls, and system health checks.
    """

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        max_historical_runs: int = 100,
        alert_callback: Optional[Callable[[Alert], None]] = None
    ):
        """
        Initialize metrics collector.

        Args:
            persist_path: Path to persist metrics (None for in-memory only)
            max_historical_runs: Maximum runs to keep in history
            alert_callback: Optional callback for alerts
        """
        self.persist_path = Path(persist_path) if persist_path else None
        self.max_historical_runs = max_historical_runs
        self.alert_callback = alert_callback

        self._lock = threading.RLock()
        self._current_run: Optional[EnrichmentRunMetrics] = None
        self._historical_runs: List[EnrichmentRunMetrics] = []
        self._api_latencies: List[float] = []
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._alerts: List[Alert] = []

        # Aggregate stats
        self._total_api_calls = 0
        self._total_tokens = 0
        self._total_cost = 0.0

        # Thresholds for alerting
        self.error_rate_threshold = 0.1
        self.latency_threshold_ms = 10000
        self.cost_threshold_per_run = 10.0

        # Load persisted data if available
        if self.persist_path and self.persist_path.exists():
            self._load()

    def start_run(self, run_id: str, metadata: Optional[Dict] = None) -> EnrichmentRunMetrics:
        """
        Start tracking a new enrichment run.

        Args:
            run_id: Unique identifier for the run
            metadata: Optional metadata for the run

        Returns:
            The run metrics object
        """
        with self._lock:
            self._current_run = EnrichmentRunMetrics(
                run_id=run_id,
                started_at=datetime.now(),
                metadata=metadata or {}
            )
            logger.info(f"Started metrics tracking for run: {run_id}")
            return self._current_run

    def record_api_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_usd: float,
        success: bool,
        error_type: Optional[str] = None
    ):
        """
        Record an API call.

        Args:
            provider: Provider name (anthropic, openai, ollama)
            model: Model identifier
            input_tokens: Input tokens used
            output_tokens: Output tokens generated
            latency_ms: Request latency in milliseconds
            cost_usd: Estimated cost in USD
            success: Whether call succeeded
            error_type: Error type if failed
        """
        with self._lock:
            # Update aggregate stats
            self._total_api_calls += 1
            self._total_tokens += input_tokens + output_tokens
            self._total_cost += cost_usd
            self._api_latencies.append(latency_ms)

            # Keep latencies bounded
            if len(self._api_latencies) > 10000:
                self._api_latencies = self._api_latencies[-5000:]

            if not success and error_type:
                self._error_counts[error_type] += 1

            # Update current run if active
            if self._current_run:
                run = self._current_run
                run.total_api_calls += 1
                run.total_input_tokens += input_tokens
                run.total_output_tokens += output_tokens
                run.total_cost_usd += cost_usd

                # Update provider breakdown
                if provider not in run.provider_breakdown:
                    run.provider_breakdown[provider] = {
                        "calls": 0,
                        "tokens": 0,
                        "cost_usd": 0.0,
                        "failures": 0,
                        "total_latency_ms": 0.0,
                        "models": {}
                    }

                breakdown = run.provider_breakdown[provider]
                breakdown["calls"] += 1
                breakdown["tokens"] += input_tokens + output_tokens
                breakdown["cost_usd"] += cost_usd
                breakdown["total_latency_ms"] += latency_ms

                if not success:
                    breakdown["failures"] += 1

                # Track per-model stats
                if model not in breakdown["models"]:
                    breakdown["models"][model] = {"calls": 0, "tokens": 0}
                breakdown["models"][model]["calls"] += 1
                breakdown["models"][model]["tokens"] += input_tokens + output_tokens

            # Check for latency alerts
            if latency_ms > self.latency_threshold_ms:
                self._create_alert(
                    AlertLevel.WARNING,
                    "High Latency Detected",
                    f"API call to {provider}/{model} took {latency_ms:.0f}ms",
                    "latency_ms",
                    latency_ms,
                    self.latency_threshold_ms
                )

    def record_error(
        self,
        error_type: str,
        message: str,
        details: Optional[Dict] = None
    ):
        """
        Record an error.

        Args:
            error_type: Type/category of error
            message: Error message
            details: Additional error details
        """
        with self._lock:
            self._error_counts[error_type] += 1

            if self._current_run:
                self._current_run.errors.append({
                    "type": error_type,
                    "message": message,
                    "details": details or {},
                    "timestamp": datetime.now().isoformat()
                })

    def record_phase_timing(self, phase: str, duration_seconds: float):
        """
        Record timing for a pipeline phase.

        Args:
            phase: Phase name
            duration_seconds: Duration in seconds
        """
        with self._lock:
            if self._current_run:
                self._current_run.phase_timings[phase] = duration_seconds

    def increment_processed(self, count: int = 1):
        """Increment processed conversation count."""
        with self._lock:
            if self._current_run:
                self._current_run.conversations_processed += count

    def increment_failed(self, count: int = 1):
        """Increment failed conversation count."""
        with self._lock:
            if self._current_run:
                self._current_run.conversations_failed += count

    def increment_skipped(self, count: int = 1):
        """Increment skipped conversation count."""
        with self._lock:
            if self._current_run:
                self._current_run.conversations_skipped += count

    def complete_run(
        self,
        success_count: Optional[int] = None,
        failure_count: Optional[int] = None,
        status: RunStatus = RunStatus.COMPLETED
    ):
        """
        Mark current run as complete.

        Args:
            success_count: Final success count (overrides incremental)
            failure_count: Final failure count (overrides incremental)
            status: Final run status
        """
        with self._lock:
            if not self._current_run:
                return

            run = self._current_run
            run.completed_at = datetime.now()
            run.status = status

            if success_count is not None:
                run.conversations_processed = success_count
            if failure_count is not None:
                run.conversations_failed = failure_count

            # Check for alerts
            if run.error_rate > self.error_rate_threshold:
                self._create_alert(
                    AlertLevel.ERROR,
                    "High Error Rate",
                    f"Run {run.run_id} had {run.error_rate:.1%} error rate",
                    "error_rate",
                    run.error_rate,
                    self.error_rate_threshold
                )

            if run.total_cost_usd > self.cost_threshold_per_run:
                self._create_alert(
                    AlertLevel.WARNING,
                    "High Cost Run",
                    f"Run {run.run_id} cost ${run.total_cost_usd:.2f}",
                    "cost_usd",
                    run.total_cost_usd,
                    self.cost_threshold_per_run
                )

            # Add to history
            self._historical_runs.append(run)

            # Trim history if needed
            if len(self._historical_runs) > self.max_historical_runs:
                self._historical_runs = self._historical_runs[-self.max_historical_runs:]

            # Persist if configured
            if self.persist_path:
                self._persist()

            logger.info(
                f"Completed run {run.run_id}: "
                f"{run.conversations_processed} processed, "
                f"{run.conversations_failed} failed, "
                f"${run.total_cost_usd:.4f} cost"
            )

            self._current_run = None

    def cancel_run(self):
        """Cancel the current run."""
        self.complete_run(status=RunStatus.CANCELLED)

    def fail_run(self, error_message: str):
        """Mark current run as failed."""
        if self._current_run:
            self.record_error("run_failure", error_message)
        self.complete_run(status=RunStatus.FAILED)

    def _create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float
    ):
        """Create and dispatch an alert."""
        alert = Alert(
            level=level,
            title=title,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )

        self._alerts.append(alert)

        # Keep alerts bounded
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-50:]

        # Call alert callback if configured
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"Alert: {title} - {message}")

    def get_dashboard_data(self) -> DashboardData:
        """
        Get data for dashboard display.

        Returns:
            DashboardData object with all metrics
        """
        with self._lock:
            # Current run
            current = None
            if self._current_run:
                current = self._current_run.to_dict()

            # Recent runs
            recent = [run.to_dict() for run in self._historical_runs[-10:]]

            # Aggregate stats
            aggregate = {
                "total_runs": len(self._historical_runs),
                "total_conversations": sum(
                    r.conversations_processed for r in self._historical_runs
                ),
                "total_api_calls": self._total_api_calls,
                "total_tokens": self._total_tokens,
                "total_cost_usd": self._total_cost,
                "avg_success_rate": self._calculate_avg_success_rate(),
                "avg_cost_per_run": self._calculate_avg_cost_per_run()
            }

            # Latency stats
            latency = self._calculate_latency_stats()

            # Cost breakdown
            costs = self._calculate_cost_breakdown()

            # Provider stats
            providers = self._calculate_provider_stats()

            # Error breakdown
            errors = dict(self._error_counts)

            # Recent alerts
            alerts = [a.to_dict() for a in self._alerts[-20:]]

            return DashboardData(
                generated_at=datetime.now(),
                current_run=current,
                recent_runs=recent,
                aggregate=aggregate,
                latency=latency,
                costs=costs,
                providers=providers,
                errors=errors,
                alerts=alerts
            )

    def _calculate_avg_success_rate(self) -> float:
        """Calculate average success rate across runs."""
        if not self._historical_runs:
            return 0.0
        rates = [r.success_rate for r in self._historical_runs]
        return sum(rates) / len(rates)

    def _calculate_avg_cost_per_run(self) -> float:
        """Calculate average cost per run."""
        if not self._historical_runs:
            return 0.0
        costs = [r.total_cost_usd for r in self._historical_runs]
        return sum(costs) / len(costs)

    def _calculate_latency_stats(self) -> Dict[str, float]:
        """Calculate latency statistics."""
        if not self._api_latencies:
            return {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "max_ms": 0}

        sorted_latencies = sorted(self._api_latencies)
        n = len(sorted_latencies)

        return {
            "avg_ms": statistics.mean(sorted_latencies),
            "p50_ms": sorted_latencies[int(n * 0.50)],
            "p95_ms": sorted_latencies[int(n * 0.95)] if n > 20 else sorted_latencies[-1],
            "p99_ms": sorted_latencies[int(n * 0.99)] if n > 100 else sorted_latencies[-1],
            "max_ms": max(sorted_latencies)
        }

    def _calculate_cost_breakdown(self) -> Dict[str, float]:
        """Calculate cost breakdown by provider."""
        breakdown = defaultdict(float)

        for run in self._historical_runs:
            for provider, stats in run.provider_breakdown.items():
                breakdown[provider] += stats.get("cost_usd", 0)

        breakdown["total"] = self._total_cost
        return dict(breakdown)

    def _calculate_provider_stats(self) -> Dict[str, Dict]:
        """Calculate per-provider statistics."""
        provider_data = defaultdict(lambda: {
            "calls": 0,
            "tokens": 0,
            "cost_usd": 0.0,
            "failures": 0,
            "avg_latency_ms": 0.0
        })

        for run in self._historical_runs:
            for provider, stats in run.provider_breakdown.items():
                pd = provider_data[provider]
                pd["calls"] += stats.get("calls", 0)
                pd["tokens"] += stats.get("tokens", 0)
                pd["cost_usd"] += stats.get("cost_usd", 0)
                pd["failures"] += stats.get("failures", 0)

        # Calculate averages
        for provider, pd in provider_data.items():
            if pd["calls"] > 0:
                pd["success_rate"] = 1 - (pd["failures"] / pd["calls"])
            else:
                pd["success_rate"] = 0

        return dict(provider_data)

    def get_health_check(self) -> Dict[str, Any]:
        """
        Get system health status.

        Returns:
            Health check dictionary
        """
        with self._lock:
            recent_errors = sum(
                1 for r in self._historical_runs[-5:]
                if r.error_rate > self.error_rate_threshold
            )

            status = "healthy"
            if recent_errors >= 3:
                status = "critical"
            elif recent_errors >= 1:
                status = "degraded"

            return {
                "status": status,
                "current_run_active": self._current_run is not None,
                "total_runs": len(self._historical_runs),
                "avg_success_rate": self._calculate_avg_success_rate(),
                "recent_error_runs": recent_errors,
                "unacknowledged_alerts": sum(
                    1 for a in self._alerts if not a.acknowledged
                ),
                "last_run": (
                    self._historical_runs[-1].to_dict()
                    if self._historical_runs else None
                )
            }

    def acknowledge_alert(self, index: int) -> bool:
        """
        Acknowledge an alert.

        Args:
            index: Alert index in the list

        Returns:
            True if acknowledged, False if not found
        """
        with self._lock:
            if 0 <= index < len(self._alerts):
                self._alerts[index].acknowledged = True
                return True
            return False

    def clear_alerts(self):
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()

    def reset_stats(self):
        """Reset all statistics (keeps history)."""
        with self._lock:
            self._api_latencies.clear()
            self._error_counts.clear()
            self._total_api_calls = 0
            self._total_tokens = 0
            self._total_cost = 0.0

    def _persist(self):
        """Persist metrics to disk."""
        if not self.persist_path:
            return

        try:
            data = {
                "historical_runs": [r.to_dict() for r in self._historical_runs],
                "error_counts": dict(self._error_counts),
                "total_api_calls": self._total_api_calls,
                "total_tokens": self._total_tokens,
                "total_cost": self._total_cost,
                "alerts": [a.to_dict() for a in self._alerts[-50:]],
                "saved_at": datetime.now().isoformat()
            }

            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")

    def _load(self):
        """Load persisted metrics from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)

            # Reconstruct historical runs
            for run_data in data.get("historical_runs", []):
                run = EnrichmentRunMetrics(
                    run_id=run_data["run_id"],
                    started_at=datetime.fromisoformat(run_data["started_at"]),
                    status=RunStatus(run_data.get("status", "completed")),
                    completed_at=(
                        datetime.fromisoformat(run_data["completed_at"])
                        if run_data.get("completed_at") else None
                    ),
                    conversations_processed=run_data.get("conversations_processed", 0),
                    conversations_failed=run_data.get("conversations_failed", 0),
                    total_api_calls=run_data.get("total_api_calls", 0),
                    total_input_tokens=run_data.get("total_tokens", 0) // 2,
                    total_output_tokens=run_data.get("total_tokens", 0) // 2,
                    total_cost_usd=run_data.get("total_cost_usd", 0),
                    provider_breakdown=run_data.get("provider_breakdown", {})
                )
                self._historical_runs.append(run)

            self._error_counts = defaultdict(int, data.get("error_counts", {}))
            self._total_api_calls = data.get("total_api_calls", 0)
            self._total_tokens = data.get("total_tokens", 0)
            self._total_cost = data.get("total_cost", 0)

            logger.info(f"Loaded {len(self._historical_runs)} historical runs from disk")

        except Exception as e:
            logger.error(f"Failed to load persisted metrics: {e}")


# =============================================================================
# Global Instance
# =============================================================================

_metrics_instance: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics_collector(
    persist_path: Optional[Path] = None,
    **kwargs
) -> MetricsCollector:
    """
    Get the global metrics collector instance.

    Args:
        persist_path: Path to persist metrics
        **kwargs: Additional configuration

    Returns:
        MetricsCollector singleton instance
    """
    global _metrics_instance

    with _metrics_lock:
        if _metrics_instance is None:
            if persist_path is None:
                persist_path = Path("data/metrics.json")
            _metrics_instance = MetricsCollector(persist_path=persist_path, **kwargs)

        return _metrics_instance


def reset_metrics_collector():
    """Reset the global metrics collector instance."""
    global _metrics_instance
    with _metrics_lock:
        _metrics_instance = None


# =============================================================================
# Context Managers
# =============================================================================

class MetricsRun:
    """
    Context manager for tracking an enrichment run.

    Usage:
        with MetricsRun("batch_001") as run:
            # Do enrichment work
            run.record_api_call(...)
    """

    def __init__(
        self,
        run_id: str,
        collector: Optional[MetricsCollector] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize metrics run context.

        Args:
            run_id: Unique run identifier
            collector: Optional metrics collector (uses global if None)
            metadata: Optional run metadata
        """
        self.run_id = run_id
        self.collector = collector or get_metrics_collector()
        self.metadata = metadata

    def __enter__(self) -> 'MetricsRun':
        """Start the run."""
        self.collector.start_run(self.run_id, self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete or fail the run."""
        if exc_type is not None:
            self.collector.fail_run(str(exc_val))
        else:
            self.collector.complete_run()
        return False

    def record_api_call(self, **kwargs):
        """Record an API call."""
        self.collector.record_api_call(**kwargs)

    def record_error(self, error_type: str, message: str, details: Optional[Dict] = None):
        """Record an error."""
        self.collector.record_error(error_type, message, details)

    def increment_processed(self, count: int = 1):
        """Increment processed count."""
        self.collector.increment_processed(count)

    def increment_failed(self, count: int = 1):
        """Increment failed count."""
        self.collector.increment_failed(count)


class PhaseTimer:
    """
    Context manager for timing a pipeline phase.

    Usage:
        with PhaseTimer("embedding_generation", collector):
            # Do embedding work
    """

    def __init__(
        self,
        phase_name: str,
        collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize phase timer.

        Args:
            phase_name: Name of the phase
            collector: Optional metrics collector
        """
        self.phase_name = phase_name
        self.collector = collector or get_metrics_collector()
        self.start_time = None

    def __enter__(self) -> 'PhaseTimer':
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record duration."""
        duration = time.time() - self.start_time
        self.collector.record_phase_timing(self.phase_name, duration)
        return False
