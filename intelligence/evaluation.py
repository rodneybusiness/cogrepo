"""
Evaluation and Feedback System for CogRepo Intelligence Features

Provides comprehensive evaluation capabilities for:
- Recommendation quality tracking
- Clustering coherence measurement
- Tag accuracy assessment
- Summary quality evaluation
- User feedback collection and analysis

Features:
- SQLite-based persistent storage
- User feedback ingestion
- Quality metrics calculation
- Satisfaction rate tracking
- KPI dashboards and reporting

Usage:
    from intelligence.evaluation import EvaluationSystem, UserFeedback, FeedbackType

    eval_system = EvaluationSystem(db_path="data/evaluation.db")

    # Record user feedback
    eval_system.record_feedback(UserFeedback(
        feedback_type=FeedbackType.RECOMMENDATION_HELPFUL,
        entity_type="recommendation",
        entity_id="rec_123",
        value=1
    ))

    # Get satisfaction metrics
    report = eval_system.generate_report()
"""

import sqlite3
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class FeedbackType(Enum):
    """Types of user feedback."""
    # Recommendations
    RECOMMENDATION_HELPFUL = "rec_helpful"
    RECOMMENDATION_IRRELEVANT = "rec_irrelevant"
    RECOMMENDATION_CLICKED = "rec_clicked"
    RECOMMENDATION_IGNORED = "rec_ignored"

    # Tags
    TAG_CORRECT = "tag_correct"
    TAG_INCORRECT = "tag_incorrect"
    TAG_MISSING = "tag_missing"
    TAG_SUGGESTED = "tag_suggested"

    # Clustering
    CLUSTER_COHERENT = "cluster_coherent"
    CLUSTER_WRONG = "cluster_wrong"
    CLUSTER_SPLIT_NEEDED = "cluster_split"
    CLUSTER_MERGE_NEEDED = "cluster_merge"

    # Summaries
    SUMMARY_ACCURATE = "summary_accurate"
    SUMMARY_INACCURATE = "summary_inaccurate"
    SUMMARY_TOO_SHORT = "summary_short"
    SUMMARY_TOO_LONG = "summary_long"

    # General
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    REPORT_ISSUE = "report_issue"


class EntityType(Enum):
    """Types of entities that can receive feedback."""
    CONVERSATION = "conversation"
    RECOMMENDATION = "recommendation"
    CLUSTER = "cluster"
    TAG = "tag"
    SUMMARY = "summary"
    SEARCH_RESULT = "search_result"


class MetricType(Enum):
    """Types of quality metrics."""
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    NDCG = "ndcg"
    SATISFACTION_RATE = "satisfaction_rate"
    CLICK_RATE = "click_rate"
    COHERENCE = "coherence"
    ACCURACY = "accuracy"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class UserFeedback:
    """A piece of user feedback."""
    feedback_type: FeedbackType
    entity_type: str
    entity_id: str
    value: int = 1  # 1 = positive, -1 = negative, 0 = neutral
    user_id: Optional[str] = None
    comment: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "feedback_type": self.feedback_type.value,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "value": self.value,
            "user_id": self.user_id,
            "comment": self.comment,
            "context": json.dumps(self.context),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class QualityMetric:
    """A quality metric measurement."""
    metric_type: MetricType
    feature: str  # "recommendations", "clustering", "tags", "summaries"
    value: float
    sample_size: int = 0
    confidence: float = 0.0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "metric_type": self.metric_type.value,
            "feature": self.feature,
            "value": self.value,
            "sample_size": self.sample_size,
            "confidence": self.confidence,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "metadata": json.dumps(self.metadata),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    generated_at: datetime
    period_days: int
    features: Dict[str, Dict[str, Any]]
    overall_health: str  # "healthy", "degraded", "critical"
    alerts: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "period_days": self.period_days,
            "features": self.features,
            "overall_health": self.overall_health,
            "alerts": self.alerts,
            "recommendations": self.recommendations
        }


# =============================================================================
# Evaluation Store
# =============================================================================

class EvaluationStore:
    """
    Persistent storage for evaluation data.

    Uses SQLite for reliable, embedded storage.
    """

    def __init__(self, db_path: Path):
        """
        Initialize evaluation store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                -- User feedback table
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    value INTEGER DEFAULT 1,
                    user_id TEXT,
                    comment TEXT,
                    context TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- Quality metrics table
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    feature TEXT NOT NULL,
                    value REAL NOT NULL,
                    sample_size INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0,
                    period_start DATETIME,
                    period_end DATETIME,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- Click/interaction tracking
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    position INTEGER,
                    context TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- A/B experiment tracking
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    entity_id TEXT,
                    outcome TEXT,
                    value REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_feedback_entity
                ON feedback(entity_type, entity_id);

                CREATE INDEX IF NOT EXISTS idx_feedback_type
                ON feedback(feedback_type, timestamp);

                CREATE INDEX IF NOT EXISTS idx_metrics_feature
                ON metrics(feature, timestamp);

                CREATE INDEX IF NOT EXISTS idx_interactions_entity
                ON interactions(entity_type, entity_id);

                CREATE INDEX IF NOT EXISTS idx_experiments_id
                ON experiments(experiment_id, variant);
            """)

    def record_feedback(self, feedback: UserFeedback) -> int:
        """
        Store user feedback.

        Args:
            feedback: UserFeedback object

        Returns:
            Inserted row ID
        """
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute("""
                    INSERT INTO feedback
                    (feedback_type, entity_type, entity_id, value, user_id, comment, context, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.feedback_type.value,
                    feedback.entity_type,
                    feedback.entity_id,
                    feedback.value,
                    feedback.user_id,
                    feedback.comment,
                    json.dumps(feedback.context),
                    feedback.timestamp.isoformat()
                ))
                return cursor.lastrowid

    def record_metric(self, metric: QualityMetric) -> int:
        """
        Store a quality metric.

        Args:
            metric: QualityMetric object

        Returns:
            Inserted row ID
        """
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute("""
                    INSERT INTO metrics
                    (metric_type, feature, value, sample_size, confidence,
                     period_start, period_end, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.metric_type.value,
                    metric.feature,
                    metric.value,
                    metric.sample_size,
                    metric.confidence,
                    metric.period_start.isoformat() if metric.period_start else None,
                    metric.period_end.isoformat() if metric.period_end else None,
                    json.dumps(metric.metadata),
                    metric.timestamp.isoformat()
                ))
                return cursor.lastrowid

    def record_interaction(
        self,
        interaction_type: str,
        entity_type: str,
        entity_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        position: Optional[int] = None,
        context: Optional[Dict] = None
    ) -> int:
        """
        Record a user interaction (click, view, etc.).

        Args:
            interaction_type: Type of interaction (click, view, dismiss)
            entity_type: Type of entity
            entity_id: Entity identifier
            user_id: Optional user ID
            session_id: Optional session ID
            position: Optional position in list
            context: Optional context data

        Returns:
            Inserted row ID
        """
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute("""
                    INSERT INTO interactions
                    (interaction_type, entity_type, entity_id, user_id, session_id, position, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    interaction_type,
                    entity_type,
                    entity_id,
                    user_id,
                    session_id,
                    position,
                    json.dumps(context) if context else None
                ))
                return cursor.lastrowid

    def get_feedback_stats(
        self,
        entity_type: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get feedback statistics for an entity type.

        Args:
            entity_type: Type of entity
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        with self._get_conn() as conn:
            result = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN value > 0 THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN value < 0 THEN 1 ELSE 0 END) as negative,
                    SUM(CASE WHEN value = 0 THEN 1 ELSE 0 END) as neutral,
                    AVG(value) as avg_value
                FROM feedback
                WHERE entity_type = ?
                AND timestamp > datetime('now', ?)
            """, (entity_type, f'-{days} days')).fetchone()

            total = result['total'] or 0
            positive = result['positive'] or 0
            negative = result['negative'] or 0

            return {
                "entity_type": entity_type,
                "period_days": days,
                "total_feedback": total,
                "positive": positive,
                "negative": negative,
                "neutral": result['neutral'] or 0,
                "satisfaction_rate": positive / total if total > 0 else None,
                "avg_value": result['avg_value']
            }

    def get_feedback_by_type(
        self,
        feedback_type: FeedbackType,
        days: int = 30,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get recent feedback of a specific type.

        Args:
            feedback_type: Type of feedback
            days: Number of days to look back
            limit: Maximum records to return

        Returns:
            List of feedback records
        """
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM feedback
                WHERE feedback_type = ?
                AND timestamp > datetime('now', ?)
                ORDER BY timestamp DESC
                LIMIT ?
            """, (feedback_type.value, f'-{days} days', limit)).fetchall()

            return [dict(row) for row in rows]

    def get_recent_metrics(
        self,
        feature: str,
        days: int = 30
    ) -> List[Dict]:
        """
        Get recent metrics for a feature.

        Args:
            feature: Feature name
            days: Number of days to look back

        Returns:
            List of metric records
        """
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM metrics
                WHERE feature = ?
                AND timestamp > datetime('now', ?)
                ORDER BY timestamp DESC
            """, (feature, f'-{days} days')).fetchall()

            return [dict(row) for row in rows]

    def get_click_through_rate(
        self,
        entity_type: str,
        days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate click-through rate for an entity type.

        Args:
            entity_type: Type of entity
            days: Number of days to analyze

        Returns:
            CTR statistics
        """
        with self._get_conn() as conn:
            # Get views and clicks
            result = conn.execute("""
                SELECT
                    interaction_type,
                    COUNT(*) as count
                FROM interactions
                WHERE entity_type = ?
                AND timestamp > datetime('now', ?)
                GROUP BY interaction_type
            """, (entity_type, f'-{days} days')).fetchall()

            counts = {row['interaction_type']: row['count'] for row in result}
            views = counts.get('view', 0) + counts.get('show', 0)
            clicks = counts.get('click', 0)

            return {
                "entity_type": entity_type,
                "period_days": days,
                "views": views,
                "clicks": clicks,
                "ctr": clicks / views if views > 0 else 0
            }

    def cleanup_old_data(self, retention_days: int = 365) -> int:
        """
        Remove old data beyond retention period.

        Args:
            retention_days: Days to retain data

        Returns:
            Number of rows deleted
        """
        with self._lock:
            with self._get_conn() as conn:
                deleted = 0

                for table in ['feedback', 'metrics', 'interactions', 'experiments']:
                    cursor = conn.execute(f"""
                        DELETE FROM {table}
                        WHERE timestamp < datetime('now', ?)
                    """, (f'-{retention_days} days',))
                    deleted += cursor.rowcount

                return deleted


# =============================================================================
# Evaluators
# =============================================================================

class RecommendationEvaluator:
    """Evaluates recommendation quality."""

    def __init__(self, store: EvaluationStore):
        """Initialize evaluator with store."""
        self.store = store

    def evaluate_session(
        self,
        recommendations: List[Dict],
        clicked: List[str],
        shown_positions: Optional[Dict[str, int]] = None
    ) -> Dict[str, float]:
        """
        Evaluate recommendations from a single session.

        Args:
            recommendations: List of recommendation dicts with 'id' field
            clicked: List of clicked recommendation IDs
            shown_positions: Optional dict mapping ID to position shown

        Returns:
            Evaluation metrics
        """
        if not recommendations:
            return {"precision": 0, "recall": 0, "ndcg": 0, "mrr": 0}

        rec_ids = [r.get('id') or r.get('conversation_id') for r in recommendations]
        clicked_set = set(clicked)

        # Precision: what % of shown recs were clicked?
        hits = len(set(rec_ids) & clicked_set)
        precision = hits / len(rec_ids) if rec_ids else 0

        # Mean Reciprocal Rank
        mrr = 0.0
        for i, rec_id in enumerate(rec_ids):
            if rec_id in clicked_set:
                mrr = 1.0 / (i + 1)
                break

        # NDCG (Normalized Discounted Cumulative Gain)
        dcg = sum(
            1.0 / np.log2(i + 2)
            for i, rec_id in enumerate(rec_ids)
            if rec_id in clicked_set
        )

        # Ideal DCG: all clicked items at top
        ideal_dcg = sum(
            1.0 / np.log2(i + 2)
            for i in range(min(len(clicked), len(rec_ids)))
        )
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

        metrics = {
            "precision": precision,
            "mrr": mrr,
            "ndcg": ndcg,
            "hits": hits,
            "shown": len(rec_ids),
            "clicked": len(clicked)
        }

        # Store metric
        self.store.record_metric(QualityMetric(
            metric_type=MetricType.NDCG,
            feature="recommendations",
            value=ndcg,
            sample_size=len(rec_ids),
            metadata=metrics
        ))

        return metrics

    def get_aggregate_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get aggregate recommendation metrics over time."""
        metrics = self.store.get_recent_metrics("recommendations", days)

        if not metrics:
            return {"available": False}

        ndcg_values = [m['value'] for m in metrics if m['metric_type'] == 'ndcg']

        feedback = self.store.get_feedback_stats("recommendation", days)
        ctr = self.store.get_click_through_rate("recommendation", days)

        return {
            "available": True,
            "period_days": days,
            "avg_ndcg": np.mean(ndcg_values) if ndcg_values else None,
            "median_ndcg": np.median(ndcg_values) if ndcg_values else None,
            "sample_count": len(metrics),
            "feedback": feedback,
            "ctr": ctr
        }


class ClusteringEvaluator:
    """Evaluates clustering quality."""

    def __init__(self, store: EvaluationStore):
        """Initialize evaluator with store."""
        self.store = store

    def evaluate_clusters(
        self,
        clusters: List[Dict],
        embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate clustering coherence using embeddings.

        Args:
            clusters: List of cluster dicts with 'conversation_ids' field
            embeddings: Dict mapping conversation ID to embedding vector

        Returns:
            Clustering metrics
        """
        if not clusters:
            return {"coherence": 0, "silhouette": 0}

        coherence_scores = []
        cluster_sizes = []

        for cluster in clusters:
            conv_ids = cluster.get('conversation_ids', [])
            if len(conv_ids) < 2:
                continue

            # Get embeddings for cluster members
            member_embeddings = []
            for cid in conv_ids:
                if cid in embeddings:
                    emb = embeddings[cid]
                    if isinstance(emb, np.ndarray):
                        member_embeddings.append(emb)

            if len(member_embeddings) < 2:
                continue

            member_embeddings = np.array(member_embeddings)

            # Calculate centroid and intra-cluster distances
            centroid = np.mean(member_embeddings, axis=0)
            distances = np.linalg.norm(member_embeddings - centroid, axis=1)

            # Coherence: inverse of normalized distance
            mean_dist = np.mean(distances)
            max_dist = np.max(distances) if len(distances) > 0 else 1
            coherence = 1 - (mean_dist / (max_dist + 1e-6))

            coherence_scores.append(coherence)
            cluster_sizes.append(len(conv_ids))

        if not coherence_scores:
            return {"coherence": 0, "silhouette": 0, "clusters_evaluated": 0}

        avg_coherence = float(np.mean(coherence_scores))
        weighted_coherence = float(np.average(coherence_scores, weights=cluster_sizes))

        metrics = {
            "coherence": avg_coherence,
            "weighted_coherence": weighted_coherence,
            "min_coherence": float(np.min(coherence_scores)),
            "max_coherence": float(np.max(coherence_scores)),
            "num_clusters": len(clusters),
            "clusters_evaluated": len(coherence_scores),
            "avg_cluster_size": float(np.mean(cluster_sizes))
        }

        # Store metric
        self.store.record_metric(QualityMetric(
            metric_type=MetricType.COHERENCE,
            feature="clustering",
            value=avg_coherence,
            sample_size=len(coherence_scores),
            metadata=metrics
        ))

        return metrics

    def get_aggregate_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get aggregate clustering metrics over time."""
        metrics = self.store.get_recent_metrics("clustering", days)
        feedback = self.store.get_feedback_stats("cluster", days)

        if not metrics:
            return {"available": False, "feedback": feedback}

        coherence_values = [m['value'] for m in metrics]

        return {
            "available": True,
            "period_days": days,
            "avg_coherence": float(np.mean(coherence_values)),
            "trend": self._calculate_trend(coherence_values),
            "sample_count": len(metrics),
            "feedback": feedback
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 3:
            return "insufficient_data"

        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        return "stable"


class TagEvaluator:
    """Evaluates tag quality."""

    def __init__(self, store: EvaluationStore):
        """Initialize evaluator with store."""
        self.store = store

    def evaluate_tags(
        self,
        predicted_tags: List[str],
        ground_truth_tags: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate tag prediction quality.

        Args:
            predicted_tags: Tags generated by the system
            ground_truth_tags: Correct tags (from user feedback)

        Returns:
            Tag quality metrics
        """
        if not predicted_tags and not ground_truth_tags:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        predicted_set = set(t.lower() for t in predicted_tags)
        truth_set = set(t.lower() for t in ground_truth_tags)

        if not predicted_set:
            return {"precision": 0, "recall": 0, "f1": 0}

        true_positives = len(predicted_set & truth_set)
        precision = true_positives / len(predicted_set)
        recall = true_positives / len(truth_set) if truth_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_count": len(predicted_set),
            "truth_count": len(truth_set),
            "overlap": true_positives
        }

        # Store metric
        self.store.record_metric(QualityMetric(
            metric_type=MetricType.F1_SCORE,
            feature="tags",
            value=f1,
            sample_size=1,
            metadata=metrics
        ))

        return metrics

    def get_aggregate_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get aggregate tag metrics."""
        feedback = self.store.get_feedback_stats("tag", days)
        metrics = self.store.get_recent_metrics("tags", days)

        if not metrics:
            return {"available": False, "feedback": feedback}

        f1_values = [m['value'] for m in metrics]

        return {
            "available": True,
            "period_days": days,
            "avg_f1": float(np.mean(f1_values)),
            "sample_count": len(metrics),
            "feedback": feedback
        }


# =============================================================================
# Main Evaluation System
# =============================================================================

class EvaluationSystem:
    """
    Main evaluation system coordinating all evaluators.

    Provides a unified interface for:
    - Recording feedback
    - Evaluating quality
    - Generating reports
    - Managing KPIs
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize evaluation system.

        Args:
            db_path: Path to evaluation database
            config: Optional configuration dictionary
        """
        self.config = config or {}

        if db_path is None:
            db_path = Path("data/evaluation.db")

        self.store = EvaluationStore(db_path)

        # Initialize evaluators
        self.recommendation_evaluator = RecommendationEvaluator(self.store)
        self.clustering_evaluator = ClusteringEvaluator(self.store)
        self.tag_evaluator = TagEvaluator(self.store)

        # Thresholds
        self.min_satisfaction_rate = self.config.get("min_satisfaction_rate", 0.7)

    def record_feedback(self, feedback: UserFeedback) -> int:
        """
        Record user feedback.

        Args:
            feedback: UserFeedback object

        Returns:
            Record ID
        """
        return self.store.record_feedback(feedback)

    def record_interaction(
        self,
        interaction_type: str,
        entity_type: str,
        entity_id: str,
        **kwargs
    ) -> int:
        """
        Record a user interaction.

        Args:
            interaction_type: Type of interaction
            entity_type: Type of entity
            entity_id: Entity identifier
            **kwargs: Additional context

        Returns:
            Record ID
        """
        return self.store.record_interaction(
            interaction_type,
            entity_type,
            entity_id,
            **kwargs
        )

    def evaluate_recommendations(
        self,
        recommendations: List[Dict],
        clicked: List[str]
    ) -> Dict[str, float]:
        """Evaluate recommendation quality."""
        return self.recommendation_evaluator.evaluate_session(recommendations, clicked)

    def evaluate_clusters(
        self,
        clusters: List[Dict],
        embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate clustering quality."""
        return self.clustering_evaluator.evaluate_clusters(clusters, embeddings)

    def evaluate_tags(
        self,
        predicted: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """Evaluate tag quality."""
        return self.tag_evaluator.evaluate_tags(predicted, ground_truth)

    def get_feature_health(self, feature: str, days: int = 30) -> Dict[str, Any]:
        """
        Get health status for a specific feature.

        Args:
            feature: Feature name
            days: Analysis period

        Returns:
            Health status dictionary
        """
        feedback = self.store.get_feedback_stats(feature, days)

        health = {
            "feature": feature,
            "period_days": days,
            "feedback": feedback,
            "status": "unknown"
        }

        sat_rate = feedback.get("satisfaction_rate")
        if sat_rate is not None:
            if sat_rate >= self.min_satisfaction_rate:
                health["status"] = "healthy"
            elif sat_rate >= self.min_satisfaction_rate * 0.8:
                health["status"] = "degraded"
            else:
                health["status"] = "critical"

        return health

    def generate_report(self, days: int = 30) -> EvaluationReport:
        """
        Generate comprehensive evaluation report.

        Args:
            days: Analysis period in days

        Returns:
            EvaluationReport object
        """
        features = {}
        alerts = []
        recommendations = []

        # Evaluate each feature
        for feature in ["recommendation", "cluster", "tag", "summary"]:
            health = self.get_feature_health(feature, days)
            features[feature] = health

            if health["status"] == "critical":
                alerts.append(f"{feature.title()} quality is critically low")
            elif health["status"] == "degraded":
                alerts.append(f"{feature.title()} quality is degraded")

        # Add specific evaluator metrics
        features["recommendation"]["metrics"] = self.recommendation_evaluator.get_aggregate_metrics(days)
        features["cluster"]["metrics"] = self.clustering_evaluator.get_aggregate_metrics(days)
        features["tag"]["metrics"] = self.tag_evaluator.get_aggregate_metrics(days)

        # Determine overall health
        statuses = [f.get("status", "unknown") for f in features.values()]
        if "critical" in statuses:
            overall_health = "critical"
        elif "degraded" in statuses:
            overall_health = "degraded"
        elif all(s == "healthy" for s in statuses):
            overall_health = "healthy"
        else:
            overall_health = "unknown"

        # Generate recommendations
        for feature, data in features.items():
            sat_rate = data.get("feedback", {}).get("satisfaction_rate")
            if sat_rate is not None and sat_rate < self.min_satisfaction_rate:
                recommendations.append(
                    f"Review {feature} algorithm - satisfaction rate is {sat_rate:.1%}"
                )

            total = data.get("feedback", {}).get("total_feedback", 0)
            if total < 10:
                recommendations.append(
                    f"Collect more feedback for {feature} (currently {total} samples)"
                )

        return EvaluationReport(
            generated_at=datetime.now(),
            period_days=days,
            features=features,
            overall_health=overall_health,
            alerts=alerts,
            recommendations=recommendations
        )

    def cleanup(self, retention_days: int = 365) -> int:
        """
        Clean up old evaluation data.

        Args:
            retention_days: Days to retain

        Returns:
            Number of records deleted
        """
        return self.store.cleanup_old_data(retention_days)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_evaluation_system(config: Optional[Dict] = None) -> EvaluationSystem:
    """
    Create an evaluation system with default or custom config.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured EvaluationSystem
    """
    config = config or {}
    db_path = Path(config.get("db_path", "data/evaluation.db"))
    return EvaluationSystem(db_path=db_path, config=config)
