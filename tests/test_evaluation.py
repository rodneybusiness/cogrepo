"""
Tests for Evaluation and Metrics System

Tests feedback collection, quality metrics, and reporting.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from intelligence.evaluation import (
    EvaluationSystem, EvaluationStore, UserFeedback, FeedbackType,
    QualityMetric, MetricType, EntityType,
    RecommendationEvaluator, ClusteringEvaluator, TagEvaluator
)

from core.metrics import (
    MetricsCollector, EnrichmentRunMetrics, RunStatus, Alert, AlertLevel,
    MetricsRun, PhaseTimer, get_metrics_collector, reset_metrics_collector
)


class TestEvaluationStore:
    """Tests for EvaluationStore."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary evaluation store."""
        db_path = tmp_path / "test_eval.db"
        return EvaluationStore(db_path)

    def test_record_feedback(self, store):
        """Test recording user feedback."""
        feedback = UserFeedback(
            feedback_type=FeedbackType.RECOMMENDATION_HELPFUL,
            entity_type="recommendation",
            entity_id="rec_123",
            value=1,
            comment="Very relevant!"
        )

        record_id = store.record_feedback(feedback)
        assert record_id > 0

    def test_get_feedback_stats(self, store):
        """Test getting feedback statistics."""
        # Add some feedback
        for i in range(10):
            feedback = UserFeedback(
                feedback_type=FeedbackType.RECOMMENDATION_HELPFUL,
                entity_type="recommendation",
                entity_id=f"rec_{i}",
                value=1 if i < 7 else -1
            )
            store.record_feedback(feedback)

        stats = store.get_feedback_stats("recommendation", days=30)

        assert stats["total_feedback"] == 10
        assert stats["positive"] == 7
        assert stats["negative"] == 3
        assert abs(stats["satisfaction_rate"] - 0.7) < 0.01

    def test_record_metric(self, store):
        """Test recording quality metric."""
        metric = QualityMetric(
            metric_type=MetricType.NDCG,
            feature="recommendations",
            value=0.85,
            sample_size=100
        )

        record_id = store.record_metric(metric)
        assert record_id > 0

    def test_record_interaction(self, store):
        """Test recording user interaction."""
        record_id = store.record_interaction(
            interaction_type="click",
            entity_type="recommendation",
            entity_id="rec_123",
            position=3
        )
        assert record_id > 0

    def test_click_through_rate(self, store):
        """Test CTR calculation."""
        # Record views and clicks
        for i in range(100):
            store.record_interaction("view", "recommendation", f"rec_{i}")

        for i in range(20):
            store.record_interaction("click", "recommendation", f"rec_{i}")

        ctr = store.get_click_through_rate("recommendation", days=30)

        assert ctr["views"] == 100
        assert ctr["clicks"] == 20
        assert abs(ctr["ctr"] - 0.2) < 0.01


class TestEvaluationSystem:
    """Tests for main EvaluationSystem."""

    @pytest.fixture
    def eval_system(self, tmp_path):
        """Create evaluation system with temp database."""
        db_path = tmp_path / "test_eval.db"
        return EvaluationSystem(db_path=db_path)

    def test_record_and_retrieve_feedback(self, eval_system):
        """Test recording and retrieving feedback."""
        feedback = UserFeedback(
            feedback_type=FeedbackType.TAG_CORRECT,
            entity_type="tag",
            entity_id="python",
            value=1
        )

        eval_system.record_feedback(feedback)
        health = eval_system.get_feature_health("tag", days=30)

        assert health["feedback"]["total_feedback"] == 1

    def test_generate_report(self, eval_system):
        """Test report generation."""
        # Add some feedback
        for i in range(5):
            eval_system.record_feedback(UserFeedback(
                feedback_type=FeedbackType.RECOMMENDATION_HELPFUL,
                entity_type="recommendation",
                entity_id=f"rec_{i}",
                value=1
            ))

        report = eval_system.generate_report(days=30)

        assert report.generated_at is not None
        assert "recommendation" in report.features


class TestRecommendationEvaluator:
    """Tests for recommendation evaluation."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with temp store."""
        db_path = tmp_path / "test_eval.db"
        store = EvaluationStore(db_path)
        return RecommendationEvaluator(store)

    def test_evaluate_perfect_recommendations(self, evaluator):
        """Test evaluation with perfect recommendations."""
        recommendations = [
            {"id": "rec_1"},
            {"id": "rec_2"},
            {"id": "rec_3"}
        ]
        clicked = ["rec_1", "rec_2", "rec_3"]

        metrics = evaluator.evaluate_session(recommendations, clicked)

        assert metrics["precision"] == 1.0
        assert metrics["ndcg"] == 1.0

    def test_evaluate_no_clicks(self, evaluator):
        """Test evaluation with no clicks."""
        recommendations = [
            {"id": "rec_1"},
            {"id": "rec_2"}
        ]
        clicked = []

        metrics = evaluator.evaluate_session(recommendations, clicked)

        assert metrics["precision"] == 0.0
        assert metrics["ndcg"] == 0.0

    def test_mrr_calculation(self, evaluator):
        """Test Mean Reciprocal Rank."""
        recommendations = [
            {"id": "rec_1"},
            {"id": "rec_2"},
            {"id": "rec_3"}
        ]

        # Click on second item
        clicked = ["rec_2"]
        metrics = evaluator.evaluate_session(recommendations, clicked)
        assert abs(metrics["mrr"] - 0.5) < 0.01  # 1/2

        # Click on first item
        clicked = ["rec_1"]
        metrics = evaluator.evaluate_session(recommendations, clicked)
        assert abs(metrics["mrr"] - 1.0) < 0.01  # 1/1


class TestTagEvaluator:
    """Tests for tag evaluation."""

    @pytest.fixture
    def evaluator(self, tmp_path):
        """Create evaluator with temp store."""
        db_path = tmp_path / "test_eval.db"
        store = EvaluationStore(db_path)
        return TagEvaluator(store)

    def test_perfect_tags(self, evaluator):
        """Test perfect tag prediction."""
        predicted = ["python", "django", "api"]
        ground_truth = ["python", "django", "api"]

        metrics = evaluator.evaluate_tags(predicted, ground_truth)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_partial_overlap(self, evaluator):
        """Test partial tag overlap."""
        predicted = ["python", "django", "flask"]
        ground_truth = ["python", "django", "api"]

        metrics = evaluator.evaluate_tags(predicted, ground_truth)

        assert metrics["precision"] == 2/3
        assert metrics["recall"] == 2/3

    def test_no_overlap(self, evaluator):
        """Test no tag overlap."""
        predicted = ["java", "spring"]
        ground_truth = ["python", "django"]

        metrics = evaluator.evaluate_tags(predicted, ground_truth)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    @pytest.fixture
    def collector(self, tmp_path):
        """Create collector with temp storage."""
        reset_metrics_collector()
        return MetricsCollector(persist_path=tmp_path / "metrics.json")

    def test_start_and_complete_run(self, collector):
        """Test starting and completing a run."""
        run = collector.start_run("test_run_1")

        assert run.run_id == "test_run_1"
        assert run.status == RunStatus.RUNNING

        collector.complete_run(success_count=10, failure_count=2)

        assert len(collector._historical_runs) == 1
        completed = collector._historical_runs[0]
        assert completed.status == RunStatus.COMPLETED
        assert completed.conversations_processed == 10
        assert completed.conversations_failed == 2

    def test_record_api_call(self, collector):
        """Test recording API calls."""
        collector.start_run("test_run")

        collector.record_api_call(
            provider="anthropic",
            model="claude-3-5-sonnet",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=1234.5,
            cost_usd=0.015,
            success=True
        )

        assert collector._current_run.total_api_calls == 1
        assert collector._current_run.total_input_tokens == 1000
        assert collector._current_run.total_output_tokens == 500
        assert collector._current_run.total_cost_usd == 0.015

    def test_record_error(self, collector):
        """Test error recording."""
        collector.start_run("test_run")

        collector.record_error(
            error_type="rate_limit",
            message="Too many requests",
            details={"retry_after": 60}
        )

        assert collector._error_counts["rate_limit"] == 1
        assert len(collector._current_run.errors) == 1

    def test_dashboard_data(self, collector):
        """Test dashboard data generation."""
        # Complete a run
        collector.start_run("run_1")
        collector.record_api_call(
            provider="anthropic",
            model="sonnet",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
            cost_usd=0.01,
            success=True
        )
        collector.complete_run(success_count=5, failure_count=1)

        dashboard = collector.get_dashboard_data()

        assert dashboard.aggregate["total_runs"] == 1
        assert dashboard.aggregate["total_api_calls"] == 1
        assert dashboard.latency["avg_ms"] > 0

    def test_health_check(self, collector):
        """Test health check."""
        collector.start_run("run_1")
        collector.complete_run(success_count=10, failure_count=0)

        health = collector.get_health_check()

        assert health["status"] == "healthy"
        assert health["total_runs"] == 1

    def test_alert_generation(self, collector):
        """Test alert generation on threshold breach."""
        collector.error_rate_threshold = 0.05
        collector.start_run("run_1")

        # Record many errors
        for _ in range(10):
            collector.record_api_call(
                provider="test",
                model="test",
                input_tokens=100,
                output_tokens=50,
                latency_ms=100,
                cost_usd=0.01,
                success=False,
                error_type="api_error"
            )

        collector.complete_run(success_count=5, failure_count=5)

        assert len(collector._alerts) > 0

    def test_persistence(self, collector, tmp_path):
        """Test metrics persistence."""
        collector.start_run("run_1")
        collector.complete_run(success_count=10, failure_count=0)

        # Load into new collector
        collector2 = MetricsCollector(persist_path=tmp_path / "metrics.json")

        assert len(collector2._historical_runs) == 1
        assert collector2._historical_runs[0].run_id == "run_1"


class TestMetricsContextManagers:
    """Tests for context managers."""

    def test_metrics_run_context(self, tmp_path):
        """Test MetricsRun context manager."""
        reset_metrics_collector()
        collector = MetricsCollector(persist_path=tmp_path / "metrics.json")

        with MetricsRun("test_run", collector) as run:
            run.record_api_call(
                provider="test",
                model="test",
                input_tokens=100,
                output_tokens=50,
                latency_ms=100,
                cost_usd=0.01,
                success=True
            )
            run.increment_processed(5)

        assert len(collector._historical_runs) == 1
        assert collector._historical_runs[0].status == RunStatus.COMPLETED

    def test_metrics_run_on_exception(self, tmp_path):
        """Test MetricsRun handles exceptions."""
        reset_metrics_collector()
        collector = MetricsCollector(persist_path=tmp_path / "metrics.json")

        with pytest.raises(ValueError):
            with MetricsRun("test_run", collector):
                raise ValueError("Test error")

        assert len(collector._historical_runs) == 1
        assert collector._historical_runs[0].status == RunStatus.FAILED

    def test_phase_timer(self, tmp_path):
        """Test PhaseTimer context manager."""
        import time

        reset_metrics_collector()
        collector = MetricsCollector(persist_path=tmp_path / "metrics.json")
        collector.start_run("test_run")

        with PhaseTimer("embedding_generation", collector):
            time.sleep(0.1)  # 100ms

        assert "embedding_generation" in collector._current_run.phase_timings
        assert collector._current_run.phase_timings["embedding_generation"] >= 0.1


class TestEnrichmentRunMetrics:
    """Tests for EnrichmentRunMetrics dataclass."""

    def test_duration_calculation(self):
        """Test duration calculation."""
        run = EnrichmentRunMetrics(
            run_id="test",
            started_at=datetime.now()
        )

        # Should have some duration
        assert run.duration_seconds >= 0

    def test_success_rate(self):
        """Test success rate calculation."""
        run = EnrichmentRunMetrics(
            run_id="test",
            started_at=datetime.now(),
            conversations_processed=80,
            conversations_failed=20
        )

        assert run.success_rate == 0.8

    def test_error_rate(self):
        """Test error rate calculation."""
        run = EnrichmentRunMetrics(
            run_id="test",
            started_at=datetime.now(),
            total_api_calls=100,
            provider_breakdown={
                "anthropic": {"failures": 10},
                "openai": {"failures": 5}
            }
        )

        assert run.error_rate == 0.15

    def test_avg_cost_per_conversation(self):
        """Test average cost calculation."""
        run = EnrichmentRunMetrics(
            run_id="test",
            started_at=datetime.now(),
            conversations_processed=100,
            total_cost_usd=5.0
        )

        assert run.avg_cost_per_conversation == 0.05

    def test_to_dict(self):
        """Test conversion to dict."""
        run = EnrichmentRunMetrics(
            run_id="test",
            started_at=datetime.now(),
            conversations_processed=10
        )

        d = run.to_dict()

        assert d["run_id"] == "test"
        assert d["conversations_processed"] == 10
        assert "started_at" in d
