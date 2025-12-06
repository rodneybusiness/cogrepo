"""
CogRepo Evaluation and Metrics API

Flask Blueprint providing endpoints for:
- User feedback collection
- Quality evaluation metrics
- Enrichment run monitoring
- System health and observability

All endpoints degrade gracefully if dependencies are missing.
"""

from flask import Blueprint, request, jsonify
from pathlib import Path
import sys
import json
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

evaluation_api = Blueprint('evaluation', __name__)


def get_data_paths():
    """Get paths to data files."""
    data_dir = Path(__file__).parent.parent / 'data'
    return {
        'evaluation_db': data_dir / 'evaluation.db',
        'metrics': data_dir / 'metrics.json',
    }


# =============================================================================
# Feedback Endpoints
# =============================================================================

@evaluation_api.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback on recommendations, tags, clusters, etc.

    Request body:
    {
        "feedback_type": "rec_helpful|rec_irrelevant|tag_correct|...",
        "entity_type": "recommendation|cluster|tag|summary",
        "entity_id": "entity_123",
        "value": 1,  // 1=positive, -1=negative, 0=neutral
        "comment": "Optional comment",
        "context": {}  // Optional context data
    }
    """
    try:
        from intelligence.evaluation import (
            EvaluationSystem, UserFeedback, FeedbackType
        )

        data = request.get_json()

        if not data:
            return jsonify({'error': 'Request body required'}), 400

        # Validate required fields
        required = ['feedback_type', 'entity_type', 'entity_id']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Parse feedback type
        try:
            feedback_type = FeedbackType(data['feedback_type'])
        except ValueError:
            valid_types = [t.value for t in FeedbackType]
            return jsonify({
                'error': f"Invalid feedback_type. Must be one of: {valid_types}"
            }), 400

        # Create feedback object
        feedback = UserFeedback(
            feedback_type=feedback_type,
            entity_type=data['entity_type'],
            entity_id=data['entity_id'],
            value=data.get('value', 1),
            user_id=data.get('user_id'),
            comment=data.get('comment'),
            context=data.get('context', {})
        )

        # Record feedback
        paths = get_data_paths()
        eval_system = EvaluationSystem(db_path=paths['evaluation_db'])
        record_id = eval_system.record_feedback(feedback)

        return jsonify({
            'status': 'recorded',
            'record_id': record_id,
            'timestamp': datetime.now().isoformat()
        })

    except ImportError:
        return jsonify({
            'error': 'Evaluation module not available'
        }), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@evaluation_api.route('/feedback/stats/<entity_type>')
def get_feedback_stats(entity_type):
    """
    Get feedback statistics for an entity type.

    Path params:
        entity_type: recommendation, cluster, tag, summary

    Query params:
        days: Number of days to analyze (default 30)
    """
    try:
        from intelligence.evaluation import EvaluationSystem

        days = request.args.get('days', 30, type=int)
        paths = get_data_paths()

        eval_system = EvaluationSystem(db_path=paths['evaluation_db'])
        stats = eval_system.store.get_feedback_stats(entity_type, days)

        return jsonify(stats)

    except ImportError:
        return jsonify({'error': 'Evaluation module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@evaluation_api.route('/feedback/recent')
def get_recent_feedback():
    """
    Get recent feedback entries.

    Query params:
        feedback_type: Filter by type (optional)
        entity_type: Filter by entity type (optional)
        days: Number of days (default 7)
        limit: Maximum results (default 50)
    """
    try:
        from intelligence.evaluation import EvaluationSystem, FeedbackType

        days = request.args.get('days', 7, type=int)
        limit = request.args.get('limit', 50, type=int)
        feedback_type_str = request.args.get('feedback_type')

        paths = get_data_paths()
        eval_system = EvaluationSystem(db_path=paths['evaluation_db'])

        if feedback_type_str:
            try:
                feedback_type = FeedbackType(feedback_type_str)
                feedback = eval_system.store.get_feedback_by_type(
                    feedback_type, days, limit
                )
            except ValueError:
                return jsonify({'error': 'Invalid feedback_type'}), 400
        else:
            # Get all recent feedback (combine types)
            feedback = []
            for ft in FeedbackType:
                feedback.extend(
                    eval_system.store.get_feedback_by_type(ft, days, limit // 10)
                )
            feedback = sorted(
                feedback,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )[:limit]

        return jsonify({
            'feedback': feedback,
            'count': len(feedback),
            'period_days': days
        })

    except ImportError:
        return jsonify({'error': 'Evaluation module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Interaction Tracking Endpoints
# =============================================================================

@evaluation_api.route('/interaction', methods=['POST'])
def record_interaction():
    """
    Record a user interaction (click, view, dismiss).

    Request body:
    {
        "interaction_type": "click|view|dismiss|hover",
        "entity_type": "recommendation|search_result|cluster",
        "entity_id": "entity_123",
        "position": 3,  // Optional position in list
        "session_id": "session_abc",  // Optional
        "context": {}  // Optional
    }
    """
    try:
        from intelligence.evaluation import EvaluationSystem

        data = request.get_json()

        if not data:
            return jsonify({'error': 'Request body required'}), 400

        required = ['interaction_type', 'entity_type', 'entity_id']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        paths = get_data_paths()
        eval_system = EvaluationSystem(db_path=paths['evaluation_db'])

        record_id = eval_system.record_interaction(
            interaction_type=data['interaction_type'],
            entity_type=data['entity_type'],
            entity_id=data['entity_id'],
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            position=data.get('position'),
            context=data.get('context')
        )

        return jsonify({
            'status': 'recorded',
            'record_id': record_id
        })

    except ImportError:
        return jsonify({'error': 'Evaluation module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@evaluation_api.route('/ctr/<entity_type>')
def get_click_through_rate(entity_type):
    """
    Get click-through rate for an entity type.

    Query params:
        days: Number of days to analyze (default 30)
    """
    try:
        from intelligence.evaluation import EvaluationSystem

        days = request.args.get('days', 30, type=int)
        paths = get_data_paths()

        eval_system = EvaluationSystem(db_path=paths['evaluation_db'])
        ctr = eval_system.store.get_click_through_rate(entity_type, days)

        return jsonify(ctr)

    except ImportError:
        return jsonify({'error': 'Evaluation module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Evaluation Report Endpoints
# =============================================================================

@evaluation_api.route('/report')
def get_evaluation_report():
    """
    Get comprehensive evaluation report.

    Query params:
        days: Analysis period in days (default 30)
    """
    try:
        from intelligence.evaluation import EvaluationSystem

        days = request.args.get('days', 30, type=int)
        paths = get_data_paths()

        eval_system = EvaluationSystem(db_path=paths['evaluation_db'])
        report = eval_system.generate_report(days)

        return jsonify(report.to_dict())

    except ImportError:
        return jsonify({'error': 'Evaluation module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@evaluation_api.route('/feature-health/<feature>')
def get_feature_health(feature):
    """
    Get health status for a specific feature.

    Path params:
        feature: recommendation, cluster, tag, summary

    Query params:
        days: Analysis period (default 30)
    """
    try:
        from intelligence.evaluation import EvaluationSystem

        days = request.args.get('days', 30, type=int)
        paths = get_data_paths()

        eval_system = EvaluationSystem(db_path=paths['evaluation_db'])
        health = eval_system.get_feature_health(feature, days)

        return jsonify(health)

    except ImportError:
        return jsonify({'error': 'Evaluation module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Metrics Endpoints
# =============================================================================

@evaluation_api.route('/metrics/dashboard')
def get_metrics_dashboard():
    """
    Get enrichment metrics dashboard data.

    Returns run history, costs, latency, and error rates.
    """
    try:
        from core.metrics import get_metrics_collector

        paths = get_data_paths()
        collector = get_metrics_collector(persist_path=paths['metrics'])
        dashboard = collector.get_dashboard_data()

        return jsonify(dashboard.to_dict())

    except ImportError:
        return jsonify({'error': 'Metrics module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@evaluation_api.route('/metrics/health')
def get_metrics_health():
    """
    Get system health check based on metrics.

    Returns status and key indicators.
    """
    try:
        from core.metrics import get_metrics_collector

        paths = get_data_paths()
        collector = get_metrics_collector(persist_path=paths['metrics'])
        health = collector.get_health_check()

        return jsonify(health)

    except ImportError:
        return jsonify({'error': 'Metrics module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@evaluation_api.route('/metrics/runs')
def get_recent_runs():
    """
    Get recent enrichment runs.

    Query params:
        limit: Maximum runs to return (default 10)
    """
    try:
        from core.metrics import get_metrics_collector

        limit = request.args.get('limit', 10, type=int)
        paths = get_data_paths()

        collector = get_metrics_collector(persist_path=paths['metrics'])
        dashboard = collector.get_dashboard_data()

        runs = dashboard.recent_runs[-limit:] if dashboard.recent_runs else []

        return jsonify({
            'runs': runs,
            'count': len(runs),
            'current_run': dashboard.current_run
        })

    except ImportError:
        return jsonify({'error': 'Metrics module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@evaluation_api.route('/metrics/costs')
def get_cost_breakdown():
    """
    Get cost breakdown by provider and time.

    Query params:
        days: Analysis period (default 30)
    """
    try:
        from core.metrics import get_metrics_collector

        paths = get_data_paths()
        collector = get_metrics_collector(persist_path=paths['metrics'])
        dashboard = collector.get_dashboard_data()

        return jsonify({
            'costs': dashboard.costs,
            'providers': dashboard.providers,
            'total_cost_usd': dashboard.aggregate.get('total_cost_usd', 0)
        })

    except ImportError:
        return jsonify({'error': 'Metrics module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@evaluation_api.route('/metrics/latency')
def get_latency_stats():
    """Get API latency statistics."""
    try:
        from core.metrics import get_metrics_collector

        paths = get_data_paths()
        collector = get_metrics_collector(persist_path=paths['metrics'])
        dashboard = collector.get_dashboard_data()

        return jsonify({
            'latency': dashboard.latency,
            'providers': {
                name: {'avg_latency_ms': stats.get('avg_latency_ms', 0)}
                for name, stats in dashboard.providers.items()
            }
        })

    except ImportError:
        return jsonify({'error': 'Metrics module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@evaluation_api.route('/metrics/errors')
def get_error_breakdown():
    """Get error breakdown by type."""
    try:
        from core.metrics import get_metrics_collector

        paths = get_data_paths()
        collector = get_metrics_collector(persist_path=paths['metrics'])
        dashboard = collector.get_dashboard_data()

        return jsonify({
            'errors': dashboard.errors,
            'total_errors': sum(dashboard.errors.values()),
            'error_rate': dashboard.aggregate.get('avg_success_rate', 1) - 1
        })

    except ImportError:
        return jsonify({'error': 'Metrics module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@evaluation_api.route('/metrics/alerts')
def get_alerts():
    """Get recent alerts."""
    try:
        from core.metrics import get_metrics_collector

        paths = get_data_paths()
        collector = get_metrics_collector(persist_path=paths['metrics'])
        dashboard = collector.get_dashboard_data()

        return jsonify({
            'alerts': dashboard.alerts,
            'unacknowledged': sum(
                1 for a in dashboard.alerts if not a.get('acknowledged', False)
            )
        })

    except ImportError:
        return jsonify({'error': 'Metrics module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@evaluation_api.route('/metrics/alerts/<int:index>/acknowledge', methods=['POST'])
def acknowledge_alert(index):
    """Acknowledge an alert."""
    try:
        from core.metrics import get_metrics_collector

        paths = get_data_paths()
        collector = get_metrics_collector(persist_path=paths['metrics'])

        if collector.acknowledge_alert(index):
            return jsonify({'status': 'acknowledged'})
        else:
            return jsonify({'error': 'Alert not found'}), 404

    except ImportError:
        return jsonify({'error': 'Metrics module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# LLM Provider Endpoints
# =============================================================================

@evaluation_api.route('/providers/health')
def get_provider_health():
    """
    Get health status of all LLM providers.

    Returns availability and performance metrics for each provider.
    """
    try:
        from core.llm_provider import get_provider_chain
        from core.config import get_config

        config = get_config()

        # Build provider config
        provider_config = {
            "primary_provider": config.llm.primary_provider,
            "fallback_providers": config.llm.fallback_providers,
            "anthropic": {"api_key": config.anthropic.api_key},
            "openai": {"api_key": config.openai.api_key} if config.openai.api_key else {},
            "ollama": {"base_url": config.ollama.base_url}
        }

        chain = get_provider_chain(provider_config)
        health = chain.get_health()

        result = {}
        for name, h in health.items():
            result[name] = {
                "available": h.is_available,
                "last_check": h.last_check.isoformat(),
                "last_error": h.last_error,
                "success_rate": h.success_rate,
                "avg_latency_ms": h.avg_latency_ms
            }

        return jsonify({
            "providers": result,
            "primary": config.llm.primary_provider,
            "fallbacks": config.llm.fallback_providers
        })

    except ImportError:
        return jsonify({'error': 'LLM provider module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@evaluation_api.route('/providers/stats')
def get_provider_stats():
    """Get usage statistics for all providers."""
    try:
        from core.metrics import get_metrics_collector

        paths = get_data_paths()
        collector = get_metrics_collector(persist_path=paths['metrics'])
        dashboard = collector.get_dashboard_data()

        return jsonify({
            "providers": dashboard.providers,
            "costs": dashboard.costs
        })

    except ImportError:
        return jsonify({'error': 'Metrics module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# PII Endpoints
# =============================================================================

@evaluation_api.route('/pii/stats')
def get_pii_stats():
    """Get PII detection and scrubbing statistics."""
    try:
        from core.pii_handler import PIIHandler

        # Get stats from handler if available
        # Note: This would need to be tracked across requests
        # For now, return configuration status

        from core.config import get_config
        config = get_config()

        return jsonify({
            "enabled": config.pii.enabled,
            "scrub_mode": config.pii.scrub_mode,
            "detection_enabled": {
                "emails": config.pii.detect_emails,
                "phones": config.pii.detect_phones,
                "ssn": config.pii.detect_ssn,
                "credit_cards": config.pii.detect_credit_cards,
                "api_keys": config.pii.detect_api_keys,
                "ip_addresses": config.pii.detect_ip_addresses
            },
            "vault_enabled": config.pii.enable_vault
        })

    except ImportError:
        return jsonify({'error': 'PII module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@evaluation_api.route('/pii/test', methods=['POST'])
def test_pii_detection():
    """
    Test PII detection on sample text.

    Request body:
    {
        "text": "Sample text with test@example.com"
    }

    Returns detected PII (without revealing actual values).
    """
    try:
        from core.pii_handler import PIIHandler, PIIConfig

        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text required'}), 400

        # Use a test config
        config = PIIConfig(enabled=True)
        handler = PIIHandler(config)

        # Detect PII
        matches = handler.detector.detect(data['text'])

        # Return sanitized results (don't reveal actual PII)
        results = []
        for match in matches:
            results.append({
                "type": match.pii_type.value,
                "severity": match.severity.value,
                "confidence": match.confidence,
                "position": {"start": match.start, "end": match.end},
                "length": len(match.original)
            })

        return jsonify({
            "pii_found": len(results) > 0,
            "count": len(results),
            "detections": results
        })

    except ImportError:
        return jsonify({'error': 'PII module not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Health Check
# =============================================================================

@evaluation_api.route('/health')
def evaluation_health():
    """Check evaluation and metrics feature availability."""
    features = {}

    # Check evaluation module
    try:
        from intelligence.evaluation import EvaluationSystem
        features['evaluation'] = 'available'
    except ImportError:
        features['evaluation'] = 'unavailable'

    # Check metrics module
    try:
        from core.metrics import MetricsCollector
        features['metrics'] = 'available'
    except ImportError:
        features['metrics'] = 'unavailable'

    # Check LLM providers
    try:
        from core.llm_provider import LLMProvider
        features['llm_providers'] = 'available'
    except ImportError:
        features['llm_providers'] = 'unavailable'

    # Check PII handler
    try:
        from core.pii_handler import PIIHandler
        features['pii_handler'] = 'available'
    except ImportError:
        features['pii_handler'] = 'unavailable'

    # Check data files
    paths = get_data_paths()
    features['evaluation_db'] = 'exists' if paths['evaluation_db'].exists() else 'not_created'
    features['metrics_file'] = 'exists' if paths['metrics'].exists() else 'not_created'

    return jsonify({
        'status': 'ok',
        'features': features
    })
