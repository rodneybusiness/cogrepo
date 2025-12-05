"""
CogRepo Intelligence API

Flask Blueprint providing endpoints for Phase 5 intelligence features:
- Knowledge graph
- Recommendations
- Insights and trends
- Topic segmentation
- Advanced scoring

All endpoints are optional and degrade gracefully if dependencies are missing.
"""

from flask import Blueprint, request, jsonify
from pathlib import Path
import sys
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

intelligence_api = Blueprint('intelligence', __name__)


def get_data_paths():
    """Get paths to data files."""
    data_dir = Path(__file__).parent.parent / 'data'
    return {
        'jsonl': data_dir / 'enriched_repository.jsonl',
        'db': data_dir / 'cogrepo.db',
        'embeddings': data_dir / 'embeddings.npy',
        'embedding_ids': data_dir / 'embedding_ids.json',
        'knowledge_graph': data_dir / 'knowledge_graph.json',
    }


def load_conversations():
    """Load conversations from JSONL with error handling."""
    paths = get_data_paths()
    conversations = []

    if paths['jsonl'].exists():
        try:
            with open(paths['jsonl'], encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            conversations.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines
        except IOError:
            pass  # Return empty list on IO errors

    return conversations


# =============================================================================
# Knowledge Graph Endpoints
# =============================================================================

@intelligence_api.route('/knowledge-graph')
def get_knowledge_graph():
    """
    Get knowledge graph data for visualization.

    Query params:
        max_nodes: Maximum nodes (default 100)
        max_edges: Maximum edges (default 200)
        min_count: Minimum entity occurrence (default 2)
    """
    try:
        from context.knowledge_graph import KnowledgeGraph

        paths = get_data_paths()

        # Try to load cached graph
        if paths['knowledge_graph'].exists():
            kg = KnowledgeGraph.load(str(paths['knowledge_graph']))
        else:
            # Build from conversations
            conversations = load_conversations()
            kg = KnowledgeGraph()
            kg.build_from_conversations(conversations)

        # Get visualization params
        max_nodes = request.args.get('max_nodes', 100, type=int)
        max_edges = request.args.get('max_edges', 200, type=int)
        min_count = request.args.get('min_count', 2, type=int)

        viz_data = kg.export_for_visualization(
            max_nodes=max_nodes,
            max_edges=max_edges,
            min_node_count=min_count
        )

        return jsonify(viz_data)

    except ImportError:
        return jsonify({
            'error': 'Knowledge graph module not available',
            'nodes': [],
            'links': []
        }), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/knowledge-graph/entity/<entity_name>')
def get_entity_relationships(entity_name):
    """Get relationships for a specific entity."""
    try:
        from context.knowledge_graph import KnowledgeGraph

        paths = get_data_paths()

        if paths['knowledge_graph'].exists():
            kg = KnowledgeGraph.load(str(paths['knowledge_graph']))
        else:
            conversations = load_conversations()
            kg = KnowledgeGraph()
            kg.build_from_conversations(conversations)

        limit = request.args.get('limit', 10, type=int)
        related = kg.get_related(entity_name, limit=limit)

        return jsonify({
            'entity': entity_name,
            'related': [
                {
                    'name': e.name,
                    'type': e.entity_type,
                    'weight': w,
                    'count': e.occurrence_count
                }
                for e, w in related
            ]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/knowledge-graph/top-entities')
def get_top_entities():
    """Get top entities by occurrence."""
    try:
        from context.knowledge_graph import KnowledgeGraph

        paths = get_data_paths()

        if paths['knowledge_graph'].exists():
            kg = KnowledgeGraph.load(str(paths['knowledge_graph']))
        else:
            conversations = load_conversations()
            kg = KnowledgeGraph()
            kg.build_from_conversations(conversations)

        entity_type = request.args.get('type')
        limit = request.args.get('limit', 20, type=int)

        entities = kg.get_top_entities(entity_type, limit=limit)

        return jsonify({
            'entities': [e.to_dict() for e in entities],
            'types': kg.get_entity_types()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Recommendation Endpoints
# =============================================================================

@intelligence_api.route('/recommendations/<convo_id>')
def get_recommendations(convo_id):
    """Get recommendations for a conversation."""
    try:
        from intelligence.recommendations import RecommendationEngine
        import numpy as np

        paths = get_data_paths()
        limit = request.args.get('limit', 5, type=int)

        # Load data
        conversations = load_conversations()
        embeddings = None
        ids = None

        if paths['embeddings'].exists() and paths['embedding_ids'].exists():
            embeddings = np.load(str(paths['embeddings']))
            with open(paths['embedding_ids']) as f:
                ids = json.load(f)

        engine = RecommendationEngine(embeddings, ids, conversations)
        recs = engine.find_similar(convo_id, limit=limit)

        # Add conversation metadata
        id_to_conv = {c.get('convo_id', ''): c for c in conversations}
        result = []

        for rec in recs:
            conv = id_to_conv.get(rec.conversation_id, {})
            result.append({
                **rec.to_dict(),
                'title': conv.get('generated_title', ''),
                'tags': conv.get('tags', [])[:5],
            })

        return jsonify({
            'source': convo_id,
            'recommendations': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/recommendations/trending')
def get_trending():
    """Get trending/recent high-quality conversations."""
    try:
        from intelligence.recommendations import RecommendationEngine

        conversations = load_conversations()
        days = request.args.get('days', 7, type=int)
        limit = request.args.get('limit', 10, type=int)

        engine = RecommendationEngine(conversations=conversations)
        trending = engine.get_trending(conversations, days=days, limit=limit)

        # Add conversation metadata
        id_to_conv = {c.get('convo_id', ''): c for c in conversations}
        result = []

        for rec in trending:
            conv = id_to_conv.get(rec.conversation_id, {})
            result.append({
                **rec.to_dict(),
                'title': conv.get('generated_title', ''),
                'tags': conv.get('tags', [])[:5],
            })

        return jsonify({
            'period_days': days,
            'trending': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Insights Endpoints
# =============================================================================

@intelligence_api.route('/insights')
def get_insights():
    """Get repository insights."""
    try:
        from intelligence.insights import InsightsEngine

        conversations = load_conversations()
        engine = InsightsEngine(conversations)

        insights = engine.generate_insights()

        return jsonify({
            'insights': [i.to_dict() for i in insights[:20]],
            'count': len(insights)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/insights/dashboard')
def get_dashboard():
    """Get full dashboard data."""
    try:
        from intelligence.insights import InsightsEngine

        conversations = load_conversations()
        engine = InsightsEngine(conversations)

        return jsonify(engine.export_dashboard_data())

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/insights/trends/<topic>')
def get_topic_trends(topic):
    """Get trends for a specific topic."""
    try:
        from intelligence.insights import InsightsEngine

        conversations = load_conversations()
        period = request.args.get('period', 'month')

        engine = InsightsEngine(conversations)
        trend = engine.get_topic_trends(topic, period=period)

        return jsonify(trend.to_dict())

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/insights/technology-trends')
def get_technology_trends():
    """Get technology adoption trends."""
    try:
        from intelligence.insights import InsightsEngine

        conversations = load_conversations()
        period = request.args.get('period', 'month')
        limit = request.args.get('limit', 20, type=int)

        engine = InsightsEngine(conversations)
        trends = engine.get_technology_trends(period=period, limit=limit)

        return jsonify({
            'trends': [t.to_dict() for t in trends],
            'period': period
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Topic Segmentation Endpoints
# =============================================================================

@intelligence_api.route('/topics/<convo_id>')
def get_conversation_topics(convo_id):
    """Get topic segments for a conversation."""
    try:
        from intelligence.topic_segmentation import TopicSegmenter

        conversations = load_conversations()
        conv = next((c for c in conversations if c.get('convo_id') == convo_id), None)

        if not conv:
            return jsonify({'error': 'Conversation not found'}), 404

        segmenter = TopicSegmenter()
        topics = segmenter.segment(conv)

        return jsonify(topics.to_dict())

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/topics/multi-topic')
def get_multi_topic_conversations():
    """Get list of conversations with multiple topics."""
    try:
        from intelligence.topic_segmentation import TopicSegmenter

        conversations = load_conversations()
        limit = request.args.get('limit', 50, type=int)

        segmenter = TopicSegmenter()
        multi_topic = []

        for conv in conversations[:limit * 2]:  # Check more to find multi-topic
            topics = segmenter.segment(conv)
            if topics.has_topic_transitions:
                multi_topic.append({
                    'conversation_id': conv.get('convo_id', ''),
                    'title': conv.get('generated_title', ''),
                    'topic_count': topics.topic_count,
                    'topics': [s.topic for s in topics.segments],
                    'summary': topics.topic_summary,
                })

            if len(multi_topic) >= limit:
                break

        return jsonify({
            'multi_topic_conversations': multi_topic,
            'count': len(multi_topic)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Advanced Scoring Endpoints
# =============================================================================

@intelligence_api.route('/score/<convo_id>')
def get_conversation_score(convo_id):
    """Get detailed score for a conversation."""
    try:
        from intelligence.scoring import ConversationScorer

        conversations = load_conversations()
        conv = next((c for c in conversations if c.get('convo_id') == convo_id), None)

        if not conv:
            return jsonify({'error': 'Conversation not found'}), 404

        scorer = ConversationScorer(conversations)
        score = scorer.score(conv)

        return jsonify({
            'conversation_id': convo_id,
            'score': score.to_dict()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/scores/top')
def get_top_scored():
    """Get top-scored conversations."""
    try:
        from intelligence.scoring import score_conversations

        conversations = load_conversations()
        limit = request.args.get('limit', 20, type=int)

        results = score_conversations(conversations, calculate_percentiles=True)
        sorted_results = sorted(results, key=lambda x: x[1].overall, reverse=True)

        # Build response
        id_to_conv = {c.get('convo_id', ''): c for c in conversations}
        top = []

        for convo_id, score in sorted_results[:limit]:
            conv = id_to_conv.get(convo_id, {})
            top.append({
                'conversation_id': convo_id,
                'title': conv.get('generated_title', ''),
                'score': score.to_dict(),
            })

        return jsonify({
            'top_conversations': top,
            'count': len(top)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@intelligence_api.route('/scores/distribution')
def get_score_distribution():
    """Get score distribution statistics."""
    try:
        from intelligence.scoring import score_conversations

        conversations = load_conversations()
        results = score_conversations(conversations, calculate_percentiles=False)

        scores = [s.overall for _, s in results]

        # Build distribution
        buckets = {
            '90-100': 0, '80-89': 0, '70-79': 0, '60-69': 0,
            '50-59': 0, '40-49': 0, '30-39': 0, '20-29': 0,
            '10-19': 0, '0-9': 0
        }

        for score in scores:
            if score >= 90: buckets['90-100'] += 1
            elif score >= 80: buckets['80-89'] += 1
            elif score >= 70: buckets['70-79'] += 1
            elif score >= 60: buckets['60-69'] += 1
            elif score >= 50: buckets['50-59'] += 1
            elif score >= 40: buckets['40-49'] += 1
            elif score >= 30: buckets['30-39'] += 1
            elif score >= 20: buckets['20-29'] += 1
            elif score >= 10: buckets['10-19'] += 1
            else: buckets['0-9'] += 1

        return jsonify({
            'distribution': buckets,
            'statistics': {
                'count': len(scores),
                'average': sum(scores) / len(scores) if scores else 0,
                'min': min(scores) if scores else 0,
                'max': max(scores) if scores else 0,
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Clustering Endpoints
# =============================================================================

@intelligence_api.route('/clusters')
def get_clusters():
    """Get conversation clusters."""
    try:
        from intelligence.clustering import ConversationClusterer
        import numpy as np

        paths = get_data_paths()

        if not paths['embeddings'].exists():
            return jsonify({
                'error': 'Embeddings not generated',
                'clusters': []
            }), 503

        conversations = load_conversations()
        embeddings = np.load(str(paths['embeddings']))

        with open(paths['embedding_ids']) as f:
            ids = json.load(f)

        clusterer = ConversationClusterer()
        clusters = clusterer.cluster(conversations, embeddings, ids)

        return jsonify({
            'clusters': [c.to_dict() for c in clusters if not c.is_outlier],
            'outlier_count': sum(1 for c in clusters if c.is_outlier),
            'total_clusters': len([c for c in clusters if not c.is_outlier])
        })

    except ImportError:
        return jsonify({
            'error': 'Clustering requires scikit-learn or hdbscan',
            'clusters': []
        }), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Health Check
# =============================================================================

@intelligence_api.route('/health')
def intelligence_health():
    """Check intelligence feature availability."""
    features = {}

    # Check each module
    try:
        from context.knowledge_graph import KnowledgeGraph
        features['knowledge_graph'] = 'available'
    except ImportError:
        features['knowledge_graph'] = 'unavailable'

    try:
        from intelligence.recommendations import RecommendationEngine
        features['recommendations'] = 'available'
    except ImportError:
        features['recommendations'] = 'unavailable'

    try:
        from intelligence.insights import InsightsEngine
        features['insights'] = 'available'
    except ImportError:
        features['insights'] = 'unavailable'

    try:
        from intelligence.topic_segmentation import TopicSegmenter
        features['topic_segmentation'] = 'available'
    except ImportError:
        features['topic_segmentation'] = 'unavailable'

    try:
        from intelligence.scoring import ConversationScorer
        features['advanced_scoring'] = 'available'
    except ImportError:
        features['advanced_scoring'] = 'unavailable'

    try:
        from intelligence.clustering import ConversationClusterer
        features['clustering'] = 'available'
    except ImportError:
        features['clustering'] = 'unavailable'

    # Check data
    paths = get_data_paths()
    features['embeddings'] = 'available' if paths['embeddings'].exists() else 'not_generated'

    return jsonify({
        'status': 'ok',
        'features': features
    })
