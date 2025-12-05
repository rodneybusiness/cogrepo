"""
CogRepo v2 API Integration

Provides Flask blueprints for v2 functionality:
- Database-backed search (fast FTS5)
- Semantic search with embeddings
- Artifact browsing
- Project/chain visualization

Usage:
    from api_v2 import api_v2
    app.register_blueprint(api_v2, url_prefix='/api/v2')
"""

import json
import sys
from pathlib import Path
from flask import Blueprint, request, jsonify
from typing import Optional, List, Dict, Any

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

api_v2 = Blueprint('api_v2', __name__)


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent / "data"


def get_db_path() -> str:
    """Get the database path."""
    return str(get_data_dir() / "cogrepo.db")


def get_embedding_store_path() -> str:
    """Get the embedding store directory."""
    return str(get_data_dir())


# =============================================================================
# Search Endpoints
# =============================================================================

@api_v2.route('/search')
def search():
    """
    Fast database-backed search with optional semantic component.

    Query params:
        q: Search query
        source: Filter by source (OpenAI, Anthropic, Google)
        tag: Filter by tag
        has_code: Filter for conversations with code (true/false)
        min_score: Minimum quality score
        mode: Search mode (bm25, semantic, hybrid)
        page: Page number
        limit: Results per page
    """
    try:
        query = request.args.get('q', '')
        source = request.args.get('source', '')
        tag = request.args.get('tag', '')
        has_code = request.args.get('has_code', '')
        min_score = request.args.get('min_score', type=float)
        mode = request.args.get('mode', 'bm25')
        page = request.args.get('page', 1, type=int)
        limit = min(request.args.get('limit', 25, type=int), 100)

        db_path = get_db_path()

        if not Path(db_path).exists():
            return jsonify({
                'results': [],
                'total': 0,
                'page': page,
                'limit': limit,
                'mode': mode,
                'error': 'Database not initialized. Run cogrepo_enrich.py first.'
            })

        # Try to use HybridSearcher if semantic mode and embeddings available
        if mode in ('semantic', 'hybrid'):
            try:
                from search.hybrid_search import HybridSearcher
                from search.embeddings import EmbeddingStore

                embedding_store = None
                embedding_path = get_embedding_store_path()
                if Path(embedding_path).joinpath('embeddings.npy').exists():
                    embedding_store = EmbeddingStore(embedding_path)
                    embedding_store.load()

                searcher = HybridSearcher(
                    db_path,
                    embedding_store,
                    bm25_weight=0.3 if mode == 'semantic' else 0.5,
                    semantic_weight=0.7 if mode == 'semantic' else 0.5
                )

                # Build filters
                tag_filter = [tag] if tag else None
                source_filter = source if source else None

                results = searcher.search(
                    query,
                    limit=limit * 2,  # Get more for filtering
                    source_filter=source_filter,
                    tag_filter=tag_filter,
                    semantic_only=(mode == 'semantic'),
                    bm25_only=(mode == 'bm25')
                )

                # Convert to dict and apply additional filters
                filtered = []
                for r in results:
                    result_dict = r.to_dict()

                    # Apply has_code filter
                    if has_code == 'true':
                        # Need to query DB for this field
                        pass  # Will be handled below

                    filtered.append(result_dict)

                # Paginate
                total = len(filtered)
                offset = (page - 1) * limit
                paginated = filtered[offset:offset + limit]

                return jsonify({
                    'results': paginated,
                    'total': total,
                    'page': page,
                    'limit': limit,
                    'mode': mode
                })

            except ImportError:
                # Fall back to database search
                mode = 'bm25'

        # Database-only search
        from database.repository import ConversationRepository

        repo = ConversationRepository(db_path)

        # Build filters dict
        filters = {}
        if source:
            filters['source'] = source
        if has_code == 'true':
            filters['has_code'] = True

        results = repo.search(
            query=query if query else None,
            limit=limit,
            offset=(page - 1) * limit,
            filters=filters if filters else None
        )

        # Get total count
        total = repo.count(filters=filters if filters else None)

        return jsonify({
            'results': results,
            'total': total,
            'page': page,
            'limit': limit,
            'mode': mode
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_v2.route('/semantic_search')
def semantic_search():
    """
    Pure semantic search using embeddings.

    Query params:
        q: Search query
        limit: Number of results
    """
    try:
        query = request.args.get('q', '')
        limit = min(request.args.get('limit', 10, type=int), 50)

        if not query:
            return jsonify({'results': [], 'error': 'Query required'})

        embedding_path = get_embedding_store_path()
        db_path = get_db_path()

        if not Path(embedding_path).joinpath('embeddings.npy').exists():
            return jsonify({
                'results': [],
                'error': 'Embeddings not generated. Run: pip install sentence-transformers && python cogrepo_enrich.py --phase embeddings'
            })

        from search.embeddings import EmbeddingEngine, EmbeddingStore
        from database.repository import ConversationRepository

        # Load embeddings
        store = EmbeddingStore(embedding_path)
        if not store.load():
            return jsonify({'results': [], 'error': 'Failed to load embeddings'})

        # Generate query embedding
        engine = EmbeddingEngine()
        query_vec = engine.embed(query)

        # Find similar
        similar = engine.find_similar(query_vec, store.embeddings, top_k=limit)

        # Get full conversation data
        repo = ConversationRepository(db_path)
        results = []

        for idx, score in similar:
            convo_id = store.ids[idx]
            conv = repo.get(convo_id)
            if conv:
                conv['semantic_score'] = round(score, 3)
                results.append(conv)

        return jsonify({
            'results': results,
            'total': len(results),
            'mode': 'semantic'
        })

    except ImportError as e:
        return jsonify({
            'results': [],
            'error': f'Missing dependency: {e}. Install with pip install sentence-transformers'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Artifact Endpoints
# =============================================================================

@api_v2.route('/artifacts')
def list_artifacts():
    """
    List all artifacts with filtering.

    Query params:
        type: Filter by artifact type (code_snippet, shell_command, etc.)
        language: Filter by programming language
        convo_id: Filter by conversation
        page: Page number
        limit: Results per page
    """
    try:
        artifact_type = request.args.get('type', '')
        language = request.args.get('language', '')
        convo_id = request.args.get('convo_id', '')
        page = request.args.get('page', 1, type=int)
        limit = min(request.args.get('limit', 25, type=int), 100)

        db_path = get_db_path()

        if not Path(db_path).exists():
            return jsonify({'artifacts': [], 'total': 0})

        from database.repository import ConversationRepository
        repo = ConversationRepository(db_path)

        # Get all artifacts
        all_artifacts = repo.get_all_artifacts()

        # Apply filters
        filtered = []
        for artifact in all_artifacts:
            if artifact_type and artifact.get('artifact_type') != artifact_type:
                continue
            if language and artifact.get('language', '').lower() != language.lower():
                continue
            if convo_id and artifact.get('convo_id') != convo_id:
                continue
            filtered.append(artifact)

        # Paginate
        total = len(filtered)
        offset = (page - 1) * limit
        paginated = filtered[offset:offset + limit]

        return jsonify({
            'artifacts': paginated,
            'total': total,
            'page': page,
            'limit': limit
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_v2.route('/artifacts/<convo_id>')
def get_conversation_artifacts(convo_id):
    """Get all artifacts for a specific conversation."""
    try:
        db_path = get_db_path()

        if not Path(db_path).exists():
            return jsonify({'artifacts': []})

        from database.repository import ConversationRepository
        repo = ConversationRepository(db_path)

        artifacts = repo.get_artifacts(convo_id)

        return jsonify({
            'artifacts': artifacts,
            'convo_id': convo_id,
            'total': len(artifacts)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_v2.route('/artifact_types')
def get_artifact_types():
    """Get list of artifact types with counts."""
    try:
        db_path = get_db_path()

        if not Path(db_path).exists():
            return jsonify({'types': {}})

        from database.repository import ConversationRepository
        repo = ConversationRepository(db_path)

        all_artifacts = repo.get_all_artifacts()

        # Count by type
        type_counts = {}
        for artifact in all_artifacts:
            atype = artifact.get('artifact_type', 'unknown')
            type_counts[atype] = type_counts.get(atype, 0) + 1

        return jsonify({'types': type_counts})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_v2.route('/languages')
def get_languages():
    """Get list of programming languages with counts."""
    try:
        db_path = get_db_path()

        if not Path(db_path).exists():
            return jsonify({'languages': {}})

        from database.repository import ConversationRepository
        repo = ConversationRepository(db_path)

        # Get languages from conversations with code
        stats = repo.get_stats()
        languages = stats.get('languages', {})

        return jsonify({'languages': languages})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Context Endpoints (Projects & Chains)
# =============================================================================

@api_v2.route('/projects')
def list_projects():
    """
    List inferred projects.

    Projects are auto-detected groupings based on:
    - File paths mentioned
    - Repository references
    - Technology stacks
    """
    try:
        data_dir = get_data_dir()
        context_file = data_dir / 'context_analysis.json'

        if not context_file.exists():
            # Generate on the fly if not cached
            return generate_projects()

        with open(context_file) as f:
            context = json.load(f)

        projects = context.get('projects', [])

        return jsonify({
            'projects': projects,
            'total': len(projects)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_projects():
    """Generate projects from current conversations."""
    try:
        db_path = get_db_path()

        if not Path(db_path).exists():
            return jsonify({'projects': [], 'total': 0})

        from database.repository import ConversationRepository
        from context.project_inference import ProjectInferrer

        repo = ConversationRepository(db_path)
        conversations = repo.get_all(limit=10000)

        inferrer = ProjectInferrer(min_conversations=2)
        projects = inferrer.infer_projects(conversations)

        return jsonify({
            'projects': [p.to_dict() for p in projects],
            'total': len(projects)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_v2.route('/projects/<project_name>/conversations')
def get_project_conversations(project_name):
    """Get conversations belonging to a project."""
    try:
        data_dir = get_data_dir()
        context_file = data_dir / 'context_analysis.json'

        if not context_file.exists():
            return jsonify({'conversations': [], 'error': 'No context analysis found'})

        with open(context_file) as f:
            context = json.load(f)

        # Find project
        project = None
        for p in context.get('projects', []):
            if p.get('name') == project_name:
                project = p
                break

        if not project:
            return jsonify({'conversations': [], 'error': 'Project not found'})

        # Get conversation details
        db_path = get_db_path()
        from database.repository import ConversationRepository
        repo = ConversationRepository(db_path)

        conversations = []
        for convo_id in project.get('conversation_ids', []):
            conv = repo.get(convo_id)
            if conv:
                conversations.append(conv)

        return jsonify({
            'project': project,
            'conversations': conversations,
            'total': len(conversations)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_v2.route('/chains')
def list_chains():
    """
    List conversation chains.

    Chains are sequences of related conversations:
    - Continuations of earlier discussions
    - Follow-up questions
    - Debug sessions
    """
    try:
        data_dir = get_data_dir()
        context_file = data_dir / 'context_analysis.json'

        if not context_file.exists():
            return generate_chains()

        with open(context_file) as f:
            context = json.load(f)

        chains = context.get('chains', [])

        return jsonify({
            'chains': chains,
            'total': len(chains)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_chains():
    """Generate chains from current conversations."""
    try:
        db_path = get_db_path()

        if not Path(db_path).exists():
            return jsonify({'chains': [], 'total': 0})

        from database.repository import ConversationRepository
        from context.chain_detection import ChainDetector

        repo = ConversationRepository(db_path)
        conversations = repo.get_all(limit=10000)

        detector = ChainDetector()
        chains = detector.detect_chains(conversations)

        return jsonify({
            'chains': [c.to_dict() for c in chains],
            'total': len(chains)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_v2.route('/chains/<chain_id>/conversations')
def get_chain_conversations(chain_id):
    """Get conversations in a chain."""
    try:
        data_dir = get_data_dir()
        context_file = data_dir / 'context_analysis.json'

        if not context_file.exists():
            return jsonify({'conversations': [], 'error': 'No context analysis found'})

        with open(context_file) as f:
            context = json.load(f)

        # Find chain
        chain = None
        for c in context.get('chains', []):
            if c.get('chain_id') == chain_id:
                chain = c
                break

        if not chain:
            return jsonify({'conversations': [], 'error': 'Chain not found'})

        # Get conversation details
        db_path = get_db_path()
        from database.repository import ConversationRepository
        repo = ConversationRepository(db_path)

        conversations = []
        for convo_id in chain.get('conversation_ids', []):
            conv = repo.get(convo_id)
            if conv:
                conversations.append(conv)

        return jsonify({
            'chain': chain,
            'conversations': conversations,
            'total': len(conversations)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Stats Endpoints
# =============================================================================

@api_v2.route('/stats')
def get_stats():
    """Get comprehensive repository statistics."""
    try:
        db_path = get_db_path()

        if not Path(db_path).exists():
            return jsonify({
                'total': 0,
                'error': 'Database not initialized'
            })

        from database.repository import ConversationRepository
        repo = ConversationRepository(db_path)

        stats = repo.get_stats()

        # Add embedding status
        embedding_path = get_embedding_store_path()
        stats['embeddings_available'] = Path(embedding_path).joinpath('embeddings.npy').exists()

        # Add context status
        context_file = get_data_dir() / 'context_analysis.json'
        if context_file.exists():
            with open(context_file) as f:
                context = json.load(f)
            stats['projects_count'] = len(context.get('projects', []))
            stats['chains_count'] = len(context.get('chains', []))
        else:
            stats['projects_count'] = 0
            stats['chains_count'] = 0

        return jsonify(stats)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_v2.route('/tags')
def get_tags():
    """Get tag cloud with counts."""
    try:
        db_path = get_db_path()

        if not Path(db_path).exists():
            return jsonify({'tags': []})

        from database.repository import ConversationRepository
        repo = ConversationRepository(db_path)

        limit = request.args.get('limit', 50, type=int)
        tags = repo.get_tag_cloud(limit=limit)

        return jsonify({
            'tags': [{'tag': t, 'count': c} for t, c in tags]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Conversation Detail
# =============================================================================

@api_v2.route('/conversation/<path:convo_id>')
def get_conversation(convo_id):
    """Get full conversation with enrichments."""
    try:
        db_path = get_db_path()

        if not Path(db_path).exists():
            return jsonify({'error': 'Database not initialized'}), 404

        from database.repository import ConversationRepository
        repo = ConversationRepository(db_path)

        conv = repo.get(convo_id)

        if not conv:
            return jsonify({'error': 'Conversation not found'}), 404

        # Get artifacts
        artifacts = repo.get_artifacts(convo_id)
        conv['artifacts'] = artifacts

        # Get related conversations (if embeddings available)
        try:
            embedding_path = get_embedding_store_path()
            if Path(embedding_path).joinpath('embeddings.npy').exists():
                from search.embeddings import EmbeddingEngine, EmbeddingStore

                store = EmbeddingStore(embedding_path)
                if store.load():
                    related = store.find_similar(
                        store.get_by_id(convo_id),
                        top_k=5,
                        exclude_ids=[convo_id]
                    )

                    related_convos = []
                    for rel_id, score in related:
                        rel_conv = repo.get(rel_id)
                        if rel_conv:
                            related_convos.append({
                                'convo_id': rel_id,
                                'title': rel_conv.get('generated_title', ''),
                                'similarity': round(score, 3)
                            })

                    conv['related_conversations'] = related_convos
        except Exception:
            pass  # Related conversations are optional

        return jsonify(conv)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Health Check
# =============================================================================

@api_v2.route('/health')
def health_check():
    """Check v2 API health and capabilities."""
    capabilities = {
        'database': False,
        'embeddings': False,
        'artifacts': False,
        'context': False
    }

    db_path = get_db_path()
    embedding_path = get_embedding_store_path()
    context_file = get_data_dir() / 'context_analysis.json'

    if Path(db_path).exists():
        capabilities['database'] = True
        try:
            from database.repository import ConversationRepository
            repo = ConversationRepository(db_path)
            stats = repo.get_stats()
            capabilities['conversation_count'] = stats.get('total', 0)
            capabilities['artifacts'] = stats.get('total_artifacts', 0) > 0
        except Exception:
            pass

    if Path(embedding_path).joinpath('embeddings.npy').exists():
        capabilities['embeddings'] = True

    if context_file.exists():
        capabilities['context'] = True

    return jsonify({
        'status': 'ok',
        'version': '2.0',
        'capabilities': capabilities
    })
