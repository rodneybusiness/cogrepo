#!/usr/bin/env python3
"""
CogRepo Web Server with Upload & Real-time Processing

Features:
- File upload with drag-and-drop support
- Real-time progress via WebSocket
- Import history and statistics
- Search and browse conversations
- Mobile-responsive modern UI
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models import ProcessingStats
from parsers import ChatGPTParser, ClaudeParser, GeminiParser
from state_manager import ProcessingStateManager
from enrichment import EnrichmentPipeline
from cogrepo_import import CogRepoImporter
import yaml

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='.')
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure upload directory exists
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

# Setup logging
try:
    from logging_config import setup_logging, setup_request_logging
    import os
    is_production = os.getenv('FLASK_ENV', 'development') == 'production'
    setup_logging(app, level='INFO', json_format=is_production)
    setup_request_logging(app)
    print("  [OK] Logging configured")
except ImportError as e:
    print(f"  [!!] Logging config not available: {e}")

# Register health endpoints
try:
    from health import health_bp
    app.register_blueprint(health_bp)
    print("  [OK] Health endpoints registered (/health, /ready)")
except ImportError as e:
    print(f"  [!!] Health endpoints not available: {e}")

# Register v2 API blueprint
try:
    from api_v2 import api_v2
    app.register_blueprint(api_v2, url_prefix='/api/v2')
    print("  [OK] v2 API registered at /api/v2")
except ImportError as e:
    print(f"  [!!] v2 API not available: {e}")

# Setup error handlers
try:
    from error_handlers import setup_error_handlers
    setup_error_handlers(app)
    print("  [OK] Error handlers configured")
except ImportError as e:
    print(f"  [!!] Error handlers not available: {e}")

# Register intelligence API blueprint
try:
    from api_intelligence import intelligence_api
    app.register_blueprint(intelligence_api, url_prefix='/api/intelligence')
    print("  [OK] Intelligence API registered at /api/intelligence")
except ImportError as e:
    print(f"  [!!] Intelligence API not available: {e}")

# Register SOTA enrichment API blueprint
try:
    from enrichment_api import enrichment_bp
    app.register_blueprint(enrichment_bp)
    print("  [OK] SOTA Enrichment API registered at /api/enrich")
except ImportError as e:
    print(f"  [!!] SOTA Enrichment API not available: {e}")

# Global state for tracking active imports
active_imports: Dict[str, Dict[str, Any]] = {}
import_lock = threading.Lock()


def get_repo_path():
    """Get the repository data path"""
    possible_paths = [
        Path(__file__).parent.parent / "data" / "enriched_repository.jsonl",
        Path(__file__).parent / "../data/enriched_repository.jsonl",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    return str(possible_paths[0])


class ProgressCallback:
    """Callback class for tracking import progress"""

    def __init__(self, import_id: str, socketio_instance):
        self.import_id = import_id
        self.socketio = socketio_instance
        self.current = 0
        self.total = 0
        self.status = "initializing"
        self.message = ""

    def update(self, current: int, total: int, status: str, message: str = ""):
        """Update progress and emit to WebSocket"""
        self.current = current
        self.total = total
        self.status = status
        self.message = message

        self.socketio.emit('import_progress', {
            'import_id': self.import_id,
            'current': current,
            'total': total,
            'percentage': int((current / total * 100) if total > 0 else 0),
            'status': status,
            'message': message
        })


def process_import_background(
    import_id: str,
    file_path: str,
    source: str,
    enrich: bool,
    config_path: str
):
    """Background thread for processing imports"""
    try:
        # Update status
        with import_lock:
            active_imports[import_id]['status'] = 'processing'
            active_imports[import_id]['start_time'] = datetime.now().isoformat()

        socketio.emit('import_status', {
            'import_id': import_id,
            'status': 'processing',
            'message': 'Starting import...'
        })

        # Create importer
        importer = CogRepoImporter(config_file=config_path)

        # Simulate progress updates (in real implementation, modify CogRepoImporter to accept callback)
        # For now, we'll use a simplified version

        socketio.emit('import_progress', {
            'import_id': import_id,
            'current': 0,
            'total': 100,
            'percentage': 0,
            'status': 'parsing',
            'message': 'Parsing export file...'
        })

        # Run the actual import
        stats = importer.import_conversations(
            file_path=file_path,
            source=source,
            enrich=enrich,
            dry_run=False
        )

        # Import completed
        with import_lock:
            active_imports[import_id]['status'] = 'completed'
            active_imports[import_id]['end_time'] = datetime.now().isoformat()
            active_imports[import_id]['stats'] = stats.to_dict()

        socketio.emit('import_complete', {
            'import_id': import_id,
            'status': 'completed',
            'stats': stats.to_dict(),
            'message': f'Successfully processed {stats.total_processed} conversations'
        })

    except Exception as e:
        # Import failed
        error_message = str(e)

        with import_lock:
            active_imports[import_id]['status'] = 'failed'
            active_imports[import_id]['end_time'] = datetime.now().isoformat()
            active_imports[import_id]['error'] = error_message

        socketio.emit('import_error', {
            'import_id': import_id,
            'status': 'failed',
            'error': error_message,
            'message': f'Import failed: {error_message}'
        })

    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass


# API Routes

@app.route('/')
def index():
    """Serve the main UI"""
    return send_from_directory('.', 'index.html')


@app.route('/api/status')
def api_status():
    """Get server status and statistics"""
    try:
        state_manager = ProcessingStateManager()
        stats = state_manager.get_stats()

        # Check if data file exists
        repo_path = get_repo_path()
        data_exists = os.path.exists(repo_path)

        if data_exists:
            # Count conversations
            with open(repo_path, 'r') as f:
                conversation_count = sum(1 for _ in f)
        else:
            conversation_count = 0

        return jsonify({
            'status': 'online',
            'data_exists': data_exists,
            'total_conversations': conversation_count,
            'stats': stats,
            'api_key_configured': bool(os.getenv('ANTHROPIC_API_KEY'))
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Handle file upload and start processing"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get options
        source = request.form.get('source', 'auto')
        enrich = request.form.get('enrich', 'true').lower() == 'true'

        # Validate source
        if source not in ['auto', 'chatgpt', 'claude', 'gemini']:
            return jsonify({'error': 'Invalid source'}), 400

        # Check API key if enriching
        if enrich and not os.getenv('ANTHROPIC_API_KEY'):
            return jsonify({
                'error': 'ANTHROPIC_API_KEY not configured. Cannot enrich without API key.'
            }), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        import_id = str(uuid.uuid4())
        file_path = app.config['UPLOAD_FOLDER'] / f"{import_id}_{filename}"
        file.save(str(file_path))

        # Create import record
        with import_lock:
            active_imports[import_id] = {
                'id': import_id,
                'filename': filename,
                'source': source,
                'enrich': enrich,
                'status': 'queued',
                'created_time': datetime.now().isoformat(),
                'start_time': None,
                'end_time': None,
                'stats': None,
                'error': None
            }

        # Start background processing
        config_path = str(Path(__file__).parent.parent / 'config' / 'enrichment_config.yaml')

        thread = threading.Thread(
            target=process_import_background,
            args=(import_id, str(file_path), source, enrich, config_path),
            daemon=True
        )
        thread.start()

        return jsonify({
            'import_id': import_id,
            'status': 'queued',
            'message': 'Import queued for processing'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/imports')
def api_imports():
    """Get list of all imports"""
    with import_lock:
        imports_list = list(active_imports.values())

    # Sort by creation time (newest first)
    imports_list.sort(key=lambda x: x['created_time'], reverse=True)

    return jsonify({'imports': imports_list})


@app.route('/api/imports/<import_id>')
def api_import_status(import_id):
    """Get status of specific import"""
    with import_lock:
        import_data = active_imports.get(import_id)

    if not import_data:
        return jsonify({'error': 'Import not found'}), 404

    return jsonify(import_data)


@app.route('/api/conversations')
def api_conversations():
    """Get all conversations (for search UI)"""
    try:
        repo_path = get_repo_path()

        if not os.path.exists(repo_path):
            return jsonify({
                'conversations': [],
                'total': 0,
                'message': 'No conversations found. Upload some exports to get started!'
            })

        conversations = []
        with open(repo_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        conv = json.loads(line)
                        conversations.append(conv)
                    except (json.JSONDecodeError, ValueError):
                        continue

        # Apply filters if provided
        query = request.args.get('q', '').lower()
        source_filter = request.args.get('source', '')

        if query:
            conversations = [
                c for c in conversations
                if query in c.get('generated_title', '').lower()
                or query in c.get('raw_text', '').lower()
                or query in str(c.get('tags', [])).lower()
            ]

        if source_filter:
            conversations = [c for c in conversations if c.get('source', '') == source_filter]

        # Sort by timestamp (newest first)
        conversations.sort(
            key=lambda x: x.get('timestamp', x.get('create_time', '')),
            reverse=True
        )

        # Limit results (cap at 1000 to prevent OOM)
        limit = min(int(request.args.get('limit', 100)), 1000)
        conversations = conversations[:limit]

        return jsonify({
            'conversations': conversations,
            'total': len(conversations)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/conversation/<path:convo_id>')
def api_conversation(convo_id):
    """Get a single conversation by ID with related conversations and projects"""
    try:
        repo_path = get_repo_path()

        if not os.path.exists(repo_path):
            return jsonify({'error': 'No conversations found'}), 404

        # Load all conversations (needed for chain/project detection)
        all_conversations = []
        target_conversation = None

        with open(repo_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        conv = json.loads(line)
                        all_conversations.append(conv)
                        # Check both convo_id and external_id
                        if conv.get('convo_id') == convo_id or conv.get('external_id') == convo_id:
                            target_conversation = conv
                    except (json.JSONDecodeError, ValueError):
                        continue

        if not target_conversation:
            return jsonify({'error': 'Conversation not found'}), 404

        # Add related conversations (chains) on-demand
        include_chains = request.args.get('include_chains', 'true').lower() == 'true'
        if include_chains:
            try:
                from context.chain_detection import find_related
                related = find_related(target_conversation, all_conversations, max_results=5)
                target_conversation['_related_conversations'] = [
                    {
                        'convo_id': r[0].get('convo_id'),
                        'title': r[0].get('title', r[0].get('generated_title', 'Untitled')),
                        'score': r[1],
                        'relation_type': r[2],
                        'timestamp': r[0].get('timestamp', r[0].get('created_at', ''))
                    }
                    for r in related
                ]
            except Exception as e:
                print(f"  [!!] Chain detection error: {e}")
                target_conversation['_related_conversations'] = []

        # Add project membership on-demand
        include_projects = request.args.get('include_projects', 'true').lower() == 'true'
        if include_projects:
            try:
                from context.project_inference import ProjectInferrer
                inferrer = ProjectInferrer(min_conversations=2)
                signals = inferrer.extract_signals(target_conversation)

                # Return project signals for this conversation
                target_conversation['_project_signals'] = {
                    'project_roots': list(signals['project_roots']),
                    'repo_names': list(signals['repo_names']),
                    'git_repos': list(signals['git_repos']),
                    'technologies': list(signals['technologies']),
                    'file_extensions': dict(signals['file_extensions'].most_common(5))
                }
            except Exception as e:
                print(f"  [!!] Project inference error: {e}")
                target_conversation['_project_signals'] = {}

        return jsonify(target_conversation)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search')
def api_search():
    """Search conversations using hybrid search (BM25 + semantic)"""
    try:
        query = request.args.get('q', '')
        source = request.args.get('source', '')
        tag = request.args.get('tag', '')
        date_from = request.args.get('date_from', '')
        date_to = request.args.get('date_to', '')
        min_score = request.args.get('min_score', type=float)
        page = request.args.get('page', 1, type=int)
        limit = min(request.args.get('limit', 25, type=int), 1000)  # Cap at 1000

        # Search mode flags
        semantic_only = request.args.get('semantic_only', 'false').lower() == 'true'
        bm25_only = request.args.get('bm25_only', 'false').lower() == 'true'

        repo_path = get_repo_path()

        if not os.path.exists(repo_path):
            return jsonify({'results': [], 'total': 0, 'page': page, 'limit': limit})

        # If no query, return recent conversations (fallback to original behavior)
        if not query:
            conversations = []
            with open(repo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            conv = json.loads(line)

                            # Apply filters
                            if source and conv.get('source', '') != source:
                                continue
                            if tag:
                                conv_tags = [t.lower() for t in conv.get('tags', [])]
                                if tag.lower() not in conv_tags:
                                    continue
                            if date_from and conv.get('create_time', '') < date_from:
                                continue
                            if date_to and conv.get('create_time', '') > date_to:
                                continue
                            if min_score is not None:
                                conv_score = conv.get('score', 0)
                                if conv_score < min_score:
                                    continue

                            conversations.append(conv)
                        except (json.JSONDecodeError, ValueError):
                            continue

            # Sort by date (newest first)
            conversations.sort(
                key=lambda x: x.get('create_time', x.get('timestamp', '')),
                reverse=True
            )

            total = len(conversations)
            offset = (page - 1) * limit
            conversations = conversations[offset:offset + limit]

            return jsonify({
                'results': conversations,
                'total': total,
                'page': page,
                'limit': limit
            })

        # Use hybrid search for queries
        try:
            from search.hybrid_search import HybridSearcher
            from search.embeddings import EmbeddingStore

            # Initialize embedding store
            embeddings_path = str(Path(__file__).parent.parent / 'data' / 'embeddings.npy')
            embedding_store = None

            if os.path.exists(embeddings_path) and not bm25_only:
                try:
                    embedding_store = EmbeddingStore(embeddings_path)
                    print(f"  [OK] Loaded embeddings from {embeddings_path}")
                except Exception as e:
                    print(f"  [!!] Failed to load embeddings: {e}")
                    if semantic_only:
                        # Can't do semantic-only without embeddings
                        return jsonify({'error': 'Embeddings not available for semantic search'}), 400

            # Initialize hybrid searcher
            db_path = str(Path(__file__).parent.parent / 'data' / 'conversations.db')
            searcher = HybridSearcher(
                db_path=db_path,
                embedding_store=embedding_store,
                bm25_weight=0.5,
                semantic_weight=0.5,
                rrf_k=60
            )

            # Build index if needed (only on first run or if stale)
            if not os.path.exists(db_path):
                print(f"  [*] Building search index from {repo_path}...")
                from search_engine import SearchEngine

                # Use SearchEngine to build the database with FTS5 index
                engine = SearchEngine(db_path=db_path)

                # Load conversations from JSONL and index them
                conversations = []
                with open(repo_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                conversations.append(json.loads(line))
                            except (json.JSONDecodeError, ValueError):
                                continue

                count = engine.index_conversations(conversations)
                print(f"  [OK] Indexed {count} conversations")

            # Perform hybrid search
            search_results = searcher.search(
                query=query,
                limit=limit * 10,  # Get more results for filtering
                source_filter=source if source else None,
                tag_filter=tag if tag else None,
                min_score=min_score if min_score is not None else 0.0,
                semantic_only=semantic_only,
                bm25_only=bm25_only
            )

            # Convert SearchResult objects to dicts and apply remaining filters
            results = []
            for sr in search_results:
                # Load full conversation from repo
                with open(repo_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                conv = json.loads(line)
                                if conv.get('convo_id') == sr.convo_id:
                                    # Apply date filters
                                    conv_date = conv.get('create_time', '')
                                    if date_from and conv_date < date_from:
                                        break
                                    if date_to and conv_date > date_to:
                                        break

                                    # Add search metadata
                                    conv['_search'] = {
                                        'score': sr.score,
                                        'bm25_score': sr.bm25_score,
                                        'semantic_score': sr.semantic_score,
                                        'matched_terms': sr.matched_terms
                                    }
                                    results.append(conv)
                                    break
                            except (json.JSONDecodeError, ValueError):
                                continue

            total = len(results)

            # Paginate
            offset = (page - 1) * limit
            results = results[offset:offset + limit]

            return jsonify({
                'results': results,
                'total': total,
                'page': page,
                'limit': limit,
                'search_mode': 'semantic_only' if semantic_only else ('bm25_only' if bm25_only else 'hybrid')
            })

        except ImportError as e:
            print(f"  [!!] Hybrid search not available: {e}")
            print("  [*] Falling back to basic text search")
            # Fall back to original search implementation
            conversations = []
            with open(repo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            conv = json.loads(line)

                            # Simple text matching
                            query_lower = query.lower()
                            title = conv.get('generated_title', '').lower()
                            text = conv.get('raw_text', '').lower()
                            summary = conv.get('summary_abstractive', '').lower()
                            tags = ' '.join(conv.get('tags', [])).lower()

                            if not any(query_lower in field for field in [title, text, summary, tags]):
                                continue

                            # Apply filters
                            if source and conv.get('source', '') != source:
                                continue
                            if tag:
                                conv_tags = [t.lower() for t in conv.get('tags', [])]
                                if tag.lower() not in conv_tags:
                                    continue
                            if date_from and conv.get('create_time', '') < date_from:
                                continue
                            if date_to and conv.get('create_time', '') > date_to:
                                continue
                            if min_score is not None:
                                conv_score = conv.get('score', 0)
                                if conv_score < min_score:
                                    continue

                            conversations.append(conv)
                        except (json.JSONDecodeError, ValueError):
                            continue

            conversations.sort(
                key=lambda x: x.get('create_time', x.get('timestamp', '')),
                reverse=True
            )

            total = len(conversations)
            offset = (page - 1) * limit
            conversations = conversations[offset:offset + limit]

            return jsonify({
                'results': conversations,
                'total': total,
                'page': page,
                'limit': limit,
                'search_mode': 'fallback'
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/semantic_search')
def api_semantic_search():
    """Semantic search using embeddings"""
    # Redirect to main search with semantic_only flag
    from flask import request as flask_request
    args = dict(flask_request.args)
    args['semantic_only'] = 'true'
    flask_request.args = args
    return api_search()


@app.route('/api/stats')
def api_stats():
    """Get repository statistics"""
    try:
        repo_path = get_repo_path()

        if not os.path.exists(repo_path):
            return jsonify({
                'total_conversations': 0,
                'sources': {},
                'date_range': None,
                'avg_score': None
            })

        conversations = []
        sources = {}
        dates = []
        scores = []
        tags = {}

        with open(repo_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        conv = json.loads(line)
                        conversations.append(conv)

                        # Count sources
                        src = conv.get('source', 'Unknown')
                        sources[src] = sources.get(src, 0) + 1

                        # Collect dates
                        dt = conv.get('create_time', conv.get('timestamp'))
                        if dt:
                            dates.append(dt)

                        # Collect scores
                        score = conv.get('score', conv.get('relevance'))
                        if score is not None:
                            scores.append(score)

                        # Collect tags
                        for tag in conv.get('tags', []):
                            tags[tag] = tags.get(tag, 0) + 1
                    except (json.JSONDecodeError, ValueError):
                        continue

        # Calculate stats
        date_range = None
        if dates:
            dates.sort()
            date_range = {
                'earliest': dates[0],
                'latest': dates[-1]
            }

        avg_score = sum(scores) / len(scores) if scores else None

        # Top tags
        top_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:20]

        return jsonify({
            'total_conversations': len(conversations),
            'sources': sources,
            'date_range': date_range,
            'avg_score': avg_score,
            'top_tags': top_tags
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export', methods=['POST'])
def api_export():
    """Export selected conversations"""
    try:
        data = request.get_json() or {}
        conversation_ids = data.get('conversation_ids', [])
        format_type = data.get('format', 'json')

        if not conversation_ids:
            return jsonify({'error': 'No conversation IDs provided'}), 400

        repo_path = get_repo_path()

        if not os.path.exists(repo_path):
            return jsonify({'error': 'No conversations found'}), 404

        # Find requested conversations
        exported = []
        with open(repo_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        conv = json.loads(line)
                        cid = conv.get('convo_id') or conv.get('external_id')
                        if cid in conversation_ids:
                            exported.append(conv)
                    except (json.JSONDecodeError, ValueError):
                        continue

        return jsonify({
            'data': exported,
            'count': len(exported),
            'format': format_type
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tags')
def api_tags():
    """Get all tags with counts"""
    try:
        repo_path = get_repo_path()
        tags = {}

        if os.path.exists(repo_path):
            with open(repo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            conv = json.loads(line)
                            for tag in conv.get('tags', []):
                                tags[tag] = tags.get(tag, 0) + 1
                        except (json.JSONDecodeError, ValueError):
                            continue

        return jsonify({'tags': tags})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sources')
def api_sources():
    """Get all sources with counts"""
    try:
        repo_path = get_repo_path()
        sources = {}

        if os.path.exists(repo_path):
            with open(repo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            conv = json.loads(line)
                            src = conv.get('source', 'Unknown')
                            sources[src] = sources.get(src, 0) + 1
                        except (json.JSONDecodeError, ValueError):
                            continue

        return jsonify({'sources': sources})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects')
def api_projects():
    """Get all detected projects across conversations"""
    try:
        repo_path = get_repo_path()

        if not os.path.exists(repo_path):
            return jsonify({'projects': []})

        # Load all conversations
        conversations = []
        with open(repo_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        conversations.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        continue

        # Infer projects
        from context.project_inference import ProjectInferrer
        inferrer = ProjectInferrer(min_conversations=2, min_confidence=0.3)
        projects = inferrer.infer_projects(conversations)

        # Format for API response
        projects_data = [
            {
                'name': p.name,
                'conversation_count': len(p.conversation_ids),
                'technologies': list(p.technologies),
                'confidence': p.confidence,
                'keywords': list(p.keywords),
                # Don't include full conversation list to keep response small
                # Clients can request individual conversations if needed
            }
            for p in projects
        ]

        return jsonify({'projects': projects_data, 'total': len(projects_data)})

    except Exception as e:
        import traceback
        print(f"  [!!] Project API error: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/suggestions')
def api_suggestions():
    """Get search suggestions/autocomplete"""
    try:
        query = request.args.get('q', '').lower()
        limit = min(int(request.args.get('limit', 10)), 50)  # Cap suggestions

        if not query or len(query) < 2:
            return jsonify({'suggestions': []})

        repo_path = get_repo_path()
        suggestions = set()

        if os.path.exists(repo_path):
            with open(repo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(suggestions) >= limit:
                        break
                    line = line.strip()
                    if line:
                        try:
                            conv = json.loads(line)
                            title = conv.get('generated_title', conv.get('title', ''))
                            if title and query in title.lower():
                                suggestions.add(title[:100])
                        except (json.JSONDecodeError, ValueError):
                            continue

        return jsonify({'suggestions': list(suggestions)[:limit]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# WebSocket Events

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to CogRepo server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    print(f"Client disconnected: {request.sid}")


@socketio.on('subscribe_import')
def handle_subscribe(data):
    """Subscribe to updates for specific import"""
    import_id = data.get('import_id')
    print(f"Client {request.sid} subscribed to import {import_id}")


# Main

if __name__ == '__main__':
    print("=" * 60)
    print("  CogRepo Web Server")
    print("=" * 60)
    print(f"  📁 Data path: {get_repo_path()}")
    print(f"  📤 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"  🔑 API key configured: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
    port = int(os.getenv('PORT', 5001))
    print(f"  🌐 Starting server on http://localhost:{port}")
    print("=" * 60)
    print()

    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)
