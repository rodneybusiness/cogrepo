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
    """Get a single conversation by ID"""
    try:
        repo_path = get_repo_path()

        if not os.path.exists(repo_path):
            return jsonify({'error': 'No conversations found'}), 404

        with open(repo_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        conv = json.loads(line)
                        # Check both convo_id and external_id
                        if conv.get('convo_id') == convo_id or conv.get('external_id') == convo_id:
                            return jsonify(conv)
                    except (json.JSONDecodeError, ValueError):
                        continue

        return jsonify({'error': 'Conversation not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search')
def api_search():
    """Search conversations"""
    try:
        query = request.args.get('q', '')
        source = request.args.get('source', '')
        tag = request.args.get('tag', '')
        date_from = request.args.get('date_from', '')
        date_to = request.args.get('date_to', '')
        min_score = request.args.get('min_score', type=float)
        page = request.args.get('page', 1, type=int)
        limit = min(request.args.get('limit', 25, type=int), 1000)  # Cap at 1000

        repo_path = get_repo_path()

        if not os.path.exists(repo_path):
            return jsonify({'results': [], 'total': 0, 'page': page, 'limit': limit})

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

        # Apply search
        results = []
        query_lower = query.lower() if query else ''

        for conv in conversations:
            # Text search
            if query_lower:
                title = conv.get('generated_title', conv.get('title', '')).lower()
                text = conv.get('raw_text', '').lower()
                summary = conv.get('summary_abstractive', '').lower()
                tags = ' '.join(conv.get('tags', [])).lower()

                if not any(query_lower in field for field in [title, text, summary, tags]):
                    continue

            # Source filter
            if source and conv.get('source', '') != source:
                continue

            # Tag filter (exact match, case-insensitive)
            if tag:
                conv_tags = [t.lower() for t in conv.get('tags', [])]
                if tag.lower() not in conv_tags:
                    continue

            # Date filters
            conv_date = conv.get('create_time', conv.get('timestamp', ''))
            if date_from and conv_date < date_from:
                continue
            if date_to and conv_date > date_to:
                continue

            # Score filter
            if min_score is not None:
                conv_score = conv.get('score', conv.get('relevance', 0))
                if conv_score < min_score:
                    continue

            results.append(conv)

        # Sort by date (newest first)
        results.sort(
            key=lambda x: x.get('create_time', x.get('timestamp', '')),
            reverse=True
        )

        total = len(results)

        # Paginate
        offset = (page - 1) * limit
        results = results[offset:offset + limit]

        return jsonify({
            'results': results,
            'total': total,
            'page': page,
            'limit': limit
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/semantic_search')
def api_semantic_search():
    """Semantic search - for now, falls back to regular search"""
    # Could be enhanced with vector embeddings later
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
