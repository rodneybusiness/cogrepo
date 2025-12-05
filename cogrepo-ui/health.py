"""
CogRepo Health Check System

Production-ready health and readiness endpoints with:
- Database connectivity checks
- Dependency availability
- Resource usage monitoring
- Detailed diagnostics

Endpoints:
- /health - Kubernetes liveness probe (is the process alive?)
- /ready - Kubernetes readiness probe (can it serve traffic?)
- /health/detailed - Full diagnostic report
"""

from flask import Blueprint, jsonify
from pathlib import Path
import os
import sys
import time
import psutil

health_bp = Blueprint('health', __name__)

# Track startup time
STARTUP_TIME = time.time()


def check_database():
    """Check if SQLite database is accessible."""
    try:
        db_path = Path(__file__).parent.parent / 'data' / 'cogrepo.db'
        if not db_path.exists():
            return {'status': 'warning', 'message': 'Database not found (using JSONL)'}

        import sqlite3
        conn = sqlite3.connect(str(db_path), timeout=5)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM conversations')
        count = cursor.fetchone()[0]
        conn.close()

        return {'status': 'ok', 'conversations': count}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def check_data_files():
    """Check if data files are accessible."""
    try:
        data_dir = Path(__file__).parent.parent / 'data'
        jsonl_path = data_dir / 'enriched_repository.jsonl'

        if not jsonl_path.exists():
            return {'status': 'warning', 'message': 'No data file found'}

        # Check file is readable and get count
        count = 0
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1

        return {'status': 'ok', 'conversations': count, 'path': str(jsonl_path)}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def check_embeddings():
    """Check if embedding files exist."""
    try:
        data_dir = Path(__file__).parent.parent / 'data'
        embeddings_path = data_dir / 'embeddings.npy'
        ids_path = data_dir / 'embedding_ids.json'

        if embeddings_path.exists() and ids_path.exists():
            import numpy as np
            embeddings = np.load(str(embeddings_path))
            return {
                'status': 'ok',
                'count': embeddings.shape[0],
                'dimension': embeddings.shape[1]
            }
        else:
            return {'status': 'unavailable', 'message': 'Embeddings not generated'}
    except ImportError:
        return {'status': 'unavailable', 'message': 'numpy not installed'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def check_api_key():
    """Check if API key is configured."""
    try:
        # Check environment variable first
        if os.getenv('ANTHROPIC_API_KEY'):
            return {'status': 'ok', 'source': 'environment'}

        # Check config file
        config_path = Path.home() / '.cogrepo' / 'config.yaml'
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if config and config.get('anthropic', {}).get('api_key'):
                    return {'status': 'ok', 'source': 'config_file'}

        return {'status': 'not_configured', 'message': 'API key not set'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def check_dependencies():
    """Check if required dependencies are available."""
    deps = {}

    # Core dependencies
    for module in ['flask', 'flask_socketio', 'flask_cors']:
        try:
            __import__(module)
            deps[module] = {'status': 'ok'}
        except ImportError:
            deps[module] = {'status': 'missing'}

    # Optional dependencies
    optional_deps = ['sentence_transformers', 'anthropic', 'numpy', 'pydantic']
    for module in optional_deps:
        try:
            __import__(module)
            deps[module] = {'status': 'ok', 'optional': True}
        except ImportError:
            deps[module] = {'status': 'not_installed', 'optional': True}

    return deps


def get_system_resources():
    """Get current system resource usage."""
    try:
        process = psutil.Process()

        return {
            'memory': {
                'rss_mb': round(process.memory_info().rss / 1024 / 1024, 2),
                'percent': round(process.memory_percent(), 2)
            },
            'cpu': {
                'percent': round(process.cpu_percent(interval=0.1), 2),
                'num_threads': process.num_threads()
            },
            'system': {
                'memory_available_mb': round(psutil.virtual_memory().available / 1024 / 1024, 2),
                'disk_free_gb': round(psutil.disk_usage('/').free / 1024 / 1024 / 1024, 2)
            }
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


# =============================================================================
# Health Endpoints
# =============================================================================

@health_bp.route('/health')
def liveness():
    """
    Kubernetes liveness probe.

    Returns 200 if the process is alive and can respond.
    Used to determine if the container needs to be restarted.
    """
    return jsonify({
        'status': 'ok',
        'uptime_seconds': int(time.time() - STARTUP_TIME)
    })


@health_bp.route('/ready')
def readiness():
    """
    Kubernetes readiness probe.

    Returns 200 if the service can accept traffic.
    Checks critical dependencies before accepting requests.
    """
    # Check database/data access
    data_check = check_data_files()
    db_check = check_database()

    # Service is ready if we have either database or data files
    is_ready = (
        data_check.get('status') == 'ok' or
        db_check.get('status') == 'ok'
    )

    response = {
        'status': 'ready' if is_ready else 'not_ready',
        'checks': {
            'data_files': data_check.get('status'),
            'database': db_check.get('status')
        }
    }

    status_code = 200 if is_ready else 503
    return jsonify(response), status_code


@health_bp.route('/health/detailed')
def detailed_health():
    """
    Detailed health check for diagnostics.

    Returns comprehensive status of all components.
    Useful for debugging and monitoring dashboards.
    """
    return jsonify({
        'status': 'ok',
        'version': '2.0.0',
        'uptime_seconds': int(time.time() - STARTUP_TIME),
        'python_version': sys.version,
        'checks': {
            'database': check_database(),
            'data_files': check_data_files(),
            'embeddings': check_embeddings(),
            'api_key': check_api_key(),
            'dependencies': check_dependencies()
        },
        'resources': get_system_resources()
    })


@health_bp.route('/health/live')
def live():
    """Alias for liveness probe."""
    return liveness()


@health_bp.route('/metrics')
def metrics():
    """
    Basic metrics endpoint for monitoring.

    Returns key metrics in a format suitable for monitoring systems.
    """
    data_check = check_data_files()
    db_check = check_database()
    resources = get_system_resources()

    return jsonify({
        'cogrepo_uptime_seconds': int(time.time() - STARTUP_TIME),
        'cogrepo_conversations_total': data_check.get('conversations', 0),
        'cogrepo_database_conversations': db_check.get('conversations', 0),
        'process_memory_mb': resources.get('memory', {}).get('rss_mb', 0),
        'process_cpu_percent': resources.get('cpu', {}).get('percent', 0),
        'system_memory_available_mb': resources.get('system', {}).get('memory_available_mb', 0)
    })
