"""
CogRepo Structured Logging Configuration

Provides:
- JSON structured logging for production
- Colorized console output for development
- Request logging middleware
- Error tracking with context

Usage:
    from logging_config import setup_logging, get_logger

    # At app startup
    setup_logging(app, level='INFO', json_format=True)

    # In modules
    logger = get_logger(__name__)
    logger.info('Message', extra={'user_id': 123})
"""

import logging
import json
import sys
import traceback
from datetime import datetime
from functools import wraps
from flask import request, g
import time
import uuid


# =============================================================================
# Custom Formatters
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON structured log formatter for production."""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add extra fields
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
        if hasattr(record, 'status_code'):
            log_entry['status_code'] = record.status_code
        if hasattr(record, 'path'):
            log_entry['path'] = record.path
        if hasattr(record, 'method'):
            log_entry['method'] = record.method

        # Add any extra attributes from the record
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                          'message', 'request_id', 'user_id', 'duration_ms',
                          'status_code', 'path', 'method'):
                if not key.startswith('_'):
                    log_entry[key] = value

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colorized console formatter for development."""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')

        # Build message
        parts = [
            f'{color}[{timestamp}]{reset}',
            f'{color}{record.levelname:8}{reset}',
            f'{record.name}:',
            record.getMessage()
        ]

        # Add request ID if present
        if hasattr(record, 'request_id'):
            parts.insert(2, f'[{record.request_id[:8]}]')

        # Add duration if present
        if hasattr(record, 'duration_ms'):
            parts.append(f'({record.duration_ms}ms)')

        message = ' '.join(parts)

        # Add exception if present
        if record.exc_info:
            message += '\n' + ''.join(traceback.format_exception(*record.exc_info))

        return message


# =============================================================================
# Logger Setup
# =============================================================================

def setup_logging(app, level='INFO', json_format=None):
    """
    Configure logging for the Flask application.

    Args:
        app: Flask application instance
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (default: True in production)
    """
    import os

    # Determine format based on environment
    if json_format is None:
        json_format = os.getenv('FLASK_ENV', 'production') == 'production'

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Set formatter
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter())

    root_logger.addHandler(console_handler)

    # Configure Flask's logger
    app.logger.handlers = []
    app.logger.addHandler(console_handler)
    app.logger.setLevel(getattr(logging, level.upper()))

    # Log startup
    app.logger.info('Logging configured', extra={
        'format': 'json' if json_format else 'colored',
        'level': level
    })

    return root_logger


def get_logger(name):
    """Get a logger with the given name."""
    return logging.getLogger(name)


# =============================================================================
# Request Logging Middleware
# =============================================================================

def setup_request_logging(app):
    """
    Set up request logging middleware.

    Logs:
    - Request start with method, path, and request ID
    - Request end with status code and duration
    - Errors with full context
    """
    logger = get_logger('cogrepo.requests')

    @app.before_request
    def before_request():
        """Log request start and set up context."""
        g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        g.start_time = time.time()

        logger.info(
            f'{request.method} {request.path}',
            extra={
                'request_id': g.request_id,
                'method': request.method,
                'path': request.path,
                'remote_addr': request.remote_addr,
                'user_agent': request.user_agent.string[:100] if request.user_agent else None
            }
        )

    @app.after_request
    def after_request(response):
        """Log request completion."""
        duration_ms = int((time.time() - g.start_time) * 1000)

        # Determine log level based on status code
        if response.status_code >= 500:
            log_method = logger.error
        elif response.status_code >= 400:
            log_method = logger.warning
        else:
            log_method = logger.info

        log_method(
            f'{request.method} {request.path} -> {response.status_code}',
            extra={
                'request_id': g.request_id,
                'method': request.method,
                'path': request.path,
                'status_code': response.status_code,
                'duration_ms': duration_ms,
                'content_length': response.content_length
            }
        )

        # Add request ID to response headers
        response.headers['X-Request-ID'] = g.request_id

        return response

    @app.errorhandler(Exception)
    def handle_exception(error):
        """Log unhandled exceptions."""
        logger.exception(
            f'Unhandled exception: {str(error)}',
            extra={
                'request_id': getattr(g, 'request_id', 'unknown'),
                'method': request.method,
                'path': request.path,
                'error_type': type(error).__name__
            }
        )

        # Return JSON error response
        from flask import jsonify
        return jsonify({
            'error': 'Internal server error',
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 500


# =============================================================================
# Performance Logging Decorator
# =============================================================================

def log_performance(logger_name=None):
    """
    Decorator to log function performance.

    Usage:
        @log_performance('cogrepo.search')
        def search_conversations(query):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)

                logger.debug(
                    f'{func.__name__} completed',
                    extra={
                        'function': func.__name__,
                        'duration_ms': duration_ms,
                        'request_id': getattr(g, 'request_id', None)
                    }
                )

                return result

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)

                logger.error(
                    f'{func.__name__} failed: {str(e)}',
                    extra={
                        'function': func.__name__,
                        'duration_ms': duration_ms,
                        'error_type': type(e).__name__,
                        'request_id': getattr(g, 'request_id', None)
                    }
                )
                raise

        return wrapper
    return decorator


# =============================================================================
# Audit Logging
# =============================================================================

class AuditLogger:
    """
    Audit logger for tracking important operations.

    Usage:
        audit = AuditLogger()
        audit.log_import(user='admin', source='chatgpt', count=100)
    """

    def __init__(self):
        self.logger = get_logger('cogrepo.audit')

    def log_import(self, source, count, user=None, duration_seconds=None):
        """Log a data import operation."""
        self.logger.info(
            f'Data import completed',
            extra={
                'audit_type': 'import',
                'source': source,
                'conversation_count': count,
                'user': user,
                'duration_seconds': duration_seconds
            }
        )

    def log_search(self, query, results_count, mode='bm25', user=None):
        """Log a search operation."""
        self.logger.info(
            f'Search performed',
            extra={
                'audit_type': 'search',
                'query': query[:100] if query else None,
                'results_count': results_count,
                'search_mode': mode,
                'user': user
            }
        )

    def log_export(self, count, format_type, user=None):
        """Log a data export operation."""
        self.logger.info(
            f'Data export completed',
            extra={
                'audit_type': 'export',
                'conversation_count': count,
                'format': format_type,
                'user': user
            }
        )

    def log_backup(self, backup_path, size_mb, user=None):
        """Log a backup operation."""
        self.logger.info(
            f'Backup created',
            extra={
                'audit_type': 'backup',
                'backup_path': backup_path,
                'size_mb': size_mb,
                'user': user
            }
        )

    def log_restore(self, backup_path, user=None):
        """Log a restore operation."""
        self.logger.info(
            f'Restore completed',
            extra={
                'audit_type': 'restore',
                'backup_path': backup_path,
                'user': user
            }
        )
