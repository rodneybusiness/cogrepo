"""
CogRepo Error Handling System

Provides:
- Custom exception classes
- Flask error handlers
- Graceful degradation
- Error recovery utilities

Usage:
    from error_handlers import setup_error_handlers, CogRepoError

    # In app.py
    setup_error_handlers(app)

    # In routes
    if not data:
        raise NotFoundError("Conversation not found", convo_id=convo_id)
"""

from flask import jsonify, request, current_app
from functools import wraps
import traceback
import logging

logger = logging.getLogger('cogrepo.errors')


# =============================================================================
# Custom Exceptions
# =============================================================================

class CogRepoError(Exception):
    """Base exception for CogRepo errors."""

    status_code = 500
    error_type = 'internal_error'
    message = 'An unexpected error occurred'

    def __init__(self, message=None, **kwargs):
        super().__init__(message or self.message)
        self.message = message or self.message
        self.details = kwargs

    def to_dict(self):
        return {
            'error': self.error_type,
            'message': self.message,
            'details': self.details
        }


class NotFoundError(CogRepoError):
    """Resource not found."""
    status_code = 404
    error_type = 'not_found'
    message = 'Resource not found'


class ValidationError(CogRepoError):
    """Invalid input data."""
    status_code = 400
    error_type = 'validation_error'
    message = 'Invalid input'


class ConfigurationError(CogRepoError):
    """Configuration issue."""
    status_code = 500
    error_type = 'configuration_error'
    message = 'Server configuration error'


class DatabaseError(CogRepoError):
    """Database operation failed."""
    status_code = 500
    error_type = 'database_error'
    message = 'Database operation failed'


class SearchError(CogRepoError):
    """Search operation failed."""
    status_code = 500
    error_type = 'search_error'
    message = 'Search operation failed'


class EnrichmentError(CogRepoError):
    """Enrichment operation failed."""
    status_code = 500
    error_type = 'enrichment_error'
    message = 'Enrichment operation failed'


class RateLimitError(CogRepoError):
    """Rate limit exceeded."""
    status_code = 429
    error_type = 'rate_limit_exceeded'
    message = 'Rate limit exceeded. Please try again later.'


class ServiceUnavailableError(CogRepoError):
    """Service temporarily unavailable."""
    status_code = 503
    error_type = 'service_unavailable'
    message = 'Service temporarily unavailable'


# =============================================================================
# Error Handlers
# =============================================================================

def setup_error_handlers(app):
    """Register error handlers with Flask app."""

    @app.errorhandler(CogRepoError)
    def handle_cogrepo_error(error):
        """Handle custom CogRepo errors."""
        logger.warning(
            f'{error.error_type}: {error.message}',
            extra={
                'error_type': error.error_type,
                'details': error.details,
                'path': request.path
            }
        )

        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.errorhandler(400)
    def handle_bad_request(error):
        """Handle bad request errors."""
        return jsonify({
            'error': 'bad_request',
            'message': str(error.description) if hasattr(error, 'description') else 'Bad request'
        }), 400

    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors."""
        return jsonify({
            'error': 'not_found',
            'message': f'Resource not found: {request.path}'
        }), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        """Handle method not allowed errors."""
        return jsonify({
            'error': 'method_not_allowed',
            'message': f'Method {request.method} not allowed for {request.path}'
        }), 405

    @app.errorhandler(413)
    def handle_request_too_large(error):
        """Handle payload too large errors."""
        return jsonify({
            'error': 'payload_too_large',
            'message': 'Request payload is too large'
        }), 413

    @app.errorhandler(429)
    def handle_rate_limit(error):
        """Handle rate limit errors."""
        return jsonify({
            'error': 'rate_limit_exceeded',
            'message': 'Too many requests. Please try again later.'
        }), 429

    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle internal server errors."""
        logger.exception(
            f'Internal server error: {str(error)}',
            extra={'path': request.path}
        )

        # Don't expose internal error details in production
        if current_app.debug:
            message = str(error)
        else:
            message = 'An internal error occurred'

        return jsonify({
            'error': 'internal_error',
            'message': message
        }), 500

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """Handle any unexpected errors."""
        logger.exception(
            f'Unexpected error: {type(error).__name__}: {str(error)}',
            extra={
                'error_type': type(error).__name__,
                'path': request.path
            }
        )

        # Don't expose error details in production
        if current_app.debug:
            return jsonify({
                'error': 'unexpected_error',
                'message': str(error),
                'type': type(error).__name__,
                'traceback': traceback.format_exc()
            }), 500
        else:
            return jsonify({
                'error': 'unexpected_error',
                'message': 'An unexpected error occurred'
            }), 500


# =============================================================================
# Error Recovery Utilities
# =============================================================================

def safe_operation(default_value=None, log_errors=True):
    """
    Decorator for safe operation execution with fallback.

    Usage:
        @safe_operation(default_value=[])
        def get_conversations():
            return db.query(...)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f'Error in {func.__name__}: {str(e)}',
                        extra={'function': func.__name__, 'error': str(e)}
                    )
                return default_value
        return wrapper
    return decorator


def with_retry(max_retries=3, delay=1.0, backoff=2.0, exceptions=(Exception,)):
    """
    Decorator for retrying operations with exponential backoff.

    Usage:
        @with_retry(max_retries=3, delay=1.0)
        def api_call():
            return requests.get(...)
    """
    import time

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(
                        f'Retry {attempt + 1}/{max_retries} for {func.__name__}: {str(e)}',
                        extra={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'max_retries': max_retries
                        }
                    )

                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff

            # All retries exhausted
            logger.error(
                f'All retries exhausted for {func.__name__}',
                extra={'function': func.__name__, 'max_retries': max_retries}
            )
            raise last_exception

        return wrapper
    return decorator


def graceful_degradation(fallback_func):
    """
    Decorator for graceful degradation with fallback function.

    Usage:
        def fallback_search(query):
            return {'results': [], 'degraded': True}

        @graceful_degradation(fallback_search)
        def semantic_search(query):
            return embedding_engine.search(query)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f'Degraded mode for {func.__name__}: {str(e)}',
                    extra={'function': func.__name__, 'error': str(e)}
                )
                return fallback_func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker for preventing cascade failures.

    Usage:
        api_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        @api_breaker
        def call_external_api():
            return requests.get(...)
    """

    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time

            # Check if circuit is open
            if self.state == 'open':
                if self.last_failure_time:
                    elapsed = time.time() - self.last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self.state = 'half-open'
                        logger.info(f'Circuit breaker half-open for {func.__name__}')
                    else:
                        raise ServiceUnavailableError(
                            f'Service temporarily unavailable (circuit open)',
                            retry_after=int(self.recovery_timeout - elapsed)
                        )

            try:
                result = func(*args, **kwargs)

                # Reset on success
                if self.state == 'half-open':
                    self.state = 'closed'
                    self.failures = 0
                    logger.info(f'Circuit breaker closed for {func.__name__}')

                return result

            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()

                if self.failures >= self.failure_threshold:
                    self.state = 'open'
                    logger.error(
                        f'Circuit breaker opened for {func.__name__}',
                        extra={
                            'function': func.__name__,
                            'failures': self.failures,
                            'threshold': self.failure_threshold
                        }
                    )

                raise

        return wrapper

    def reset(self):
        """Manually reset the circuit breaker."""
        self.failures = 0
        self.last_failure_time = None
        self.state = 'closed'


# =============================================================================
# Request Validation
# =============================================================================

def validate_request_json(*required_fields):
    """
    Decorator to validate required JSON fields in request.

    Usage:
        @validate_request_json('query', 'limit')
        def search():
            data = request.get_json()
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = request.get_json(silent=True)

            if not data:
                raise ValidationError('Request body must be valid JSON')

            missing = [f for f in required_fields if f not in data]
            if missing:
                raise ValidationError(
                    f'Missing required fields: {", ".join(missing)}',
                    missing_fields=missing
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_query_params(**param_specs):
    """
    Decorator to validate query parameters.

    Usage:
        @validate_query_params(
            page={'type': int, 'default': 1, 'min': 1},
            limit={'type': int, 'default': 25, 'max': 100}
        )
        def search():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for param, spec in param_specs.items():
                value = request.args.get(param, spec.get('default'))

                if value is None and spec.get('required'):
                    raise ValidationError(f'Missing required parameter: {param}')

                if value is not None:
                    try:
                        # Type conversion
                        value = spec['type'](value)

                        # Range validation
                        if 'min' in spec and value < spec['min']:
                            raise ValidationError(
                                f'{param} must be at least {spec["min"]}',
                                param=param,
                                min_value=spec['min']
                            )
                        if 'max' in spec and value > spec['max']:
                            raise ValidationError(
                                f'{param} must be at most {spec["max"]}',
                                param=param,
                                max_value=spec['max']
                            )

                    except (ValueError, TypeError):
                        raise ValidationError(
                            f'{param} must be of type {spec["type"].__name__}',
                            param=param,
                            expected_type=spec['type'].__name__
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator
