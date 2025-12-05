"""
Logging Configuration for CogRepo

Provides structured logging with:
- Console and file output
- Log rotation
- JSON formatting option for production
- Context-aware logging
- Performance metrics

Usage:
    from core.logging_config import get_logger, setup_logging

    # Initialize at app start
    setup_logging(level="INFO", log_file="logs/cogrepo.log")

    # Get logger in each module
    logger = get_logger(__name__)
    logger.info("Processing started", extra={"conversation_count": 100})
"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps
import time


# =============================================================================
# Custom Formatters
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Colored console output for development.

    Colors:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red background
    """

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m',   # Red background
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"

        # Format timestamp
        record.asctime = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for production/structured logging.

    Output format:
    {
        "timestamp": "2025-11-29T10:30:00.000Z",
        "level": "INFO",
        "logger": "cogrepo.enrichment",
        "message": "Enrichment complete",
        "context": {...}
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra context if present
        if hasattr(record, '__dict__'):
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in (
                    'name', 'msg', 'args', 'created', 'filename', 'funcName',
                    'levelname', 'levelno', 'lineno', 'module', 'msecs',
                    'pathname', 'process', 'processName', 'relativeCreated',
                    'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                    'message', 'asctime'
                )
            }
            if extra_fields:
                log_data["context"] = extra_fields

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class ContextualFormatter(logging.Formatter):
    """
    Formatter that includes context from extra fields.

    Output: "2025-11-29 10:30:00 | INFO | module | message | {context}"
    """

    def format(self, record: logging.LogRecord) -> str:
        # Get extra context
        extra = {}
        for key, value in record.__dict__.items():
            if key not in (
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'asctime'
            ):
                extra[key] = value

        # Build message
        base = super().format(record)

        if extra:
            context_str = " | " + " ".join(f"{k}={v}" for k, v in extra.items())
            return base + context_str

        return base


# =============================================================================
# Logger Setup
# =============================================================================

_initialized = False
_default_level = logging.INFO


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> None:
    """
    Configure logging for the application.

    Should be called once at application startup.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        json_format: Use JSON formatting (for production)
        console: Enable console output
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    global _initialized, _default_level

    # Convert level string to constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    _default_level = numeric_level

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        if json_format:
            console_handler.setFormatter(JSONFormatter())
        else:
            # Use colored formatter for terminal
            if sys.stdout.isatty():
                formatter = ColoredFormatter(
                    '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
                )
            else:
                formatter = ContextualFormatter(
                    '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            console_handler.setFormatter(formatter)

        root_logger.addHandler(console_handler)

    # File handler (with rotation)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)

        # Always use structured format for files
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(ContextualFormatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))

        root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Starting process", extra={"count": 10})
    """
    global _initialized

    if not _initialized:
        # Auto-initialize with defaults
        setup_logging()

    return logging.getLogger(name)


# =============================================================================
# Logging Utilities
# =============================================================================

class LogContext:
    """
    Context manager for adding context to log messages.

    Usage:
        with LogContext(logger, operation="import", source="chatgpt"):
            logger.info("Starting")  # Includes operation and source
            do_work()
            logger.info("Complete")
    """

    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self._original_extra = {}

    def __enter__(self):
        # Store original and update
        for key, value in self.context.items():
            self._original_extra[key] = getattr(self.logger, key, None)
            setattr(self.logger, key, value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original
        for key, original in self._original_extra.items():
            if original is None:
                delattr(self.logger, key)
            else:
                setattr(self.logger, key, original)


def log_execution_time(logger: logging.Logger = None, level: int = logging.DEBUG):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger to use (default: module logger)
        level: Log level for timing messages

    Usage:
        @log_execution_time()
        def slow_function():
            ...
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time

                logger.log(
                    level,
                    f"{func.__name__} completed",
                    extra={"duration_ms": round(elapsed * 1000, 2)}
                )

                return result

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(
                    f"{func.__name__} failed after {elapsed:.2f}s: {e}",
                    extra={"duration_ms": round(elapsed * 1000, 2)}
                )
                raise

        return wrapper
    return decorator


class ProgressLogger:
    """
    Logger for tracking progress of long-running operations.

    Usage:
        progress = ProgressLogger(logger, total=1000, operation="importing")
        for item in items:
            process(item)
            progress.update()
        progress.complete()
    """

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        operation: str = "processing",
        log_every: int = 100
    ):
        self.logger = logger
        self.total = total
        self.operation = operation
        self.log_every = log_every
        self.current = 0
        self.start_time = time.perf_counter()
        self.errors = 0

    def update(self, count: int = 1, error: bool = False):
        """Update progress."""
        self.current += count
        if error:
            self.errors += count

        # Log at intervals
        if self.current % self.log_every == 0 or self.current == self.total:
            elapsed = time.perf_counter() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.current) / rate if rate > 0 else 0

            self.logger.info(
                f"{self.operation}: {self.current}/{self.total}",
                extra={
                    "progress_pct": round(100 * self.current / self.total, 1),
                    "rate_per_sec": round(rate, 2),
                    "remaining_sec": round(remaining, 1),
                    "errors": self.errors
                }
            )

    def complete(self):
        """Log completion."""
        elapsed = time.perf_counter() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0

        self.logger.info(
            f"{self.operation} complete",
            extra={
                "total_processed": self.current,
                "duration_sec": round(elapsed, 2),
                "rate_per_sec": round(rate, 2),
                "errors": self.errors,
                "success_rate": round(100 * (self.current - self.errors) / max(self.current, 1), 1)
            }
        )


# =============================================================================
# Specialized Loggers
# =============================================================================

class APICallLogger:
    """
    Specialized logger for tracking API calls.

    Tracks:
    - Request/response times
    - Token usage
    - Costs
    - Error rates
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.errors = 0

    def log_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        cost: float = 0.0,
        success: bool = True
    ):
        """Log an API call."""
        self.total_calls += 1
        self.total_tokens += input_tokens + output_tokens
        self.total_cost += cost

        if not success:
            self.errors += 1

        self.logger.debug(
            f"API call to {model}",
            extra={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "duration_ms": round(duration_ms, 2),
                "cost_usd": round(cost, 4),
                "success": success
            }
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get cumulative statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 2),
            "errors": self.errors,
            "success_rate": round(100 * (self.total_calls - self.errors) / max(self.total_calls, 1), 1)
        }

    def log_summary(self):
        """Log summary of all API calls."""
        stats = self.get_stats()
        self.logger.info(
            "API usage summary",
            extra=stats
        )
