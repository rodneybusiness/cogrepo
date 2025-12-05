"""
Custom Exception Hierarchy for CogRepo

Provides structured error handling with:
- Clear error categorization
- Rich error context
- Proper exception chaining
- User-friendly error messages

Usage:
    from core.exceptions import ParsingError, EnrichmentError

    try:
        conversations = parser.parse()
    except ParsingError as e:
        logger.error(f"Failed to parse: {e.message}", extra=e.context)
        # Handle gracefully
"""

from typing import Dict, Any, Optional
from datetime import datetime


class CogRepoException(Exception):
    """
    Base exception for all CogRepo errors.

    Provides:
    - Structured error message
    - Context dictionary for debugging
    - Timestamp for error tracking
    - Optional error code for programmatic handling
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize exception.

        Args:
            message: Human-readable error message
            context: Additional context for debugging
            error_code: Machine-readable error code (e.g., "PARSE_001")
            cause: Original exception that caused this error
        """
        self.message = message
        self.context = context or {}
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()
        self.cause = cause

        # Build full message
        full_message = message
        if error_code:
            full_message = f"[{error_code}] {message}"

        super().__init__(full_message)

        # Chain the cause if provided
        if cause:
            self.__cause__ = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "timestamp": self.timestamp,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, error_code={self.error_code!r})"


# =============================================================================
# Parsing Errors
# =============================================================================

class ParsingError(CogRepoException):
    """
    Raised when parsing conversation exports fails.

    Error Codes:
    - PARSE_001: Invalid JSON format
    - PARSE_002: Missing required fields
    - PARSE_003: Unsupported file format
    - PARSE_004: Encoding error
    - PARSE_005: Malformed conversation structure
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        source_platform: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            "file_path": file_path,
            "line_number": line_number,
            "source_platform": source_platform
        })
        super().__init__(message, context=context, **kwargs)
        self.file_path = file_path
        self.line_number = line_number
        self.source_platform = source_platform


class InvalidJSONError(ParsingError):
    """Raised when JSON parsing fails."""

    def __init__(self, message: str, file_path: str = None, **kwargs):
        super().__init__(message, file_path=file_path, error_code="PARSE_001", **kwargs)


class MissingFieldError(ParsingError):
    """Raised when required fields are missing."""

    def __init__(self, field_name: str, file_path: str = None, **kwargs):
        message = f"Missing required field: {field_name}"
        context = kwargs.pop('context', {})
        context['missing_field'] = field_name
        super().__init__(message, file_path=file_path, error_code="PARSE_002", context=context, **kwargs)


class UnsupportedFormatError(ParsingError):
    """Raised when file format is not supported."""

    def __init__(self, file_path: str, detected_format: str = None, **kwargs):
        message = f"Unsupported file format: {file_path}"
        context = kwargs.pop('context', {})
        context['detected_format'] = detected_format
        super().__init__(message, file_path=file_path, error_code="PARSE_003", context=context, **kwargs)


# =============================================================================
# Enrichment Errors
# =============================================================================

class EnrichmentError(CogRepoException):
    """
    Raised when AI enrichment fails.

    Error Codes:
    - ENRICH_001: API call failed
    - ENRICH_002: Response parsing failed
    - ENRICH_003: Rate limit exceeded
    - ENRICH_004: Invalid response format
    - ENRICH_005: Conversation too short for enrichment
    """

    def __init__(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        enrichment_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            "conversation_id": conversation_id,
            "enrichment_type": enrichment_type
        })
        super().__init__(message, context=context, **kwargs)
        self.conversation_id = conversation_id
        self.enrichment_type = enrichment_type


class APIError(EnrichmentError):
    """Raised when API call fails."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        api_error_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            "status_code": status_code,
            "api_error_type": api_error_type
        })
        super().__init__(message, error_code="ENRICH_001", context=context, **kwargs)
        self.status_code = status_code
        self.api_error_type = api_error_type


class RateLimitError(EnrichmentError):
    """Raised when API rate limit is exceeded."""

    def __init__(
        self,
        message: str = "API rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context['retry_after_seconds'] = retry_after
        super().__init__(message, error_code="ENRICH_003", context=context, **kwargs)
        self.retry_after = retry_after


class ResponseParsingError(EnrichmentError):
    """Raised when API response cannot be parsed."""

    def __init__(
        self,
        message: str,
        raw_response: Optional[str] = None,
        expected_format: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            "raw_response_preview": raw_response[:500] if raw_response else None,
            "expected_format": expected_format
        })
        super().__init__(message, error_code="ENRICH_004", context=context, **kwargs)


# =============================================================================
# Storage Errors
# =============================================================================

class StorageError(CogRepoException):
    """
    Raised when storage operations fail.

    Error Codes:
    - STORE_001: File write failed
    - STORE_002: File read failed
    - STORE_003: Database error
    - STORE_004: Index corruption
    - STORE_005: Insufficient disk space
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            "file_path": file_path,
            "operation": operation
        })
        super().__init__(message, context=context, **kwargs)
        self.file_path = file_path
        self.operation = operation


class FileWriteError(StorageError):
    """Raised when file write fails."""

    def __init__(self, file_path: str, **kwargs):
        message = f"Failed to write file: {file_path}"
        super().__init__(message, file_path=file_path, operation="write", error_code="STORE_001", **kwargs)


class FileReadError(StorageError):
    """Raised when file read fails."""

    def __init__(self, file_path: str, **kwargs):
        message = f"Failed to read file: {file_path}"
        super().__init__(message, file_path=file_path, operation="read", error_code="STORE_002", **kwargs)


class DatabaseError(StorageError):
    """Raised when database operations fail."""

    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        context = kwargs.pop('context', {})
        context['query'] = query
        super().__init__(message, error_code="STORE_003", context=context, **kwargs)


# =============================================================================
# Search Errors
# =============================================================================

class SearchError(CogRepoException):
    """
    Raised when search operations fail.

    Error Codes:
    - SEARCH_001: Invalid query syntax
    - SEARCH_002: Index not available
    - SEARCH_003: Search timeout
    - SEARCH_004: Too many results
    """

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context['query'] = query
        super().__init__(message, context=context, **kwargs)
        self.query = query


class InvalidQueryError(SearchError):
    """Raised when search query is invalid."""

    def __init__(self, query: str, reason: str = None, **kwargs):
        message = f"Invalid search query: {query}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, query=query, error_code="SEARCH_001", **kwargs)


class IndexNotAvailableError(SearchError):
    """Raised when search index is not available."""

    def __init__(self, index_type: str = "full-text", **kwargs):
        message = f"Search index not available: {index_type}"
        context = kwargs.pop('context', {})
        context['index_type'] = index_type
        super().__init__(message, error_code="SEARCH_002", context=context, **kwargs)


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(CogRepoException):
    """
    Raised when configuration is invalid.

    Error Codes:
    - CONFIG_001: Missing required setting
    - CONFIG_002: Invalid value
    - CONFIG_003: File not found
    - CONFIG_004: YAML parse error
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            "config_key": config_key,
            "config_file": config_file
        })
        super().__init__(message, context=context, **kwargs)
        self.config_key = config_key
        self.config_file = config_file


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str, **kwargs):
        message = f"Missing required configuration: {config_key}"
        super().__init__(message, config_key=config_key, error_code="CONFIG_001", **kwargs)


class InvalidConfigValueError(ConfigurationError):
    """Raised when configuration value is invalid."""

    def __init__(self, config_key: str, value: Any, expected: str, **kwargs):
        message = f"Invalid value for {config_key}: {value} (expected {expected})"
        context = kwargs.pop('context', {})
        context.update({
            "invalid_value": str(value),
            "expected": expected
        })
        super().__init__(message, config_key=config_key, error_code="CONFIG_002", context=context, **kwargs)


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(CogRepoException):
    """
    Raised when data validation fails.

    Error Codes:
    - VALID_001: Required field missing
    - VALID_002: Type mismatch
    - VALID_003: Value out of range
    - VALID_004: Format invalid
    """

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Any = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        context.update({
            "field_name": field_name,
            "invalid_value": str(invalid_value) if invalid_value is not None else None
        })
        super().__init__(message, context=context, **kwargs)
        self.field_name = field_name
        self.invalid_value = invalid_value


# =============================================================================
# Utility Functions
# =============================================================================

def handle_exception(exc: Exception, logger=None, reraise: bool = True) -> Optional[CogRepoException]:
    """
    Handle an exception, converting to CogRepoException if needed.

    Args:
        exc: The exception to handle
        logger: Optional logger for recording the error
        reraise: Whether to re-raise the exception

    Returns:
        CogRepoException if converted, None if not reraising
    """
    if isinstance(exc, CogRepoException):
        cogrepo_exc = exc
    else:
        # Wrap in generic CogRepoException
        cogrepo_exc = CogRepoException(
            message=str(exc),
            cause=exc,
            context={"original_type": type(exc).__name__}
        )

    if logger:
        logger.error(
            f"{cogrepo_exc.error_code or 'ERROR'}: {cogrepo_exc.message}",
            extra=cogrepo_exc.context
        )

    if reraise:
        raise cogrepo_exc from exc

    return cogrepo_exc
