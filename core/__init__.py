"""
CogRepo Core Module

Foundation components shared across the application:
- Custom exceptions for proper error handling
- Logging configuration
- Configuration management with validation
- Validated data models (Pydantic)
- Type definitions
"""

from .exceptions import (
    CogRepoException,
    ParsingError,
    EnrichmentError,
    StorageError,
    SearchError,
    ConfigurationError,
    ValidationError,
    APIError,
    RateLimitError,
)

from .logging_config import get_logger, setup_logging

from .config import (
    Config,
    PathConfig,
    AnthropicConfig,
    EnrichmentConfig,
    SearchConfig,
    WebUIConfig,
    LoggingConfig,
    ArchiveConfig,
    get_config,
    init_config,
    reset_config,
    validate_environment,
)

from .validated_models import (
    MessageRole,
    ConversationSource,
    ConversationStatus,
    PrimaryDomain,
    Message,
    RawConversation,
    BrillianceScore,
    FuturePotential,
    EnrichmentMetadata,
    EnrichedConversation,
    ProcessingStats,
    ArchiveInfo,
    SearchResult,
    SearchResponse,
    SyncResult,
    validate_conversation,
    validate_message,
    safe_parse_conversation,
)

__all__ = [
    # Exceptions
    'CogRepoException',
    'ParsingError',
    'EnrichmentError',
    'StorageError',
    'SearchError',
    'ConfigurationError',
    'ValidationError',
    'APIError',
    'RateLimitError',
    # Logging
    'get_logger',
    'setup_logging',
    # Configuration
    'Config',
    'PathConfig',
    'AnthropicConfig',
    'EnrichmentConfig',
    'SearchConfig',
    'WebUIConfig',
    'LoggingConfig',
    'ArchiveConfig',
    'get_config',
    'init_config',
    'reset_config',
    'validate_environment',
    # Validated Models
    'MessageRole',
    'ConversationSource',
    'ConversationStatus',
    'PrimaryDomain',
    'Message',
    'RawConversation',
    'BrillianceScore',
    'FuturePotential',
    'EnrichmentMetadata',
    'EnrichedConversation',
    'ProcessingStats',
    'ArchiveInfo',
    'SearchResult',
    'SearchResponse',
    'SyncResult',
    'validate_conversation',
    'validate_message',
    'safe_parse_conversation',
]
