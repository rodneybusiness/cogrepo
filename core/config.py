"""
Unified Configuration System for CogRepo

Provides centralized, validated configuration with:
- Environment variable support
- YAML file configuration
- Pydantic validation
- Type-safe access
- Sensible defaults

Usage:
    from core.config import get_config, Config

    config = get_config()
    print(config.anthropic.api_key)
    print(config.paths.repository)
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


# =============================================================================
# Path Configuration
# =============================================================================

class PathConfig(BaseModel):
    """File path configuration."""

    # Base directory (usually the cogrepo project root)
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Base directory for CogRepo"
    )

    # Data paths
    data_dir: Optional[Path] = Field(
        default=None,
        description="Data directory (defaults to base_dir/data)"
    )
    repository_file: str = Field(
        default="enriched_repository.jsonl",
        description="Name of the enriched repository file"
    )

    # Database paths
    search_db: str = Field(
        default="cogrepo_search.db",
        description="Search database filename"
    )
    archive_registry_file: str = Field(
        default="archive_registry.json",
        description="Archive registry filename"
    )

    # Upload directory
    upload_dir: Optional[Path] = Field(
        default=None,
        description="Upload directory for web UI"
    )

    # Log directory
    log_dir: Optional[Path] = Field(
        default=None,
        description="Directory for log files"
    )

    @model_validator(mode='after')
    def set_defaults(self):
        """Set default paths based on base_dir."""
        if self.data_dir is None:
            object.__setattr__(self, 'data_dir', self.base_dir / 'data')
        if self.upload_dir is None:
            object.__setattr__(self, 'upload_dir', self.base_dir / 'uploads')
        if self.log_dir is None:
            object.__setattr__(self, 'log_dir', self.base_dir / 'logs')
        return self

    @property
    def repository(self) -> Path:
        """Full path to the enriched repository."""
        return self.data_dir / self.repository_file

    @property
    def search_database(self) -> Path:
        """Full path to the search database."""
        return self.data_dir / self.search_db

    @property
    def archive_registry(self) -> Path:
        """Full path to the archive registry."""
        return self.data_dir / self.archive_registry_file

    def ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# API Configuration
# =============================================================================

class AnthropicConfig(BaseModel):
    """Anthropic API configuration."""

    api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default model for enrichment"
    )
    haiku_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Fast/cheap model for simple tasks"
    )
    max_tokens: int = Field(
        default=4096,
        ge=100,
        le=200000,
        description="Maximum tokens for API responses"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for API calls"
    )

    # Rate limiting
    requests_per_minute: int = Field(
        default=50,
        ge=1,
        description="Rate limit for API requests"
    )

    # Retry configuration
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts"
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        description="Initial delay between retries (seconds)"
    )
    retry_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        description="Multiplier for exponential backoff"
    )

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    @model_validator(mode='after')
    def load_api_key_from_sources(self):
        """
        Load API key from resolution chain:
        1. Environment variable (highest priority)
        2. ~/.cogrepo/config.yaml (user config)
        3. ./.cogrepo/config.yaml (project config)
        """
        if self.api_key is not None:
            return self

        # 1. Environment variable
        env_key = os.getenv('ANTHROPIC_API_KEY')
        if env_key:
            object.__setattr__(self, 'api_key', env_key)
            return self

        # 2. User config (~/.cogrepo/config.yaml)
        user_config = Path.home() / '.cogrepo' / 'config.yaml'
        if user_config.exists():
            try:
                import yaml
                with open(user_config) as f:
                    config_data = yaml.safe_load(f)
                    if config_data and 'anthropic' in config_data:
                        key = config_data['anthropic'].get('api_key')
                        if key:
                            object.__setattr__(self, 'api_key', key)
                            return self
            except Exception:
                pass

        # 3. Project config (./.cogrepo/config.yaml)
        project_config = Path('.cogrepo/config.yaml')
        if project_config.exists():
            try:
                import yaml
                with open(project_config) as f:
                    config_data = yaml.safe_load(f)
                    if config_data and 'anthropic' in config_data:
                        key = config_data['anthropic'].get('api_key')
                        if key:
                            object.__setattr__(self, 'api_key', key)
                            return self
            except Exception:
                pass

        return self


# =============================================================================
# Enrichment Configuration
# =============================================================================

class EnrichmentConfig(BaseModel):
    """Enrichment pipeline configuration."""

    # Processing options
    batch_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of conversations per batch"
    )
    max_text_length: int = Field(
        default=50000,
        ge=1000,
        description="Maximum text length for enrichment"
    )
    skip_short_conversations: bool = Field(
        default=True,
        description="Skip conversations with fewer than min_messages"
    )
    min_messages: int = Field(
        default=2,
        ge=1,
        description="Minimum messages for a valid conversation"
    )

    # Output options
    include_raw_text: bool = Field(
        default=True,
        description="Include raw text in enriched output"
    )

    # Quality thresholds
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )

    # Prompts configuration
    use_structured_prompts: bool = Field(
        default=True,
        description="Use JSON-structured prompts"
    )


# =============================================================================
# Search Configuration
# =============================================================================

class SearchConfig(BaseModel):
    """Search engine configuration."""

    # FTS configuration
    use_fts5: bool = Field(
        default=True,
        description="Use SQLite FTS5 for search"
    )
    fts_tokenizer: str = Field(
        default="porter unicode61",
        description="FTS5 tokenizer configuration"
    )

    # Search options
    default_limit: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Default result limit"
    )
    max_limit: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum result limit"
    )

    # BM25 tuning
    bm25_k1: float = Field(
        default=1.2,
        ge=0.0,
        description="BM25 k1 parameter"
    )
    bm25_b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 b parameter"
    )

    # Fuzzy search
    enable_fuzzy: bool = Field(
        default=True,
        description="Enable fuzzy search"
    )
    fuzzy_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Fuzzy match threshold"
    )


# =============================================================================
# Web UI Configuration
# =============================================================================

class WebUIConfig(BaseModel):
    """Web UI configuration."""

    host: str = Field(
        default="127.0.0.1",
        description="Host to bind to"
    )
    port: int = Field(
        default=5000,
        ge=1,
        le=65535,
        description="Port to listen on"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    # Security
    secret_key: Optional[str] = Field(
        default=None,
        description="Flask secret key"
    )

    # Upload limits
    max_upload_size_mb: int = Field(
        default=100,
        ge=1,
        description="Maximum upload size in MB"
    )
    allowed_extensions: List[str] = Field(
        default_factory=lambda: ['json', 'jsonl'],
        description="Allowed file extensions"
    )

    @model_validator(mode='after')
    def generate_secret_key(self):
        """Generate secret key if not provided."""
        if self.secret_key is None:
            import secrets
            object.__setattr__(self, 'secret_key', secrets.token_hex(32))
        return self


# =============================================================================
# Logging Configuration
# =============================================================================

class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(
        default="INFO",
        description="Log level"
    )
    format: str = Field(
        default="colored",
        description="Log format: colored, json, plain"
    )

    # File logging
    enable_file_logging: bool = Field(
        default=True,
        description="Enable file logging"
    )
    log_file: str = Field(
        default="cogrepo.log",
        description="Log file name"
    )
    max_file_size_mb: int = Field(
        default=10,
        ge=1,
        description="Max log file size before rotation"
    )
    backup_count: int = Field(
        default=5,
        ge=0,
        description="Number of backup log files"
    )

    # Console logging
    enable_console_logging: bool = Field(
        default=True,
        description="Enable console logging"
    )

    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


# =============================================================================
# Archive Configuration
# =============================================================================

class ArchiveConfig(BaseModel):
    """Archive management configuration."""

    # Auto-sync settings
    auto_sync_enabled: bool = Field(
        default=True,
        description="Enable automatic syncing"
    )
    sync_interval_minutes: int = Field(
        default=60,
        ge=1,
        description="Auto-sync interval"
    )

    # Processing settings
    incremental_only: bool = Field(
        default=True,
        description="Only process new conversations"
    )
    track_file_changes: bool = Field(
        default=True,
        description="Track file hash/size changes"
    )

    # Default archive locations (can be overridden per archive)
    default_chatgpt_path: Optional[str] = Field(
        default=None,
        description="Default ChatGPT export path"
    )
    default_claude_path: Optional[str] = Field(
        default=None,
        description="Default Claude export path"
    )
    default_gemini_path: Optional[str] = Field(
        default=None,
        description="Default Gemini export path"
    )


# =============================================================================
# Main Configuration Class
# =============================================================================

class Config(BaseSettings):
    """
    Main configuration class for CogRepo.

    Loads configuration from:
    1. Environment variables (highest priority)
    2. .env file
    3. Default values (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_prefix='COGREPO_',
        env_file='.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__',
        extra='ignore'
    )

    # Sub-configurations
    paths: PathConfig = Field(default_factory=PathConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    web_ui: WebUIConfig = Field(default_factory=WebUIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    archive: ArchiveConfig = Field(default_factory=ArchiveConfig)

    # Application metadata
    app_name: str = Field(default="CogRepo")
    version: str = Field(default="2.0.0")
    debug: bool = Field(default=False)

    def __init__(self, **data):
        """Initialize config, loading from .env file."""
        # Load .env file if it exists
        env_file = Path(__file__).parent.parent / '.env'
        if env_file.exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)

        super().__init__(**data)

    @property
    def has_api_key(self) -> bool:
        """Check if Anthropic API key is configured."""
        return self.anthropic.is_configured

    @classmethod
    def from_env(cls, base_dir: Optional[Path] = None) -> 'Config':
        """Create config from environment variables."""
        config_data = {}

        if base_dir:
            config_data['paths'] = PathConfig(base_dir=base_dir)

        # Load API key from environment
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            config_data['anthropic'] = AnthropicConfig(api_key=api_key)

        # Load debug from environment
        if os.getenv('COGREPO_DEBUG', '').lower() in ('true', '1', 'yes'):
            config_data['debug'] = True

        return cls(**config_data)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Config':
        """Load config from YAML file."""
        import yaml

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_dict(self, exclude_secrets: bool = True) -> Dict[str, Any]:
        """Convert to dictionary, optionally excluding secrets."""
        data = self.model_dump()

        if exclude_secrets:
            # Mask sensitive values
            if data.get('anthropic', {}).get('api_key'):
                key = data['anthropic']['api_key']
                data['anthropic']['api_key'] = f"{key[:10]}...{key[-4:]}" if len(key) > 14 else "***"
            if data.get('web_ui', {}).get('secret_key'):
                data['web_ui']['secret_key'] = "***"

        return data

    def validate_for_enrichment(self) -> List[str]:
        """Validate config is ready for enrichment. Returns list of issues."""
        issues = []

        if not self.anthropic.is_configured:
            issues.append("ANTHROPIC_API_KEY not configured")

        if not self.paths.repository.parent.exists():
            issues.append(f"Data directory does not exist: {self.paths.data_dir}")

        return issues

    def validate_for_web(self) -> List[str]:
        """Validate config is ready for web UI. Returns list of issues."""
        issues = []

        if not self.paths.repository.exists():
            issues.append(f"Repository file not found: {self.paths.repository}")

        return issues

    def print_summary(self) -> None:
        """Print configuration summary."""
        print(f"\n{'=' * 60}")
        print(f"  {self.app_name} Configuration v{self.version}")
        print(f"{'=' * 60}")
        print(f"  Debug Mode:     {self.debug}")
        print(f"  Base Directory: {self.paths.base_dir}")
        print(f"  Data Directory: {self.paths.data_dir}")
        print(f"  Repository:     {self.paths.repository}")
        print(f"  Search DB:      {self.paths.search_database}")
        print(f"  API Key:        {'✓ Configured' if self.anthropic.is_configured else '✗ Not configured'}")
        print(f"  Model:          {self.anthropic.model}")
        print(f"  Log Level:      {self.logging.level}")
        print(f"{'=' * 60}\n")


# =============================================================================
# Global Config Access
# =============================================================================

_config_instance: Optional[Config] = None


@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Get the global configuration instance.

    Uses LRU cache to ensure single instance.
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config()

    return _config_instance


def reset_config() -> None:
    """Reset the configuration (useful for testing)."""
    global _config_instance
    _config_instance = None
    get_config.cache_clear()


def init_config(
    base_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    debug: bool = False,
    **kwargs
) -> Config:
    """
    Initialize configuration with custom values.

    Args:
        base_dir: Base directory for CogRepo
        api_key: Anthropic API key
        debug: Enable debug mode
        **kwargs: Additional configuration overrides

    Returns:
        Configured Config instance
    """
    global _config_instance

    config_data = {'debug': debug}

    if base_dir:
        config_data['paths'] = PathConfig(base_dir=base_dir)

    if api_key:
        config_data['anthropic'] = AnthropicConfig(api_key=api_key)

    config_data.update(kwargs)

    _config_instance = Config(**config_data)
    get_config.cache_clear()

    return _config_instance


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_environment() -> Dict[str, Any]:
    """
    Validate the current environment and return status.

    Returns:
        Dictionary with validation results
    """
    config = get_config()

    return {
        'api_key_configured': config.anthropic.is_configured,
        'data_dir_exists': config.paths.data_dir.exists(),
        'repository_exists': config.paths.repository.exists(),
        'search_db_exists': config.paths.search_database.exists(),
        'archive_registry_exists': config.paths.archive_registry.exists(),
        'upload_dir_exists': config.paths.upload_dir.exists(),
        'log_dir_exists': config.paths.log_dir.exists(),
        'enrichment_ready': len(config.validate_for_enrichment()) == 0,
        'web_ready': len(config.validate_for_web()) == 0,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CogRepo Configuration Utility")
    parser.add_argument('--validate', action='store_true', help="Validate environment")
    parser.add_argument('--print', action='store_true', help="Print configuration")
    parser.add_argument('--init', action='store_true', help="Initialize directories")

    args = parser.parse_args()

    config = get_config()

    if args.validate:
        print("\nEnvironment Validation:")
        print("-" * 40)
        status = validate_environment()
        for key, value in status.items():
            icon = "✓" if value else "✗"
            print(f"  {icon} {key}: {value}")

        # Check for issues
        enrichment_issues = config.validate_for_enrichment()
        if enrichment_issues:
            print("\nEnrichment Issues:")
            for issue in enrichment_issues:
                print(f"  ✗ {issue}")

        web_issues = config.validate_for_web()
        if web_issues:
            print("\nWeb UI Issues:")
            for issue in web_issues:
                print(f"  ✗ {issue}")

    elif args.init:
        print("\nInitializing directories...")
        config.paths.ensure_dirs()
        print(f"  ✓ Data dir: {config.paths.data_dir}")
        print(f"  ✓ Upload dir: {config.paths.upload_dir}")
        print(f"  ✓ Log dir: {config.paths.log_dir}")

    else:
        config.print_summary()
