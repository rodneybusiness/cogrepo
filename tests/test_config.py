"""
Tests for Configuration System

Tests:
- PathConfig path resolution
- AnthropicConfig validation
- EnrichmentConfig defaults
- SearchConfig parameters
- WebUIConfig security
- LoggingConfig validation
- Config loading and merging
- Environment variable handling
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import (
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


class TestPathConfig:
    """Tests for PathConfig."""

    def test_default_paths(self):
        """Test default path configuration."""
        config = PathConfig()

        assert config.data_dir is not None
        assert config.upload_dir is not None
        assert config.log_dir is not None

    def test_custom_base_dir(self):
        """Test setting custom base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = PathConfig(base_dir=base)

            assert config.base_dir == base
            assert config.data_dir == base / 'data'

    def test_repository_path(self):
        """Test repository path property."""
        config = PathConfig()

        assert config.repository == config.data_dir / config.repository_file
        assert str(config.repository).endswith('enriched_repository.jsonl')

    def test_ensure_dirs(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = PathConfig(base_dir=base)

            config.ensure_dirs()

            assert config.data_dir.exists()
            assert config.upload_dir.exists()
            assert config.log_dir.exists()


class TestAnthropicConfig:
    """Tests for AnthropicConfig."""

    def test_default_model(self):
        """Test default model setting."""
        config = AnthropicConfig()

        assert 'claude' in config.model.lower() or 'sonnet' in config.model.lower()

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid
        config = AnthropicConfig(max_tokens=4096)
        assert config.max_tokens == 4096

        # Too low
        with pytest.raises(ValueError):
            AnthropicConfig(max_tokens=50)

        # Too high
        with pytest.raises(ValueError):
            AnthropicConfig(max_tokens=500000)

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid
        config = AnthropicConfig(temperature=0.5)
        assert config.temperature == 0.5

        # Too low
        with pytest.raises(ValueError):
            AnthropicConfig(temperature=-0.1)

        # Too high
        with pytest.raises(ValueError):
            AnthropicConfig(temperature=1.5)

    def test_is_configured(self):
        """Test API key configuration check."""
        config = AnthropicConfig(api_key=None)
        assert config.is_configured is False

        config = AnthropicConfig(api_key="test-key")
        assert config.is_configured is True

    def test_api_key_from_env(self):
        """Test API key loading from environment."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'env-test-key'}):
            config = AnthropicConfig()
            assert config.api_key == 'env-test-key'

    def test_retry_config(self):
        """Test retry configuration."""
        config = AnthropicConfig(max_retries=5, retry_delay=2.0, retry_multiplier=3.0)

        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.retry_multiplier == 3.0


class TestEnrichmentConfig:
    """Tests for EnrichmentConfig."""

    def test_default_batch_size(self):
        """Test default batch size."""
        config = EnrichmentConfig()
        assert config.batch_size == 5

    def test_batch_size_validation(self):
        """Test batch size validation."""
        # Valid
        config = EnrichmentConfig(batch_size=10)
        assert config.batch_size == 10

        # Too low
        with pytest.raises(ValueError):
            EnrichmentConfig(batch_size=0)

        # Too high
        with pytest.raises(ValueError):
            EnrichmentConfig(batch_size=200)

    def test_min_confidence_validation(self):
        """Test confidence threshold validation."""
        # Valid
        config = EnrichmentConfig(min_confidence=0.8)
        assert config.min_confidence == 0.8

        # Invalid
        with pytest.raises(ValueError):
            EnrichmentConfig(min_confidence=1.5)


class TestSearchConfig:
    """Tests for SearchConfig."""

    def test_default_fts5(self):
        """Test FTS5 enabled by default."""
        config = SearchConfig()
        assert config.use_fts5 is True

    def test_limit_validation(self):
        """Test result limit validation."""
        config = SearchConfig(default_limit=50, max_limit=500)

        assert config.default_limit == 50
        assert config.max_limit == 500

    def test_bm25_parameters(self):
        """Test BM25 tuning parameters."""
        config = SearchConfig(bm25_k1=1.5, bm25_b=0.6)

        assert config.bm25_k1 == 1.5
        assert config.bm25_b == 0.6


class TestWebUIConfig:
    """Tests for WebUIConfig."""

    def test_default_host_port(self):
        """Test default host and port."""
        config = WebUIConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 5000

    def test_port_validation(self):
        """Test port validation."""
        # Valid
        config = WebUIConfig(port=8080)
        assert config.port == 8080

        # Invalid
        with pytest.raises(ValueError):
            WebUIConfig(port=0)

        with pytest.raises(ValueError):
            WebUIConfig(port=70000)

    def test_secret_key_generation(self):
        """Test automatic secret key generation."""
        config = WebUIConfig()

        assert config.secret_key is not None
        assert len(config.secret_key) == 64  # 32 bytes hex

    def test_allowed_extensions(self):
        """Test allowed file extensions."""
        config = WebUIConfig()

        assert 'json' in config.allowed_extensions
        assert 'jsonl' in config.allowed_extensions


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_default_level(self):
        """Test default log level."""
        config = LoggingConfig()
        assert config.level == "INFO"

    def test_level_validation(self):
        """Test log level validation."""
        # Valid levels
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            config = LoggingConfig(level=level)
            assert config.level == level

        # Case insensitive
        config = LoggingConfig(level='debug')
        assert config.level == 'DEBUG'

        # Invalid
        with pytest.raises(ValueError):
            LoggingConfig(level='INVALID')

    def test_file_rotation_config(self):
        """Test file rotation configuration."""
        config = LoggingConfig(max_file_size_mb=20, backup_count=10)

        assert config.max_file_size_mb == 20
        assert config.backup_count == 10


class TestConfig:
    """Tests for main Config class."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_default_config(self):
        """Test creating config with defaults."""
        config = Config()

        assert config.app_name == "CogRepo"
        assert config.version == "2.0.0"
        assert config.debug is False

    def test_sub_configs(self):
        """Test sub-configuration access."""
        config = Config()

        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.anthropic, AnthropicConfig)
        assert isinstance(config.enrichment, EnrichmentConfig)
        assert isinstance(config.search, SearchConfig)
        assert isinstance(config.web_ui, WebUIConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.archive, ArchiveConfig)

    def test_to_dict_masks_secrets(self):
        """Test that secrets are masked in dict output."""
        config = Config()
        config.anthropic.api_key = "sk-ant-api03-very-secret-key-here"

        data = config.to_dict(exclude_secrets=True)

        # API key should be masked
        assert 'very-secret' not in str(data['anthropic']['api_key'])
        assert '***' in str(data['anthropic']['api_key']) or '...' in str(data['anthropic']['api_key'])

    def test_validate_for_enrichment(self):
        """Test enrichment validation."""
        config = Config()
        config.anthropic.api_key = None

        issues = config.validate_for_enrichment()

        assert len(issues) > 0
        assert any('API_KEY' in issue.upper() for issue in issues)

    def test_validate_for_enrichment_with_key(self):
        """Test enrichment validation with API key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.anthropic.api_key = "test-key"
            config.paths._data_dir = Path(tmpdir)

            # Create the data dir
            Path(tmpdir).mkdir(exist_ok=True)

            issues = config.validate_for_enrichment()

            # Should have no API key issue
            assert not any('API_KEY' in issue.upper() for issue in issues)


class TestGlobalConfig:
    """Tests for global config functions."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_get_config_singleton(self):
        """Test get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_init_config(self):
        """Test init_config with custom values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = init_config(
                base_dir=Path(tmpdir),
                api_key="test-init-key",
                debug=True
            )

            assert config.debug is True
            assert config.anthropic.api_key == "test-init-key"
            assert config.paths.base_dir == Path(tmpdir)

    def test_reset_config(self):
        """Test config reset."""
        config1 = get_config()
        config1.debug = True

        reset_config()

        config2 = get_config()
        assert config2.debug is False  # Back to default

    def test_validate_environment(self):
        """Test environment validation."""
        reset_config()

        status = validate_environment()

        assert 'api_key_configured' in status
        assert 'data_dir_exists' in status
        assert 'repository_exists' in status
        assert 'enrichment_ready' in status
        assert 'web_ready' in status


class TestArchiveConfig:
    """Tests for ArchiveConfig."""

    def test_default_settings(self):
        """Test default archive settings."""
        config = ArchiveConfig()

        assert config.auto_sync_enabled is True
        assert config.incremental_only is True
        assert config.track_file_changes is True

    def test_sync_interval_validation(self):
        """Test sync interval validation."""
        config = ArchiveConfig(sync_interval_minutes=30)
        assert config.sync_interval_minutes == 30

        with pytest.raises(ValueError):
            ArchiveConfig(sync_interval_minutes=0)

    def test_default_archive_paths(self):
        """Test default archive paths are None."""
        config = ArchiveConfig()

        assert config.default_chatgpt_path is None
        assert config.default_claude_path is None
        assert config.default_gemini_path is None
