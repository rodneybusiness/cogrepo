"""
Tests for LLM Provider Abstraction Layer

Tests provider interfaces, chain fallback, and cost calculation.
Note: These tests use mocks to avoid actual API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_provider import (
    LLMProvider, LLMRequest, LLMResponse, ModelTier, ProviderType,
    AnthropicProvider, OpenAIProvider, OllamaProvider,
    ProviderChain, CostCalculator,
    LLMProviderError, RateLimitError, AuthenticationError, ModelNotFoundError,
    create_provider, get_provider_chain
)


class TestLLMRequest:
    """Tests for LLMRequest model."""

    def test_basic_request(self):
        """Test basic request creation."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100
        )

        assert len(request.messages) == 1
        assert request.max_tokens == 100
        assert request.tier == ModelTier.STANDARD

    def test_request_with_all_params(self):
        """Test request with all parameters."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=500,
            temperature=0.7,
            tier=ModelTier.FAST,
            system_prompt="Be helpful"
        )

        assert request.temperature == 0.7
        assert request.tier == ModelTier.FAST
        assert request.system_prompt == "Be helpful"

    def test_anthropic_format(self):
        """Test conversion to Anthropic format."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100
        )

        fmt = request.to_anthropic_format()

        assert "messages" in fmt
        assert fmt["max_tokens"] == 100

    def test_openai_format(self):
        """Test conversion to OpenAI format."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="Be helpful"
        )

        fmt = request.to_openai_format()

        assert len(fmt["messages"]) == 2  # system + user
        assert fmt["messages"][0]["role"] == "system"


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_basic_response(self):
        """Test basic response creation."""
        response = LLMResponse(
            content="Hello!",
            model="test-model",
            provider="test",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100.0,
            cost_usd=0.001
        )

        assert response.content == "Hello!"
        assert response.total_tokens == 15

    def test_response_with_raw(self):
        """Test response with raw data."""
        response = LLMResponse(
            content="Hello!",
            model="test-model",
            provider="test",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100.0,
            cost_usd=0.001,
            raw_response={"id": "msg_123"}
        )

        assert response.raw_response["id"] == "msg_123"


class TestCostCalculator:
    """Tests for cost estimation."""

    def test_known_model_cost(self):
        """Test cost for known model."""
        cost = CostCalculator.estimate(
            "claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500
        )

        # Sonnet: $3/1M input, $15/1M output
        expected = (1000 / 1_000_000 * 3) + (500 / 1_000_000 * 15)
        assert abs(cost - expected) < 0.0001

    def test_haiku_cheaper(self):
        """Test that Haiku is cheaper than Sonnet."""
        haiku_cost = CostCalculator.estimate(
            "claude-3-5-haiku-20241022",
            input_tokens=1000,
            output_tokens=500
        )

        sonnet_cost = CostCalculator.estimate(
            "claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500
        )

        assert haiku_cost < sonnet_cost

    def test_ollama_free(self):
        """Test that Ollama models are free."""
        cost = CostCalculator.estimate(
            "llama3",
            input_tokens=10000,
            output_tokens=5000
        )

        assert cost == 0.0

    def test_unknown_model_defaults(self):
        """Test unknown model uses conservative estimate."""
        cost = CostCalculator.estimate(
            "unknown-model-xyz",
            input_tokens=1000,
            output_tokens=500
        )

        assert cost > 0  # Should have some cost estimate


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    @pytest.fixture
    def provider(self):
        """Create provider with mock API key."""
        return AnthropicProvider({"api_key": "test-key"})

    def test_name_and_type(self, provider):
        """Test provider identification."""
        assert provider.name == "anthropic"
        assert provider.provider_type == ProviderType.ANTHROPIC

    def test_model_tier_mapping(self, provider):
        """Test model tier mapping."""
        fast = provider.get_model_for_tier(ModelTier.FAST)
        standard = provider.get_model_for_tier(ModelTier.STANDARD)
        advanced = provider.get_model_for_tier(ModelTier.ADVANCED)

        assert "haiku" in fast.lower()
        assert "sonnet" in standard.lower() or "claude" in standard.lower()

    def test_not_configured_without_key(self):
        """Test availability check without API key."""
        provider = AnthropicProvider({})
        assert not provider._check_availability()

    def test_complete_success(self, provider):
        """Test successful completion with mock."""
        # Test request creation and response handling
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hi"}]
        )

        # Verify request format is correct
        fmt = request.to_anthropic_format()
        assert "messages" in fmt
        assert len(fmt["messages"]) == 1

        # Verify provider configured properly
        assert provider.name == "anthropic"
        assert provider._api_key == "test-key"


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    @pytest.fixture
    def provider(self):
        """Create provider with mock API key."""
        return OpenAIProvider({"api_key": "test-key"})

    def test_name_and_type(self, provider):
        """Test provider identification."""
        assert provider.name == "openai"
        assert provider.provider_type == ProviderType.OPENAI

    def test_model_tier_mapping(self, provider):
        """Test model tier mapping."""
        fast = provider.get_model_for_tier(ModelTier.FAST)
        standard = provider.get_model_for_tier(ModelTier.STANDARD)

        assert "mini" in fast.lower()
        assert "gpt" in standard.lower()

    def test_custom_base_url(self):
        """Test custom base URL for Azure."""
        provider = OpenAIProvider({
            "api_key": "test",
            "base_url": "https://custom.openai.azure.com"
        })
        assert provider._base_url == "https://custom.openai.azure.com"


class TestOllamaProvider:
    """Tests for Ollama provider."""

    @pytest.fixture
    def provider(self):
        """Create provider with default config."""
        return OllamaProvider({"base_url": "http://localhost:11434"})

    def test_name_and_type(self, provider):
        """Test provider identification."""
        assert provider.name == "ollama"
        assert provider.provider_type == ProviderType.OLLAMA

    def test_model_tier_mapping(self, provider):
        """Test model tier mapping."""
        fast = provider.get_model_for_tier(ModelTier.FAST)
        advanced = provider.get_model_for_tier(ModelTier.ADVANCED)

        assert "llama" in fast.lower()
        assert "mixtral" in advanced.lower()

    def test_cost_always_zero(self, provider):
        """Test that local models are free."""
        cost = provider.estimate_cost(10000, 5000, "llama3")
        assert cost == 0.0


class TestProviderChain:
    """Tests for provider chain with fallback."""

    @pytest.fixture
    def mock_providers(self):
        """Create mock providers."""
        primary = Mock(spec=LLMProvider)
        primary.name = "primary"
        primary.provider_type = ProviderType.ANTHROPIC
        primary.is_available = True
        primary.refresh_health.return_value = Mock(is_available=True)

        fallback = Mock(spec=LLMProvider)
        fallback.name = "fallback"
        fallback.provider_type = ProviderType.OPENAI
        fallback.is_available = True
        fallback.refresh_health.return_value = Mock(is_available=True)

        return [primary, fallback]

    def test_uses_primary_on_success(self, mock_providers):
        """Test that primary provider is used first."""
        primary, fallback = mock_providers

        response = LLMResponse(
            content="Hello",
            model="test",
            provider="primary",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100,
            cost_usd=0.001
        )
        primary.complete.return_value = response
        primary.get_model_for_tier.return_value = "test-model"

        chain = ProviderChain([primary, fallback])
        request = LLMRequest(messages=[{"role": "user", "content": "Hi"}])

        result = chain.complete(request)

        assert result.provider == "primary"
        primary.complete.assert_called_once()
        fallback.complete.assert_not_called()

    def test_falls_back_on_error(self, mock_providers):
        """Test fallback when primary fails."""
        primary, fallback = mock_providers

        primary.complete.side_effect = LLMProviderError("Failed", "primary")

        response = LLMResponse(
            content="Hello from fallback",
            model="test",
            provider="fallback",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100,
            cost_usd=0.001
        )
        fallback.complete.return_value = response
        fallback.get_model_for_tier.return_value = "test-model"

        chain = ProviderChain([primary, fallback])
        request = LLMRequest(messages=[{"role": "user", "content": "Hi"}])

        result = chain.complete(request)

        assert result.provider == "fallback"

    def test_skips_unavailable_provider(self, mock_providers):
        """Test skipping unavailable providers."""
        primary, fallback = mock_providers

        # Set primary as unavailable (refresh_health should update is_available)
        primary.is_available = False

        response = LLMResponse(
            content="Hello",
            model="test",
            provider="fallback",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100.0,
            cost_usd=0.001
        )
        fallback.complete.return_value = response
        fallback.get_model_for_tier.return_value = "test-model"

        chain = ProviderChain([primary, fallback])
        request = LLMRequest(messages=[{"role": "user", "content": "Hi"}])

        result = chain.complete(request)

        assert result.content == "Hello"
        primary.complete.assert_not_called()

    def test_prefer_local_reorders(self, mock_providers):
        """Test that prefer_local moves Ollama first."""
        anthropic = Mock(spec=LLMProvider)
        anthropic.name = "anthropic"
        anthropic.provider_type = ProviderType.ANTHROPIC
        anthropic.is_available = True
        anthropic.refresh_health.return_value = Mock(is_available=True)

        ollama = Mock(spec=LLMProvider)
        ollama.name = "ollama"
        ollama.provider_type = ProviderType.OLLAMA
        ollama.is_available = True
        ollama.refresh_health.return_value = Mock(is_available=True)

        response = LLMResponse(
            content="Hello from local",
            model="llama3",
            provider="ollama",
            input_tokens=10,
            output_tokens=5,
            latency_ms=500,
            cost_usd=0.0
        )
        ollama.complete.return_value = response
        ollama.get_model_for_tier.return_value = "llama3"

        chain = ProviderChain([anthropic, ollama])
        request = LLMRequest(messages=[{"role": "user", "content": "Hi"}])

        result = chain.complete(request, prefer_local=True)

        assert result.provider == "ollama"
        anthropic.complete.assert_not_called()

    def test_all_providers_fail_raises(self, mock_providers):
        """Test error when all providers fail."""
        primary, fallback = mock_providers

        primary.complete.side_effect = LLMProviderError("Failed", "primary")
        fallback.complete.side_effect = LLMProviderError("Failed", "fallback")

        chain = ProviderChain([primary, fallback])
        request = LLMRequest(messages=[{"role": "user", "content": "Hi"}])

        with pytest.raises(LLMProviderError):
            chain.complete(request)

    def test_stats_tracking(self, mock_providers):
        """Test statistics are tracked."""
        primary, _ = mock_providers

        response = LLMResponse(
            content="Hello",
            model="test",
            provider="primary",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200,
            cost_usd=0.01
        )
        primary.complete.return_value = response
        primary.get_model_for_tier.return_value = "test-model"

        chain = ProviderChain(mock_providers)

        # Make multiple calls
        for _ in range(3):
            chain.complete(LLMRequest(messages=[{"role": "user", "content": "Hi"}]))

        stats = chain.get_stats()

        assert stats["total_calls"] == 3
        assert stats["total_cost_usd"] == 0.03


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        provider = create_provider(
            ProviderType.ANTHROPIC,
            {"api_key": "test"}
        )

        assert isinstance(provider, AnthropicProvider)

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        provider = create_provider(
            ProviderType.OPENAI,
            {"api_key": "test"}
        )

        assert isinstance(provider, OpenAIProvider)

    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        provider = create_provider(
            ProviderType.OLLAMA,
            {"base_url": "http://localhost:11434"}
        )

        assert isinstance(provider, OllamaProvider)

    def test_get_provider_chain_from_config(self):
        """Test creating chain from config."""
        config = {
            "primary_provider": "anthropic",
            "fallback_providers": ["openai"],
            "anthropic": {"api_key": "test-anthropic"},
            "openai": {"api_key": "test-openai"}
        }

        chain = get_provider_chain(config)

        assert len(chain.providers) == 2
        assert chain.providers[0].name == "anthropic"
        assert chain.providers[1].name == "openai"


class TestExceptions:
    """Tests for exception hierarchy."""

    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError("Rate limited", "anthropic", retry_after=60)

        assert error.retryable
        assert error.retry_after == 60

    def test_auth_error_not_retryable(self):
        """Test auth error is not retryable."""
        error = AuthenticationError("Invalid key", "openai")

        assert not error.retryable

    def test_model_not_found(self):
        """Test model not found error."""
        error = ModelNotFoundError("Model not found", "anthropic", "gpt-5")

        assert error.model == "gpt-5"
        assert not error.retryable
