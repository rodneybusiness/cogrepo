"""
Provider-Agnostic LLM Abstraction Layer for CogRepo

Provides a unified interface for multiple LLM providers with:
- Standardized request/response models
- Automatic fallback chains
- Cost tracking and optimization
- Smart routing based on task complexity and PII
- Rate limiting and retry logic

Supported providers:
- Anthropic (Claude models)
- OpenAI (GPT models)
- Ollama (local models)

Usage:
    from core.llm_provider import get_provider_chain, LLMRequest, ModelTier

    chain = get_provider_chain(config)
    response = chain.complete(LLMRequest(
        messages=[{"role": "user", "content": "Hello"}],
        tier=ModelTier.FAST
    ))
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from datetime import datetime
import time
import logging
import threading
from functools import wraps

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class ModelTier(Enum):
    """
    Model capability tiers for task-based routing.

    FAST: Quick, cheap models for simple tasks (tags, titles)
    STANDARD: Balanced models for most tasks (summaries, scoring)
    ADVANCED: Best quality for complex tasks (insights, analysis)
    """
    FAST = "fast"
    STANDARD = "standard"
    ADVANCED = "advanced"


class ProviderType(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class LLMMessage:
    """A single message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class LLMRequest:
    """
    Standardized request to any LLM provider.

    Attributes:
        messages: Conversation messages
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0-1)
        tier: Model capability tier for routing
        system_prompt: Optional system prompt
        metadata: Additional provider-specific options
    """
    messages: List[Dict[str, str]]
    max_tokens: int = 1024
    temperature: float = 0.3
    tier: ModelTier = ModelTier.STANDARD
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic API format."""
        return {
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API format."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.messages)
        return {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


@dataclass
class LLMResponse:
    """
    Standardized response from any LLM provider.

    Attributes:
        content: The generated text content
        model: Model identifier used
        provider: Provider name (anthropic, openai, ollama)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        latency_ms: Request latency in milliseconds
        cost_usd: Estimated cost in USD
        raw_response: Original provider response (for debugging)
    """
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    raw_response: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


@dataclass
class ProviderHealth:
    """Health status of a provider."""
    provider: str
    is_available: bool
    last_check: datetime
    last_error: Optional[str] = None
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0


# =============================================================================
# Cost Estimation
# =============================================================================

class CostCalculator:
    """
    Calculates estimated costs for LLM API calls.

    Prices as of 2024 (may need updates):
    """

    # Prices per 1M tokens (input, output)
    PRICING = {
        # Anthropic
        "claude-3-5-sonnet-20241022": (3.0, 15.0),
        "claude-sonnet-4-20250514": (3.0, 15.0),
        "claude-3-5-haiku-20241022": (0.25, 1.25),
        "claude-3-opus-20240229": (15.0, 75.0),

        # OpenAI
        "gpt-4o": (2.5, 10.0),
        "gpt-4o-mini": (0.15, 0.6),
        "gpt-4-turbo": (10.0, 30.0),
        "gpt-3.5-turbo": (0.5, 1.5),

        # Ollama (local - free)
        "llama3": (0.0, 0.0),
        "llama3.1": (0.0, 0.0),
        "llama3.2": (0.0, 0.0),
        "mistral": (0.0, 0.0),
        "mixtral": (0.0, 0.0),
        "codellama": (0.0, 0.0),
        "phi3": (0.0, 0.0),
    }

    @classmethod
    def estimate(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for a request.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # Try exact match
        if model in cls.PRICING:
            input_price, output_price = cls.PRICING[model]
        else:
            # Try prefix match
            for model_prefix, prices in cls.PRICING.items():
                if model.startswith(model_prefix.split('-')[0]):
                    input_price, output_price = prices
                    break
            else:
                # Default conservative estimate
                input_price, output_price = (5.0, 15.0)

        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price

        return input_cost + output_cost


# =============================================================================
# Abstract Provider Base Class
# =============================================================================

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All provider implementations must inherit from this class
    and implement the required abstract methods.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize provider with configuration.

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._health = ProviderHealth(
            provider=self.name,
            is_available=False,
            last_check=datetime.now()
        )
        self._call_count = 0
        self._error_count = 0
        self._total_latency = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/metrics."""
        pass

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Provider type enum."""
        pass

    @property
    def is_available(self) -> bool:
        """Check if provider is configured and reachable."""
        return self._health.is_available

    @abstractmethod
    def _check_availability(self) -> bool:
        """
        Check if provider is available.

        Returns:
            True if provider is configured and reachable
        """
        pass

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Execute a completion request.

        Args:
            request: Standardized LLM request

        Returns:
            Standardized LLM response

        Raises:
            LLMProviderError: If request fails
        """
        pass

    @abstractmethod
    def get_model_for_tier(self, tier: ModelTier) -> str:
        """
        Get the model name for a given capability tier.

        Args:
            tier: Desired model capability tier

        Returns:
            Model identifier string
        """
        pass

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """
        Estimate cost for a request.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model identifier

        Returns:
            Estimated cost in USD
        """
        return CostCalculator.estimate(model, input_tokens, output_tokens)

    def refresh_health(self) -> ProviderHealth:
        """Refresh and return health status."""
        try:
            self._health.is_available = self._check_availability()
            self._health.last_error = None
        except Exception as e:
            self._health.is_available = False
            self._health.last_error = str(e)

        self._health.last_check = datetime.now()

        if self._call_count > 0:
            self._health.success_rate = 1 - (self._error_count / self._call_count)
            self._health.avg_latency_ms = self._total_latency / self._call_count

        return self._health

    def _record_call(self, latency_ms: float, success: bool):
        """Record call metrics."""
        self._call_count += 1
        self._total_latency += latency_ms
        if not success:
            self._error_count += 1


# =============================================================================
# Provider Exceptions
# =============================================================================

class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        retryable: bool = True,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.original_error = original_error


class RateLimitError(LLMProviderError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str,
        provider: str,
        retry_after: Optional[float] = None
    ):
        super().__init__(message, provider, retryable=True)
        self.retry_after = retry_after


class AuthenticationError(LLMProviderError):
    """Authentication failed."""

    def __init__(self, message: str, provider: str):
        super().__init__(message, provider, retryable=False)


class ModelNotFoundError(LLMProviderError):
    """Requested model not available."""

    def __init__(self, message: str, provider: str, model: str):
        super().__init__(message, provider, retryable=False)
        self.model = model


# =============================================================================
# Anthropic Provider
# =============================================================================

class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider implementation.

    Supports Claude 3.5 Sonnet, Haiku, and Opus models.
    """

    # Model mappings for tiers
    TIER_MODELS = {
        ModelTier.FAST: "claude-3-5-haiku-20241022",
        ModelTier.STANDARD: "claude-sonnet-4-20250514",
        ModelTier.ADVANCED: "claude-3-opus-20240229",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
        self._api_key = config.get("api_key")

        # Allow tier model overrides from config
        if "models" in config:
            for tier_name, model in config["models"].items():
                try:
                    tier = ModelTier(tier_name)
                    self.TIER_MODELS[tier] = model
                except ValueError:
                    pass

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self._api_key)
            except ImportError:
                raise LLMProviderError(
                    "anthropic package not installed. Run: pip install anthropic",
                    self.name,
                    retryable=False
                )
        return self._client

    def _check_availability(self) -> bool:
        """Check if Anthropic API is available."""
        if not self._api_key:
            return False
        try:
            # Just check if we can create a client
            self._get_client()
            return True
        except Exception:
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError,))
    )
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute completion request against Anthropic API."""
        client = self._get_client()
        model = self.get_model_for_tier(request.tier)

        start_time = time.time()

        try:
            # Build request
            api_request = {
                "model": model,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "messages": request.messages,
            }

            if request.system_prompt:
                api_request["system"] = request.system_prompt

            # Make API call
            response = client.messages.create(**api_request)

            latency_ms = (time.time() - start_time) * 1000
            self._record_call(latency_ms, success=True)

            # Extract response data
            content = response.content[0].text if response.content else ""
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            return LLMResponse(
                content=content,
                model=model,
                provider=self.name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=self.estimate_cost(input_tokens, output_tokens, model),
                raw_response={"id": response.id, "stop_reason": response.stop_reason}
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._record_call(latency_ms, success=False)

            error_str = str(e).lower()

            if "rate" in error_str or "429" in error_str:
                raise RateLimitError(str(e), self.name)
            elif "auth" in error_str or "401" in error_str or "api_key" in error_str:
                raise AuthenticationError(str(e), self.name)
            elif "model" in error_str and "not found" in error_str:
                raise ModelNotFoundError(str(e), self.name, model)
            else:
                raise LLMProviderError(str(e), self.name, original_error=e)

    def get_model_for_tier(self, tier: ModelTier) -> str:
        """Get model for capability tier."""
        return self.TIER_MODELS.get(tier, self.TIER_MODELS[ModelTier.STANDARD])


# =============================================================================
# OpenAI Provider
# =============================================================================

class OpenAIProvider(LLMProvider):
    """
    OpenAI GPT provider implementation.

    Supports GPT-4o, GPT-4o-mini, GPT-4-turbo, and GPT-3.5-turbo models.
    """

    TIER_MODELS = {
        ModelTier.FAST: "gpt-4o-mini",
        ModelTier.STANDARD: "gpt-4o",
        ModelTier.ADVANCED: "gpt-4-turbo",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
        self._api_key = config.get("api_key")
        self._base_url = config.get("base_url")  # For Azure or compatible APIs

        # Allow tier model overrides
        if "models" in config:
            for tier_name, model in config["models"].items():
                try:
                    tier = ModelTier(tier_name)
                    self.TIER_MODELS[tier] = model
                except ValueError:
                    pass

    @property
    def name(self) -> str:
        return "openai"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                kwargs = {"api_key": self._api_key}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = OpenAI(**kwargs)
            except ImportError:
                raise LLMProviderError(
                    "openai package not installed. Run: pip install openai",
                    self.name,
                    retryable=False
                )
        return self._client

    def _check_availability(self) -> bool:
        """Check if OpenAI API is available."""
        if not self._api_key:
            return False
        try:
            self._get_client()
            return True
        except Exception:
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError,))
    )
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute completion request against OpenAI API."""
        client = self._get_client()
        model = self.get_model_for_tier(request.tier)

        start_time = time.time()

        try:
            # Build messages
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.extend(request.messages)

            # Make API call
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            latency_ms = (time.time() - start_time) * 1000
            self._record_call(latency_ms, success=True)

            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            return LLMResponse(
                content=content,
                model=model,
                provider=self.name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=self.estimate_cost(input_tokens, output_tokens, model),
                raw_response={"id": response.id, "finish_reason": choice.finish_reason}
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._record_call(latency_ms, success=False)

            error_str = str(e).lower()

            if "rate" in error_str or "429" in error_str:
                raise RateLimitError(str(e), self.name)
            elif "auth" in error_str or "401" in error_str or "api_key" in error_str:
                raise AuthenticationError(str(e), self.name)
            elif "model" in error_str and "not found" in error_str:
                raise ModelNotFoundError(str(e), self.name, model)
            else:
                raise LLMProviderError(str(e), self.name, original_error=e)

    def get_model_for_tier(self, tier: ModelTier) -> str:
        """Get model for capability tier."""
        return self.TIER_MODELS.get(tier, self.TIER_MODELS[ModelTier.STANDARD])


# =============================================================================
# Ollama Provider (Local Models)
# =============================================================================

class OllamaProvider(LLMProvider):
    """
    Ollama local model provider implementation.

    Supports Llama, Mistral, CodeLlama, and other local models.
    Runs locally with no API costs.
    """

    TIER_MODELS = {
        ModelTier.FAST: "llama3.2",
        ModelTier.STANDARD: "llama3.1",
        ModelTier.ADVANCED: "mixtral",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._base_url = config.get("base_url", "http://localhost:11434")
        self._timeout = config.get("timeout", 120)

        # Allow tier model overrides
        if "models" in config:
            for tier_name, model in config["models"].items():
                try:
                    tier = ModelTier(tier_name)
                    self.TIER_MODELS[tier] = model
                except ValueError:
                    pass

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA

    def _check_availability(self) -> bool:
        """Check if Ollama server is available."""
        try:
            import requests
            response = requests.get(
                f"{self._base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute completion request against local Ollama server."""
        import requests

        model = self.get_model_for_tier(request.tier)
        start_time = time.time()

        try:
            # Build prompt from messages
            prompt_parts = []
            if request.system_prompt:
                prompt_parts.append(f"System: {request.system_prompt}")

            for msg in request.messages:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                prompt_parts.append(f"{role}: {content}")

            prompt = "\n\n".join(prompt_parts)

            # Make API call
            response = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    }
                },
                timeout=self._timeout
            )
            response.raise_for_status()

            latency_ms = (time.time() - start_time) * 1000
            self._record_call(latency_ms, success=True)

            data = response.json()
            content = data.get("response", "")

            # Ollama provides eval counts
            input_tokens = data.get("prompt_eval_count", len(prompt) // 4)
            output_tokens = data.get("eval_count", len(content) // 4)

            return LLMResponse(
                content=content,
                model=model,
                provider=self.name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=0.0,  # Local models are free
                raw_response=data
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._record_call(latency_ms, success=False)

            error_str = str(e).lower()

            if "connection" in error_str or "timeout" in error_str:
                raise LLMProviderError(
                    f"Ollama server not reachable at {self._base_url}: {e}",
                    self.name,
                    retryable=True,
                    original_error=e
                )
            elif "model" in error_str:
                raise ModelNotFoundError(str(e), self.name, model)
            else:
                raise LLMProviderError(str(e), self.name, original_error=e)

    def get_model_for_tier(self, tier: ModelTier) -> str:
        """Get model for capability tier."""
        return self.TIER_MODELS.get(tier, self.TIER_MODELS[ModelTier.STANDARD])

    def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        try:
            import requests
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


# =============================================================================
# Provider Chain (Fallback Logic)
# =============================================================================

class ProviderChain:
    """
    Manages multiple providers with fallback logic.

    Tries providers in order until one succeeds.
    Supports smart routing based on:
    - Provider health
    - Cost optimization
    - PII sensitivity
    """

    def __init__(
        self,
        providers: List[LLMProvider],
        metrics_callback: Optional[Callable[[LLMResponse], None]] = None
    ):
        """
        Initialize provider chain.

        Args:
            providers: Ordered list of providers (primary first)
            metrics_callback: Optional callback for metrics collection
        """
        self.providers = providers
        self.metrics_callback = metrics_callback
        self._lock = threading.Lock()
        self._total_calls = 0
        self._total_cost = 0.0
        self._provider_stats: Dict[str, Dict] = {}

    def complete(
        self,
        request: LLMRequest,
        prefer_local: bool = False,
        max_cost_usd: Optional[float] = None
    ) -> LLMResponse:
        """
        Try providers in order until one succeeds.

        Args:
            request: The LLM request to execute
            prefer_local: If True, try local providers first (for PII)
            max_cost_usd: Maximum allowed cost for this request

        Returns:
            LLM response from first successful provider

        Raises:
            LLMProviderError: If all providers fail
        """
        providers_to_try = self._get_ordered_providers(prefer_local)
        last_error = None

        for provider in providers_to_try:
            # Check availability
            provider.refresh_health()
            if not provider.is_available:
                logger.debug(f"Skipping unavailable provider: {provider.name}")
                continue

            # Check cost constraint
            if max_cost_usd is not None:
                model = provider.get_model_for_tier(request.tier)
                # Rough estimate based on max tokens
                estimated_cost = provider.estimate_cost(
                    request.max_tokens,  # Assume input roughly equals max output
                    request.max_tokens,
                    model
                )
                if estimated_cost > max_cost_usd:
                    logger.debug(
                        f"Skipping {provider.name} due to cost "
                        f"(estimated ${estimated_cost:.4f} > max ${max_cost_usd:.4f})"
                    )
                    continue

            try:
                logger.debug(f"Trying provider: {provider.name}")
                response = provider.complete(request)

                # Record metrics
                self._record_success(provider.name, response)

                # Call metrics callback if provided
                if self.metrics_callback:
                    self.metrics_callback(response)

                return response

            except AuthenticationError as e:
                # Don't retry auth errors
                logger.warning(f"Auth error with {provider.name}: {e}")
                last_error = e
                continue

            except LLMProviderError as e:
                logger.warning(f"Error with {provider.name}: {e}")
                last_error = e
                if not e.retryable:
                    continue
                # Continue to next provider

        raise LLMProviderError(
            f"All providers failed. Last error: {last_error}",
            provider="chain",
            retryable=False,
            original_error=last_error
        )

    def _get_ordered_providers(self, prefer_local: bool) -> List[LLMProvider]:
        """Get providers in optimal order."""
        if prefer_local:
            # Put local providers (Ollama) first
            local = [p for p in self.providers if p.provider_type == ProviderType.OLLAMA]
            remote = [p for p in self.providers if p.provider_type != ProviderType.OLLAMA]
            return local + remote
        return self.providers

    def _record_success(self, provider_name: str, response: LLMResponse):
        """Record successful call metrics."""
        with self._lock:
            self._total_calls += 1
            self._total_cost += response.cost_usd

            if provider_name not in self._provider_stats:
                self._provider_stats[provider_name] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "total_latency_ms": 0.0
                }

            stats = self._provider_stats[provider_name]
            stats["calls"] += 1
            stats["tokens"] += response.total_tokens
            stats["cost_usd"] += response.cost_usd
            stats["total_latency_ms"] += response.latency_ms

    def get_stats(self) -> Dict[str, Any]:
        """Get chain statistics."""
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "total_cost_usd": self._total_cost,
                "providers": {
                    name: {
                        **stats,
                        "avg_latency_ms": (
                            stats["total_latency_ms"] / stats["calls"]
                            if stats["calls"] > 0 else 0
                        )
                    }
                    for name, stats in self._provider_stats.items()
                }
            }

    def get_health(self) -> Dict[str, ProviderHealth]:
        """Get health status of all providers."""
        return {
            provider.name: provider.refresh_health()
            for provider in self.providers
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_provider(
    provider_type: ProviderType,
    config: Dict[str, Any]
) -> LLMProvider:
    """
    Create a provider instance.

    Args:
        provider_type: Type of provider to create
        config: Provider-specific configuration

    Returns:
        Configured provider instance
    """
    providers = {
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.OLLAMA: OllamaProvider,
    }

    provider_class = providers.get(provider_type)
    if not provider_class:
        raise ValueError(f"Unknown provider type: {provider_type}")

    return provider_class(config)


def get_provider_chain(config: Dict[str, Any]) -> ProviderChain:
    """
    Create a provider chain from configuration.

    Expected config structure:
    {
        "primary_provider": "anthropic",
        "fallback_providers": ["openai", "ollama"],
        "anthropic": {"api_key": "..."},
        "openai": {"api_key": "..."},
        "ollama": {"base_url": "http://localhost:11434"}
    }

    Args:
        config: LLM configuration dictionary

    Returns:
        Configured provider chain
    """
    providers = []

    # Add primary provider
    primary = config.get("primary_provider", "anthropic")
    try:
        primary_type = ProviderType(primary)
        primary_config = config.get(primary, {})
        providers.append(create_provider(primary_type, primary_config))
    except (ValueError, KeyError) as e:
        logger.warning(f"Could not create primary provider {primary}: {e}")

    # Add fallback providers
    fallbacks = config.get("fallback_providers", [])
    for fallback_name in fallbacks:
        try:
            fallback_type = ProviderType(fallback_name)
            fallback_config = config.get(fallback_name, {})
            providers.append(create_provider(fallback_type, fallback_config))
        except (ValueError, KeyError) as e:
            logger.warning(f"Could not create fallback provider {fallback_name}: {e}")

    if not providers:
        raise ValueError("No providers configured")

    return ProviderChain(providers)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_complete(
    prompt: str,
    config: Dict[str, Any],
    tier: ModelTier = ModelTier.STANDARD,
    system_prompt: Optional[str] = None
) -> str:
    """
    Quick completion helper for simple use cases.

    Args:
        prompt: User prompt
        config: LLM configuration
        tier: Model capability tier
        system_prompt: Optional system prompt

    Returns:
        Generated text content
    """
    chain = get_provider_chain(config)
    request = LLMRequest(
        messages=[{"role": "user", "content": prompt}],
        tier=tier,
        system_prompt=system_prompt
    )
    response = chain.complete(request)
    return response.content
