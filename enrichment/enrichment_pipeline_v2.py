"""
Enrichment Pipeline v2 - Provider-Agnostic with PII Handling

This is the next-generation enrichment pipeline that uses:
- Provider-agnostic LLM abstraction (Anthropic, OpenAI, Ollama)
- PII detection and scrubbing before sending to external APIs
- Comprehensive metrics collection
- Smart routing based on task complexity and PII sensitivity

Usage:
    from enrichment.enrichment_pipeline_v2 import EnrichmentPipelineV2

    pipeline = EnrichmentPipelineV2(config)
    enriched = pipeline.enrich_conversation(raw_conversation)
"""

import re
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import RawConversation, EnrichedConversation

# Import new modules
from core.llm_provider import (
    LLMProvider, ProviderChain, LLMRequest, LLMResponse,
    ModelTier, get_provider_chain, LLMProviderError
)
from core.pii_handler import (
    PIIHandler, PIIConfig, PIIAudit, ScrubMode
)
from core.metrics import (
    MetricsCollector, get_metrics_collector, MetricsRun, PhaseTimer
)

logger = logging.getLogger(__name__)


class EnrichmentPipelineV2:
    """
    Next-generation AI-powered enrichment pipeline.

    Key improvements over v1:
    - Provider-agnostic: Works with Anthropic, OpenAI, or local Ollama
    - PII-aware: Scrubs sensitive data before external API calls
    - Observable: Full metrics collection for monitoring
    - Resilient: Automatic fallback between providers
    """

    def __init__(
        self,
        config: Dict[str, Any],
        provider_chain: Optional[ProviderChain] = None,
        pii_handler: Optional[PIIHandler] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize enrichment pipeline.

        Args:
            config: Configuration dictionary
            provider_chain: Optional pre-configured provider chain
            pii_handler: Optional pre-configured PII handler
            metrics_collector: Optional pre-configured metrics collector
        """
        self.config = config

        # Get sub-configs
        self.processing_config = config.get("processing", {})
        self.enrichment_config = config.get("enrichment", {})
        self.llm_config = config.get("llm", {})
        self.pii_config = config.get("pii", {})

        # Initialize provider chain
        if provider_chain:
            self.provider_chain = provider_chain
        else:
            self.provider_chain = self._init_provider_chain()

        # Initialize PII handler
        if pii_handler:
            self.pii_handler = pii_handler
        else:
            self.pii_handler = self._init_pii_handler()

        # Initialize metrics
        self.metrics = metrics_collector or get_metrics_collector()

        # Statistics
        self.total_enrichments = 0
        self.total_api_calls = 0
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0
        self.failed_enrichments = 0
        self.pii_scrubbed_count = 0

    def _init_provider_chain(self) -> ProviderChain:
        """Initialize provider chain from config."""
        # Build provider config from our config structure
        provider_config = {
            "primary_provider": self.llm_config.get("primary_provider", "anthropic"),
            "fallback_providers": self.llm_config.get("fallback_providers", []),
        }

        # Add provider-specific configs
        if "anthropic" in self.config:
            provider_config["anthropic"] = self.config["anthropic"]
        if "openai" in self.config:
            provider_config["openai"] = self.config["openai"]
        if "ollama" in self.config:
            provider_config["ollama"] = self.config["ollama"]

        return get_provider_chain(provider_config)

    def _init_pii_handler(self) -> PIIHandler:
        """Initialize PII handler from config."""
        pii_config = PIIConfig(
            enabled=self.pii_config.get("enabled", True),
            scrub_mode=ScrubMode(self.pii_config.get("scrub_mode", "redact")),
            detect_emails=self.pii_config.get("detect_emails", True),
            detect_phones=self.pii_config.get("detect_phones", True),
            detect_api_keys=self.pii_config.get("detect_api_keys", True),
            detect_credit_cards=self.pii_config.get("detect_credit_cards", True),
        )
        return PIIHandler(pii_config)

    def enrich_conversation(
        self,
        raw: RawConversation,
        skip_pii_check: bool = False
    ) -> EnrichedConversation:
        """
        Enrich a single conversation with AI-generated metadata.

        Args:
            raw: RawConversation object
            skip_pii_check: Skip PII handling (for local models only)

        Returns:
            EnrichedConversation object
        """
        self.total_enrichments += 1

        # Check if conversation meets minimum requirements
        if not self._should_enrich(raw):
            return EnrichedConversation.from_raw(raw, enrichments=None)

        # Prepare conversation text
        original_text = raw.raw_text
        conversation_text = self._prepare_conversation_text(raw)

        # Handle PII
        pii_audit = None
        prefer_local = False

        if not skip_pii_check and self.pii_handler.config.enabled:
            # Check for PII
            if self.pii_handler.has_pii(conversation_text):
                self.pii_scrubbed_count += 1

                # Option 1: Prefer local model for PII content
                if self.llm_config.get("prefer_local_for_pii", True):
                    prefer_local = True
                    logger.debug(f"PII detected, preferring local model for {raw.external_id}")

                # Option 2: Scrub PII before sending to external API
                else:
                    conversation_text, pii_audit = self.pii_handler.process(conversation_text)
                    logger.debug(f"Scrubbed {pii_audit.total_matches} PII instances from {raw.external_id}")

        # Generate enrichments
        enrichments = {}
        enrichment_errors = []

        try:
            # Generate title (fast model)
            if self.enrichment_config.get("generate_titles", True):
                try:
                    enrichments["generated_title"] = self._generate_title(
                        conversation_text,
                        raw.title,
                        prefer_local=prefer_local
                    )
                except LLMProviderError as e:
                    enrichment_errors.append(f"title: {e}")

            # Generate summaries (standard model)
            if self.enrichment_config.get("generate_summaries", True):
                try:
                    summaries = self._generate_summaries(
                        conversation_text,
                        prefer_local=prefer_local
                    )
                    enrichments["summary_abstractive"] = summaries["abstractive"]
                    enrichments["summary_extractive"] = summaries["extractive"]
                except LLMProviderError as e:
                    enrichment_errors.append(f"summaries: {e}")

            # Extract tags and topics (fast model)
            if self.enrichment_config.get("extract_tags", True):
                try:
                    tags_topics = self._extract_tags_and_topics(
                        conversation_text,
                        prefer_local=prefer_local
                    )
                    enrichments["tags"] = tags_topics["tags"]
                    enrichments["key_topics"] = tags_topics["key_topics"]
                    enrichments["primary_domain"] = tags_topics["primary_domain"]
                except LLMProviderError as e:
                    enrichment_errors.append(f"tags: {e}")

            # Calculate score (standard model)
            if self.enrichment_config.get("calculate_scores", True):
                try:
                    scoring = self._calculate_score(
                        conversation_text,
                        prefer_local=prefer_local
                    )
                    enrichments["brilliance_score"] = {
                        "score": scoring["score"],
                        "reasoning": scoring["reasoning"]
                    }
                    enrichments["score"] = scoring["score"]
                    enrichments["score_reasoning"] = scoring["reasoning"]
                except LLMProviderError as e:
                    enrichment_errors.append(f"scoring: {e}")

            # Extract insights (standard model)
            if self.enrichment_config.get("extract_insights", True):
                try:
                    insights = self._extract_insights(
                        conversation_text,
                        prefer_local=prefer_local
                    )
                    enrichments["key_insights"] = insights["key_insights"]
                    enrichments["status"] = insights["status"]
                    enrichments["future_potential"] = insights["future_potential"]
                except LLMProviderError as e:
                    enrichment_errors.append(f"insights: {e}")

            # Add PII audit to metadata
            if pii_audit and pii_audit.pii_found:
                enrichments["_pii_audit"] = pii_audit.to_dict()

            # Add any errors to metadata
            if enrichment_errors:
                enrichments["_enrichment_errors"] = enrichment_errors

            # Create enriched conversation
            return EnrichedConversation.from_raw(raw, enrichments=enrichments)

        except Exception as e:
            self.failed_enrichments += 1
            logger.error(f"Enrichment failed for {raw.external_id}: {e}")
            return EnrichedConversation.from_raw(raw, enrichments=None)

    def _should_enrich(self, raw: RawConversation) -> bool:
        """Check if conversation meets minimum requirements."""
        min_length = self.processing_config.get("min_conversation_length_chars", 100)
        min_messages = self.processing_config.get("min_message_count", 2)

        if len(raw.raw_text) < min_length:
            return False

        if len(raw.messages) < min_messages:
            return False

        return True

    def _prepare_conversation_text(
        self,
        raw: RawConversation,
        max_chars: int = 15000
    ) -> str:
        """Prepare conversation text for API calls."""
        text = raw.raw_text

        if len(text) > max_chars:
            # Truncate but keep beginning and end
            half = max_chars // 2
            text = text[:half] + "\n\n[... middle truncated ...]\n\n" + text[-half:]

        return text

    def _call_llm(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.STANDARD,
        max_tokens: int = 1024,
        prefer_local: bool = False,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Make an LLM call through the provider chain.

        Args:
            prompt: User prompt
            tier: Model tier (FAST, STANDARD, ADVANCED)
            max_tokens: Maximum response tokens
            prefer_local: Prefer local models (for PII)
            system_prompt: Optional system prompt

        Returns:
            LLMResponse object
        """
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.config.get("api", {}).get("temperature", 0.3),
            tier=tier,
            system_prompt=system_prompt
        )

        # Get cost limit
        max_cost = self.llm_config.get("max_cost_per_conversation_usd")

        response = self.provider_chain.complete(
            request,
            prefer_local=prefer_local,
            max_cost_usd=max_cost
        )

        # Track metrics
        self.total_api_calls += 1
        self.total_tokens_used += response.total_tokens
        self.total_cost_usd += response.cost_usd

        # Record in metrics collector
        self.metrics.record_api_call(
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
            success=True
        )

        return response

    def _generate_title(
        self,
        conversation_text: str,
        original_title: str,
        prefer_local: bool = False
    ) -> str:
        """Generate a meaningful title for the conversation."""
        if not conversation_text.strip():
            return original_title or "Empty Conversation"

        prompt = f"""Analyze this conversation and generate a concise, descriptive title (5-8 words).

The title should capture the main topic or purpose of the conversation.

Conversation:
{conversation_text[:3000]}

Original title: {original_title}

Generate ONLY the title text, no quotes or formatting."""

        response = self._call_llm(
            prompt,
            tier=ModelTier.FAST,
            max_tokens=100,
            prefer_local=prefer_local
        )

        title = response.content.strip().strip('"').strip("'")

        # Truncate if too long
        max_length = self.enrichment_config.get("title", {}).get("max_length", 100)
        if len(title) > max_length:
            title = title[:max_length - 3] + "..."

        return title or original_title

    def _generate_summaries(
        self,
        conversation_text: str,
        prefer_local: bool = False
    ) -> Dict[str, str]:
        """Generate abstractive and extractive summaries."""
        target_length = self.enrichment_config.get("summary", {}).get("abstractive_length", 280)
        extractive_sentences = self.enrichment_config.get("summary", {}).get("extractive_sentences", 3)

        prompt = f"""Summarize this conversation in two ways:

1. ABSTRACTIVE SUMMARY: Write a concise summary in {target_length} characters or less that captures the essence and key points.

2. EXTRACTIVE SUMMARY: Extract {extractive_sentences} key sentences from the conversation that best represent its content.

Conversation:
{conversation_text[:8000]}

Format your response as:
ABSTRACTIVE:
[your abstractive summary]

EXTRACTIVE:
[key sentence 1]
[key sentence 2]
[key sentence 3]"""

        response = self._call_llm(
            prompt,
            tier=ModelTier.STANDARD,
            max_tokens=600,
            prefer_local=prefer_local
        )

        result_text = response.content

        # Parse response
        abstractive = ""
        extractive = ""

        if "ABSTRACTIVE:" in result_text and "EXTRACTIVE:" in result_text:
            parts = result_text.split("EXTRACTIVE:")
            abstractive = parts[0].replace("ABSTRACTIVE:", "").strip()
            extractive = parts[1].strip()
        else:
            abstractive = result_text.strip()
            extractive = result_text[:500]

        return {
            "abstractive": abstractive[:target_length + 50],
            "extractive": extractive[:1000]
        }

    def _extract_tags_and_topics(
        self,
        conversation_text: str,
        prefer_local: bool = False
    ) -> Dict[str, Any]:
        """Extract tags, key topics, and primary domain."""
        max_tags = self.enrichment_config.get("tags", {}).get("max_tags", 10)
        min_tags = self.enrichment_config.get("tags", {}).get("min_tags", 3)

        prompt = f"""Analyze this conversation and extract:

1. PRIMARY DOMAIN: The main category (e.g., Business Strategy, Technical, Creative Writing, Personal Development, Science, etc.)

2. TAGS: {min_tags}-{max_tags} relevant tags/keywords

3. KEY TOPICS: 3-5 main topics discussed

Conversation:
{conversation_text[:6000]}

Format your response as:
DOMAIN: [primary domain]
TAGS: tag1, tag2, tag3, ...
TOPICS: topic1, topic2, topic3, ..."""

        response = self._call_llm(
            prompt,
            tier=ModelTier.FAST,
            max_tokens=300,
            prefer_local=prefer_local
        )

        result_text = response.content

        # Parse response
        domain = "Uncategorized"
        tags = []
        topics = []

        for line in result_text.split('\n'):
            line = line.strip()
            if line.startswith("DOMAIN:"):
                domain = line.replace("DOMAIN:", "").strip()
            elif line.startswith("TAGS:"):
                tags_str = line.replace("TAGS:", "").strip()
                tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
            elif line.startswith("TOPICS:"):
                topics_str = line.replace("TOPICS:", "").strip()
                topics = [topic.strip() for topic in topics_str.split(',') if topic.strip()]

        return {
            "primary_domain": domain,
            "tags": tags[:max_tags],
            "key_topics": topics[:5]
        }

    def _calculate_score(
        self,
        conversation_text: str,
        prefer_local: bool = False
    ) -> Dict[str, Any]:
        """Calculate conversation quality/importance score."""
        scale = self.enrichment_config.get("scoring", {}).get("brilliance_scale", 10)

        prompt = f"""Evaluate this conversation's quality, depth, and value on a scale of 1-{scale}.

Consider:
- Depth of thinking and insights
- Practical value and actionability
- Creativity or novelty
- Problem-solving effectiveness
- Overall usefulness

Conversation:
{conversation_text[:6000]}

Format your response as:
SCORE: [number 1-{scale}]
REASONING: [brief explanation of the score]"""

        response = self._call_llm(
            prompt,
            tier=ModelTier.STANDARD,
            max_tokens=200,
            prefer_local=prefer_local
        )

        result_text = response.content

        # Parse response
        score = 5
        reasoning = ""

        for line in result_text.split('\n'):
            line = line.strip()
            if line.startswith("SCORE:"):
                score_str = line.replace("SCORE:", "").strip()
                match = re.search(r'(\d+)', score_str)
                if match:
                    score = min(int(match.group(1)), scale)
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        if not reasoning:
            reasoning = result_text[:200]

        return {"score": score, "reasoning": reasoning}

    def _extract_insights(
        self,
        conversation_text: str,
        prefer_local: bool = False
    ) -> Dict[str, Any]:
        """Extract key insights, status, and future potential."""
        max_insights = self.enrichment_config.get("insights", {}).get("max_insights", 5)

        prompt = f"""Analyze this conversation and extract:

1. KEY INSIGHTS: {max_insights} most important insights or takeaways (bullet points)

2. STATUS: Current state (e.g., Completed, Ongoing, Reference, Planning, Resolved)

3. FUTURE POTENTIAL: Brief description of value proposition and suggested next steps

Conversation:
{conversation_text[:6000]}

Format your response as:
INSIGHTS:
- insight 1
- insight 2
...

STATUS: [status]

FUTURE VALUE: [value proposition]
NEXT STEPS: [suggested actions]"""

        response = self._call_llm(
            prompt,
            tier=ModelTier.STANDARD,
            max_tokens=500,
            prefer_local=prefer_local
        )

        result_text = response.content

        # Parse response
        insights = []
        status = "Completed"
        future_value = ""
        next_steps = ""

        current_section = None
        for line in result_text.split('\n'):
            line = line.strip()

            if line.startswith("INSIGHTS:"):
                current_section = "insights"
            elif line.startswith("STATUS:"):
                status = line.replace("STATUS:", "").strip()
                current_section = None
            elif line.startswith("FUTURE VALUE:"):
                future_value = line.replace("FUTURE VALUE:", "").strip()
                current_section = "future"
            elif line.startswith("NEXT STEPS:"):
                next_steps = line.replace("NEXT STEPS:", "").strip()
                current_section = "steps"
            elif line.startswith("-") or line.startswith("•"):
                if current_section == "insights":
                    insight = line.lstrip("-•").strip()
                    if insight:
                        insights.append(insight)
            elif current_section == "future" and line:
                future_value += " " + line
            elif current_section == "steps" and line:
                next_steps += " " + line

        return {
            "key_insights": insights[:max_insights],
            "status": status,
            "future_potential": {
                "value_proposition": future_value.strip(),
                "next_steps": next_steps.strip()
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get enrichment statistics."""
        chain_stats = self.provider_chain.get_stats()
        pii_stats = self.pii_handler.get_stats()

        return {
            "total_enrichments": self.total_enrichments,
            "failed_enrichments": self.failed_enrichments,
            "success_rate": (
                (self.total_enrichments - self.failed_enrichments) / self.total_enrichments
                if self.total_enrichments > 0 else 0
            ),
            "total_api_calls": self.total_api_calls,
            "total_tokens_used": self.total_tokens_used,
            "total_cost_usd": self.total_cost_usd,
            "pii_scrubbed_count": self.pii_scrubbed_count,
            "provider_chain": chain_stats,
            "pii_handler": pii_stats
        }


def create_pipeline_from_config(config_path: Optional[Path] = None) -> EnrichmentPipelineV2:
    """
    Create an enrichment pipeline from configuration file.

    Args:
        config_path: Path to config YAML (uses default if None)

    Returns:
        Configured EnrichmentPipelineV2 instance
    """
    import yaml

    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "enrichment_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return EnrichmentPipelineV2(config.get("enrichment", config))
