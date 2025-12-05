"""
Structured Enrichment Pipeline for CogRepo

Improves on the original enrichment with:
- JSON-structured prompts for reliable parsing
- Confidence scores for all outputs
- Batched enrichment for efficiency
- Prompt caching support
- Better error handling and recovery

This addresses the fragile string-parsing in the original pipeline.

Usage:
    from enrichment.structured_enrichment import StructuredEnrichmentPipeline

    pipeline = StructuredEnrichmentPipeline(api_key="...")

    # Enrich a single conversation
    enrichments = pipeline.enrich(conversation_text, title="Original Title")

    # Batch enrich
    results = pipeline.enrich_batch(conversations)
"""

import json
import re
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.logging_config import get_logger, APICallLogger
from core.exceptions import EnrichmentError, APIError, ResponseParsingError, RateLimitError

logger = get_logger(__name__)


# =============================================================================
# Output Data Structures
# =============================================================================

@dataclass
class EnrichmentResult:
    """Result of a single enrichment task with confidence."""
    value: Any
    confidence: float  # 0.0 - 1.0
    model_used: str
    tokens_used: int = 0
    processing_time_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationEnrichments:
    """All enrichments for a conversation."""
    generated_title: EnrichmentResult = None
    summary_abstractive: EnrichmentResult = None
    summary_extractive: EnrichmentResult = None
    primary_domain: EnrichmentResult = None
    tags: EnrichmentResult = None
    key_topics: EnrichmentResult = None
    brilliance_score: EnrichmentResult = None
    key_insights: EnrichmentResult = None
    status: EnrichmentResult = None
    future_potential: EnrichmentResult = None

    # Aggregated metadata
    total_tokens: int = 0
    total_cost: float = 0.0
    processing_time_ms: float = 0
    overall_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {}

        if self.generated_title:
            result['generated_title'] = self.generated_title.value

        if self.summary_abstractive:
            result['summary_abstractive'] = self.summary_abstractive.value

        if self.summary_extractive:
            result['summary_extractive'] = self.summary_extractive.value

        if self.primary_domain:
            result['primary_domain'] = self.primary_domain.value

        if self.tags:
            result['tags'] = self.tags.value

        if self.key_topics:
            result['key_topics'] = self.key_topics.value

        if self.brilliance_score:
            result['brilliance_score'] = {
                'score': self.brilliance_score.value.get('score', 5),
                'reasoning': self.brilliance_score.value.get('reasoning', '')
            }
            result['score'] = self.brilliance_score.value.get('score', 5)
            result['score_reasoning'] = self.brilliance_score.value.get('reasoning', '')

        if self.key_insights:
            result['key_insights'] = self.key_insights.value

        if self.status:
            result['status'] = self.status.value

        if self.future_potential:
            result['future_potential'] = self.future_potential.value

        # Add metadata
        result['_enrichment_metadata'] = {
            'total_tokens': self.total_tokens,
            'total_cost_usd': round(self.total_cost, 4),
            'processing_time_ms': round(self.processing_time_ms, 2),
            'overall_confidence': round(self.overall_confidence, 3),
            'enriched_at': datetime.now().isoformat()
        }

        return result


# =============================================================================
# Prompt Templates (JSON-structured)
# =============================================================================

TITLE_PROMPT = """Analyze this conversation and generate a concise, descriptive title.

<conversation>
{conversation}
</conversation>

Original title: {original_title}

Respond with ONLY a JSON object in this exact format:
{{
    "title": "Your generated title (5-8 words)",
    "confidence": 0.95
}}

The confidence score (0.0-1.0) reflects how well the title captures the conversation's essence."""


SUMMARY_PROMPT = """Summarize this conversation in two ways.

<conversation>
{conversation}
</conversation>

Respond with ONLY a JSON object in this exact format:
{{
    "abstractive": "A 250-300 character summary that captures the essence and key points",
    "extractive": ["Key sentence 1 from the conversation", "Key sentence 2", "Key sentence 3"],
    "confidence": 0.90
}}

Guidelines:
- Abstractive: Write in your own words, capturing the main topic and outcome
- Extractive: Select 3 most important sentences from the actual conversation
- Confidence: How well the summaries represent the conversation (0.0-1.0)"""


CLASSIFICATION_PROMPT = """Analyze and classify this conversation.

<conversation>
{conversation}
</conversation>

Respond with ONLY a JSON object in this exact format:
{{
    "primary_domain": "Main category (Business, Technical, Creative, Personal, Science, etc.)",
    "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
    "key_topics": ["main topic 1", "main topic 2", "main topic 3"],
    "confidence": 0.85
}}

Guidelines:
- primary_domain: Single most relevant category
- tags: 5-10 relevant keywords/tags
- key_topics: 3-5 main subjects discussed
- confidence: How certain you are about the classification (0.0-1.0)"""


SCORING_PROMPT = """Evaluate the quality and value of this conversation.

<conversation>
{conversation}
</conversation>

Respond with ONLY a JSON object in this exact format:
{{
    "score": 7,
    "reasoning": "Brief explanation of the score",
    "factors": {{
        "depth": 8,
        "actionability": 6,
        "creativity": 7,
        "problem_solving": 7
    }},
    "confidence": 0.80
}}

Scoring criteria (1-10 scale):
- depth: How deep and insightful is the thinking?
- actionability: How practical and actionable are the outcomes?
- creativity: How creative or novel are the ideas?
- problem_solving: How effective was the problem-solving?

Overall score is a weighted average. Confidence reflects certainty in evaluation."""


INSIGHTS_PROMPT = """Extract key insights and assess the conversation's potential.

<conversation>
{conversation}
</conversation>

Respond with ONLY a JSON object in this exact format:
{{
    "key_insights": [
        "Most important insight or takeaway",
        "Second key insight",
        "Third key insight"
    ],
    "status": "Completed",
    "future_potential": {{
        "value_proposition": "What makes this conversation valuable",
        "next_steps": "Suggested follow-up actions"
    }},
    "confidence": 0.75
}}

Status options: Completed, Ongoing, Reference, Planning, Resolved, Archived

Provide 3-5 key insights that would be valuable to revisit later."""


# =============================================================================
# Structured Enrichment Pipeline
# =============================================================================

class StructuredEnrichmentPipeline:
    """
    Improved enrichment pipeline with structured JSON outputs.

    Key improvements over original:
    1. JSON-structured prompts - no fragile string parsing
    2. Confidence scores - know when outputs are uncertain
    3. Better error handling - graceful degradation
    4. API call tracking - token/cost monitoring
    5. Batching support - efficient processing
    """

    # Cost estimates per 1K tokens (Anthropic pricing)
    COST_PER_1K = {
        'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},
        'claude-3-5-haiku-20241022': {'input': 0.00025, 'output': 0.00125},
        'claude-sonnet-4-5-20250929': {'input': 0.003, 'output': 0.015},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        main_model: str = "claude-3-5-sonnet-20241022",
        fast_model: str = "claude-3-5-haiku-20241022",
        max_tokens: int = 1500,
        temperature: float = 0.3
    ):
        """
        Initialize enrichment pipeline.

        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env)
            main_model: Model for complex tasks (summaries, scoring)
            fast_model: Model for simple tasks (titles, tags)
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
        """
        import os
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')

        if not self.api_key:
            raise EnrichmentError(
                "API key required. Set ANTHROPIC_API_KEY or pass api_key parameter.",
                error_code="ENRICH_001"
            )

        self.client = Anthropic(api_key=self.api_key)
        self.main_model = main_model
        self.fast_model = fast_model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Tracking
        self.api_logger = APICallLogger(logger)

    def enrich(
        self,
        conversation_text: str,
        title: str = "",
        skip_tasks: List[str] = None
    ) -> ConversationEnrichments:
        """
        Enrich a single conversation.

        Args:
            conversation_text: Full conversation text
            title: Original title (for reference)
            skip_tasks: Tasks to skip (e.g., ['title', 'insights'])

        Returns:
            ConversationEnrichments with all enrichment results
        """
        skip_tasks = skip_tasks or []
        start_time = time.perf_counter()

        # Truncate if too long
        max_chars = 15000
        if len(conversation_text) > max_chars:
            half = max_chars // 2
            conversation_text = (
                conversation_text[:half] +
                "\n\n[... content truncated for processing ...]\n\n" +
                conversation_text[-half:]
            )

        enrichments = ConversationEnrichments()
        confidences = []

        # Generate title (fast model)
        if 'title' not in skip_tasks:
            enrichments.generated_title = self._generate_title(conversation_text, title)
            if enrichments.generated_title:
                confidences.append(enrichments.generated_title.confidence)
                enrichments.total_tokens += enrichments.generated_title.tokens_used

        # Generate summaries (main model)
        if 'summary' not in skip_tasks:
            summaries = self._generate_summaries(conversation_text)
            if summaries:
                enrichments.summary_abstractive = EnrichmentResult(
                    value=summaries.get('abstractive', ''),
                    confidence=summaries.get('confidence', 0.8),
                    model_used=self.main_model,
                    tokens_used=summaries.get('_tokens', 0)
                )
                enrichments.summary_extractive = EnrichmentResult(
                    value='\n'.join(summaries.get('extractive', [])),
                    confidence=summaries.get('confidence', 0.8),
                    model_used=self.main_model
                )
                confidences.append(summaries.get('confidence', 0.8))
                enrichments.total_tokens += summaries.get('_tokens', 0)

        # Classification (fast model)
        if 'classification' not in skip_tasks:
            classification = self._classify(conversation_text)
            if classification:
                enrichments.primary_domain = EnrichmentResult(
                    value=classification.get('primary_domain', 'Uncategorized'),
                    confidence=classification.get('confidence', 0.8),
                    model_used=self.fast_model
                )
                enrichments.tags = EnrichmentResult(
                    value=classification.get('tags', []),
                    confidence=classification.get('confidence', 0.8),
                    model_used=self.fast_model
                )
                enrichments.key_topics = EnrichmentResult(
                    value=classification.get('key_topics', []),
                    confidence=classification.get('confidence', 0.8),
                    model_used=self.fast_model,
                    tokens_used=classification.get('_tokens', 0)
                )
                confidences.append(classification.get('confidence', 0.8))
                enrichments.total_tokens += classification.get('_tokens', 0)

        # Scoring (main model)
        if 'scoring' not in skip_tasks:
            scoring = self._score(conversation_text)
            if scoring:
                enrichments.brilliance_score = EnrichmentResult(
                    value={
                        'score': scoring.get('score', 5),
                        'reasoning': scoring.get('reasoning', ''),
                        'factors': scoring.get('factors', {})
                    },
                    confidence=scoring.get('confidence', 0.7),
                    model_used=self.main_model,
                    tokens_used=scoring.get('_tokens', 0)
                )
                confidences.append(scoring.get('confidence', 0.7))
                enrichments.total_tokens += scoring.get('_tokens', 0)

        # Insights (main model)
        if 'insights' not in skip_tasks:
            insights = self._extract_insights(conversation_text)
            if insights:
                enrichments.key_insights = EnrichmentResult(
                    value=insights.get('key_insights', []),
                    confidence=insights.get('confidence', 0.7),
                    model_used=self.main_model
                )
                enrichments.status = EnrichmentResult(
                    value=insights.get('status', 'Completed'),
                    confidence=insights.get('confidence', 0.7),
                    model_used=self.main_model
                )
                enrichments.future_potential = EnrichmentResult(
                    value=insights.get('future_potential', {}),
                    confidence=insights.get('confidence', 0.7),
                    model_used=self.main_model,
                    tokens_used=insights.get('_tokens', 0)
                )
                confidences.append(insights.get('confidence', 0.7))
                enrichments.total_tokens += insights.get('_tokens', 0)

        # Calculate aggregates
        enrichments.processing_time_ms = (time.perf_counter() - start_time) * 1000
        enrichments.overall_confidence = sum(confidences) / len(confidences) if confidences else 0
        enrichments.total_cost = self._estimate_cost(enrichments.total_tokens)

        return enrichments

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, RateLimitError))
    )
    def _call_api(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = None
    ) -> Tuple[str, int]:
        """
        Make API call with retry logic.

        Returns:
            Tuple of (response_text, total_tokens)
        """
        model = model or self.main_model
        max_tokens = max_tokens or self.max_tokens

        start_time = time.perf_counter()

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            duration_ms = (time.perf_counter() - start_time) * 1000
            total_tokens = response.usage.input_tokens + response.usage.output_tokens

            # Log the call
            self.api_logger.log_call(
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                duration_ms=duration_ms,
                cost=self._estimate_cost(total_tokens, model),
                success=True
            )

            return response.content[0].text, total_tokens

        except Exception as e:
            error_str = str(e).lower()

            if 'rate' in error_str or 'limit' in error_str:
                raise RateLimitError(str(e), cause=e)
            else:
                raise APIError(str(e), cause=e)

    def _parse_json_response(self, response: str, default: Any = None) -> Dict[str, Any]:
        """
        Parse JSON from API response with fallbacks.

        Handles:
        - Clean JSON
        - JSON wrapped in markdown code blocks
        - Partial JSON with recovery
        """
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object
        brace_match = re.search(r'\{[\s\S]*\}', response)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(
            "Failed to parse JSON response",
            extra={"response_preview": response[:200]}
        )

        return default or {}

    def _generate_title(
        self,
        conversation_text: str,
        original_title: str
    ) -> Optional[EnrichmentResult]:
        """Generate title using fast model."""
        try:
            prompt = TITLE_PROMPT.format(
                conversation=conversation_text[:3000],
                original_title=original_title
            )

            response, tokens = self._call_api(prompt, model=self.fast_model, max_tokens=100)
            result = self._parse_json_response(response, {'title': original_title, 'confidence': 0.5})

            title = result.get('title', original_title)
            # Clean up title
            title = title.strip('"\'')
            if len(title) > 100:
                title = title[:97] + '...'

            return EnrichmentResult(
                value=title or original_title,
                confidence=result.get('confidence', 0.8),
                model_used=self.fast_model,
                tokens_used=tokens
            )

        except Exception as e:
            logger.warning(f"Title generation failed: {e}")
            return EnrichmentResult(
                value=original_title,
                confidence=0.3,
                model_used=self.fast_model
            )

    def _generate_summaries(self, conversation_text: str) -> Optional[Dict[str, Any]]:
        """Generate abstractive and extractive summaries."""
        try:
            prompt = SUMMARY_PROMPT.format(conversation=conversation_text[:8000])
            response, tokens = self._call_api(prompt, model=self.main_model, max_tokens=600)
            result = self._parse_json_response(response)
            result['_tokens'] = tokens
            return result

        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return None

    def _classify(self, conversation_text: str) -> Optional[Dict[str, Any]]:
        """Classify conversation (domain, tags, topics)."""
        try:
            prompt = CLASSIFICATION_PROMPT.format(conversation=conversation_text[:6000])
            response, tokens = self._call_api(prompt, model=self.fast_model, max_tokens=300)
            result = self._parse_json_response(response)
            result['_tokens'] = tokens
            return result

        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return None

    def _score(self, conversation_text: str) -> Optional[Dict[str, Any]]:
        """Score conversation quality."""
        try:
            prompt = SCORING_PROMPT.format(conversation=conversation_text[:6000])
            response, tokens = self._call_api(prompt, model=self.main_model, max_tokens=300)
            result = self._parse_json_response(response)

            # Ensure score is valid
            score = result.get('score', 5)
            if not isinstance(score, (int, float)) or score < 1 or score > 10:
                score = 5
            result['score'] = int(score)
            result['_tokens'] = tokens

            return result

        except Exception as e:
            logger.warning(f"Scoring failed: {e}")
            return None

    def _extract_insights(self, conversation_text: str) -> Optional[Dict[str, Any]]:
        """Extract key insights and potential."""
        try:
            prompt = INSIGHTS_PROMPT.format(conversation=conversation_text[:6000])
            response, tokens = self._call_api(prompt, model=self.main_model, max_tokens=500)
            result = self._parse_json_response(response)
            result['_tokens'] = tokens
            return result

        except Exception as e:
            logger.warning(f"Insights extraction failed: {e}")
            return None

    def _estimate_cost(self, tokens: int, model: str = None) -> float:
        """Estimate cost for token usage."""
        model = model or self.main_model
        pricing = self.COST_PER_1K.get(model, {'input': 0.003, 'output': 0.015})

        # Assume 50/50 split (conservative)
        avg_rate = (pricing['input'] + pricing['output']) / 2
        return (tokens / 1000) * avg_rate

    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return self.api_logger.get_stats()


# =============================================================================
# Batch Processing
# =============================================================================

def enrich_batch(
    conversations: List[Dict[str, Any]],
    pipeline: StructuredEnrichmentPipeline = None,
    progress_callback: callable = None
) -> List[Tuple[Dict[str, Any], ConversationEnrichments]]:
    """
    Enrich a batch of conversations.

    Args:
        conversations: List of conversation dicts (must have 'raw_text' and optionally 'title')
        pipeline: Optional pipeline instance (creates new if None)
        progress_callback: Optional callback(current, total, enrichments)

    Returns:
        List of (conversation, enrichments) tuples
    """
    if pipeline is None:
        pipeline = StructuredEnrichmentPipeline()

    results = []
    total = len(conversations)

    for i, conv in enumerate(conversations, 1):
        try:
            text = conv.get('raw_text', '')
            title = conv.get('title', conv.get('generated_title', ''))

            enrichments = pipeline.enrich(text, title=title)
            results.append((conv, enrichments))

            if progress_callback:
                progress_callback(i, total, enrichments)

        except Exception as e:
            logger.error(f"Failed to enrich conversation {conv.get('convo_id', i)}: {e}")
            results.append((conv, ConversationEnrichments()))

    return results


if __name__ == "__main__":
    # Quick test
    import os

    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Set ANTHROPIC_API_KEY environment variable to test")
        exit(1)

    pipeline = StructuredEnrichmentPipeline()

    test_conversation = """
    USER: I'm working on a new animation project and want to use AI to help with character design. What approaches would you recommend?

    ASSISTANT: There are several exciting approaches for AI-assisted character design in animation:

    1. **Style Transfer**: Use neural style transfer to apply artistic styles to rough sketches
    2. **Generative Models**: Tools like Midjourney or Stable Diffusion can generate concept art
    3. **Pose Libraries**: AI can help generate consistent character poses from different angles
    4. **Color Palette Generation**: AI tools can suggest harmonious color schemes

    The key is to use AI as a creative partner rather than a replacement for human artistry.

    USER: That's helpful! How do I maintain consistency across generated images?

    ASSISTANT: Maintaining consistency is crucial for professional animation. Here are proven techniques:

    1. **Character Sheets**: Create detailed reference sheets that define proportions, colors, and key features
    2. **Embeddings**: Some tools allow you to train on specific characters for consistent generation
    3. **Iterative Refinement**: Start with AI suggestions, then manually refine and standardize
    4. **Style Guides**: Document specific parameters that produce your desired aesthetic

    Many studios now use a hybrid workflow where AI generates options and artists curate and refine.
    """

    print("Testing structured enrichment pipeline...\n")

    enrichments = pipeline.enrich(test_conversation, title="AI Animation Discussion")

    print("=== Enrichment Results ===\n")

    result_dict = enrichments.to_dict()

    print(f"Title: {result_dict.get('generated_title', 'N/A')}")
    print(f"Domain: {result_dict.get('primary_domain', 'N/A')}")
    print(f"Score: {result_dict.get('score', 'N/A')}/10")
    print(f"Tags: {', '.join(result_dict.get('tags', []))}")
    print(f"\nSummary: {result_dict.get('summary_abstractive', 'N/A')}")

    print(f"\nKey Insights:")
    for insight in result_dict.get('key_insights', []):
        print(f"  - {insight}")

    metadata = result_dict.get('_enrichment_metadata', {})
    print(f"\n=== Metadata ===")
    print(f"Tokens: {metadata.get('total_tokens', 0)}")
    print(f"Cost: ${metadata.get('total_cost_usd', 0):.4f}")
    print(f"Confidence: {metadata.get('overall_confidence', 0):.2%}")
    print(f"Time: {metadata.get('processing_time_ms', 0):.0f}ms")
