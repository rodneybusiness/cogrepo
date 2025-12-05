"""
Main enrichment pipeline for processing conversations with AI.

Coordinates all enrichment tasks:
- Title generation
- Summarization (abstractive and extractive)
- Tag extraction
- Topic identification
- Scoring and insights
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
from datetime import datetime
import re

from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models import RawConversation, EnrichedConversation


class EnrichmentPipeline:
    """
    AI-powered enrichment pipeline for conversations.

    Uses Claude API to generate metadata for conversations.
    """

    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        """
        Initialize enrichment pipeline.

        Args:
            config: Configuration dictionary from YAML
            api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY env var)
        """
        self.config = config
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = Anthropic(api_key=self.api_key)

        # Get config settings
        self.api_config = config.get("api", {})
        self.processing_config = config.get("processing", {})
        self.enrichment_config = config.get("enrichment", {})

        # Model selection
        self.main_model = self.api_config.get("model", "claude-3-5-sonnet-20241022")
        self.haiku_model = self.api_config.get("haiku_model", "claude-3-5-haiku-20241022")
        self.use_haiku_for_simple = self.api_config.get("use_haiku_for_simple_tasks", True)

        # Processing settings
        self.max_tokens = self.api_config.get("max_tokens", 1500)
        self.temperature = self.api_config.get("temperature", 0.3)

        # Statistics
        self.total_api_calls = 0
        self.total_tokens_used = 0
        self.failed_enrichments = 0

    def enrich_conversation(self, raw: RawConversation) -> EnrichedConversation:
        """
        Enrich a single conversation with AI-generated metadata.

        Args:
            raw: RawConversation object

        Returns:
            EnrichedConversation object

        Raises:
            Exception: If enrichment fails after retries
        """
        # Check if conversation meets minimum requirements
        if not self._should_enrich(raw):
            # Return minimal enrichment
            return EnrichedConversation.from_raw(raw, enrichments=None)

        # Prepare conversation text
        conversation_text = self._prepare_conversation_text(raw)

        # Generate enrichments
        enrichments = {}

        try:
            # Generate title
            if self.enrichment_config.get("generate_titles", True):
                enrichments["generated_title"] = self._generate_title(conversation_text, raw.title)

            # Generate summaries
            if self.enrichment_config.get("generate_summaries", True):
                summaries = self._generate_summaries(conversation_text)
                enrichments["summary_abstractive"] = summaries["abstractive"]
                enrichments["summary_extractive"] = summaries["extractive"]

            # Extract tags and topics
            if self.enrichment_config.get("extract_tags", True):
                tags_topics = self._extract_tags_and_topics(conversation_text)
                enrichments["tags"] = tags_topics["tags"]
                enrichments["key_topics"] = tags_topics["key_topics"]
                enrichments["primary_domain"] = tags_topics["primary_domain"]

            # Calculate score and extract insights
            if self.enrichment_config.get("calculate_scores", True):
                scoring = self._calculate_score(conversation_text)
                enrichments["brilliance_score"] = {
                    "score": scoring["score"],
                    "reasoning": scoring["reasoning"]
                }
                enrichments["score"] = scoring["score"]
                enrichments["score_reasoning"] = scoring["reasoning"]

            # Extract key insights
            if self.enrichment_config.get("extract_insights", True):
                insights = self._extract_insights(conversation_text)
                enrichments["key_insights"] = insights["key_insights"]
                enrichments["status"] = insights["status"]
                enrichments["future_potential"] = insights["future_potential"]

            # Create enriched conversation
            return EnrichedConversation.from_raw(raw, enrichments=enrichments)

        except Exception as e:
            self.failed_enrichments += 1
            print(f"Warning: Enrichment failed for conversation {raw.external_id}: {e}")
            # Return minimal enrichment on failure
            return EnrichedConversation.from_raw(raw, enrichments=None)

    def _should_enrich(self, raw: RawConversation) -> bool:
        """Check if conversation meets minimum requirements for enrichment."""
        min_length = self.processing_config.get("min_conversation_length_chars", 100)
        min_messages = self.processing_config.get("min_message_count", 2)

        if len(raw.raw_text) < min_length:
            return False

        if len(raw.messages) < min_messages:
            return False

        return True

    def _prepare_conversation_text(self, raw: RawConversation, max_chars: int = 15000) -> str:
        """
        Prepare conversation text for API calls.

        Truncates if too long to stay within context limits.
        """
        text = raw.raw_text

        if len(text) > max_chars:
            # Truncate but keep beginning and end
            half = max_chars // 2
            text = text[:half] + "\n\n[... middle truncated ...]\n\n" + text[-half:]

        return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _generate_title(self, conversation_text: str, original_title: str) -> str:
        """
        Generate a meaningful title for the conversation.

        Uses Haiku model for cost efficiency.
        """
        if not conversation_text.strip():
            return original_title or "Empty Conversation"

        use_haiku = self.enrichment_config.get("title", {}).get("use_haiku", True)
        model = self.haiku_model if use_haiku else self.main_model

        prompt = f"""Analyze this conversation and generate a concise, descriptive title (5-8 words).

The title should capture the main topic or purpose of the conversation.

Conversation:
{conversation_text[:3000]}

Original title: {original_title}

Generate ONLY the title text, no quotes or formatting."""

        response = self.client.messages.create(
            model=model,
            max_tokens=100,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        self.total_api_calls += 1
        self.total_tokens_used += response.usage.input_tokens + response.usage.output_tokens

        title = response.content[0].text.strip()

        # Remove quotes if present
        title = title.strip('"').strip("'")

        # Truncate if too long
        max_length = self.enrichment_config.get("title", {}).get("max_length", 100)
        if len(title) > max_length:
            title = title[:max_length-3] + "..."

        return title or original_title

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _generate_summaries(self, conversation_text: str) -> Dict[str, str]:
        """
        Generate abstractive and extractive summaries.
        """
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

        use_haiku = self.enrichment_config.get("summary", {}).get("use_haiku", False)
        model = self.haiku_model if use_haiku else self.main_model

        response = self.client.messages.create(
            model=model,
            max_tokens=600,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        self.total_api_calls += 1
        self.total_tokens_used += response.usage.input_tokens + response.usage.output_tokens

        result_text = response.content[0].text

        # Parse response
        abstractive = ""
        extractive = ""

        if "ABSTRACTIVE:" in result_text and "EXTRACTIVE:" in result_text:
            parts = result_text.split("EXTRACTIVE:")
            abstractive = parts[0].replace("ABSTRACTIVE:", "").strip()
            extractive = parts[1].strip()
        else:
            # Fallback: use entire response as abstractive
            abstractive = result_text.strip()
            extractive = result_text[:500]

        return {
            "abstractive": abstractive[:target_length+50],  # Allow slight overflow
            "extractive": extractive[:1000]
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _extract_tags_and_topics(self, conversation_text: str) -> Dict[str, Any]:
        """
        Extract tags, key topics, and primary domain.
        """
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

        use_haiku = self.enrichment_config.get("tags", {}).get("use_haiku", True)
        model = self.haiku_model if use_haiku else self.main_model

        response = self.client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        self.total_api_calls += 1
        self.total_tokens_used += response.usage.input_tokens + response.usage.output_tokens

        result_text = response.content[0].text

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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _calculate_score(self, conversation_text: str) -> Dict[str, Any]:
        """
        Calculate conversation quality/importance score.
        """
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

        use_haiku = self.enrichment_config.get("scoring", {}).get("use_haiku", False)
        model = self.haiku_model if use_haiku else self.main_model

        response = self.client.messages.create(
            model=model,
            max_tokens=200,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        self.total_api_calls += 1
        self.total_tokens_used += response.usage.input_tokens + response.usage.output_tokens

        result_text = response.content[0].text

        # Parse response
        score = 5  # Default
        reasoning = ""

        for line in result_text.split('\n'):
            line = line.strip()
            if line.startswith("SCORE:"):
                score_str = line.replace("SCORE:", "").strip()
                # Extract number
                match = re.search(r'(\d+)', score_str)
                if match:
                    score = min(int(match.group(1)), scale)
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        if not reasoning:
            reasoning = result_text[:200]

        return {
            "score": score,
            "reasoning": reasoning
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _extract_insights(self, conversation_text: str) -> Dict[str, Any]:
        """
        Extract key insights, status, and future potential.
        """
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

        use_haiku = self.enrichment_config.get("insights", {}).get("use_haiku", False)
        model = self.haiku_model if use_haiku else self.main_model

        response = self.client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        self.total_api_calls += 1
        self.total_tokens_used += response.usage.input_tokens + response.usage.output_tokens

        result_text = response.content[0].text

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
        return {
            "total_api_calls": self.total_api_calls,
            "total_tokens_used": self.total_tokens_used,
            "failed_enrichments": self.failed_enrichments,
            "estimated_cost_usd": self._estimate_cost()
        }

    def _estimate_cost(self) -> float:
        """
        Estimate API cost based on token usage.

        Rough estimates:
        - Sonnet: $3/1M input, $15/1M output
        - Haiku: $0.25/1M input, $1.25/1M output

        Simplified: assume 50/50 split and average cost
        """
        # Conservative estimate assuming mostly Sonnet
        avg_cost_per_1k_tokens = 0.01  # ~$10 per million tokens
        return (self.total_tokens_used / 1000) * avg_cost_per_1k_tokens
