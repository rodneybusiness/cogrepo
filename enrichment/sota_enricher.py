"""
SOTA 2026 Enrichment Engine

Uses best-in-class models for maximum quality:
- Claude Sonnet 4 for text generation (titles, summaries, tags)
- OpenAI text-embedding-3-large for semantic embeddings (1536d)
- Streaming for real-time UI updates
- Preview/approval workflow
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, AsyncIterator
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from anthropic import Anthropic, AsyncAnthropic
import openai

# Cost tracking
CLAUDE_SONNET_4_INPUT_COST = 3.0  # $ per MTok
CLAUDE_SONNET_4_OUTPUT_COST = 15.0  # $ per MTok
EMBEDDING_3_LARGE_COST = 0.13  # $ per MTok


@dataclass
class EnrichmentResult:
    """Result of enrichment operation."""
    conversation_id: str
    field: str
    value: Any
    original_value: Any
    confidence: float
    cost: float
    model: str
    timestamp: str
    partial: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class EnrichmentDiff:
    """Diff between original and enriched conversation."""
    conversation_id: str
    changes: List[Dict[str, Any]]
    total_cost: float
    models_used: List[str]
    timestamp: str

    def to_dict(self):
        return asdict(self)


class SOTAEnricher:
    """
    State-of-the-art enrichment using best 2026 models.

    Features:
    - Streaming generation for immediate feedback
    - High-quality embeddings (1536d vs 384d)
    - Cost tracking and estimation
    - Preview mode (don't persist)
    """

    def __init__(
        self,
        anthropic_key: Optional[str] = None,
        openai_key: Optional[str] = None
    ):
        self.anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")

        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY required")
        # OpenAI key is optional - only needed for embeddings
        # if not self.openai_key:
        #     raise ValueError("OPENAI_API_KEY required for embeddings")

        self.anthropic = AsyncAnthropic(api_key=self.anthropic_key)
        openai.api_key = self.openai_key

        # Model configs
        self.text_model = "claude-sonnet-4-20250514"
        self.embedding_model = "text-embedding-3-large"

    async def enrich_single_streaming(
        self,
        conversation: Dict[str, Any],
        fields: List[str]
    ) -> AsyncIterator[EnrichmentResult]:
        """
        Enrich a single conversation with streaming results.

        Yields EnrichmentResult objects as each field is processed.
        """
        conversation_id = conversation.get("convo_id", "unknown")

        for field in fields:
            if field == "title":
                async for result in self._enrich_title_streaming(conversation, conversation_id):
                    yield result

            elif field == "summary":
                async for result in self._enrich_summary_streaming(conversation, conversation_id):
                    yield result

            elif field == "tags":
                result = await self._enrich_tags(conversation, conversation_id)
                yield result

            elif field == "embedding":
                result = await self._enrich_embedding(conversation, conversation_id)
                yield result

    async def _enrich_title_streaming(
        self,
        conversation: Dict[str, Any],
        conversation_id: str
    ) -> AsyncIterator[EnrichmentResult]:
        """Generate title with streaming."""

        # Build prompt
        raw_text = conversation.get("raw_text", "")[:3000]
        existing_title = conversation.get("generated_title", "")

        prompt = f"""Analyze this conversation and generate a concise, descriptive title.

CONVERSATION:
{raw_text}

CURRENT TITLE: {existing_title if existing_title else "None"}

INSTRUCTIONS:
- Output ONLY the title text, nothing else
- Do NOT include explanations like "This title is better because..."
- Do NOT use quotes around the title
- Make it 40-80 characters
- Capture the main topic/purpose
- Be specific and descriptive
- Use clear, professional language

Title:"""

        # Stream generation
        accumulated_title = ""
        input_tokens = len(prompt.split()) * 1.3  # Rough estimate
        output_tokens = 0

        async with self.anthropic.messages.stream(
            model=self.text_model,
            max_tokens=100,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            async for text in stream.text_stream:
                accumulated_title += text
                output_tokens += 1

                # Yield partial result
                yield EnrichmentResult(
                    conversation_id=conversation_id,
                    field="title",
                    value=accumulated_title.strip(),
                    original_value=existing_title,
                    confidence=0.85,
                    cost=0.0,  # Will calculate at end
                    model=self.text_model,
                    timestamp=datetime.utcnow().isoformat(),
                    partial=True
                )

        # Calculate final cost
        cost = (
            (input_tokens / 1_000_000) * CLAUDE_SONNET_4_INPUT_COST +
            (output_tokens / 1_000_000) * CLAUDE_SONNET_4_OUTPUT_COST
        )

        # Clean up title - remove explanations, quotes, etc.
        final_title = accumulated_title.strip()

        # Remove common explanation patterns
        if "This title" in final_title or "because" in final_title.lower():
            # Extract just the title part before any explanation
            parts = final_title.split('\n')
            final_title = parts[0].strip()

        # Remove quotes if present
        final_title = final_title.strip('"\'')

        # Truncate to 80 chars if too long
        if len(final_title) > 80:
            final_title = final_title[:77] + "..."

        # Final result
        yield EnrichmentResult(
            conversation_id=conversation_id,
            field="title",
            value=final_title,
            original_value=existing_title,
            confidence=0.9,
            cost=cost,
            model=self.text_model,
            timestamp=datetime.utcnow().isoformat(),
            partial=False
        )

    async def _enrich_summary_streaming(
        self,
        conversation: Dict[str, Any],
        conversation_id: str
    ) -> AsyncIterator[EnrichmentResult]:
        """Generate summary with streaming."""

        raw_text = conversation.get("raw_text", "")[:5000]
        existing_summary = conversation.get("summary_abstractive", "")

        prompt = f"""Create a comprehensive 2-3 sentence summary of this conversation.

CONVERSATION:
{raw_text}

CURRENT SUMMARY: {existing_summary if existing_summary else "None"}

Generate a better summary that:
1. Captures key points and outcomes
2. Highlights important insights or decisions
3. Uses clear, concise language
4. Is 2-3 sentences (100-200 words)

Summary:"""

        accumulated_summary = ""
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = 0

        async with self.anthropic.messages.stream(
            model=self.text_model,
            max_tokens=300,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            async for text in stream.text_stream:
                accumulated_summary += text
                output_tokens += 1

                yield EnrichmentResult(
                    conversation_id=conversation_id,
                    field="summary",
                    value=accumulated_summary.strip(),
                    original_value=existing_summary,
                    confidence=0.85,
                    cost=0.0,
                    model=self.text_model,
                    timestamp=datetime.utcnow().isoformat(),
                    partial=True
                )

        cost = (
            (input_tokens / 1_000_000) * CLAUDE_SONNET_4_INPUT_COST +
            (output_tokens / 1_000_000) * CLAUDE_SONNET_4_OUTPUT_COST
        )

        yield EnrichmentResult(
            conversation_id=conversation_id,
            field="summary",
            value=accumulated_summary.strip(),
            original_value=existing_summary,
            confidence=0.9,
            cost=cost,
            model=self.text_model,
            timestamp=datetime.utcnow().isoformat(),
            partial=False
        )

    async def _enrich_tags(
        self,
        conversation: Dict[str, Any],
        conversation_id: str
    ) -> EnrichmentResult:
        """Generate high-quality tags."""

        raw_text = conversation.get("raw_text", "")[:5000]
        existing_tags = conversation.get("tags", [])

        prompt = f"""Extract 10-15 precise, descriptive tags from this conversation.

CONVERSATION:
{raw_text}

CURRENT TAGS: {', '.join(existing_tags) if existing_tags else "None"}

Generate better tags that are:
1. Specific and descriptive (not generic)
2. Technical concepts, frameworks, methodologies
3. Key topics and domain terminology
4. Problem types and solution patterns

Format: Return comma-separated list only.

Tags:"""

        response = await self.anthropic.messages.create(
            model=self.text_model,
            max_tokens=200,
            temperature=0.2,  # Lower for consistency
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse tags
        tags_text = response.content[0].text.strip()
        new_tags = [
            tag.strip()
            for tag in tags_text.split(',')
            if tag.strip() and len(tag.strip()) > 2
        ][:15]

        # Calculate cost
        cost = (
            (response.usage.input_tokens / 1_000_000) * CLAUDE_SONNET_4_INPUT_COST +
            (response.usage.output_tokens / 1_000_000) * CLAUDE_SONNET_4_OUTPUT_COST
        )

        return EnrichmentResult(
            conversation_id=conversation_id,
            field="tags",
            value=new_tags,
            original_value=existing_tags,
            confidence=0.88,
            cost=cost,
            model=self.text_model,
            timestamp=datetime.utcnow().isoformat(),
            partial=False
        )

    async def _enrich_embedding(
        self,
        conversation: Dict[str, Any],
        conversation_id: str
    ) -> EnrichmentResult:
        """Generate SOTA embedding (1536 dimensions)."""

        # Prepare text for embedding
        title = conversation.get("generated_title", "")
        summary = conversation.get("summary_abstractive", "")
        tags = conversation.get("tags", [])

        # Combine fields for rich embedding
        text = f"{title}. {summary}"
        if tags:
            text += f" Tags: {', '.join(tags[:10])}"

        # Truncate to ~8000 tokens
        text = text[:30000]

        # Generate embedding using OpenAI's best model
        response = await asyncio.to_thread(
            openai.embeddings.create,
            model=self.embedding_model,
            input=text,
            dimensions=1536  # Full dimensionality
        )

        embedding = response.data[0].embedding

        # Calculate cost
        tokens = response.usage.total_tokens
        cost = (tokens / 1_000_000) * EMBEDDING_3_LARGE_COST

        return EnrichmentResult(
            conversation_id=conversation_id,
            field="embedding",
            value=embedding,
            original_value=None,  # Don't include old embedding (too large)
            confidence=0.95,
            cost=cost,
            model=self.embedding_model,
            timestamp=datetime.utcnow().isoformat(),
            partial=False
        )

    async def enrich_single(
        self,
        conversation: Dict[str, Any],
        fields: List[str]
    ) -> Dict[str, Any]:
        """
        Enrich a single conversation (non-streaming).

        Returns complete enriched conversation data.
        """
        results = {}
        total_cost = 0.0

        async for result in self.enrich_single_streaming(conversation, fields):
            if not result.partial:
                results[result.field] = result.value
                total_cost += result.cost

        return {
            "enriched_data": results,
            "total_cost": total_cost,
            "timestamp": datetime.utcnow().isoformat()
        }

    def calculate_diff(
        self,
        original: Dict[str, Any],
        enriched: Dict[str, Any]
    ) -> EnrichmentDiff:
        """Calculate diff between original and enriched."""

        changes = []

        for field in enriched:
            original_value = original.get(field)
            new_value = enriched[field]

            if original_value != new_value:
                change = {
                    "field": field,
                    "original": original_value,
                    "new": new_value,
                    "change_type": self._determine_change_type(field, original_value, new_value)
                }
                changes.append(change)

        return EnrichmentDiff(
            conversation_id=original.get("convo_id", "unknown"),
            changes=changes,
            total_cost=0.0,  # Will be filled by caller
            models_used=[self.text_model, self.embedding_model],
            timestamp=datetime.utcnow().isoformat()
        )

    def _determine_change_type(self, field: str, old: Any, new: Any) -> str:
        """Determine type of change."""
        if old is None or old == "" or old == []:
            return "added"
        elif new is None or new == "" or new == []:
            return "removed"
        elif isinstance(old, list) and isinstance(new, list):
            added = set(new) - set(old)
            removed = set(old) - set(new)
            if added and removed:
                return "modified"
            elif added:
                return "added_items"
            elif removed:
                return "removed_items"
        return "modified"
