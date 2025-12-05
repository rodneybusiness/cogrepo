#!/usr/bin/env python3
"""
High-Quality Tag Enrichment Script

Adds exceptional tags to conversations that are missing them.
Uses Claude Sonnet 4 for maximum quality and detail.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import time
from anthropic import Anthropic
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from models import EnrichedConversation


class TagEnricher:
    """Generate high-quality tags for conversations."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"  # Use Sonnet 4 for quality
        self.total_api_calls = 0
        self.total_cost = 0.0

    def generate_tags(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate rich, detailed tags for a conversation.

        Aims to match or exceed the quality of ChatGPT conversation tags.
        """
        # Build conversation text
        conv_text = self._build_conversation_text(conversation)

        # Enhanced prompt for exceptional tag quality
        prompt = f"""Analyze this conversation and extract comprehensive, high-quality metadata.

CONVERSATION:
{conv_text[:8000]}

Generate:

1. PRIMARY DOMAIN: The main category (be specific - e.g., "Software Architecture", "Machine Learning", "Film Financing", "Creative Writing")

2. TAGS (10-15 tags): Be specific, descriptive, and comprehensive. Include:
   - Technical concepts or frameworks mentioned
   - Key topics and subtopics
   - Methodologies or approaches discussed
   - Tools, technologies, or platforms referenced
   - Problem types or solution patterns
   - Domain-specific terminology

   Examples of GOOD tags:
   - "TypeScript", "React Hooks", "State Management", "Performance Optimization"
   - "Film Budgeting", "Independent Production", "Distribution Strategy"
   - "Narrative Structure", "Character Development", "Third Act Resolution"

   Examples of BAD tags (too generic):
   - "Programming", "Movies", "Writing"

3. KEY TOPICS (5-7 topics): Main discussion points, each 2-4 words

4. SUBTOPICS (3-5): Specific aspects explored in detail

Format as:
DOMAIN: [specific primary domain]
TAGS: tag1, tag2, tag3, ...
TOPICS: topic1, topic2, topic3, ...
SUBTOPICS: subtopic1, subtopic2, subtopic3, ..."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            self.total_api_calls += 1

            # Calculate cost (Sonnet pricing: $3/MTok input, $15/MTok output)
            input_cost = (response.usage.input_tokens / 1_000_000) * 3
            output_cost = (response.usage.output_tokens / 1_000_000) * 15
            self.total_cost += input_cost + output_cost

            result_text = response.content[0].text

            # Parse response
            return self._parse_tag_response(result_text)

        except Exception as e:
            print(f"Warning: Tag generation failed: {e}")
            return {
                "primary_domain": "Uncategorized",
                "tags": [],
                "key_topics": [],
                "subtopics": []
            }

    def _build_conversation_text(self, conversation: Dict[str, Any]) -> str:
        """Build conversation text from messages."""
        parts = []

        # Add title if available
        title = conversation.get("title") or conversation.get("generated_title", "")
        if title:
            parts.append(f"Title: {title}\n")

        # Add messages
        messages = conversation.get("messages", [])
        for i, msg in enumerate(messages[:20]):  # Limit to first 20 messages
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}\n")

        return "\n".join(parts)

    def _parse_tag_response(self, result_text: str) -> Dict[str, Any]:
        """Parse AI response into structured metadata."""
        domain = "Uncategorized"
        tags = []
        topics = []
        subtopics = []

        for line in result_text.split('\n'):
            line = line.strip()

            if line.startswith("DOMAIN:"):
                domain = line.replace("DOMAIN:", "").strip()

            elif line.startswith("TAGS:"):
                tags_str = line.replace("TAGS:", "").strip()
                tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
                # Clean up tags
                tags = [tag for tag in tags if len(tag) > 1 and len(tag) < 50]

            elif line.startswith("TOPICS:"):
                topics_str = line.replace("TOPICS:", "").strip()
                topics = [topic.strip() for topic in topics_str.split(',') if topic.strip()]

            elif line.startswith("SUBTOPICS:"):
                subtopics_str = line.replace("SUBTOPICS:", "").strip()
                subtopics = [sub.strip() for sub in subtopics_str.split(',') if sub.strip()]

        return {
            "primary_domain": domain,
            "tags": tags[:15],  # Keep top 15
            "key_topics": topics[:7],
            "subtopics": subtopics[:5]
        }


def load_conversations(file_path: Path) -> List[Dict[str, Any]]:
    """Load conversations from JSONL file."""
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    return conversations


def save_conversations(conversations: List[Dict[str, Any]], file_path: Path):
    """Save conversations to JSONL file."""
    # Backup original
    backup_path = file_path.parent / f"{file_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    if file_path.exists():
        import shutil
        shutil.copy(file_path, backup_path)
        print(f"✓ Backup created: {backup_path}")

    # Write updated conversations
    with open(file_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')


def main():
    """Main execution."""
    print("=" * 80)
    print("CogRepo Tag Enrichment - High Quality Mode")
    print("=" * 80)
    print()

    # Initialize
    repo_file = Path(__file__).parent / "data" / "enriched_repository.jsonl"

    if not repo_file.exists():
        print(f"Error: Repository file not found: {repo_file}")
        sys.exit(1)

    print(f"📁 Loading conversations from: {repo_file}")
    conversations = load_conversations(repo_file)
    print(f"✓ Loaded {len(conversations)} total conversations")
    print()

    # Filter conversations needing tags
    needs_tags = []
    for conv in conversations:
        tags = conv.get("tags", [])
        source = conv.get("source", "")

        # Need tags if: empty tags AND (Claude or Gemini source)
        if (not tags or len(tags) == 0) and source in ["Anthropic", "Google"]:
            needs_tags.append(conv)

    print(f"📊 Analysis:")
    print(f"  Conversations needing tags: {len(needs_tags)}")
    print(f"  Claude (Anthropic): {sum(1 for c in needs_tags if c.get('source') == 'Anthropic')}")
    print(f"  Gemini (Google): {sum(1 for c in needs_tags if c.get('source') == 'Google')}")
    print()

    if not needs_tags:
        print("✓ All conversations already have tags!")
        return

    # Estimate cost
    # Sonnet: ~$3/MTok input, ~$15/MTok output
    # Avg: 2K input tokens + 150 output tokens per conversation
    # Cost per conv: (2000/1M * $3) + (150/1M * $15) = $0.006 + $0.00225 = ~$0.008
    estimated_cost = len(needs_tags) * 0.008
    print(f"💰 Estimated cost: ${estimated_cost:.2f} ({len(needs_tags)} conversations × $0.008)")
    print(f"⏱️  Estimated time: {len(needs_tags) * 3 / 60:.1f} minutes")
    print()

    # Auto-proceed (no confirmation needed for automation)
    print("✓ Auto-proceeding with tag enrichment...")
    print()

    print()
    print("🚀 Starting tag enrichment...")
    print()

    # Initialize enricher
    enricher = TagEnricher()

    # Process conversations
    enriched_count = 0
    failed_count = 0

    for conv in tqdm(needs_tags, desc="Enriching tags", unit="conv"):
        try:
            # Generate tags
            tag_data = enricher.generate_tags(conv)

            # Update conversation
            conv["tags"] = tag_data["tags"]
            conv["key_topics"] = tag_data["key_topics"]
            conv["primary_domain"] = tag_data["primary_domain"]

            # Add subtopics if not present
            if "subtopics" not in conv:
                conv["subtopics"] = tag_data["subtopics"]

            enriched_count += 1

            # Rate limiting: ~20 requests per minute
            time.sleep(3)

        except Exception as e:
            print(f"\nError processing {conv.get('external_id', 'unknown')}: {e}")
            failed_count += 1
            continue

    print()
    print("=" * 80)
    print("📊 Enrichment Complete!")
    print("=" * 80)
    print(f"✓ Successfully enriched: {enriched_count} conversations")
    print(f"✗ Failed: {failed_count} conversations")
    print(f"💰 Actual cost: ${enricher.total_cost:.2f}")
    print(f"📞 API calls: {enricher.total_api_calls}")
    print()

    # Save updated repository
    print("💾 Saving updated repository...")
    save_conversations(conversations, repo_file)
    print(f"✓ Saved to: {repo_file}")
    print()

    # Show sample tags
    print("📝 Sample enriched tags:")
    for conv in needs_tags[:3]:
        if conv.get("tags"):
            print(f"\n  {conv.get('title', 'Untitled')[:60]}...")
            print(f"  Domain: {conv.get('primary_domain', 'N/A')}")
            print(f"  Tags: {', '.join(conv.get('tags', [])[:5])}...")

    print()
    print("✅ Tag enrichment complete! Your repository now has high-quality tags.")


if __name__ == "__main__":
    main()
