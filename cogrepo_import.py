#!/usr/bin/env python3
"""
CogRepo Import Tool

Import and process conversation exports from ChatGPT, Claude, and Gemini.

Usage:
    python cogrepo_import.py --source chatgpt --file conversations.json
    python cogrepo_import.py --source claude --file claude_export.json --enrich
    python cogrepo_import.py --source gemini --file gemini_export.json --dry-run
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / '.env')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models import RawConversation, EnrichedConversation, ProcessingStats
from parsers import ChatGPTParser, ClaudeParser, GeminiParser
from state_manager import ProcessingStateManager
from enrichment import EnrichmentPipeline


class CogRepoImporter:
    """Main importer for CogRepo conversations."""

    def __init__(self, config_file: str = "config/enrichment_config.yaml"):
        """
        Initialize importer.

        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(__file__).parent / config_file
        self.config = self._load_config()
        self.state_manager = ProcessingStateManager(
            self.config.get("output", {}).get("processing_state", "data/processing_state.json")
        )

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_file.exists():
            print(f"Warning: Config file not found: {self.config_file}")
            print("Using default configuration")
            return self._default_config()

        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)

    def _default_config(self) -> dict:
        """Return default configuration if file not found."""
        return {
            "api": {"model": "claude-3-5-sonnet-20241022", "max_tokens": 1500, "temperature": 0.3},
            "processing": {"batch_size": 50, "max_retries": 3},
            "enrichment": {
                "generate_titles": True,
                "generate_summaries": True,
                "extract_tags": True,
                "calculate_scores": True
            },
            "output": {
                "enriched_repository": "data/enriched_repository.jsonl",
                "processing_state": "data/processing_state.json"
            }
        }

    def import_conversations(
        self,
        file_path: str,
        source: str,
        enrich: bool = True,
        dry_run: bool = False
    ) -> ProcessingStats:
        """
        Import conversations from export file.

        Args:
            file_path: Path to export file
            source: Source platform (chatgpt, claude, gemini)
            enrich: Whether to enrich with AI metadata
            dry_run: If True, parse and show stats but don't save

        Returns:
            ProcessingStats object
        """
        stats = ProcessingStats(start_time=datetime.now())

        # Step 1: Parse conversations
        print(f"\n📥 Parsing {source} export file...")
        parser = self._get_parser(file_path, source)

        try:
            raw_conversations = parser.parse()
            stats.total_found = len(raw_conversations)
            print(f"✓ Found {stats.total_found} conversations")
        except Exception as e:
            print(f"✗ Failed to parse file: {e}")
            return stats

        # Step 2: Deduplicate
        print(f"\n🔍 Checking for duplicates...")
        new_conversations = self._deduplicate(raw_conversations)
        stats.total_new = len(new_conversations)
        stats.total_duplicates = stats.total_found - stats.total_new

        print(f"✓ New conversations: {stats.total_new}")
        print(f"  Already processed: {stats.total_duplicates}")

        if dry_run:
            print(f"\n🔍 DRY RUN MODE - No changes will be made")
            self._show_dry_run_preview(new_conversations)
            stats.end_time = datetime.now()
            return stats

        if stats.total_new == 0:
            print("\n✓ No new conversations to process")
            stats.end_time = datetime.now()
            return stats

        # Step 3: Enrich (if requested)
        enriched_conversations = []

        if enrich:
            print(f"\n🤖 Enriching conversations with AI metadata...")
            enrichment_pipeline = EnrichmentPipeline(self.config)

            for raw in tqdm(new_conversations, desc="Enriching"):
                try:
                    enriched = enrichment_pipeline.enrich_conversation(raw)
                    enriched_conversations.append(enriched)
                    stats.total_processed += 1

                    # Update state
                    content_hash = self.state_manager.get_content_hash(raw)
                    self.state_manager.add_processed(
                        external_id=raw.external_id,
                        internal_uuid=enriched.convo_id,
                        source=raw.source,
                        content_hash=content_hash,
                        conversation_date=raw.create_time
                    )

                except Exception as e:
                    print(f"\n✗ Failed to enrich conversation {raw.external_id}: {e}")
                    self.state_manager.add_failed(raw.external_id, raw.source, str(e))
                    stats.total_failed += 1

            # Show enrichment stats
            enrich_stats = enrichment_pipeline.get_stats()
            print(f"\n📊 Enrichment Statistics:")
            print(f"  API calls: {enrich_stats['total_api_calls']}")
            print(f"  Tokens used: {enrich_stats['total_tokens_used']:,}")
            print(f"  Estimated cost: ${enrich_stats['estimated_cost_usd']:.2f}")

        else:
            print(f"\n📝 Creating minimal enrichments (no AI)...")
            for raw in tqdm(new_conversations, desc="Processing"):
                enriched = EnrichedConversation.from_raw(raw, enrichments=None)
                enriched_conversations.append(enriched)
                stats.total_processed += 1

                # Update state
                content_hash = self.state_manager.get_content_hash(raw)
                self.state_manager.add_processed(
                    external_id=raw.external_id,
                    internal_uuid=enriched.convo_id,
                    source=raw.source,
                    content_hash=content_hash,
                    conversation_date=raw.create_time
                )

        # Step 4: Save to file
        if enriched_conversations:
            print(f"\n💾 Saving conversations...")
            self._save_conversations(enriched_conversations)
            print(f"✓ Saved {len(enriched_conversations)} conversations")

        # Step 5: Update state
        print(f"\n💾 Updating processing state...")
        source_map = {"chatgpt": "OpenAI", "claude": "Anthropic", "gemini": "Google"}
        self.state_manager.update_source_import_date(source_map.get(source.lower(), "OpenAI"))
        self.state_manager.update_batch_stats(stats)
        self.state_manager.save()
        print(f"✓ State saved")

        stats.end_time = datetime.now()

        # Show final summary
        self._show_summary(stats)

        return stats

    def _get_parser(self, file_path: str, source: str):
        """Get appropriate parser for source."""
        source = source.lower()

        if source == "chatgpt":
            return ChatGPTParser(file_path)
        elif source == "claude":
            return ClaudeParser(file_path)
        elif source == "gemini":
            return GeminiParser(file_path)
        else:
            # Auto-detect
            print(f"Unknown source '{source}', attempting auto-detection...")
            for Parser in [ChatGPTParser, ClaudeParser, GeminiParser]:
                try:
                    parser = Parser(file_path)
                    if parser.detect_format():
                        print(f"✓ Detected format: {parser.platform_name}")
                        return parser
                except (FileNotFoundError, ValueError, json.JSONDecodeError, KeyError):
                    continue

            raise ValueError(f"Could not detect format for file: {file_path}")

    def _deduplicate(self, conversations: List[RawConversation]) -> List[RawConversation]:
        """
        Filter out already-processed conversations.

        Returns:
            List of new conversations only
        """
        new_conversations = []

        for conv in conversations:
            # Check external ID
            if self.state_manager.is_processed(conv.external_id, conv.source):
                continue

            # Check content hash
            content_hash = self.state_manager.get_content_hash(conv)
            if self.state_manager.is_duplicate_content(content_hash):
                continue

            new_conversations.append(conv)

        return new_conversations

    def _save_conversations(self, conversations: List[EnrichedConversation]):
        """Save enriched conversations to JSONL file."""
        output_file = Path(__file__).parent / self.config.get("output", {}).get(
            "enriched_repository", "data/enriched_repository.jsonl"
        )

        # Ensure directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Append to file
        with open(output_file, 'a', encoding='utf-8') as f:
            for conv in conversations:
                f.write(conv.to_jsonl() + '\n')

    def _show_dry_run_preview(self, conversations: List[RawConversation], limit: int = 10):
        """Show preview of conversations in dry run mode."""
        print(f"\n📋 Sample of new conversations (showing up to {limit}):")
        for i, conv in enumerate(conversations[:limit], 1):
            date_str = conv.create_time.strftime('%Y-%m-%d')
            print(f"  {i}. \"{conv.title}\" ({conv.source}, {date_str})")

        if len(conversations) > limit:
            print(f"  ... and {len(conversations) - limit} more")

        # Estimate cost
        if self.config.get("enrichment", {}).get("generate_titles", True):
            estimated_cost = len(conversations) * 0.025  # Rough estimate
            estimated_time = (len(conversations) / 50) * 120  # ~2 min per 50 conversations

            print(f"\n💰 Estimated enrichment cost: ${estimated_cost:.2f}")
            print(f"⏱️  Estimated processing time: {int(estimated_time)} seconds (~{int(estimated_time/60)} minutes)")

    def _show_summary(self, stats: ProcessingStats):
        """Show final processing summary."""
        print(f"\n" + "="*60)
        print(f"✓ IMPORT COMPLETE")
        print(f"="*60)
        print(f"  Total found: {stats.total_found}")
        print(f"  New: {stats.total_new}")
        print(f"  Duplicates: {stats.total_duplicates}")
        print(f"  Successfully processed: {stats.total_processed}")
        print(f"  Failed: {stats.total_failed}")
        print(f"  Duration: {int(stats.duration_seconds)}s")

        if stats.total_processed > 0:
            print(f"  Processing rate: {stats.processing_rate:.1f} conversations/second")

        # Show total counts
        total_stats = self.state_manager.get_stats()
        print(f"\n📊 Total Repository Statistics:")
        print(f"  Total conversations: {total_stats['total_processed']}")
        print(f"  By source:")
        for source, count in total_stats['by_source'].items():
            print(f"    {source}: {count}")

        print(f"\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import conversations into CogRepo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import ChatGPT conversations with AI enrichment
  python cogrepo_import.py --source chatgpt --file conversations.json --enrich

  # Import Claude conversations without enrichment (faster, cheaper)
  python cogrepo_import.py --source claude --file claude_export.json

  # Preview what would be imported (dry run)
  python cogrepo_import.py --source gemini --file gemini.json --dry-run

  # Auto-detect source format
  python cogrepo_import.py --file export.json --enrich
        """
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to conversation export file"
    )

    parser.add_argument(
        "--source",
        choices=["chatgpt", "claude", "gemini", "auto"],
        default="auto",
        help="Source platform (default: auto-detect)"
    )

    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Enrich conversations with AI metadata (requires API key)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and show stats without saving"
    )

    parser.add_argument(
        "--config",
        default="config/enrichment_config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Validate file exists
    if not Path(args.file).exists():
        print(f"✗ Error: File not found: {args.file}")
        sys.exit(1)

    # Check API key if enriching
    if args.enrich and not args.dry_run:
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("✗ Error: ANTHROPIC_API_KEY environment variable not set")
            print("  Set it with: export ANTHROPIC_API_KEY='your-key-here'")
            sys.exit(1)

    # Create importer and run
    try:
        importer = CogRepoImporter(config_file=args.config)
        stats = importer.import_conversations(
            file_path=args.file,
            source=args.source,
            enrich=args.enrich,
            dry_run=args.dry_run
        )

        sys.exit(0 if stats.total_failed == 0 else 1)

    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
