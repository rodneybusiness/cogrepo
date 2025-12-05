#!/usr/bin/env python3
"""
Quick Sync Utility for CogRepo

One-command sync for all your LLM archives.
Detects what's new and processes only the new conversations.

Usage:
    python quick_sync.py              # Sync all archives
    python quick_sync.py --dry-run    # Preview what would be synced
    python quick_sync.py --no-enrich  # Skip AI enrichment
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

# Load environment
load_dotenv(Path(__file__).parent / '.env')
sys.path.append(str(Path(__file__).parent))

from archive_registry import ArchiveRegistry, Archive, FileChangeReport
from smart_parser import SmartParser
from state_manager import ProcessingStateManager
from models import ProcessingStats, EnrichedConversation
from enrichment import EnrichmentPipeline


def format_duration(seconds: float) -> str:
    """Format duration for display."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def format_size(bytes: int) -> str:
    """Format byte size for display."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(bytes) < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"


class QuickSync:
    """
    Unified sync manager for all LLM archives.

    Provides a single, streamlined interface for:
    1. Detecting changes in all archives
    2. Processing only new conversations
    3. Updating registry state
    4. Reporting results
    """

    def __init__(
        self,
        enrich: bool = True,
        dry_run: bool = False,
        verbose: bool = True
    ):
        """
        Initialize QuickSync.

        Args:
            enrich: Whether to enrich with AI metadata
            dry_run: If True, show what would be done but don't do it
            verbose: If True, print progress messages
        """
        self.enrich = enrich
        self.dry_run = dry_run
        self.verbose = verbose

        self.registry = ArchiveRegistry()
        self.state_manager = ProcessingStateManager()

        # Stats tracking
        self.total_processed = 0
        self.total_failed = 0
        self.total_cost = 0.0
        self.sync_results = []

    def log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)

    def sync_all(self, archive_names: Optional[List[str]] = None) -> dict:
        """
        Sync all (or specified) archives.

        Args:
            archive_names: Optional list of archive names to sync.
                          If None, syncs all auto-sync enabled archives.

        Returns:
            Dictionary with sync results
        """
        start_time = datetime.now()

        self.log("\n" + "="*60)
        self.log("  COGREPO QUICK SYNC")
        self.log("="*60)

        # Get archives to sync
        if archive_names:
            archives = [self.registry.get_archive(name) for name in archive_names]
            archives = [a for a in archives if a is not None]
        else:
            archives = [a for a in self.registry.list_archives()
                       if a.auto_sync and a.enabled]

        if not archives:
            self.log("\nNo archives to sync.")
            self.log("Register archives with: python cogrepo_manage.py register ...")
            return {"success": False, "message": "No archives to sync"}

        # Check for changes
        self.log(f"\nChecking {len(archives)} archives for changes...")

        archives_to_sync = []
        for archive in archives:
            changes = self.registry.detect_changes(archive.name)
            if changes.has_changed:
                archives_to_sync.append((archive, changes))
                self.log(f"  {archive.name}: ~{changes.estimated_new_conversations} new")
            else:
                self.log(f"  {archive.name}: up to date")

        if not archives_to_sync:
            self.log("\nAll archives are up to date!")
            return {"success": True, "message": "All up to date", "processed": 0}

        # Calculate totals
        total_estimated = sum(c.estimated_new_conversations for _, c in archives_to_sync)
        estimated_cost = total_estimated * 0.025 if self.enrich else 0

        self.log(f"\nArchives needing sync: {len(archives_to_sync)}")
        self.log(f"Estimated new conversations: {total_estimated}")
        if self.enrich:
            self.log(f"Estimated cost: ${estimated_cost:.2f}")

        # Dry run mode
        if self.dry_run:
            self.log("\n[DRY RUN] No changes will be made.")
            return {
                "success": True,
                "dry_run": True,
                "archives": len(archives_to_sync),
                "estimated_new": total_estimated,
                "estimated_cost": estimated_cost
            }

        # Check API key
        if self.enrich and not os.getenv("ANTHROPIC_API_KEY"):
            self.log("\nWarning: ANTHROPIC_API_KEY not set")
            self.log("Proceeding without AI enrichment.")
            self.enrich = False

        # Process each archive
        self.log("\n" + "-"*40)
        self.log("Processing...")
        self.log("-"*40)

        for archive, changes in archives_to_sync:
            self._sync_archive(archive, changes)

        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self.log("\n" + "="*60)
        self.log("  SYNC COMPLETE")
        self.log("="*60)
        self.log(f"\n  Total processed: {self.total_processed}")
        self.log(f"  Total failed: {self.total_failed}")
        self.log(f"  Duration: {format_duration(duration)}")
        if self.enrich:
            self.log(f"  Cost: ${self.total_cost:.2f}")

        return {
            "success": self.total_failed == 0,
            "processed": self.total_processed,
            "failed": self.total_failed,
            "duration": duration,
            "cost": self.total_cost,
            "results": self.sync_results
        }

    def _sync_archive(self, archive: Archive, changes: FileChangeReport):
        """Sync a single archive."""
        self.log(f"\n{archive.name} ({archive.source})")
        self.log(f"  File: {archive.file_path}")
        self.log(f"  Expected: ~{changes.estimated_new_conversations} new conversations")

        archive_start = datetime.now()
        processed = 0
        failed = 0

        try:
            # Get processed IDs from state manager for deduplication
            source_map = {"OpenAI": "OpenAI", "Anthropic": "Anthropic", "Google": "Google"}
            source_key = source_map.get(archive.source, archive.source)
            processed_ids = set(
                self.state_manager.state["processed_conversations"].get(source_key, {}).keys()
            )

            # Create smart parser with cursor
            source_cli = {"OpenAI": "chatgpt", "Anthropic": "claude", "Google": "gemini"}
            parser = SmartParser(
                source_cli.get(archive.source, "chatgpt"),
                archive.file_path
            )

            # Parse new conversations
            new_conversations = list(parser.parse_incremental(
                cursor=archive.cursor,
                processed_ids=processed_ids
            ))

            self.log(f"  Found: {len(new_conversations)} new conversations")

            if not new_conversations:
                self.log(f"  No new conversations to process")
                return

            # Enrich if enabled
            enriched_conversations = []

            if self.enrich:
                import yaml
                config_path = Path(__file__).parent / "config/enrichment_config.yaml"
                if config_path.exists():
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                else:
                    config = {
                        "api": {"model": "claude-3-5-sonnet-20241022"},
                        "processing": {},
                        "enrichment": {"generate_titles": True}
                    }

                pipeline = EnrichmentPipeline(config)

                for i, raw in enumerate(new_conversations, 1):
                    try:
                        enriched = pipeline.enrich_conversation(raw)
                        enriched_conversations.append(enriched)
                        processed += 1

                        # Update state
                        content_hash = self.state_manager.get_content_hash(raw)
                        self.state_manager.add_processed(
                            external_id=raw.external_id,
                            internal_uuid=enriched.convo_id,
                            source=raw.source,
                            content_hash=content_hash,
                            conversation_date=raw.create_time
                        )

                        if i % 10 == 0:
                            self.log(f"  Progress: {i}/{len(new_conversations)}")

                    except Exception as e:
                        self.log(f"  Failed to enrich {raw.external_id}: {e}")
                        failed += 1

                # Update cost estimate
                stats = pipeline.get_stats()
                self.total_cost += stats.get('estimated_cost_usd', 0)

            else:
                # No enrichment - just create minimal records
                for raw in new_conversations:
                    enriched = EnrichedConversation.from_raw(raw, enrichments=None)
                    enriched_conversations.append(enriched)
                    processed += 1

                    content_hash = self.state_manager.get_content_hash(raw)
                    self.state_manager.add_processed(
                        external_id=raw.external_id,
                        internal_uuid=enriched.convo_id,
                        source=raw.source,
                        content_hash=content_hash,
                        conversation_date=raw.create_time
                    )

            # Save to repository
            if enriched_conversations:
                output_file = Path(__file__).parent / "data/enriched_repository.jsonl"
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with open(output_file, 'a', encoding='utf-8') as f:
                    for conv in enriched_conversations:
                        f.write(conv.to_jsonl() + '\n')

            # Update registry
            archive_end = datetime.now()
            duration = (archive_end - archive_start).total_seconds()

            final_cursor = parser.get_final_cursor()
            self.registry.update_after_sync(
                name=archive.name,
                conversations_processed=processed,
                last_external_id=final_cursor.last_external_id,
                last_timestamp=final_cursor.last_timestamp,
                duration_seconds=duration
            )

            # Save state
            self.state_manager.save()

            self.log(f"  Processed: {processed} ({format_duration(duration)})")

        except Exception as e:
            self.log(f"  Error: {e}")
            failed += 1

        self.total_processed += processed
        self.total_failed += failed
        self.sync_results.append({
            "archive": archive.name,
            "processed": processed,
            "failed": failed
        })


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick sync all CogRepo archives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_sync.py              # Sync all archives
  python quick_sync.py --dry-run    # Preview without syncing
  python quick_sync.py --no-enrich  # Skip AI enrichment
  python quick_sync.py ChatGPT      # Sync specific archive
        """
    )

    parser.add_argument("archives", nargs="*", help="Specific archives to sync")
    parser.add_argument("--dry-run", action="store_true", help="Preview without syncing")
    parser.add_argument("--no-enrich", action="store_true", help="Skip AI enrichment")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    syncer = QuickSync(
        enrich=not args.no_enrich,
        dry_run=args.dry_run,
        verbose=not args.quiet
    )

    archive_names = args.archives if args.archives else None
    results = syncer.sync_all(archive_names)

    return 0 if results.get("success", False) else 1


if __name__ == "__main__":
    sys.exit(main())
