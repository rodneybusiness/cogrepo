#!/usr/bin/env python3
"""
CogRepo Update Tool

Incrementally update CogRepo with new conversations from exports.
This is a convenience wrapper around cogrepo_import.py that's optimized
for incremental updates.

Usage:
    python cogrepo_update.py --source chatgpt --file new_export.json
    python cogrepo_update.py --source claude --file claude_new.json --dry-run
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / '.env')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from cogrepo_import import CogRepoImporter


def main():
    """Main entry point for incremental updates."""
    parser = argparse.ArgumentParser(
        description="Incrementally update CogRepo with new conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update with new ChatGPT export
  python cogrepo_update.py --source chatgpt --file new_conversations.json

  # Update with new Claude conversations
  python cogrepo_update.py --source claude --file claude_new.json

  # First-time full import of Gemini conversations
  python cogrepo_update.py --source gemini --file gemini_export.json --full

  # Preview what would be updated (dry run)
  python cogrepo_update.py --source chatgpt --file export.json --dry-run

Notes:
  - Automatically deduplicates based on conversation IDs
  - AI enrichment is enabled by default (requires ANTHROPIC_API_KEY)
  - Use --no-enrich for faster import without AI metadata
  - Safe to run multiple times (won't re-process existing conversations)
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
        "--full",
        action="store_true",
        help="Full import (for first-time imports of a new platform)"
    )

    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip AI enrichment (faster, no API costs)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without saving"
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
    enrich = not args.no_enrich
    if enrich and not args.dry_run:
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("✗ Error: ANTHROPIC_API_KEY environment variable not set")
            print("  Set it with: export ANTHROPIC_API_KEY='your-key-here'")
            print("  Or use --no-enrich to skip AI enrichment")
            sys.exit(1)

    # Show mode
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made\n")
    elif args.full:
        print("📦 FULL IMPORT MODE\n")
    else:
        print("♻️  INCREMENTAL UPDATE MODE\n")

    # Create importer and run
    try:
        importer = CogRepoImporter(config_file=args.config)

        # Load state to show current stats
        state_stats = importer.state_manager.get_stats()
        print(f"📊 Current Repository Status:")
        print(f"  Total conversations: {state_stats['total_processed']}")
        for source, count in state_stats['by_source'].items():
            if count > 0:
                print(f"    {source}: {count}")
        print()

        # Run import
        stats = importer.import_conversations(
            file_path=args.file,
            source=args.source,
            enrich=enrich,
            dry_run=args.dry_run
        )

        # Success message
        if not args.dry_run and stats.total_processed > 0:
            new_total = state_stats['total_processed'] + stats.total_processed
            print(f"\n✓ Repository updated successfully!")
            print(f"  Total conversations now: {new_total} (+{stats.total_processed})")

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
