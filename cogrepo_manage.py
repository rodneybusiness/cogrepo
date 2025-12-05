#!/usr/bin/env python3
"""
CogRepo Management CLI

The primary interface for managing your LLM conversation archives.
Enables easy registration, status checking, and smart incremental syncing.

Usage:
    # Register your three archives (one-time setup)
    python cogrepo_manage.py register --name "ChatGPT" --source chatgpt --file ~/exports/chatgpt.json
    python cogrepo_manage.py register --name "Claude" --source claude --file ~/exports/claude.jsonl
    python cogrepo_manage.py register --name "Gemini" --source gemini --file ~/exports/gemini.json

    # Check status of all archives
    python cogrepo_manage.py status

    # Quick sync all archives (processes only new conversations)
    python cogrepo_manage.py sync

    # Sync specific archive
    python cogrepo_manage.py sync --name "ChatGPT"

    # Preview what would be synced (dry run)
    python cogrepo_manage.py sync --dry-run
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / '.env')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from archive_registry import ArchiveRegistry, FileChangeReport
from cogrepo_import import CogRepoImporter


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char*60}")
    print(f"  {text}")
    print(f"{char*60}")


def print_section(text: str):
    """Print a section header."""
    print(f"\n{'-'*40}")
    print(f"  {text}")
    print(f"{'-'*40}")


def format_time_ago(iso_timestamp: Optional[str]) -> str:
    """Format a timestamp as 'X ago'."""
    if not iso_timestamp:
        return "Never"

    try:
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        diff = now - dt

        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds >= 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds >= 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "Just now"
    except (ValueError, TypeError):
        return iso_timestamp[:10]


def cmd_register(args):
    """Register a new archive."""
    registry = ArchiveRegistry()

    print_header("REGISTER ARCHIVE")

    # Validate source
    valid_sources = ["chatgpt", "claude", "gemini", "openai", "anthropic", "google"]
    if args.source.lower() not in valid_sources:
        print(f"Error: Invalid source '{args.source}'")
        print(f"Valid sources: {', '.join(valid_sources)}")
        return 1

    # Expand file path
    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        print(f"Error: File not found: {args.file}")
        return 1

    try:
        archive = registry.register(
            name=args.name,
            source=args.source,
            file_path=str(file_path),
            auto_sync=not args.no_auto_sync
        )

        print(f"\nRegistered archive successfully!")
        print(f"  Name: {archive.name}")
        print(f"  Source: {archive.source}")
        print(f"  File: {archive.file_path}")
        print(f"  Total conversations: {archive.total_conversations}")
        print(f"  Auto-sync: {'Yes' if archive.auto_sync else 'No'}")

        print(f"\nRun 'python cogrepo_manage.py sync --name \"{args.name}\"' to process conversations.")
        return 0

    except Exception as e:
        print(f"Error registering archive: {e}")
        return 1


def cmd_unregister(args):
    """Unregister an archive."""
    registry = ArchiveRegistry()

    if registry.unregister(args.name):
        print(f"Unregistered archive: {args.name}")
        return 0
    else:
        print(f"Archive not found: {args.name}")
        return 1


def cmd_status(args):
    """Show status of all archives."""
    registry = ArchiveRegistry()
    status = registry.get_status()

    print_header("COGREPO ARCHIVE STATUS")

    # Summary
    print(f"\n  Total Conversations: {status['total_conversations']:,}")
    print(f"  Archives Registered: {status['archive_count']}")

    if status['archives_needing_sync']:
        print(f"\n  Archives needing sync: {', '.join(status['archives_needing_sync'])}")

    # Archive details
    if status['archives']:
        print_section("ARCHIVES")

        for arch in status['archives']:
            # Status indicator
            if not arch['enabled']:
                indicator = "[DISABLED]"
            elif arch['has_changes']:
                indicator = "[SYNC NEEDED]"
            else:
                indicator = "[OK]"

            print(f"\n  {arch['name']} ({arch['source']}) {indicator}")
            print(f"    Processed: {arch['processed']:,} conversations")

            if arch['has_changes']:
                print(f"    Pending: ~{arch['pending']:,} new conversations")
                print(f"    Changes: {arch['change_details']}")

            print(f"    Last sync: {format_time_ago(arch['last_sync'])}")
            print(f"    Auto-sync: {'Yes' if arch['auto_sync'] else 'No'}")

    else:
        print("\n  No archives registered.")
        print("\n  Register archives with:")
        print("    python cogrepo_manage.py register --name \"ChatGPT\" --source chatgpt --file ~/export.json")

    print()
    return 0


def cmd_sync(args):
    """Sync archives - process only new conversations."""
    registry = ArchiveRegistry()

    print_header("COGREPO SYNC")

    # Determine which archives to sync
    if args.name:
        archive = registry.get_archive(args.name)
        if not archive:
            print(f"Error: Archive not found: {args.name}")
            return 1
        archives_to_sync = [archive]
    else:
        # Sync all auto-sync enabled archives
        archives_to_sync = [a for a in registry.list_archives() if a.auto_sync and a.enabled]

    if not archives_to_sync:
        print("\nNo archives to sync.")
        print("Register archives with: python cogrepo_manage.py register ...")
        return 0

    # Check for changes
    archives_with_changes = []
    for archive in archives_to_sync:
        changes = registry.detect_changes(archive.name)
        if changes.has_changed or args.force:
            archives_with_changes.append((archive, changes))

    if not archives_with_changes and not args.force:
        print("\nAll archives are up to date!")
        print("Use --force to re-process anyway.")
        return 0

    # Show what will be synced
    print(f"\nArchives to sync: {len(archives_with_changes)}")
    total_new = 0

    for archive, changes in archives_with_changes:
        print(f"  - {archive.name}: ~{changes.estimated_new_conversations} new conversations")
        total_new += changes.estimated_new_conversations

    print(f"\nTotal estimated new conversations: {total_new}")

    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN] No changes will be made.")

        for archive, changes in archives_with_changes:
            print(f"\n  Would sync {archive.name}:")
            print(f"    Source: {archive.source}")
            print(f"    File: {archive.file_path}")
            print(f"    New conversations: ~{changes.estimated_new_conversations}")

        # Cost estimate
        estimated_cost = total_new * 0.025  # ~$0.025 per conversation
        print(f"\n  Estimated enrichment cost: ${estimated_cost:.2f}")
        return 0

    # Check API key if enriching
    enrich = not args.no_enrich
    if enrich:
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("\nWarning: ANTHROPIC_API_KEY not set")
            print("  Conversations will be imported without AI enrichment.")
            print("  Set key with: export ANTHROPIC_API_KEY='your-key'")
            print("  Or use --no-enrich to skip enrichment explicitly.")
            enrich = False

    # Process each archive
    total_processed = 0
    total_failed = 0
    start_time = datetime.now()

    for archive, changes in archives_with_changes:
        print(f"\n{'='*40}")
        print(f"Syncing: {archive.name}")
        print(f"{'='*40}")

        try:
            # Create importer
            importer = CogRepoImporter()

            # Import new conversations
            stats = importer.import_conversations(
                file_path=archive.file_path,
                source=_source_to_cli(archive.source),
                enrich=enrich,
                dry_run=False
            )

            # Update registry
            if stats.total_processed > 0:
                # Get last processed conversation info
                # (In production, the importer would track this)
                registry.update_after_sync(
                    name=archive.name,
                    conversations_processed=stats.total_processed,
                    last_external_id=None,  # Would come from actual processing
                    last_timestamp=datetime.now().isoformat(),
                    duration_seconds=stats.duration_seconds
                )

            total_processed += stats.total_processed
            total_failed += stats.total_failed

            print(f"\nProcessed: {stats.total_processed}")
            if stats.total_failed > 0:
                print(f"Failed: {stats.total_failed}")

        except Exception as e:
            print(f"\nError syncing {archive.name}: {e}")
            total_failed += 1

    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_header("SYNC COMPLETE")
    print(f"\n  Total processed: {total_processed}")
    print(f"  Total failed: {total_failed}")
    print(f"  Duration: {duration:.1f}s")

    if enrich and total_processed > 0:
        estimated_cost = total_processed * 0.025
        print(f"  Estimated cost: ${estimated_cost:.2f}")

    return 0 if total_failed == 0 else 1


def cmd_list(args):
    """List all registered archives."""
    registry = ArchiveRegistry()
    archives = registry.list_archives()

    print_header("REGISTERED ARCHIVES")

    if not archives:
        print("\n  No archives registered.")
        return 0

    for arch in archives:
        enabled_str = "" if arch.enabled else " [DISABLED]"
        print(f"\n  {arch.name}{enabled_str}")
        print(f"    ID: {arch.id}")
        print(f"    Source: {arch.source}")
        print(f"    File: {arch.file_path}")
        print(f"    Processed: {arch.processed_conversations}")
        print(f"    Total: {arch.total_conversations}")
        print(f"    Auto-sync: {arch.auto_sync}")

    print()
    return 0


def cmd_check(args):
    """Check a specific archive for changes."""
    registry = ArchiveRegistry()

    archive = registry.get_archive(args.name)
    if not archive:
        print(f"Archive not found: {args.name}")
        return 1

    changes = registry.detect_changes(args.name)

    print_header(f"ARCHIVE CHECK: {args.name}")
    print(f"\n  Has changes: {'Yes' if changes.has_changed else 'No'}")
    print(f"  Change type: {changes.change_type or 'N/A'}")
    print(f"  Size change: {changes.size_change:+,} bytes")
    print(f"  Est. new conversations: {changes.estimated_new_conversations}")
    print(f"  Details: {changes.details}")

    return 0


def _source_to_cli(source: str) -> str:
    """Convert source name to CLI format."""
    mapping = {
        "OpenAI": "chatgpt",
        "Anthropic": "claude",
        "Google": "gemini"
    }
    return mapping.get(source, source.lower())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CogRepo Archive Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register your archives (one-time)
  python cogrepo_manage.py register --name "ChatGPT" --source chatgpt --file ~/exports/chatgpt.json
  python cogrepo_manage.py register --name "Claude" --source claude --file ~/exports/claude.jsonl

  # Check status
  python cogrepo_manage.py status

  # Quick sync all archives
  python cogrepo_manage.py sync

  # Sync specific archive
  python cogrepo_manage.py sync --name "Claude"

  # Preview sync (dry run)
  python cogrepo_manage.py sync --dry-run
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new archive")
    register_parser.add_argument("--name", required=True, help="Display name for the archive")
    register_parser.add_argument("--source", required=True,
                                 choices=["chatgpt", "claude", "gemini"],
                                 help="Source platform")
    register_parser.add_argument("--file", required=True, help="Path to export file")
    register_parser.add_argument("--no-auto-sync", action="store_true",
                                 help="Don't include in quick sync")

    # Unregister command
    unreg_parser = subparsers.add_parser("unregister", help="Remove an archive")
    unreg_parser.add_argument("--name", required=True, help="Archive name to remove")

    # Status command
    subparsers.add_parser("status", help="Show status of all archives")

    # List command
    subparsers.add_parser("list", help="List all registered archives")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check archive for changes")
    check_parser.add_argument("--name", required=True, help="Archive name to check")

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync archives (process new conversations)")
    sync_parser.add_argument("--name", help="Specific archive to sync (default: all)")
    sync_parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    sync_parser.add_argument("--no-enrich", action="store_true", help="Skip AI enrichment")
    sync_parser.add_argument("--force", action="store_true", help="Force sync even if no changes")

    args = parser.parse_args()

    # Execute command
    if args.command == "register":
        return cmd_register(args)
    elif args.command == "unregister":
        return cmd_unregister(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "check":
        return cmd_check(args)
    elif args.command == "sync":
        return cmd_sync(args)
    else:
        # No command - show status by default
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
