#!/usr/bin/env python3
"""
Index Builder for CogRepo

Updates search indexes after importing new conversations:
- repository.index.meta.json (conversation ID and title index)
- focus_list.jsonl (high-priority conversations)
- standardized_conversations.parquet (optional, for data analysis)

Usage:
    python index_builder.py --rebuild
    python index_builder.py --update
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class IndexBuilder:
    """Build and update search indexes for CogRepo."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize index builder.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(__file__).parent / data_dir
        self.enriched_file = self.data_dir / "enriched_repository.jsonl"
        self.index_meta_file = self.data_dir / "repository.index.meta.json"
        self.focus_list_file = self.data_dir / "focus_list.jsonl"

    def rebuild_indexes(self, min_score_for_focus: int = 7):
        """
        Rebuild all indexes from scratch.

        Args:
            min_score_for_focus: Minimum score for inclusion in focus list
        """
        print("🔨 Rebuilding indexes from enriched_repository.jsonl...")

        if not self.enriched_file.exists():
            print(f"✗ Error: {self.enriched_file} not found")
            print("  Run cogrepo_import.py first to create the repository")
            return

        # Load all conversations
        conversations = self._load_all_conversations()
        print(f"✓ Loaded {len(conversations)} conversations")

        # Build index metadata (for semantic search)
        print("📝 Building repository.index.meta.json...")
        self._build_index_meta(conversations)
        print(f"✓ Created index with {len(conversations)} entries")

        # Build focus list (high-priority conversations)
        print(f"📝 Building focus_list.jsonl (score >= {min_score_for_focus})...")
        focus_count = self._build_focus_list(conversations, min_score_for_focus)
        print(f"✓ Created focus list with {focus_count} conversations")

        print("\n✓ Index rebuild complete!")

    def update_indexes(self, min_score_for_focus: int = 7):
        """
        Update indexes incrementally.

        For now, this just rebuilds since it's fast enough.
        In future, could optimize to only process new entries.

        Args:
            min_score_for_focus: Minimum score for inclusion in focus list
        """
        print("♻️  Updating indexes...")
        self.rebuild_indexes(min_score_for_focus)

    def _load_all_conversations(self) -> List[Dict[str, Any]]:
        """Load all conversations from enriched_repository.jsonl."""
        conversations = []

        with open(self.enriched_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        conversations.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line: {e}")
                        continue

        return conversations

    def _build_index_meta(self, conversations: List[Dict[str, Any]]):
        """
        Build repository.index.meta.json.

        This file contains conversation IDs and titles for the search interface.
        """
        index_data = []

        for conv in conversations:
            index_data.append({
                "convo_id": conv.get("convo_id"),
                "title": conv.get("generated_title") or conv.get("title", "Untitled")
            })

        # Sort by title for easier browsing
        index_data.sort(key=lambda x: x["title"].lower())

        # Save
        with open(self.index_meta_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

    def _build_focus_list(self, conversations: List[Dict[str, Any]], min_score: int) -> int:
        """
        Build focus_list.jsonl with high-priority conversations.

        Args:
            conversations: List of conversation dicts
            min_score: Minimum score for inclusion

        Returns:
            Count of conversations in focus list
        """
        focus_conversations = []

        for conv in conversations:
            score = conv.get("score", 0)
            brilliance_score = conv.get("brilliance_score", {}).get("score", 0)

            # Use whichever score is available
            final_score = max(score, brilliance_score)

            if final_score >= min_score:
                focus_conversations.append(conv)

        # Sort by score (descending)
        focus_conversations.sort(
            key=lambda x: max(x.get("score", 0), x.get("brilliance_score", {}).get("score", 0)),
            reverse=True
        )

        # Save as JSONL
        with open(self.focus_list_file, 'w', encoding='utf-8') as f:
            for conv in focus_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')

        return len(focus_conversations)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexes."""
        stats = {
            "enriched_repository": {
                "exists": self.enriched_file.exists(),
                "count": 0,
                "size_mb": 0
            },
            "index_meta": {
                "exists": self.index_meta_file.exists(),
                "count": 0,
                "size_mb": 0
            },
            "focus_list": {
                "exists": self.focus_list_file.exists(),
                "count": 0,
                "size_mb": 0
            }
        }

        # Count conversations in enriched_repository
        if self.enriched_file.exists():
            count = 0
            with open(self.enriched_file, 'r', encoding='utf-8') as f:
                for _ in f:
                    count += 1
            stats["enriched_repository"]["count"] = count
            stats["enriched_repository"]["size_mb"] = self.enriched_file.stat().st_size / 1024 / 1024

        # Count entries in index_meta
        if self.index_meta_file.exists():
            with open(self.index_meta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                stats["index_meta"]["count"] = len(data)
            stats["index_meta"]["size_mb"] = self.index_meta_file.stat().st_size / 1024 / 1024

        # Count conversations in focus_list
        if self.focus_list_file.exists():
            count = 0
            with open(self.focus_list_file, 'r', encoding='utf-8') as f:
                for _ in f:
                    count += 1
            stats["focus_list"]["count"] = count
            stats["focus_list"]["size_mb"] = self.focus_list_file.stat().st_size / 1024 / 1024

        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build and update CogRepo search indexes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild all indexes from scratch"
    )

    parser.add_argument(
        "--update",
        action="store_true",
        help="Update indexes incrementally (same as rebuild for now)"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics"
    )

    parser.add_argument(
        "--min-score",
        type=int,
        default=7,
        help="Minimum score for focus list (default: 7)"
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory (default: data)"
    )

    args = parser.parse_args()

    builder = IndexBuilder(data_dir=args.data_dir)

    if args.stats:
        # Show statistics
        stats = builder.get_stats()
        print("\n📊 Index Statistics:")
        print(f"\nEnriched Repository:")
        print(f"  Exists: {stats['enriched_repository']['exists']}")
        print(f"  Conversations: {stats['enriched_repository']['count']:,}")
        print(f"  Size: {stats['enriched_repository']['size_mb']:.1f} MB")

        print(f"\nIndex Metadata:")
        print(f"  Exists: {stats['index_meta']['exists']}")
        print(f"  Entries: {stats['index_meta']['count']:,}")
        print(f"  Size: {stats['index_meta']['size_mb']:.2f} MB")

        print(f"\nFocus List:")
        print(f"  Exists: {stats['focus_list']['exists']}")
        print(f"  Conversations: {stats['focus_list']['count']:,}")
        print(f"  Size: {stats['focus_list']['size_mb']:.1f} MB")

    elif args.rebuild:
        builder.rebuild_indexes(min_score_for_focus=args.min_score)

    elif args.update:
        builder.update_indexes(min_score_for_focus=args.min_score)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
