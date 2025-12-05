#!/usr/bin/env python3
"""
Fix missing external_id fields in enriched_repository.jsonl

This script adds external_id to all conversations that are missing it.
The external_id is set to be the same as convo_id for consistency.
"""

import json
from pathlib import Path
import shutil
from datetime import datetime

def fix_external_ids(repo_path: str, backup: bool = True):
    """
    Add external_id to conversations missing it.

    Args:
        repo_path: Path to enriched_repository.jsonl
        backup: Create backup before modifying
    """
    repo_path = Path(repo_path)

    if not repo_path.exists():
        print(f"Error: {repo_path} not found")
        return

    # Create backup
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = repo_path.parent / f"{repo_path.stem}_backup_{timestamp}{repo_path.suffix}"
        print(f"Creating backup: {backup_path}")
        shutil.copy2(repo_path, backup_path)

    # Process conversations
    fixed_count = 0
    total_count = 0
    temp_path = repo_path.parent / f"{repo_path.stem}_temp{repo_path.suffix}"

    print(f"Processing {repo_path}...")

    with open(repo_path, 'r', encoding='utf-8') as infile, \
         open(temp_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                conv = json.loads(line)
                total_count += 1

                # Check if external_id is missing or null
                if 'external_id' not in conv or conv['external_id'] is None:
                    # Use convo_id as external_id
                    conv['external_id'] = conv.get('convo_id', f"conv_{total_count}")
                    fixed_count += 1

                # Write updated conversation
                outfile.write(json.dumps(conv, ensure_ascii=False) + '\n')

                if total_count % 1000 == 0:
                    print(f"  Processed {total_count} conversations, fixed {fixed_count}")

            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping malformed JSON at line {total_count}: {e}")
                continue

    # Replace original with fixed version
    shutil.move(temp_path, repo_path)

    print(f"\nComplete!")
    print(f"  Total conversations: {total_count}")
    print(f"  Fixed (added external_id): {fixed_count}")
    print(f"  Already had external_id: {total_count - fixed_count}")

    if backup:
        print(f"  Backup saved to: {backup_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Fix missing external_id fields")
    parser.add_argument(
        '--repo',
        default='data/enriched_repository.jsonl',
        help='Path to enriched_repository.jsonl'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup'
    )

    args = parser.parse_args()

    fix_external_ids(args.repo, backup=not args.no_backup)
