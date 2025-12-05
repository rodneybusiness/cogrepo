#!/usr/bin/env python3
"""
Extract only new Claude conversations from an export file.
Compares against existing enriched_repository.jsonl by convo_id.
"""
import json
from pathlib import Path

def extract_new_conversations(input_file, existing_repo, output_file):
    """Extract only new conversations not in the repository."""

    # Load existing conversation IDs
    print(f"📖 Loading existing repository...")
    existing_ids = set()
    with open(existing_repo, 'r') as f:
        for line in f:
            try:
                conv = json.loads(line)
                existing_ids.add(conv['convo_id'])
            except:
                continue

    print(f"✓ Found {len(existing_ids)} existing conversations")

    # Load new export file
    print(f"\n📥 Loading new Claude export file...")
    with open(input_file, 'r') as f:
        all_conversations = json.load(f)

    print(f"✓ Found {len(all_conversations)} total conversations")

    # Filter to only new ones (Claude uses 'uuid' field)
    new_conversations = []
    for conv in all_conversations:
        conv_id = conv.get('uuid')  # Claude uses 'uuid' not 'id'
        if conv_id and conv_id not in existing_ids:
            new_conversations.append(conv)

    print(f"\n🆕 New conversations: {len(new_conversations)}")
    print(f"📊 Duplicates skipped: {len(all_conversations) - len(new_conversations)}")

    # Save new conversations to a separate file
    print(f"\n💾 Saving new conversations to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(new_conversations, f, indent=2)

    print(f"✓ Saved {len(new_conversations)} new conversations")

    return len(new_conversations)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_new_claude.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.json', '_new_only.json')

    repo_path = Path(__file__).parent / 'data' / 'enriched_repository.jsonl'

    count = extract_new_conversations(input_file, repo_path, output_file)

    print(f"\n✅ Done! {count} new conversations ready for enrichment.")
    print(f"\nNext step:")
    print(f"  python cogrepo_import.py --source claude --file {output_file} --enrich")
