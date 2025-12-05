#!/usr/bin/env python3
"""
Analyze Empty Conversations

Investigates conversations with 0 messages to understand:
- Why they exist
- What sources they come from
- Whether they should be kept or filtered
"""

import json
from pathlib import Path
from collections import Counter
from datetime import datetime

def main():
    print("=" * 80)
    print("Empty Conversations Analysis")
    print("=" * 80)
    print()

    repo_file = Path("data/enriched_repository.jsonl")

    # Load conversations
    print(f"📁 Loading conversations from: {repo_file}")
    all_convos = []
    with open(repo_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_convos.append(json.loads(line))

    print(f"✓ Loaded {len(all_convos)} total conversations")
    print()

    # Analyze empty conversations
    empty_convos = [c for c in all_convos if len(c.get('messages', [])) == 0]
    non_empty_convos = [c for c in all_convos if len(c.get('messages', [])) > 0]

    print("📊 OVERALL STATISTICS:")
    print(f"  Total conversations: {len(all_convos)}")
    print(f"  Non-empty (with messages): {len(non_empty_convos)} ({len(non_empty_convos)/len(all_convos)*100:.1f}%)")
    print(f"  Empty (0 messages): {len(empty_convos)} ({len(empty_convos)/len(all_convos)*100:.1f}%)")
    print()

    if not empty_convos:
        print("✅ No empty conversations found!")
        return

    # Analyze by source
    empty_by_source = Counter(c.get('source', 'Unknown') for c in empty_convos)
    print("📦 EMPTY CONVERSATIONS BY SOURCE:")
    for source, count in sorted(empty_by_source.items(), key=lambda x: -x[1]):
        pct = count / len(empty_convos) * 100
        total_from_source = sum(1 for c in all_convos if c.get('source') == source)
        pct_of_source = count / total_from_source * 100 if total_from_source > 0 else 0
        print(f"  {source:15s}: {count:5d} ({pct:5.1f}% of empty, {pct_of_source:5.1f}% of {source} total)")
    print()

    # Check what fields empty conversations DO have
    print("🔍 FIELDS PRESENT IN EMPTY CONVERSATIONS:")
    field_counts = Counter()
    for conv in empty_convos[:100]:  # Sample first 100
        for field in conv.keys():
            if conv[field]:  # Non-empty value
                field_counts[field] += 1

    for field, count in field_counts.most_common(15):
        print(f"  {field:25s}: {count:3d}/{min(100, len(empty_convos)):3d}")
    print()

    # Show examples from each source
    print("📝 EXAMPLES OF EMPTY CONVERSATIONS:")
    for source in empty_by_source:
        examples = [c for c in empty_convos if c.get('source') == source][:2]
        print(f"\n  {source}:")
        for i, ex in enumerate(examples, 1):
            print(f"    Example {i}:")
            print(f"      ID: {ex.get('convo_id', 'N/A')}")
            print(f"      External ID: {ex.get('external_id', 'N/A')}")
            print(f"      Title: {ex.get('title', 'N/A')}")
            print(f"      Generated Title: {ex.get('generated_title', 'N/A')}")
            print(f"      Created: {ex.get('created_at', 'N/A')}")
            print(f"      Updated: {ex.get('updated_at', 'N/A')}")
            print(f"      Tags: {ex.get('tags', [])}")
            print(f"      Messages: {len(ex.get('messages', []))}")
    print()

    # Check if they have metadata but no content
    print("🔬 DETAILED ANALYSIS:")

    has_title = sum(1 for c in empty_convos if c.get('title') or c.get('generated_title'))
    has_created_date = sum(1 for c in empty_convos if c.get('created_at'))
    has_tags = sum(1 for c in empty_convos if c.get('tags') and len(c.get('tags', [])) > 0)

    print(f"  Empty convos with title: {has_title} ({has_title/len(empty_convos)*100:.1f}%)")
    print(f"  Empty convos with created date: {has_created_date} ({has_created_date/len(empty_convos)*100:.1f}%)")
    print(f"  Empty convos with tags: {has_tags} ({has_tags/len(empty_convos)*100:.1f}%)")
    print()

    # Recommendation
    print("=" * 80)
    print("💡 ANALYSIS & RECOMMENDATION")
    print("=" * 80)
    print()

    if has_title < len(empty_convos) * 0.1:
        print("❌ ISSUE: Most empty conversations lack even basic metadata")
        print("   These are likely placeholder or deleted conversations.")
        print()
        print("   RECOMMENDATION: Filter out conversations with 0 messages during import.")
        print("   They provide no value for search or analysis.")
    else:
        print("⚠️  MIXED: Empty conversations have metadata but no content")
        print("   These might be:")
        print("   - Deleted conversations (metadata remains)")
        print("   - Conversations created but never used")
        print("   - Conversations where messages were cleared")
        print()
        print("   RECOMMENDATION: Consider filtering, but review examples first.")
    print()

    print("✅ To filter empty conversations in future imports:")
    print("   Add to your parser: conversations = [c for c in convos if len(c.get('messages', [])) > 0]")
    print()


if __name__ == "__main__":
    main()
