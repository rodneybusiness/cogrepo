#!/usr/bin/env python3
"""
Build Knowledge Graph from Conversations

Extracts entities and relationships from all conversations
and builds a knowledge graph for discovery and navigation.
"""

import json
import sys
from pathlib import Path
from context.knowledge_graph import KnowledgeGraph

def main():
    """Build knowledge graph from enriched repository."""
    print("=" * 80)
    print("CogRepo Knowledge Graph Builder")
    print("=" * 80)
    print()

    # Paths
    repo_file = Path(__file__).parent / "data" / "enriched_repository.jsonl"
    graph_file = Path(__file__).parent / "data" / "knowledge_graph.json"

    if not repo_file.exists():
        print(f"Error: Repository file not found: {repo_file}")
        sys.exit(1)

    # Load conversations
    print(f"📁 Loading conversations from: {repo_file}")
    conversations = []
    with open(repo_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    print(f"✓ Loaded {len(conversations)} conversations")
    print()

    # Build knowledge graph
    print("🔨 Building knowledge graph...")
    print("  - Extracting entities from conversation content")
    print("  - Identifying co-occurrence relationships")
    print("  - Computing relationship weights")
    print()

    kg = KnowledgeGraph()
    kg.build_from_conversations(conversations)

    # Statistics
    stats = kg.to_dict()['stats']
    print("=" * 80)
    print("📊 Knowledge Graph Statistics")
    print("=" * 80)
    print(f"Entities: {stats['entity_count']}")
    print()
    print("Entities by type:")
    for entity_type, count in sorted(stats['entity_types'].items(), key=lambda x: -x[1]):
        print(f"  {entity_type}: {count}")
    print()
    print(f"Relationships: {stats['relationship_count']}")
    print()

    # Show top entities
    all_entities = list(kg.entities.values())
    all_entities.sort(key=lambda e: e.occurrence_count, reverse=True)

    print("🏆 Top 20 Most Mentioned Entities:")
    for i, entity in enumerate(all_entities[:20], 1):
        print(f"  {i:2d}. {entity.name:20s} ({entity.entity_type:12s}) - {entity.occurrence_count:3d} mentions")
    print()

    # Save
    print(f"💾 Saving knowledge graph to: {graph_file}")
    kg.save(str(graph_file))

    # File size
    file_size = graph_file.stat().st_size / (1024 * 1024)
    print(f"✓ Saved ({file_size:.1f} MB)")
    print()

    print("✅ Knowledge graph built successfully!")
    print()
    print("You can now:")
    print("  - Query entity relationships")
    print("  - Discover related conversations")
    print("  - Visualize the knowledge graph")
    print("  - Use hybrid search with entity context")


if __name__ == "__main__":
    main()
