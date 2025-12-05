"""
CogRepo Knowledge Graph

Builds and manages a knowledge graph of entities across conversations:
- Technologies, frameworks, libraries
- Files, directories, projects
- Concepts and patterns
- Errors and solutions

Provides:
- Entity extraction and linking
- Relationship discovery
- Graph traversal and querying
- Visualization data export

Usage:
    from context.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    kg.build_from_conversations(conversations)

    # Find related entities
    related = kg.get_related('python', limit=10)

    # Export for visualization
    graph_data = kg.export_for_visualization()
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import json
import math


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    entity_type: str  # 'technology', 'file', 'concept', 'error', 'pattern'
    occurrence_count: int = 0
    conversation_ids: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.entity_type,
            'count': self.occurrence_count,
            'conversations': list(self.conversation_ids),
            'properties': self.properties,
        }


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_id: str
    target_id: str
    relationship_type: str  # 'co_occurs', 'uses', 'solves', 'related_to'
    weight: float = 1.0
    conversation_ids: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            'source': self.source_id,
            'target': self.target_id,
            'type': self.relationship_type,
            'weight': self.weight,
            'conversations': list(self.conversation_ids),
        }


class KnowledgeGraph:
    """
    Knowledge graph for entity relationships across conversations.

    Builds a graph where:
    - Nodes are entities (technologies, concepts, files)
    - Edges represent co-occurrence or semantic relationships
    """

    # Entity extraction patterns
    TECHNOLOGY_PATTERNS = {
        # Languages
        'python': re.compile(r'\bpython\b', re.I),
        'javascript': re.compile(r'\bjavascript\b|\bjs\b', re.I),
        'typescript': re.compile(r'\btypescript\b|\bts\b', re.I),
        'rust': re.compile(r'\brust\b', re.I),
        'go': re.compile(r'\bgolang\b|\bgo\b(?!\s+(?:to|for|ahead|back))', re.I),
        'java': re.compile(r'\bjava\b(?!script)', re.I),
        'c++': re.compile(r'\bc\+\+\b|\bcpp\b', re.I),
        'ruby': re.compile(r'\bruby\b', re.I),
        'php': re.compile(r'\bphp\b', re.I),
        'swift': re.compile(r'\bswift\b', re.I),
        'kotlin': re.compile(r'\bkotlin\b', re.I),

        # Frameworks
        'react': re.compile(r'\breact\b', re.I),
        'vue': re.compile(r'\bvue(?:\.?js)?\b', re.I),
        'angular': re.compile(r'\bangular\b', re.I),
        'django': re.compile(r'\bdjango\b', re.I),
        'flask': re.compile(r'\bflask\b', re.I),
        'fastapi': re.compile(r'\bfastapi\b', re.I),
        'express': re.compile(r'\bexpress\b', re.I),
        'nextjs': re.compile(r'\bnext\.?js\b', re.I),
        'rails': re.compile(r'\brails\b|\bruby on rails\b', re.I),

        # Tools
        'docker': re.compile(r'\bdocker\b', re.I),
        'kubernetes': re.compile(r'\bkubernetes\b|\bk8s\b', re.I),
        'git': re.compile(r'\bgit\b(?!hub)', re.I),
        'github': re.compile(r'\bgithub\b', re.I),
        'npm': re.compile(r'\bnpm\b', re.I),
        'pip': re.compile(r'\bpip\b', re.I),
        'webpack': re.compile(r'\bwebpack\b', re.I),
        'vite': re.compile(r'\bvite\b', re.I),

        # Databases
        'postgresql': re.compile(r'\bpostgres(?:ql)?\b', re.I),
        'mysql': re.compile(r'\bmysql\b', re.I),
        'mongodb': re.compile(r'\bmongodb\b|\bmongo\b', re.I),
        'redis': re.compile(r'\bredis\b', re.I),
        'sqlite': re.compile(r'\bsqlite\b', re.I),

        # Cloud
        'aws': re.compile(r'\baws\b|\bamazon web services\b', re.I),
        'gcp': re.compile(r'\bgcp\b|\bgoogle cloud\b', re.I),
        'azure': re.compile(r'\bazure\b', re.I),
    }

    CONCEPT_PATTERNS = {
        'api': re.compile(r'\bapi\b|\brest\b|\bgraphql\b', re.I),
        'authentication': re.compile(r'\bauth(?:entication)?\b|\boauth\b|\bjwt\b', re.I),
        'testing': re.compile(r'\btest(?:ing|s)?\b|\bunit test\b|\bpytest\b|\bjest\b', re.I),
        'deployment': re.compile(r'\bdeploy(?:ment)?\b|\bci/?cd\b', re.I),
        'debugging': re.compile(r'\bdebug(?:ging)?\b|\btroubleshoot', re.I),
        'performance': re.compile(r'\bperformance\b|\boptimiz(?:e|ation)\b', re.I),
        'security': re.compile(r'\bsecurity\b|\bvulnerabilit(?:y|ies)\b', re.I),
        'database': re.compile(r'\bdatabase\b|\bsql\b|\bquery\b', re.I),
        'frontend': re.compile(r'\bfrontend\b|\bclient.?side\b|\bui\b', re.I),
        'backend': re.compile(r'\bbackend\b|\bserver.?side\b', re.I),
        'async': re.compile(r'\basync\b|\bpromise\b|\bconcurrency\b', re.I),
        'caching': re.compile(r'\bcach(?:e|ing)\b', re.I),
        'logging': re.compile(r'\blog(?:ging|s)?\b', re.I),
        'error_handling': re.compile(r'\berror handling\b|\bexception\b', re.I),
    }

    ERROR_PATTERN = re.compile(r'(\w+(?:Error|Exception|Failure))(?:\s*:|:\s*|\s+)', re.I)
    FILE_PATTERN = re.compile(r'[\w./\\-]+\.(?:py|js|ts|jsx|tsx|java|go|rs|rb|php|vue|json|yaml|yml|toml|sql|md|html|css|scss)')

    def __init__(self):
        """Initialize empty knowledge graph."""
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[Tuple[str, str], Relationship] = {}
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)  # type -> entity_ids

    def build_from_conversations(self, conversations: List[dict]) -> 'KnowledgeGraph':
        """
        Build knowledge graph from conversations.

        Args:
            conversations: List of conversation dicts

        Returns:
            Self for chaining
        """
        for conv in conversations:
            convo_id = conv.get('convo_id', '')
            text = self._get_text(conv)

            # Extract entities
            entities_found = self._extract_entities(text, convo_id)

            # Add from enrichment data
            if 'technical_terms' in conv:
                for term in conv['technical_terms']:
                    self._add_entity(term.lower(), 'concept', convo_id)

            if 'code_languages' in conv:
                for lang in conv['code_languages']:
                    self._add_entity(lang.lower(), 'technology', convo_id)

            # Build relationships (co-occurrence)
            entity_list = list(entities_found)
            for i, e1 in enumerate(entity_list):
                for e2 in entity_list[i + 1:]:
                    self._add_relationship(e1, e2, 'co_occurs', convo_id)

        return self

    def _extract_entities(self, text: str, convo_id: str) -> Set[str]:
        """Extract entities from text."""
        found = set()

        # Technologies
        for name, pattern in self.TECHNOLOGY_PATTERNS.items():
            if pattern.search(text):
                self._add_entity(name, 'technology', convo_id)
                found.add(name)

        # Concepts
        for name, pattern in self.CONCEPT_PATTERNS.items():
            if pattern.search(text):
                self._add_entity(name, 'concept', convo_id)
                found.add(name)

        # Errors
        for match in self.ERROR_PATTERN.finditer(text):
            error_name = match.group(1).lower()
            self._add_entity(error_name, 'error', convo_id)
            found.add(error_name)

        # Files (limit to avoid noise)
        files_found = 0
        for match in self.FILE_PATTERN.finditer(text):
            if files_found >= 10:
                break
            file_name = match.group(0)
            self._add_entity(file_name, 'file', convo_id)
            found.add(file_name)
            files_found += 1

        return found

    def _add_entity(self, name: str, entity_type: str, convo_id: str):
        """Add or update an entity."""
        entity_id = f"{entity_type}:{name}"

        if entity_id not in self.entities:
            self.entities[entity_id] = Entity(
                id=entity_id,
                name=name,
                entity_type=entity_type,
            )
            self._entity_index[entity_type].add(entity_id)

        entity = self.entities[entity_id]
        entity.occurrence_count += 1
        entity.conversation_ids.add(convo_id)

    def _add_relationship(
        self,
        entity1: str,
        entity2: str,
        rel_type: str,
        convo_id: str
    ):
        """Add or update a relationship."""
        # Ensure consistent ordering
        if entity1 > entity2:
            entity1, entity2 = entity2, entity1

        # Find entity IDs
        e1_id = self._find_entity_id(entity1)
        e2_id = self._find_entity_id(entity2)

        if not e1_id or not e2_id:
            return

        key = (e1_id, e2_id)

        if key not in self.relationships:
            self.relationships[key] = Relationship(
                source_id=e1_id,
                target_id=e2_id,
                relationship_type=rel_type,
            )

        rel = self.relationships[key]
        rel.weight += 1
        rel.conversation_ids.add(convo_id)

    def _find_entity_id(self, name: str) -> Optional[str]:
        """Find entity ID by name."""
        for entity_type in ['technology', 'concept', 'error', 'file', 'pattern']:
            entity_id = f"{entity_type}:{name}"
            if entity_id in self.entities:
                return entity_id
        return None

    def _get_text(self, conv: dict) -> str:
        """Extract text from conversation."""
        parts = []
        if 'raw_text' in conv:
            parts.append(conv['raw_text'])
        if 'generated_title' in conv:
            parts.append(conv['generated_title'])
        if 'summary_abstractive' in conv:
            parts.append(conv['summary_abstractive'])
        return '\n'.join(parts)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def get_related(
        self,
        entity_name: str,
        limit: int = 10,
        min_weight: float = 1.0
    ) -> List[Tuple[Entity, float]]:
        """
        Find entities related to the given entity.

        Args:
            entity_name: Name of the entity
            limit: Maximum results
            min_weight: Minimum relationship weight

        Returns:
            List of (entity, weight) tuples
        """
        entity_id = self._find_entity_id(entity_name)
        if not entity_id:
            return []

        related = []

        for (e1_id, e2_id), rel in self.relationships.items():
            if rel.weight < min_weight:
                continue

            if e1_id == entity_id:
                other_id = e2_id
            elif e2_id == entity_id:
                other_id = e1_id
            else:
                continue

            other = self.entities.get(other_id)
            if other:
                related.append((other, rel.weight))

        # Sort by weight
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:limit]

    def get_top_entities(
        self,
        entity_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Entity]:
        """
        Get top entities by occurrence count.

        Args:
            entity_type: Filter by type (optional)
            limit: Maximum results

        Returns:
            List of entities sorted by occurrence
        """
        if entity_type:
            entity_ids = self._entity_index.get(entity_type, set())
            entities = [self.entities[eid] for eid in entity_ids]
        else:
            entities = list(self.entities.values())

        entities.sort(key=lambda e: e.occurrence_count, reverse=True)
        return entities[:limit]

    def get_entity_types(self) -> Dict[str, int]:
        """Get count of entities by type."""
        return {
            entity_type: len(entity_ids)
            for entity_type, entity_ids in self._entity_index.items()
        }

    def export_for_visualization(
        self,
        max_nodes: int = 100,
        max_edges: int = 200,
        min_node_count: int = 2,
        min_edge_weight: float = 2.0
    ) -> dict:
        """
        Export graph data for visualization (D3.js format).

        Args:
            max_nodes: Maximum number of nodes
            max_edges: Maximum number of edges
            min_node_count: Minimum entity occurrence count
            min_edge_weight: Minimum relationship weight

        Returns:
            Dict with 'nodes' and 'links' for D3.js
        """
        # Filter and sort nodes
        filtered_entities = [
            e for e in self.entities.values()
            if e.occurrence_count >= min_node_count
        ]
        filtered_entities.sort(key=lambda e: e.occurrence_count, reverse=True)
        top_entities = filtered_entities[:max_nodes]

        node_ids = {e.id for e in top_entities}

        # Build nodes
        nodes = []
        for entity in top_entities:
            nodes.append({
                'id': entity.id,
                'name': entity.name,
                'type': entity.entity_type,
                'count': entity.occurrence_count,
                'size': math.log(entity.occurrence_count + 1) * 10,
            })

        # Filter and sort edges
        filtered_rels = [
            rel for rel in self.relationships.values()
            if rel.weight >= min_edge_weight
            and rel.source_id in node_ids
            and rel.target_id in node_ids
        ]
        filtered_rels.sort(key=lambda r: r.weight, reverse=True)
        top_rels = filtered_rels[:max_edges]

        # Build links
        links = []
        for rel in top_rels:
            links.append({
                'source': rel.source_id,
                'target': rel.target_id,
                'type': rel.relationship_type,
                'weight': rel.weight,
            })

        return {
            'nodes': nodes,
            'links': links,
            'stats': {
                'total_entities': len(self.entities),
                'total_relationships': len(self.relationships),
                'shown_nodes': len(nodes),
                'shown_links': len(links),
            }
        }

    def find_paths(
        self,
        start: str,
        end: str,
        max_depth: int = 3
    ) -> List[List[str]]:
        """
        Find paths between two entities.

        Args:
            start: Starting entity name
            end: Ending entity name
            max_depth: Maximum path length

        Returns:
            List of paths (each path is list of entity IDs)
        """
        start_id = self._find_entity_id(start)
        end_id = self._find_entity_id(end)

        if not start_id or not end_id:
            return []

        # Build adjacency list
        adj: Dict[str, Set[str]] = defaultdict(set)
        for (e1_id, e2_id), _ in self.relationships.items():
            adj[e1_id].add(e2_id)
            adj[e2_id].add(e1_id)

        # BFS for paths
        paths = []
        queue = [(start_id, [start_id])]
        visited = {start_id}

        while queue and len(paths) < 10:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            for neighbor in adj[current]:
                if neighbor == end_id:
                    paths.append(path + [end_id])
                elif neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return paths

    def get_clusters(self, min_cluster_size: int = 3) -> List[Set[str]]:
        """
        Find clusters of highly connected entities.

        Uses connected components on filtered graph.
        """
        # Build adjacency for strong connections only
        adj: Dict[str, Set[str]] = defaultdict(set)
        for (e1_id, e2_id), rel in self.relationships.items():
            if rel.weight >= 3:  # Strong connection threshold
                adj[e1_id].add(e2_id)
                adj[e2_id].add(e1_id)

        # Find connected components
        visited = set()
        clusters = []

        for start in adj.keys():
            if start in visited:
                continue

            # BFS to find component
            component = set()
            queue = [start]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)

                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) >= min_cluster_size:
                clusters.append(component)

        return sorted(clusters, key=len, reverse=True)

    def to_dict(self) -> dict:
        """Export entire graph as dict."""
        return {
            'entities': [e.to_dict() for e in self.entities.values()],
            'relationships': [r.to_dict() for r in self.relationships.values()],
            'stats': {
                'entity_count': len(self.entities),
                'relationship_count': len(self.relationships),
                'entity_types': self.get_entity_types(),
            }
        }

    def save(self, path: str):
        """Save graph to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'KnowledgeGraph':
        """Load graph from JSON file."""
        kg = cls()

        with open(path) as f:
            data = json.load(f)

        for e_data in data.get('entities', []):
            entity = Entity(
                id=e_data['id'],
                name=e_data['name'],
                entity_type=e_data['type'],
                occurrence_count=e_data['count'],
                conversation_ids=set(e_data.get('conversations', [])),
                properties=e_data.get('properties', {}),
            )
            kg.entities[entity.id] = entity
            kg._entity_index[entity.entity_type].add(entity.id)

        for r_data in data.get('relationships', []):
            rel = Relationship(
                source_id=r_data['source'],
                target_id=r_data['target'],
                relationship_type=r_data['type'],
                weight=r_data['weight'],
                conversation_ids=set(r_data.get('conversations', [])),
            )
            kg.relationships[(rel.source_id, rel.target_id)] = rel

        return kg


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Build knowledge graph from conversations")
    parser.add_argument('input', help="Input JSONL file")
    parser.add_argument('--output', '-o', help="Output JSON file")
    parser.add_argument('--visualize', '-v', help="Output visualization JSON")

    args = parser.parse_args()

    # Load conversations
    conversations = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    print(f"Loaded {len(conversations)} conversations")

    # Build graph
    kg = KnowledgeGraph()
    kg.build_from_conversations(conversations)

    print(f"\nKnowledge Graph:")
    print(f"  Entities: {len(kg.entities)}")
    print(f"  Relationships: {len(kg.relationships)}")

    print("\nEntity types:")
    for etype, count in kg.get_entity_types().items():
        print(f"  {etype}: {count}")

    print("\nTop technologies:")
    for entity in kg.get_top_entities('technology', limit=10):
        print(f"  {entity.name}: {entity.occurrence_count}")

    if args.output:
        kg.save(args.output)
        print(f"\nSaved to {args.output}")

    if args.visualize:
        viz = kg.export_for_visualization()
        with open(args.visualize, 'w') as f:
            json.dump(viz, f, indent=2)
        print(f"Saved visualization to {args.visualize}")
