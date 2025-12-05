"""
Conversation Chain Detection for CogRepo v2

Detects linked conversations that form problem-solving journeys:
- Direct continuations ("as we discussed earlier")
- Follow-up questions on same topic
- Related debugging sessions
- Iterative development cycles

Uses multiple signals:
1. Temporal proximity
2. Semantic similarity
3. Shared entities (files, errors, terms)
4. Explicit references

Zero-token for basic chains, embeddings for semantic linking.

Usage:
    from context.chain_detection import ChainDetector

    detector = ChainDetector()
    chains = detector.detect_chains(conversations)

    for chain in chains:
        print(f"Chain: {len(chain.conversation_ids)} conversations")
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


@dataclass
class ConversationChain:
    """Represents a chain of related conversations."""
    chain_id: str
    conversation_ids: List[str]
    chain_type: str  # 'continuation', 'follow_up', 'debug_session', 'development'
    confidence: float
    shared_entities: Set[str] = field(default_factory=set)
    time_span_hours: float = 0.0
    topic_summary: str = ""

    def to_dict(self) -> dict:
        return {
            'chain_id': self.chain_id,
            'conversation_ids': self.conversation_ids,
            'chain_type': self.chain_type,
            'confidence': self.confidence,
            'shared_entities': list(self.shared_entities),
            'time_span_hours': self.time_span_hours,
            'topic_summary': self.topic_summary,
        }


class ChainDetector:
    """
    Detect chains of related conversations.

    Detection strategies:
    1. Explicit references: "as I mentioned", "continuing from", etc.
    2. Temporal + entity overlap: Same files/errors within 24 hours
    3. Semantic similarity: Similar embeddings
    4. Error trails: Same error appearing across conversations
    """

    # Patterns for explicit continuation references
    CONTINUATION_PATTERNS = [
        re.compile(r'(?:as|like)\s+(?:I|we)\s+(?:mentioned|discussed|said)\s+(?:earlier|before|previously)', re.I),
        re.compile(r'continuing\s+(?:from|with|on)\s+(?:my|our|the)\s+(?:previous|earlier|last)', re.I),
        re.compile(r'(?:back|returning)\s+to\s+(?:the|my|our)\s+(?:earlier|previous)', re.I),
        re.compile(r'(?:following|building)\s+(?:up|on)\s+(?:from|what)', re.I),
        re.compile(r'(?:still|again)\s+(?:having|getting|seeing)\s+(?:the|this|that)\s+(?:same|similar)', re.I),
        re.compile(r'(?:remember|recall)\s+(?:when|that|the)\s+(?:we|I)', re.I),
    ]

    # Time windows for chain detection
    CONTINUATION_WINDOW_HOURS = 4  # Direct continuations
    FOLLOWUP_WINDOW_HOURS = 24  # Same-day follow-ups
    SESSION_WINDOW_HOURS = 72  # Extended problem-solving

    def __init__(
        self,
        embedding_store=None,
        similarity_threshold: float = 0.7,
        min_entity_overlap: int = 2
    ):
        """
        Initialize the chain detector.

        Args:
            embedding_store: Optional EmbeddingStore for semantic matching
            similarity_threshold: Min cosine similarity for semantic chains
            min_entity_overlap: Min shared entities for entity-based chains
        """
        self.embedding_store = embedding_store
        self.similarity_threshold = similarity_threshold
        self.min_entity_overlap = min_entity_overlap

    def detect_chains(self, conversations: List[dict]) -> List[ConversationChain]:
        """
        Detect all conversation chains.

        Args:
            conversations: List of conversation dicts

        Returns:
            List of ConversationChain objects
        """
        # Sort by creation time
        sorted_convos = sorted(
            conversations,
            key=lambda c: c.get('created_at', ''),
        )

        chains = []

        # 1. Detect explicit continuations
        explicit_chains = self._detect_explicit_continuations(sorted_convos)
        chains.extend(explicit_chains)

        # 2. Detect entity-based chains
        entity_chains = self._detect_entity_chains(sorted_convos)
        chains.extend(entity_chains)

        # 3. Detect semantic chains (if embeddings available)
        if self.embedding_store:
            semantic_chains = self._detect_semantic_chains(sorted_convos)
            chains.extend(semantic_chains)

        # 4. Detect debug session chains
        debug_chains = self._detect_debug_sessions(sorted_convos)
        chains.extend(debug_chains)

        # Merge overlapping chains
        merged = self._merge_chains(chains)

        return sorted(merged, key=lambda c: len(c.conversation_ids), reverse=True)

    def _detect_explicit_continuations(
        self,
        conversations: List[dict]
    ) -> List[ConversationChain]:
        """Find conversations with explicit continuation references."""
        chains = []
        used = set()

        for i, conv in enumerate(conversations):
            if conv.get('convo_id') in used:
                continue

            text = self._get_text(conv)

            # Check for continuation patterns
            is_continuation = any(p.search(text) for p in self.CONTINUATION_PATTERNS)

            if is_continuation:
                # Find likely predecessor
                chain_convos = [conv]
                created_at = self._parse_time(conv.get('created_at', ''))

                if created_at:
                    # Look back for related conversations
                    for j in range(i - 1, -1, -1):
                        prev = conversations[j]
                        prev_time = self._parse_time(prev.get('created_at', ''))

                        if not prev_time:
                            continue

                        hours_diff = (created_at - prev_time).total_seconds() / 3600

                        if hours_diff > self.CONTINUATION_WINDOW_HOURS:
                            break

                        # Check for entity overlap
                        if self._has_entity_overlap(conv, prev):
                            chain_convos.insert(0, prev)
                            used.add(prev.get('convo_id'))

                if len(chain_convos) >= 2:
                    used.add(conv.get('convo_id'))
                    chains.append(self._create_chain(
                        chain_convos,
                        'continuation',
                        confidence=0.9
                    ))

        return chains

    def _detect_entity_chains(
        self,
        conversations: List[dict]
    ) -> List[ConversationChain]:
        """Find conversations linked by shared entities."""
        chains = []

        # Build entity index
        entity_to_convos: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)

        for i, conv in enumerate(conversations):
            entities = self._extract_entities(conv)
            for entity in entities:
                entity_to_convos[entity].append((i, conv))

        # Find chains based on entity overlap
        used_pairs = set()

        for entity, conv_list in entity_to_convos.items():
            if len(conv_list) < 2:
                continue

            # Check pairs within time window
            for i, (idx1, conv1) in enumerate(conv_list):
                for idx2, conv2 in conv_list[i + 1:]:
                    pair_key = (conv1.get('convo_id'), conv2.get('convo_id'))
                    if pair_key in used_pairs:
                        continue

                    time1 = self._parse_time(conv1.get('created_at', ''))
                    time2 = self._parse_time(conv2.get('created_at', ''))

                    if not (time1 and time2):
                        continue

                    hours_diff = abs((time2 - time1).total_seconds() / 3600)

                    if hours_diff <= self.FOLLOWUP_WINDOW_HOURS:
                        overlap = self._count_entity_overlap(conv1, conv2)
                        if overlap >= self.min_entity_overlap:
                            used_pairs.add(pair_key)
                            chains.append(self._create_chain(
                                [conv1, conv2],
                                'follow_up',
                                confidence=min(0.5 + overlap * 0.1, 0.9)
                            ))

        return chains

    def _detect_semantic_chains(
        self,
        conversations: List[dict]
    ) -> List[ConversationChain]:
        """Find conversations linked by semantic similarity."""
        if not self.embedding_store or self.embedding_store.embeddings is None:
            return []

        chains = []
        embeddings = self.embedding_store.embeddings
        ids = self.embedding_store.ids

        # Build ID to index map
        id_to_idx = {cid: i for i, cid in enumerate(ids)}

        for i, conv in enumerate(conversations):
            convo_id = conv.get('convo_id')
            if convo_id not in id_to_idx:
                continue

            idx = id_to_idx[convo_id]
            query_vec = embeddings[idx]

            # Find similar conversations
            from search.embeddings import EmbeddingEngine
            engine = EmbeddingEngine()

            similar = engine.find_similar(
                query_vec,
                embeddings,
                top_k=5,
                threshold=self.similarity_threshold
            )

            # Filter to within time window
            created_at = self._parse_time(conv.get('created_at', ''))
            if not created_at:
                continue

            related = []
            for sim_idx, score in similar:
                if sim_idx == idx:
                    continue

                sim_id = ids[sim_idx]
                sim_conv = next((c for c in conversations if c.get('convo_id') == sim_id), None)

                if sim_conv:
                    sim_time = self._parse_time(sim_conv.get('created_at', ''))
                    if sim_time:
                        hours_diff = abs((created_at - sim_time).total_seconds() / 3600)
                        if hours_diff <= self.SESSION_WINDOW_HOURS:
                            related.append((sim_conv, score))

            if related:
                chain_convos = [conv] + [c for c, _ in related]
                avg_score = sum(s for _, s in related) / len(related)
                chains.append(self._create_chain(
                    chain_convos,
                    'semantic_link',
                    confidence=avg_score
                ))

        return chains

    def _detect_debug_sessions(
        self,
        conversations: List[dict]
    ) -> List[ConversationChain]:
        """Detect debugging sessions based on error patterns."""
        chains = []

        # Build error index
        error_to_convos: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)

        for i, conv in enumerate(conversations):
            errors = self._extract_errors(conv)
            for error in errors:
                error_to_convos[error].append((i, conv))

        # Find debug chains
        for error, conv_list in error_to_convos.items():
            if len(conv_list) >= 2:
                # Check if within time window
                convos = [c for _, c in conv_list]
                times = [self._parse_time(c.get('created_at', '')) for c in convos]
                times = [t for t in times if t]

                if len(times) >= 2:
                    time_span = (max(times) - min(times)).total_seconds() / 3600

                    if time_span <= self.SESSION_WINDOW_HOURS:
                        chains.append(self._create_chain(
                            convos,
                            'debug_session',
                            confidence=0.8
                        ))

        return chains

    def _merge_chains(self, chains: List[ConversationChain]) -> List[ConversationChain]:
        """Merge overlapping chains."""
        if not chains:
            return []

        # Group by overlapping conversation sets
        merged = []
        used = set()

        for i, chain in enumerate(chains):
            if i in used:
                continue

            # Start new merged chain
            merged_ids = set(chain.conversation_ids)
            merged_types = {chain.chain_type}
            merged_entities = set(chain.shared_entities)
            max_confidence = chain.confidence

            # Find overlapping chains
            for j, other in enumerate(chains):
                if j == i or j in used:
                    continue

                other_ids = set(other.conversation_ids)
                overlap = len(merged_ids & other_ids)

                if overlap >= 1:
                    merged_ids.update(other_ids)
                    merged_types.add(other.chain_type)
                    merged_entities.update(other.shared_entities)
                    max_confidence = max(max_confidence, other.confidence)
                    used.add(j)

            used.add(i)

            # Determine chain type
            if 'continuation' in merged_types:
                chain_type = 'continuation'
            elif 'debug_session' in merged_types:
                chain_type = 'debug_session'
            elif 'follow_up' in merged_types:
                chain_type = 'follow_up'
            else:
                chain_type = 'semantic_link'

            merged.append(ConversationChain(
                chain_id=f"chain_{len(merged)}",
                conversation_ids=list(merged_ids),
                chain_type=chain_type,
                confidence=max_confidence,
                shared_entities=merged_entities,
            ))

        return merged

    def _create_chain(
        self,
        conversations: List[dict],
        chain_type: str,
        confidence: float
    ) -> ConversationChain:
        """Create a ConversationChain from conversations."""
        # Collect shared entities
        entity_counts: Dict[str, int] = defaultdict(int)
        for conv in conversations:
            for entity in self._extract_entities(conv):
                entity_counts[entity] += 1

        shared = {e for e, c in entity_counts.items() if c >= 2}

        # Calculate time span
        times = [self._parse_time(c.get('created_at', '')) for c in conversations]
        times = [t for t in times if t]
        time_span = 0.0
        if len(times) >= 2:
            time_span = (max(times) - min(times)).total_seconds() / 3600

        return ConversationChain(
            chain_id=f"chain_{id(conversations)}",
            conversation_ids=[c.get('convo_id', '') for c in conversations],
            chain_type=chain_type,
            confidence=confidence,
            shared_entities=shared,
            time_span_hours=time_span,
        )

    def _get_text(self, conv: dict) -> str:
        """Extract text from conversation."""
        parts = []
        if 'raw_text' in conv:
            parts.append(conv['raw_text'])
        if 'generated_title' in conv:
            parts.append(conv['generated_title'])
        return '\n'.join(parts)

    def _parse_time(self, time_str: str) -> Optional[datetime]:
        """Parse ISO timestamp."""
        if not time_str:
            return None

        try:
            # Handle various formats
            for fmt in [
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
            ]:
                try:
                    return datetime.strptime(time_str, fmt)
                except ValueError:
                    continue
        except Exception:
            pass

        return None

    def _extract_entities(self, conv: dict) -> Set[str]:
        """Extract entities from conversation."""
        entities = set()

        # File paths
        if 'file_paths' in conv:
            entities.update(conv['file_paths'])

        # Technical terms
        if 'technical_terms' in conv:
            entities.update(conv['technical_terms'])

        # Error types
        entities.update(self._extract_errors(conv))

        # Technologies
        if 'code_languages' in conv:
            entities.update(conv['code_languages'])

        return entities

    def _extract_errors(self, conv: dict) -> Set[str]:
        """Extract error signatures from conversation."""
        errors = set()
        text = self._get_text(conv)

        # Common error patterns
        patterns = [
            re.compile(r'(\w+Error):', re.I),
            re.compile(r'(\w+Exception)', re.I),
            re.compile(r'error:\s*(\w+)', re.I),
            re.compile(r'failed:\s*(\w+)', re.I),
        ]

        for pattern in patterns:
            for match in pattern.finditer(text):
                errors.add(match.group(1).lower())

        return errors

    def _has_entity_overlap(self, conv1: dict, conv2: dict) -> bool:
        """Check if two conversations share entities."""
        return self._count_entity_overlap(conv1, conv2) >= 1

    def _count_entity_overlap(self, conv1: dict, conv2: dict) -> int:
        """Count shared entities between conversations."""
        entities1 = self._extract_entities(conv1)
        entities2 = self._extract_entities(conv2)
        return len(entities1 & entities2)


def find_related(
    conversation: dict,
    all_conversations: List[dict],
    max_results: int = 5
) -> List[Tuple[dict, float, str]]:
    """
    Find conversations related to a given one.

    Args:
        conversation: The source conversation
        all_conversations: All conversations to search
        max_results: Maximum number of results

    Returns:
        List of (conversation, score, relation_type) tuples
    """
    detector = ChainDetector()
    results = []

    source_id = conversation.get('convo_id')
    source_entities = detector._extract_entities(conversation)
    source_time = detector._parse_time(conversation.get('created_at', ''))

    for other in all_conversations:
        other_id = other.get('convo_id')
        if other_id == source_id:
            continue

        # Calculate entity overlap
        overlap = detector._count_entity_overlap(conversation, other)

        # Calculate time proximity
        other_time = detector._parse_time(other.get('created_at', ''))
        time_score = 0.0
        if source_time and other_time:
            hours_diff = abs((source_time - other_time).total_seconds() / 3600)
            if hours_diff <= 24:
                time_score = 1.0 - (hours_diff / 24)

        # Combined score
        score = (overlap * 0.5) + (time_score * 0.5)

        if score > 0:
            relation = 'related'
            if overlap >= 3:
                relation = 'follow_up'
            elif time_score > 0.8:
                relation = 'continuation'

            results.append((other, score, relation))

    # Sort by score and limit
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:max_results]


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Detect conversation chains")
    parser.add_argument('input', help="Input JSONL file")
    parser.add_argument('--output', '-o', help="Output JSON file (optional)")
    parser.add_argument('--similarity', '-s', type=float, default=0.7,
                        help="Similarity threshold for semantic matching")

    args = parser.parse_args()

    # Load conversations
    conversations = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    print(f"Loaded {len(conversations)} conversations")

    # Detect chains
    detector = ChainDetector(similarity_threshold=args.similarity)
    chains = detector.detect_chains(conversations)

    print(f"\nFound {len(chains)} chains:")
    for chain in chains:
        print(f"\n  {chain.chain_id} ({chain.chain_type})")
        print(f"    Conversations: {len(chain.conversation_ids)}")
        print(f"    Confidence: {chain.confidence:.2f}")
        if chain.shared_entities:
            print(f"    Shared: {', '.join(list(chain.shared_entities)[:5])}")
        if chain.time_span_hours:
            print(f"    Time span: {chain.time_span_hours:.1f} hours")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump([c.to_dict() for c in chains], f, indent=2)
        print(f"\nSaved to {args.output}")
