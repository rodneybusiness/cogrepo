"""
CogRepo Recommendation Engine

Provides intelligent recommendations based on:
- Vector similarity (find similar conversations)
- Entity co-occurrence (related topics)
- User patterns (frequently visited topics)
- Temporal patterns (recently active topics)

Usage:
    from intelligence.recommendations import RecommendationEngine

    engine = RecommendationEngine(embeddings, ids)

    # Find similar conversations
    similar = engine.find_similar('convo_id', limit=5)

    # Get recommendations based on history
    recs = engine.recommend_for_history(['id1', 'id2', 'id3'])
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict
import json


@dataclass
class Recommendation:
    """A recommended conversation."""
    conversation_id: str
    score: float
    reason: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> dict:
        return {
            'conversation_id': self.conversation_id,
            'score': self.score,
            'reason': self.reason,
            'metadata': self.metadata or {},
        }


class RecommendationEngine:
    """
    Multi-signal recommendation engine.

    Combines multiple signals:
    1. Semantic similarity (embedding cosine distance)
    2. Entity overlap (shared technologies, concepts)
    3. Tag similarity (shared tags)
    4. Temporal patterns (related by time)
    """

    def __init__(
        self,
        embeddings: np.ndarray = None,
        ids: List[str] = None,
        conversations: List[dict] = None
    ):
        """
        Initialize recommendation engine.

        Args:
            embeddings: Numpy array of embeddings
            ids: List of conversation IDs matching embeddings
            conversations: List of conversation dicts for metadata
        """
        self.embeddings = embeddings
        self.ids = ids or []
        self.conversations = conversations or []

        # Build indices
        self._id_to_idx = {cid: i for i, cid in enumerate(self.ids)} if ids else {}
        self._id_to_conv = {c.get('convo_id', ''): c for c in self.conversations}

        # Build entity index
        self._entity_to_ids = self._build_entity_index()

    def _build_entity_index(self) -> Dict[str, Set[str]]:
        """Build index of entities to conversation IDs."""
        index = defaultdict(set)

        for conv in self.conversations:
            cid = conv.get('convo_id', '')

            # Index by tags
            for tag in conv.get('tags', []):
                index[f"tag:{tag.lower()}"].add(cid)

            # Index by technologies
            for lang in conv.get('code_languages', []):
                index[f"lang:{lang.lower()}"].add(cid)

            # Index by technical terms
            for term in conv.get('technical_terms', []):
                index[f"term:{term.lower()}"].add(cid)

            # Index by domain
            domain = conv.get('primary_domain', '')
            if domain:
                index[f"domain:{domain.lower()}"].add(cid)

        return dict(index)

    def find_similar(
        self,
        conversation_id: str,
        limit: int = 5,
        exclude_self: bool = True
    ) -> List[Recommendation]:
        """
        Find conversations similar to the given one.

        Args:
            conversation_id: Source conversation ID
            limit: Maximum recommendations
            exclude_self: Exclude the source conversation

        Returns:
            List of Recommendation objects
        """
        if self.embeddings is None or conversation_id not in self._id_to_idx:
            return self._fallback_recommendations(conversation_id, limit)

        # Get embedding
        idx = self._id_to_idx[conversation_id]
        query_vec = self.embeddings[idx]

        # Calculate similarities
        similarities = self._cosine_similarities(query_vec, self.embeddings)

        # Get top matches
        top_indices = np.argsort(similarities)[::-1]

        recommendations = []
        for i in top_indices:
            if len(recommendations) >= limit:
                break

            cid = self.ids[i]
            if exclude_self and cid == conversation_id:
                continue

            score = float(similarities[i])
            if score < 0.3:  # Minimum similarity threshold
                break

            recommendations.append(Recommendation(
                conversation_id=cid,
                score=score,
                reason="Similar content",
                metadata={'similarity': score}
            ))

        return recommendations

    def _fallback_recommendations(
        self,
        conversation_id: str,
        limit: int
    ) -> List[Recommendation]:
        """Fallback to entity-based recommendations when no embeddings."""
        conv = self._id_to_conv.get(conversation_id, {})

        # Collect entities from source conversation
        source_entities = set()
        source_entities.update(f"tag:{t.lower()}" for t in conv.get('tags', []))
        source_entities.update(f"lang:{l.lower()}" for l in conv.get('code_languages', []))
        source_entities.update(f"term:{t.lower()}" for t in conv.get('technical_terms', []))

        # Find conversations with overlapping entities
        scores = defaultdict(float)
        for entity in source_entities:
            for cid in self._entity_to_ids.get(entity, set()):
                if cid != conversation_id:
                    scores[cid] += 1.0

        # Normalize and convert to recommendations
        max_score = max(scores.values()) if scores else 1.0
        recommendations = []

        for cid, score in sorted(scores.items(), key=lambda x: -x[1])[:limit]:
            recommendations.append(Recommendation(
                conversation_id=cid,
                score=score / max_score,
                reason="Related topic",
                metadata={'entity_overlap': int(score)}
            ))

        return recommendations

    def recommend_for_history(
        self,
        history: List[str],
        limit: int = 5
    ) -> List[Recommendation]:
        """
        Get recommendations based on viewing history.

        Args:
            history: List of recently viewed conversation IDs
            limit: Maximum recommendations

        Returns:
            List of Recommendation objects
        """
        if not history:
            return []

        exclude_set = set(history)

        if self.embeddings is None:
            return self._history_fallback(history, limit, exclude_set)

        # Get embeddings for history
        history_vectors = []
        for cid in history:
            if cid in self._id_to_idx:
                idx = self._id_to_idx[cid]
                history_vectors.append(self.embeddings[idx])

        if not history_vectors:
            return self._history_fallback(history, limit, exclude_set)

        # Calculate centroid of history
        centroid = np.mean(history_vectors, axis=0)

        # Find similar to centroid
        similarities = self._cosine_similarities(centroid, self.embeddings)

        # Get top matches
        top_indices = np.argsort(similarities)[::-1]

        recommendations = []
        for i in top_indices:
            if len(recommendations) >= limit:
                break

            cid = self.ids[i]
            if cid in exclude_set:
                continue

            score = float(similarities[i])
            if score < 0.3:
                break

            recommendations.append(Recommendation(
                conversation_id=cid,
                score=score,
                reason="Based on your history",
                metadata={'similarity': score}
            ))

        return recommendations

    def _history_fallback(
        self,
        history: List[str],
        limit: int,
        exclude_set: Set[str]
    ) -> List[Recommendation]:
        """Fallback history recommendations using entity overlap."""
        # Collect all entities from history
        history_entities = set()
        for cid in history:
            conv = self._id_to_conv.get(cid, {})
            history_entities.update(f"tag:{t.lower()}" for t in conv.get('tags', []))
            history_entities.update(f"lang:{l.lower()}" for l in conv.get('code_languages', []))

        # Score candidates
        scores = defaultdict(float)
        for entity in history_entities:
            for cid in self._entity_to_ids.get(entity, set()):
                if cid not in exclude_set:
                    scores[cid] += 1.0

        # Convert to recommendations
        max_score = max(scores.values()) if scores else 1.0
        recommendations = []

        for cid, score in sorted(scores.items(), key=lambda x: -x[1])[:limit]:
            recommendations.append(Recommendation(
                conversation_id=cid,
                score=score / max_score,
                reason="Based on your history",
                metadata={'entity_matches': int(score)}
            ))

        return recommendations

    def recommend_by_entity(
        self,
        entity: str,
        entity_type: str = 'tag',
        limit: int = 10
    ) -> List[Recommendation]:
        """
        Get recommendations for a specific entity.

        Args:
            entity: Entity name
            entity_type: Type (tag, lang, term, domain)
            limit: Maximum recommendations

        Returns:
            List of Recommendation objects
        """
        key = f"{entity_type}:{entity.lower()}"
        matching_ids = self._entity_to_ids.get(key, set())

        recommendations = []
        for cid in list(matching_ids)[:limit]:
            conv = self._id_to_conv.get(cid, {})
            score = conv.get('score', 50) / 100.0  # Use quality score

            recommendations.append(Recommendation(
                conversation_id=cid,
                score=score,
                reason=f"Tagged with {entity}",
                metadata={'entity': entity, 'type': entity_type}
            ))

        # Sort by score
        recommendations.sort(key=lambda r: r.score, reverse=True)
        return recommendations

    def get_trending(
        self,
        conversations: List[dict],
        days: int = 7,
        limit: int = 10
    ) -> List[Recommendation]:
        """
        Get trending/recent high-quality conversations.

        Args:
            conversations: All conversations
            days: Look back period
            limit: Maximum recommendations

        Returns:
            List of Recommendation objects
        """
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(days=days)
        recent = []

        for conv in conversations:
            # Parse timestamp
            timestamp_str = conv.get('created_at', conv.get('timestamp', ''))
            if not timestamp_str:
                continue

            try:
                for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d']:
                    try:
                        timestamp = datetime.strptime(timestamp_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue

                if timestamp >= cutoff:
                    score = conv.get('score', 50)
                    recent.append((conv, score, timestamp))

            except Exception:
                continue

        # Sort by score then recency
        recent.sort(key=lambda x: (x[1], x[2]), reverse=True)

        recommendations = []
        for conv, score, timestamp in recent[:limit]:
            recommendations.append(Recommendation(
                conversation_id=conv.get('convo_id', ''),
                score=score / 100.0,
                reason="Recent & high quality",
                metadata={'timestamp': timestamp.isoformat(), 'quality_score': score}
            ))

        return recommendations

    def _cosine_similarities(
        self,
        query: np.ndarray,
        corpus: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarities between query and corpus."""
        # Normalize
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        corpus_norms = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)

        # Dot product
        return np.dot(corpus_norms, query_norm)


def get_recommendations(
    conversation_id: str,
    embeddings_path: str,
    ids_path: str,
    conversations_path: str,
    limit: int = 5
) -> List[dict]:
    """
    Get recommendations for a conversation.

    Args:
        conversation_id: Source conversation ID
        embeddings_path: Path to embeddings .npy
        ids_path: Path to IDs JSON
        conversations_path: Path to conversations JSONL
        limit: Maximum recommendations

    Returns:
        List of recommendation dicts
    """
    # Load data
    embeddings = np.load(embeddings_path)

    with open(ids_path) as f:
        ids = json.load(f)

    conversations = []
    with open(conversations_path) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    # Get recommendations
    engine = RecommendationEngine(embeddings, ids, conversations)
    recs = engine.find_similar(conversation_id, limit=limit)

    return [r.to_dict() for r in recs]


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Get conversation recommendations")
    parser.add_argument('conversation_id', help="Source conversation ID")
    parser.add_argument('--embeddings', '-e', required=True, help="Embeddings .npy file")
    parser.add_argument('--ids', '-i', required=True, help="IDs JSON file")
    parser.add_argument('--conversations', '-c', required=True, help="Conversations JSONL")
    parser.add_argument('--limit', '-l', type=int, default=5, help="Max recommendations")

    args = parser.parse_args()

    recs = get_recommendations(
        args.conversation_id,
        args.embeddings,
        args.ids,
        args.conversations,
        args.limit
    )

    print(f"\nRecommendations for {args.conversation_id}:")
    for rec in recs:
        print(f"  {rec['conversation_id']}: {rec['score']:.2f} - {rec['reason']}")
