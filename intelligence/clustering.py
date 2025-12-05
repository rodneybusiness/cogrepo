"""
CogRepo Conversation Clustering

Auto-clusters conversations based on semantic similarity using embeddings.
Supports multiple clustering algorithms with fallback.

Features:
- HDBSCAN for density-based clustering (preferred)
- K-Means fallback for simpler clustering
- Cluster labeling with topic extraction
- Outlier detection

Usage:
    from intelligence.clustering import ConversationClusterer

    clusterer = ConversationClusterer()
    clusters = clusterer.cluster(conversations, embeddings)

    for cluster in clusters:
        print(f"Cluster {cluster.id}: {cluster.label}")
        print(f"  Size: {len(cluster.conversation_ids)}")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import Counter
import json


@dataclass
class Cluster:
    """Represents a cluster of conversations."""
    id: str
    label: str
    conversation_ids: List[str]
    centroid: Optional[np.ndarray] = None
    keywords: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    coherence_score: float = 0.0
    is_outlier: bool = False

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'label': self.label,
            'conversation_ids': self.conversation_ids,
            'keywords': self.keywords,
            'technologies': self.technologies,
            'coherence_score': self.coherence_score,
            'size': len(self.conversation_ids),
            'is_outlier': self.is_outlier,
        }


class ConversationClusterer:
    """
    Cluster conversations using embeddings.

    Supports:
    - HDBSCAN (density-based, handles noise)
    - K-Means (simpler, requires k)
    - Agglomerative (hierarchical)
    """

    def __init__(
        self,
        algorithm: str = 'auto',
        min_cluster_size: int = 3,
        n_clusters: int = 10
    ):
        """
        Initialize clusterer.

        Args:
            algorithm: 'hdbscan', 'kmeans', 'agglomerative', or 'auto'
            min_cluster_size: Minimum conversations per cluster (HDBSCAN)
            n_clusters: Number of clusters (K-Means/Agglomerative)
        """
        self.algorithm = algorithm
        self.min_cluster_size = min_cluster_size
        self.n_clusters = n_clusters
        self._algorithm_impl = None

    def cluster(
        self,
        conversations: List[dict],
        embeddings: np.ndarray,
        ids: List[str]
    ) -> List[Cluster]:
        """
        Cluster conversations.

        Args:
            conversations: List of conversation dicts
            embeddings: Numpy array of embeddings (n_samples, n_dims)
            ids: List of conversation IDs matching embeddings

        Returns:
            List of Cluster objects
        """
        if len(embeddings) < self.min_cluster_size:
            return []

        # Build ID to conversation map
        id_to_conv = {c.get('convo_id', ''): c for c in conversations}

        # Get clustering labels
        labels = self._cluster_embeddings(embeddings)

        # Group by cluster
        clusters = self._build_clusters(
            labels, ids, embeddings, id_to_conv
        )

        # Generate labels for clusters
        self._label_clusters(clusters, id_to_conv)

        return sorted(clusters, key=lambda c: len(c.conversation_ids), reverse=True)

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform clustering on embeddings."""
        algorithm = self.algorithm

        # Auto-select algorithm
        if algorithm == 'auto':
            try:
                import hdbscan
                algorithm = 'hdbscan'
            except ImportError:
                try:
                    from sklearn.cluster import KMeans
                    algorithm = 'kmeans'
                except ImportError:
                    raise ImportError(
                        "No clustering library available. "
                        "Install hdbscan or scikit-learn."
                    )

        if algorithm == 'hdbscan':
            return self._cluster_hdbscan(embeddings)
        elif algorithm == 'kmeans':
            return self._cluster_kmeans(embeddings)
        elif algorithm == 'agglomerative':
            return self._cluster_agglomerative(embeddings)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _cluster_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster using HDBSCAN."""
        try:
            import hdbscan
        except ImportError:
            # Fallback to kmeans
            return self._cluster_kmeans(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom',
        )

        labels = clusterer.fit_predict(embeddings)
        self._algorithm_impl = clusterer
        return labels

    def _cluster_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster using K-Means."""
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("scikit-learn required for K-Means clustering")

        # Determine number of clusters
        n_clusters = min(self.n_clusters, len(embeddings) // self.min_cluster_size)
        n_clusters = max(2, n_clusters)

        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )

        labels = clusterer.fit_predict(embeddings)
        self._algorithm_impl = clusterer
        return labels

    def _cluster_agglomerative(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster using Agglomerative Clustering."""
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            raise ImportError("scikit-learn required for Agglomerative clustering")

        n_clusters = min(self.n_clusters, len(embeddings) // self.min_cluster_size)
        n_clusters = max(2, n_clusters)

        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='average',
        )

        labels = clusterer.fit_predict(embeddings)
        self._algorithm_impl = clusterer
        return labels

    def _build_clusters(
        self,
        labels: np.ndarray,
        ids: List[str],
        embeddings: np.ndarray,
        id_to_conv: Dict[str, dict]
    ) -> List[Cluster]:
        """Build Cluster objects from labels."""
        clusters = []

        # Group by label
        label_to_indices: Dict[int, List[int]] = {}
        for i, label in enumerate(labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(i)

        for label, indices in label_to_indices.items():
            # -1 is noise/outliers in HDBSCAN
            is_outlier = label == -1

            # Get conversation IDs for this cluster
            cluster_ids = [ids[i] for i in indices]

            # Calculate centroid
            cluster_embeddings = embeddings[indices]
            centroid = np.mean(cluster_embeddings, axis=0)

            # Calculate coherence (average similarity to centroid)
            if len(cluster_embeddings) > 1:
                similarities = []
                centroid_norm = centroid / np.linalg.norm(centroid)
                for emb in cluster_embeddings:
                    emb_norm = emb / np.linalg.norm(emb)
                    sim = np.dot(centroid_norm, emb_norm)
                    similarities.append(sim)
                coherence = np.mean(similarities)
            else:
                coherence = 1.0

            clusters.append(Cluster(
                id=f"cluster_{label}" if not is_outlier else "outliers",
                label="",  # Will be generated
                conversation_ids=cluster_ids,
                centroid=centroid,
                coherence_score=float(coherence),
                is_outlier=is_outlier,
            ))

        return clusters

    def _label_clusters(
        self,
        clusters: List[Cluster],
        id_to_conv: Dict[str, dict]
    ):
        """Generate descriptive labels for clusters."""
        for cluster in clusters:
            if cluster.is_outlier:
                cluster.label = "Miscellaneous"
                continue

            # Collect data from conversations
            all_tags = []
            all_terms = []
            all_titles = []
            all_languages = []

            for cid in cluster.conversation_ids:
                conv = id_to_conv.get(cid, {})
                all_tags.extend(conv.get('tags', []))
                all_terms.extend(conv.get('technical_terms', []))
                all_languages.extend(conv.get('code_languages', []))
                if 'generated_title' in conv:
                    all_titles.append(conv['generated_title'])

            # Find most common
            top_tags = Counter(all_tags).most_common(3)
            top_terms = Counter(all_terms).most_common(5)
            top_languages = Counter(all_languages).most_common(3)

            # Set keywords
            cluster.keywords = [t for t, _ in top_terms]
            cluster.technologies = [l for l, _ in top_languages]

            # Generate label
            if top_tags:
                cluster.label = top_tags[0][0].title()
            elif top_terms:
                cluster.label = top_terms[0][0].title()
            elif top_languages:
                cluster.label = f"{top_languages[0][0].title()} Development"
            else:
                cluster.label = f"Cluster {cluster.id.split('_')[-1]}"


def cluster_conversations(
    jsonl_path: str,
    embeddings_path: str,
    ids_path: str,
    output_path: str = None
) -> List[Cluster]:
    """
    Cluster conversations from files.

    Args:
        jsonl_path: Path to conversations JSONL
        embeddings_path: Path to embeddings .npy
        ids_path: Path to IDs JSON
        output_path: Optional output path for clusters JSON

    Returns:
        List of clusters
    """
    # Load data
    conversations = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    embeddings = np.load(embeddings_path)

    with open(ids_path) as f:
        ids = json.load(f)

    print(f"Loaded {len(conversations)} conversations with {len(embeddings)} embeddings")

    # Cluster
    clusterer = ConversationClusterer()
    clusters = clusterer.cluster(conversations, embeddings, ids)

    print(f"\nFound {len(clusters)} clusters:")
    for cluster in clusters:
        if not cluster.is_outlier:
            print(f"  {cluster.label}: {len(cluster.conversation_ids)} conversations")

    # Save if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump([c.to_dict() for c in clusters], f, indent=2)
        print(f"\nSaved to {output_path}")

    return clusters


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Cluster conversations by similarity")
    parser.add_argument('conversations', help="Input JSONL file")
    parser.add_argument('embeddings', help="Embeddings .npy file")
    parser.add_argument('ids', help="IDs JSON file")
    parser.add_argument('--output', '-o', help="Output JSON file")
    parser.add_argument('--algorithm', '-a', default='auto',
                        choices=['auto', 'hdbscan', 'kmeans', 'agglomerative'])
    parser.add_argument('--min-size', type=int, default=3,
                        help="Minimum cluster size")

    args = parser.parse_args()

    cluster_conversations(
        args.conversations,
        args.embeddings,
        args.ids,
        args.output
    )
