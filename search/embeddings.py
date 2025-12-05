"""
Local Embeddings Engine for Semantic Search

Uses sentence-transformers for local embedding generation.
No API calls, no cost, fast processing.

Features:
- Single and batch embedding generation
- Cosine similarity search
- Numpy-based vector storage
- Automatic model loading/caching

Usage:
    from search.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    vec = engine.embed("search query")
    similar = engine.find_similar(vec, corpus_vectors, top_k=5)
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np

# Lazy import for sentence-transformers
_model = None
_model_name = None


def get_model(model_name: str = 'all-MiniLM-L6-v2'):
    """
    Get or create the sentence-transformers model.

    Uses lazy loading to avoid import cost until needed.
    """
    global _model, _model_name

    if _model is not None and _model_name == model_name:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        return _model
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )


class EmbeddingEngine:
    """
    Generate embeddings for semantic search using local models.

    Uses sentence-transformers which runs entirely locally.
    No API calls, no cost, fast processing.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding engine.

        Args:
            model_name: Name of the sentence-transformers model to use.
                        Default is 'all-MiniLM-L6-v2' (fast, good quality).
                        Other options:
                        - 'all-mpnet-base-v2' (higher quality, slower)
                        - 'paraphrase-MiniLM-L6-v2' (good for paraphrasing)
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._model = get_model(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Numpy array of shape (dimension,)
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            Numpy array of shape (n_texts, dimension)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (-1 to 1)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def find_similar(
        self,
        query_vec: np.ndarray,
        corpus_vecs: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[int, float]]:
        """
        Find most similar vectors in a corpus.

        Args:
            query_vec: Query vector of shape (dimension,)
            corpus_vecs: Corpus vectors of shape (n_docs, dimension)
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (index, similarity_score) tuples, sorted by score descending
        """
        # Normalize vectors
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        corpus_norms = np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
        corpus_normalized = corpus_vecs / (corpus_norms + 1e-8)

        # Compute similarities
        similarities = np.dot(corpus_normalized, query_norm)

        # Apply threshold
        valid_indices = np.where(similarities >= threshold)[0]

        if len(valid_indices) == 0:
            return []

        # Get top k
        valid_sims = similarities[valid_indices]
        top_k = min(top_k, len(valid_indices))
        top_indices = np.argsort(valid_sims)[-top_k:][::-1]

        results = [
            (int(valid_indices[i]), float(valid_sims[i]))
            for i in top_indices
        ]

        return results


# =============================================================================
# Embedding Storage
# =============================================================================

class EmbeddingStore:
    """
    Store and retrieve embeddings efficiently.

    Uses numpy files for storage:
    - embeddings.npy: The actual vectors
    - embeddings_ids.json: Mapping from index to conversation ID
    """

    def __init__(self, store_dir: Union[str, Path]):
        """
        Initialize the embedding store.

        Args:
            store_dir: Directory to store embedding files
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_file = self.store_dir / 'embeddings.npy'
        self.ids_file = self.store_dir / 'embedding_ids.json'

        self._embeddings = None
        self._ids = None

    def load(self) -> bool:
        """
        Load embeddings from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.embeddings_file.exists() or not self.ids_file.exists():
            return False

        try:
            self._embeddings = np.load(self.embeddings_file)
            with open(self.ids_file) as f:
                self._ids = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False

    def save(self, embeddings: np.ndarray, ids: List[str]):
        """
        Save embeddings to disk.

        Args:
            embeddings: Numpy array of embeddings
            ids: List of conversation IDs (same order as embeddings)
        """
        np.save(self.embeddings_file, embeddings)
        with open(self.ids_file, 'w') as f:
            json.dump(ids, f)

        self._embeddings = embeddings
        self._ids = ids

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        """Get loaded embeddings."""
        if self._embeddings is None:
            self.load()
        return self._embeddings

    @property
    def ids(self) -> Optional[List[str]]:
        """Get loaded IDs."""
        if self._ids is None:
            self.load()
        return self._ids

    def get_by_id(self, convo_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific conversation ID."""
        if self.ids is None or self.embeddings is None:
            return None

        try:
            idx = self.ids.index(convo_id)
            return self.embeddings[idx]
        except ValueError:
            return None

    def find_similar(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        exclude_ids: List[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Find similar conversations by embedding.

        Args:
            query_vec: Query embedding
            top_k: Number of results
            exclude_ids: Conversation IDs to exclude

        Returns:
            List of (convo_id, similarity) tuples
        """
        if self.embeddings is None or self.ids is None:
            return []

        engine = EmbeddingEngine()
        results = engine.find_similar(query_vec, self.embeddings, top_k=top_k * 2)

        # Convert indices to IDs and filter
        exclude_set = set(exclude_ids or [])
        filtered = []

        for idx, score in results:
            convo_id = self.ids[idx]
            if convo_id not in exclude_set:
                filtered.append((convo_id, score))
                if len(filtered) >= top_k:
                    break

        return filtered


# =============================================================================
# Batch Processing
# =============================================================================

def generate_embeddings(
    input_path: str,
    output_dir: str = None,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 64,
    text_field: str = None
) -> dict:
    """
    Generate embeddings for all conversations in a JSONL file.

    Args:
        input_path: Path to input JSONL file
        output_dir: Directory to store embeddings (defaults to same as input)
        model_name: Name of the embedding model
        batch_size: Batch size for processing
        text_field: Field to embed (defaults to 'summary_abstractive' + 'generated_title')

    Returns:
        Statistics about the generation
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir or input_path.parent)

    print(f"Loading conversations from {input_path}")

    # Load conversations
    conversations = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    print(f"Found {len(conversations)} conversations")

    # Extract text to embed
    texts = []
    ids = []

    for conv in conversations:
        if text_field:
            text = conv.get(text_field, '')
        else:
            # Default: combine title and summary
            title = conv.get('generated_title', '')
            summary = conv.get('summary_abstractive', '')
            text = f"{title}. {summary}"

        texts.append(text)
        ids.append(conv.get('convo_id', ''))

    # Generate embeddings
    print(f"Generating embeddings with {model_name}...")
    engine = EmbeddingEngine(model_name)
    embeddings = engine.embed_batch(texts, batch_size=batch_size, show_progress=True)

    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Save
    store = EmbeddingStore(output_dir)
    store.save(embeddings, ids)

    print(f"Saved to {output_dir}")

    return {
        'count': len(embeddings),
        'dimension': embeddings.shape[1],
        'model': model_name,
        'output_dir': str(output_dir)
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate embeddings for conversations")
    parser.add_argument('input', help="Input JSONL file")
    parser.add_argument('--output', '-o', help="Output directory (defaults to input dir)")
    parser.add_argument('--model', '-m', default='all-MiniLM-L6-v2',
                        help="Model name (default: all-MiniLM-L6-v2)")
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help="Batch size (default: 64)")

    args = parser.parse_args()

    stats = generate_embeddings(
        args.input,
        args.output,
        model_name=args.model,
        batch_size=args.batch_size
    )
