"""
Search System for CogRepo v2

Provides:
- Local embeddings for semantic search
- Hybrid search combining BM25 and semantic similarity
- Fast vector operations with numpy

Usage:
    from search import EmbeddingEngine, HybridSearcher

    # Generate embeddings
    engine = EmbeddingEngine()
    vec = engine.embed("my query")

    # Hybrid search
    searcher = HybridSearcher(db_path, embedding_store)
    results = searcher.search("find react components", limit=10)
"""

from .embeddings import EmbeddingEngine, EmbeddingStore, generate_embeddings
from .hybrid_search import HybridSearcher, SearchResult

__all__ = [
    'EmbeddingEngine',
    'EmbeddingStore',
    'generate_embeddings',
    'HybridSearcher',
    'SearchResult',
]
