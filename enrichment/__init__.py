"""
Enrichment System for CogRepo v2

Multi-tier enrichment:
1. Zero-token: Local pattern extraction (free)
2. Artifact extraction: Code, commands, solutions (Haiku - low cost)
3. Deep analysis: Full enrichment (Sonnet - for high-value)

Usage:
    from enrichment import ZeroTokenEnricher, ArtifactExtractor

    # Zero-cost enrichment
    enricher = ZeroTokenEnricher()
    metrics = enricher.extract(text)

    # Artifact extraction
    extractor = ArtifactExtractor()
    artifacts = extractor.extract(convo_id, text)
"""

# Graceful imports - handle missing dependencies
__all__ = []

# Zero-token (no external dependencies)
from .zero_token import ZeroTokenEnricher, ZeroTokenMetrics
__all__.extend(["ZeroTokenEnricher", "ZeroTokenMetrics"])

# Artifact extraction
from .artifact_extractor import Artifact, LocalArtifactExtractor
__all__.extend(["Artifact", "LocalArtifactExtractor"])

try:
    from .artifact_extractor import ArtifactExtractor
    __all__.append("ArtifactExtractor")
except ImportError:
    ArtifactExtractor = None

# Original pipeline (requires anthropic)
try:
    from .enrichment_pipeline import EnrichmentPipeline
    __all__.append("EnrichmentPipeline")
except ImportError:
    EnrichmentPipeline = None
