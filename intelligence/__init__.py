"""
CogRepo Intelligence Layer

Advanced analytics and insights for the conversation repository:
- Auto-clustering of similar conversations
- Recommendation engine
- Trend analysis
- Knowledge graph insights

All features work with optional dependencies and degrade gracefully.
"""

from typing import TYPE_CHECKING

# Lazy imports for optional features
if TYPE_CHECKING:
    from .clustering import ConversationClusterer
    from .recommendations import RecommendationEngine
    from .insights import InsightsEngine

__all__ = [
    'ConversationClusterer',
    'RecommendationEngine',
    'InsightsEngine',
]
