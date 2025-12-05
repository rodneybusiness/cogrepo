"""
Context Analysis System for CogRepo v2

This module provides:
- Project inference: Auto-grouping conversations by project
- Chain detection: Linking related conversations
- Knowledge graph: Entity relationships (future)

Usage:
    from context import ProjectInferrer, ChainDetector

    # Group conversations by project
    inferrer = ProjectInferrer()
    projects = inferrer.infer_projects(conversations)

    # Detect conversation chains
    detector = ChainDetector()
    chains = detector.detect_chains(conversations)
"""

from .project_inference import ProjectInferrer, Project
from .chain_detection import ChainDetector, ConversationChain

__all__ = [
    'ProjectInferrer',
    'Project',
    'ChainDetector',
    'ConversationChain',
]
