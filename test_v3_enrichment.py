"""
Test v3 Enrichment Integration

Tests the new v3 enrichment fields:
- advanced_score (1-100 with 5 dimensions)
- chains (related conversations)
- projects (project membership)
"""

import asyncio
import json
from enrichment.sota_enricher import SOTAEnricher


# Sample conversation for testing
SAMPLE_CONVERSATION = {
    "convo_id": "test-123",
    "created_at": "2024-12-05T00:00:00Z",
    "generated_title": "Testing React Hooks with TypeScript",
    "raw_text": """
User: I'm trying to implement a custom React hook for managing form state with TypeScript.
The hook should handle validation and submission.