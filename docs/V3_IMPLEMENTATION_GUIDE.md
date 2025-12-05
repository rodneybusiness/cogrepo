# CogRepo v3 Implementation Guide

> **Purpose:** Fix 100% of broken features, then add MCP integration
> **Created:** December 2024
> **Status:** Ready for implementation

---

## Executive Summary

CogRepo v2 has well-designed code that is **60% functional**. This guide covers fixing **all 12 broken features** to achieve 100% functionality.

**Goal:** Every feature advertised in the README actually works.

---

## The 12 Broken Features

| # | Feature | Status | Root Cause |
|---|---------|--------|------------|
| 1 | Dependencies | BROKEN | Missing packages in requirements.txt |
| 2 | Embeddings | MISSING | Never generated - no .npy files |
| 3 | Hybrid Search | UNUSED | Code exists, not wired in |
| 4 | Clustering | BROKEN | Missing deps + no embeddings |
| 5 | Recommendations | BROKEN | Missing deps + no embeddings |
| 6 | Knowledge Graph | EMPTY | Code exists, never called |
| 7 | Project Inference | NOT INTEGRATED | Code exists, never called |
| 8 | Chain Detection | NOT INTEGRATED | Code exists, never called |
| 9 | Intelligence APIs | 503 ERRORS | All above issues |
| 10 | AI Enrichment | DISABLED | Opt-in instead of opt-out |
| 11 | UI Dashboard | INCOMPLETE | Doesn't use intelligence endpoints |
| 12 | Test Coverage | MISSING | Intelligence layer untested |

---

## Fix #1: Dependencies (requirements.txt)

### Problem
These packages are referenced in code but not in requirements.txt:
- `sentence-transformers` (embeddings)
- `scikit-learn` (clustering fallback)
- `hdbscan` (primary clustering)

### Solution

Add to `requirements.txt`:

```
# Semantic Search & Embeddings
sentence-transformers>=2.2.0

# Clustering & ML
scikit-learn>=1.3.0
hdbscan>=0.8.33
```

### Verification

```bash
pip install -r requirements.txt
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
python -c "import hdbscan; print('OK')"
python -c "from sklearn.cluster import KMeans; print('OK')"
```

---

## Fix #2: Generate Embeddings

### Problem
- `data/embeddings.npy` does not exist
- `data/embedding_ids.json` does not exist
- Semantic search, clustering, recommendations all require embeddings

### Solution

Create `generate_embeddings.py`:

```python
#!/usr/bin/env python3
"""Generate embeddings for all conversations."""

import json
import numpy as np
from pathlib import Path
from search.embeddings import EmbeddingEngine

def main():
    data_dir = Path('data')
    jsonl_path = data_dir / 'enriched_repository.jsonl'

    if not jsonl_path.exists():
        print("No conversations found. Import some first.")
        return

    # Load conversations
    conversations = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    print(f"Generating embeddings for {len(conversations)} conversations...")

    # Initialize embedding engine
    engine = EmbeddingEngine()

    # Generate embeddings
    embeddings = []
    ids = []

    for i, conv in enumerate(conversations):
        # Combine title + summary + raw_text for better embeddings
        text = ' '.join(filter(None, [
            conv.get('generated_title', ''),
            conv.get('summary_abstractive', ''),
            conv.get('raw_text', '')[:2000]  # Limit raw text
        ]))

        embedding = engine.embed(text)
        embeddings.append(embedding)
        ids.append(conv.get('convo_id', str(i)))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(conversations)}")

    # Save
    np.save(data_dir / 'embeddings.npy', np.array(embeddings))
    with open(data_dir / 'embedding_ids.json', 'w') as f:
        json.dump(ids, f)

    print(f"Saved embeddings to {data_dir}/embeddings.npy")
    print(f"Saved IDs to {data_dir}/embedding_ids.json")

if __name__ == '__main__':
    main()
```

### Run

```bash
python generate_embeddings.py
```

### Verification

```bash
ls -la data/embeddings.npy data/embedding_ids.json
python -c "import numpy as np; e = np.load('data/embeddings.npy'); print(f'Shape: {e.shape}')"
```

---

## Fix #3: Wire Up Hybrid Search

### Problem
- `search/hybrid_search.py` is complete (300+ lines)
- `cogrepo-ui/api_v2.py` only uses FTS5, ignores hybrid search

### Solution

Update `cogrepo-ui/api_v2.py` to use hybrid search when embeddings available:

```python
# At top of file, add:
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# In the search endpoint, replace FTS5-only with:
def get_searcher():
    """Get hybrid searcher if embeddings available, else FTS5."""
    embeddings_path = Path(__file__).parent.parent / 'data' / 'embeddings.npy'

    if embeddings_path.exists():
        try:
            from search.hybrid_search import HybridSearcher
            from search.embeddings import EmbeddingStore

            db_path = str(Path(__file__).parent.parent / 'data' / 'cogrepo.db')
            store = EmbeddingStore.load(
                str(embeddings_path),
                str(embeddings_path.parent / 'embedding_ids.json')
            )
            return HybridSearcher(db_path, store)
        except Exception as e:
            print(f"Hybrid search unavailable: {e}")

    # Fallback to FTS5
    from search_engine import SearchEngine
    return SearchEngine()
```

### Verification

```bash
# Start server
cd cogrepo-ui && python app.py

# Search should now use semantic matching
curl "http://localhost:5001/api/v2/search?q=react+hooks"
# Results should be semantically relevant, not just keyword matches
```

---

## Fix #4: Clustering

### Problem
- `intelligence/clustering.py` exists (350+ lines)
- Requires embeddings (now fixed)
- Requires hdbscan/sklearn (now in requirements)

### Solution

The code should work after fixes #1 and #2. Verify:

```python
# Test clustering
python -c "
from intelligence.clustering import ConversationClusterer
import numpy as np
import json

# Load embeddings
embeddings = np.load('data/embeddings.npy')
with open('data/embedding_ids.json') as f:
    ids = json.load(f)

# Load conversations
convos = []
with open('data/enriched_repository.jsonl') as f:
    for line in f:
        if line.strip():
            convos.append(json.loads(line))

# Cluster
clusterer = ConversationClusterer()
clusters = clusterer.cluster(convos, embeddings, ids)

print(f'Found {len(clusters)} clusters')
for c in clusters[:5]:
    print(f'  {c.label}: {c.size} conversations')
"
```

### Verification

```bash
curl http://localhost:5001/api/intelligence/clusters
# Should return actual clusters, not 503
```

---

## Fix #5: Recommendations

### Problem
- `intelligence/recommendations.py` exists (350+ lines)
- Requires embeddings (now fixed)

### Solution

The code should work after fix #2. Verify:

```python
# Test recommendations
python -c "
from intelligence.recommendations import RecommendationEngine
import numpy as np
import json

embeddings = np.load('data/embeddings.npy')
with open('data/embedding_ids.json') as f:
    ids = json.load(f)
with open('data/enriched_repository.jsonl') as f:
    convos = [json.loads(l) for l in f if l.strip()]

engine = RecommendationEngine(
    embeddings=embeddings,
    conversation_ids=ids,
    conversations=convos
)

# Get recommendations for first conversation
recs = engine.find_similar(ids[0], limit=5)
print(f'Recommendations for {ids[0]}:')
for r in recs:
    print(f'  {r.conversation_id}: {r.score:.2f}')
"
```

### Verification

```bash
# Get a conversation ID
ID=$(head -1 data/enriched_repository.jsonl | python -c "import json,sys; print(json.load(sys.stdin)['convo_id'])")

curl "http://localhost:5001/api/intelligence/recommendations/$ID"
# Should return similar conversations, not 503
```

---

## Fix #6: Knowledge Graph

### Problem
- `context/knowledge_graph.py` exists (500+ lines)
- Never called from import pipeline
- No graph data exists

### Solution

Create `build_knowledge_graph.py`:

```python
#!/usr/bin/env python3
"""Build knowledge graph from conversations."""

import json
from pathlib import Path
from context.knowledge_graph import KnowledgeGraph

def main():
    data_dir = Path('data')
    jsonl_path = data_dir / 'enriched_repository.jsonl'

    # Load conversations
    conversations = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    print(f"Building knowledge graph from {len(conversations)} conversations...")

    # Build graph
    kg = KnowledgeGraph()
    kg.build_from_conversations(conversations)

    # Save
    kg.save(str(data_dir / 'knowledge_graph.json'))

    print(f"Entities: {len(kg.entities)}")
    print(f"Relationships: {len(kg.relationships)}")
    print(f"Saved to {data_dir}/knowledge_graph.json")

if __name__ == '__main__':
    main()
```

### Run

```bash
python build_knowledge_graph.py
```

### Verification

```bash
curl http://localhost:5001/api/intelligence/knowledge-graph
# Should return nodes and links, not empty arrays
```

---

## Fix #7 & #8: Project Inference & Chain Detection

### Problem
- `context/project_inference.py` exists
- `context/chain_detection.py` exists
- Neither integrated into import pipeline

### Solution

Update `cogrepo_import.py` to call context modules after enrichment:

```python
# Add after enrichment completes:

def run_context_analysis(conversations):
    """Run context analysis on imported conversations."""
    from context.project_inference import ProjectInferencer
    from context.chain_detection import ChainDetector

    print("Running context analysis...")

    # Detect projects
    try:
        inferencer = ProjectInferencer()
        projects = inferencer.infer_projects(conversations)
        print(f"  Detected {len(projects)} projects")

        # Update conversations with project_id
        for conv in conversations:
            for project in projects:
                if conv.get('convo_id') in project.conversation_ids:
                    conv['project_id'] = project.project_id
                    conv['project_name'] = project.name
                    break
    except Exception as e:
        print(f"  Project inference failed: {e}")

    # Detect chains
    try:
        detector = ChainDetector()
        chains = detector.detect_chains(conversations)
        print(f"  Detected {len(chains)} conversation chains")

        # Update conversations with chain_id
        for conv in conversations:
            for chain in chains:
                if conv.get('convo_id') in chain.conversation_ids:
                    conv['chain_id'] = chain.chain_id
                    conv['chain_position'] = chain.conversation_ids.index(conv.get('convo_id'))
                    break
    except Exception as e:
        print(f"  Chain detection failed: {e}")

    return conversations
```

### Verification

```bash
# After re-importing, check for project_id and chain_id
head -1 data/enriched_repository.jsonl | python -c "
import json,sys
d = json.load(sys.stdin)
print('project_id:', d.get('project_id', 'NOT SET'))
print('chain_id:', d.get('chain_id', 'NOT SET'))
"
```

---

## Fix #9: Intelligence APIs

### Problem
All `/api/intelligence/*` endpoints return 503 errors.

### Solution
Fixes #1-8 should resolve this. The API gracefully degrades when dependencies are missing.

### Verification

```bash
# All should return 200 with data
curl http://localhost:5001/api/intelligence/health
curl http://localhost:5001/api/intelligence/clusters
curl http://localhost:5001/api/intelligence/knowledge-graph
curl http://localhost:5001/api/intelligence/insights
curl http://localhost:5001/api/intelligence/insights/dashboard
```

Expected `/health` response:
```json
{
  "status": "ok",
  "features": {
    "knowledge_graph": "available",
    "recommendations": "available",
    "insights": "available",
    "topic_segmentation": "available",
    "advanced_scoring": "available",
    "clustering": "available",
    "embeddings": "available"
  }
}
```

---

## Fix #10: AI Enrichment Default

### Problem
- Enrichment is disabled by default
- User must pass `--enrich` flag explicitly
- Most users don't know to do this

### Solution

Update `cogrepo_import.py` to prompt or default to enrichment:

```python
# Change argument default
parser.add_argument('--enrich', action='store_true', default=True,
                    help='Enrich with AI (default: True)')
parser.add_argument('--no-enrich', action='store_false', dest='enrich',
                    help='Skip AI enrichment')
```

Or prompt interactively:

```python
if not args.enrich and os.getenv('ANTHROPIC_API_KEY'):
    response = input("API key found. Enable AI enrichment? [Y/n]: ")
    if response.lower() != 'n':
        args.enrich = True
```

---

## Fix #11: UI Dashboard

### Problem
- Web UI (`index.html`) doesn't call intelligence endpoints
- No visualization of clusters, knowledge graph, insights

### Solution

Add dashboard tab to `cogrepo-ui/index.html`:

1. Add navigation tab for "Intelligence"
2. Create dashboard view with:
   - Insights summary from `/api/intelligence/insights`
   - Cluster visualization from `/api/intelligence/clusters`
   - Knowledge graph visualization from `/api/intelligence/knowledge-graph`
   - Trending topics from `/api/intelligence/insights/technology-trends`

Add to `static/js/app.js`:

```javascript
async function loadDashboard() {
    const [insights, clusters, graph] = await Promise.all([
        fetch('/api/intelligence/insights').then(r => r.json()),
        fetch('/api/intelligence/clusters').then(r => r.json()),
        fetch('/api/intelligence/knowledge-graph').then(r => r.json()),
    ]);

    renderInsights(insights);
    renderClusters(clusters);
    renderKnowledgeGraph(graph);
}
```

### Verification

- Open http://localhost:5001
- Navigate to Intelligence/Dashboard tab
- Should show insights, clusters, graph visualization

---

## Fix #12: Test Coverage

### Problem
Intelligence layer has no tests.

### Solution

Create `tests/test_intelligence.py`:

```python
"""Tests for intelligence layer."""

import pytest
import json
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_conversations():
    return [
        {
            'convo_id': 'test-1',
            'raw_text': 'How do I use React hooks?',
            'generated_title': 'React Hooks Tutorial',
            'tags': ['react', 'javascript'],
        },
        {
            'convo_id': 'test-2',
            'raw_text': 'Explain Python decorators',
            'generated_title': 'Python Decorators Guide',
            'tags': ['python'],
        },
    ]

class TestScoring:
    def test_score_conversation(self, sample_conversations):
        from intelligence.scoring import ConversationScorer

        scorer = ConversationScorer(sample_conversations)
        score = scorer.score(sample_conversations[0])

        assert 0 <= score.overall <= 100
        assert score.grade in ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F']

class TestTopicSegmentation:
    def test_segment_conversation(self, sample_conversations):
        from intelligence.topic_segmentation import TopicSegmenter

        segmenter = TopicSegmenter()
        result = segmenter.segment(sample_conversations[0])

        assert result.topic_count >= 1
        assert len(result.segments) >= 1

class TestClustering:
    def test_cluster_with_embeddings(self, sample_conversations):
        from intelligence.clustering import ConversationClusterer

        # Create dummy embeddings
        embeddings = np.random.rand(2, 384)
        ids = ['test-1', 'test-2']

        clusterer = ConversationClusterer()
        clusters = clusterer.cluster(sample_conversations, embeddings, ids)

        assert isinstance(clusters, list)

class TestKnowledgeGraph:
    def test_build_graph(self, sample_conversations):
        from context.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.build_from_conversations(sample_conversations)

        assert len(kg.entities) > 0
```

### Run Tests

```bash
pytest tests/test_intelligence.py -v
```

---

## Complete Verification Checklist

After all fixes, verify:

```bash
# 1. Dependencies
pip install -r requirements.txt
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
python -c "import hdbscan; print('OK')"

# 2. Embeddings
ls data/embeddings.npy data/embedding_ids.json

# 3. Knowledge Graph
ls data/knowledge_graph.json

# 4. Start server
cd cogrepo-ui && python app.py &

# 5. Intelligence health (all should be "available")
curl http://localhost:5001/api/intelligence/health

# 6. Each endpoint returns data
curl http://localhost:5001/api/intelligence/clusters
curl http://localhost:5001/api/intelligence/knowledge-graph
curl http://localhost:5001/api/intelligence/insights
curl http://localhost:5001/api/intelligence/insights/dashboard

# 7. Recommendations work
ID=$(head -1 ../data/enriched_repository.jsonl | python -c "import json,sys; print(json.load(sys.stdin)['convo_id'])")
curl "http://localhost:5001/api/intelligence/recommendations/$ID"

# 8. Search uses hybrid
curl "http://localhost:5001/api/v2/search?q=python+async"

# 9. Tests pass
cd .. && pytest tests/test_intelligence.py -v
```

---

## Post-Fix: MCP Integration

Once all 12 features work, add MCP server for Claude Desktop integration:

See separate section in original guide for `cogrepo_mcp.py` implementation.

---

## Success Criteria

| Feature | Test | Expected |
|---------|------|----------|
| Dependencies | `pip install` | No errors |
| Embeddings | `ls data/*.npy` | File exists |
| Hybrid Search | API search | Semantic results |
| Clustering | `/api/intelligence/clusters` | Returns clusters |
| Recommendations | `/api/intelligence/recommendations/ID` | Returns similar |
| Knowledge Graph | `/api/intelligence/knowledge-graph` | Nodes + edges |
| Project Inference | Check `project_id` in data | Field populated |
| Chain Detection | Check `chain_id` in data | Field populated |
| Intelligence APIs | `/api/intelligence/health` | All "available" |
| AI Enrichment | Import without flags | Prompts or enriches |
| UI Dashboard | Open in browser | Shows intelligence |
| Tests | `pytest` | All pass |
