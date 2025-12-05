# CogRepo v3 Integration - Execution Plan

**Status**: 🔨 IN PROGRESS
**Date**: 2025-12-05
**Scope**: Complete v3 intelligence layer integration

---

## Executive Summary

CogRepo has ALL v3 modules built but NONE are wired up. This document guides systematic integration to avoid breaking existing functionality.

### Current State
- ✅ Dependencies installed (sentence-transformers, scikit-learn, hdbscan)
- ✅ Embeddings generated (14MB, 9363 conversations)
- ✅ Knowledge graph built (1727 entities, 22985 relationships, 9.9MB)
- ✅ All intelligence modules exist: scoring.py, chain_detection.py, project_inference.py, clustering.py, recommendations.py
- ❌ NONE integrated into enrichment pipeline
- ❌ NONE exposed via APIs
- ❌ NONE visible in UI

### Issues to Fix
1. **Tags**: 30K unique tags (too granular) - need broader, reusable tags
2. **Scoring**: Using 1-10 scale, need 1-100 system with 5 dimensions
3. **Multi-subject titles**: Need clear indicators (e.g., "Topic A + Topic B + Topic C")

---

## Phase 1: Enrichment Pipeline Integration

### 1.1 Fix Tag Generation (PRIORITY)
**File**: `enrichment/sota_enricher.py:288-303`

**Current Prompt Problem**:
```python
prompt = f"""Extract 10-15 precise, descriptive tags from this conversation.
Generate better tags that are:
1. Specific and descriptive (not generic)  # ← Creates 30K unique tags!
```

**Solution**: Change to broader, hierarchical tags
```python
prompt = f"""Extract 5-8 broad, reusable tags from this conversation.

CONVERSATION:
{raw_text}

Tag Selection Criteria:
1. USE EXISTING COMMON TAGS when possible (Animation, Python, React, Testing, Design, etc.)
2. Prefer technology names, methodologies, domains (not overly specific use-cases)
3. Hierarchical: Start broad (Web Development), add specifics only if critical (GraphQL API)
4. NO quotes, NO brackets, NO meta-commentary
5. Reusable across multiple similar conversations

Examples of GOOD tags: Python, React Hooks, API Design, Performance Optimization, Testing Strategy
Examples of BAD tags: "1980s Nostalgia", [10-15 specific, "Custom useEffect Pattern for Modal State"

Format: Comma-separated list only.

Tags:"""
```

**Test Plan**:
- Run on 10 sample conversations
- Verify tags are broader and reusable
- Check tag count drops from 30K to ~500-1000 total

### 1.2 Wire 1-100 Scoring System
**File**: `enrichment/sota_enricher.py` (new method)

**Integration Point**: Add `_enrich_advanced_score()` method

```python
async def _enrich_advanced_score(
    self,
    conversation: Dict[str, Any],
    conversation_id: str
) -> EnrichmentResult:
    """Generate 1-100 score using intelligence/scoring.py"""

    from intelligence.scoring import ConversationScorer

    # Build scorer with all conversations for percentile context
    scorer = ConversationScorer([conversation])  # Simplified for single-convo scoring
    score_result = scorer.score(conversation)

    return EnrichmentResult(
        conversation_id=conversation_id,
        field="advanced_score",
        value={
            "overall": score_result.overall,
            "grade": score_result.grade,
            "dimensions": {
                "technical_depth": score_result.technical_depth.to_dict(),
                "practical_value": score_result.practical_value.to_dict(),
                "completeness": score_result.completeness.to_dict(),
                "clarity": score_result.clarity.to_dict(),
                "uniqueness": score_result.uniqueness.to_dict()
            }
        },
        confidence=0.92,
        cost=0.0,  # No API cost, local computation
        model="intelligence/scoring.py",
        timestamp=datetime.utcnow().isoformat(),
        partial=False
    )
```

**Add to enrichment fields**: `["title", "summary", "tags", "advanced_score", "embedding"]`

### 1.3 Wire Chain Detection
**File**: `enrichment/sota_enricher.py` (new method)

```python
async def _enrich_chains(
    self,
    conversation: Dict[str, Any],
    conversation_id: str
) -> EnrichmentResult:
    """Detect conversation chains."""

    from context.chain_detection import ChainDetector

    # Load all conversations for chain context
    detector = ChainDetector(repository_path="data/enriched_repository.jsonl")
    chains = detector.detect_chains_for_conversation(conversation_id)

    return EnrichmentResult(
        conversation_id=conversation_id,
        field="chains",
        value=chains,
        confidence=0.85,
        cost=0.0,
        model="context/chain_detection.py",
        timestamp=datetime.utcnow().isoformat(),
        partial=False
    )
```

### 1.4 Wire Project Inference
**File**: `enrichment/sota_enricher.py` (new method)

```python
async def _enrich_projects(
    self,
    conversation: Dict[str, Any],
    conversation_id: str
) -> EnrichmentResult:
    """Infer project membership."""

    from context.project_inference import ProjectInferencer

    inferencer = ProjectInferencer(repository_path="data/enriched_repository.jsonl")
    projects = inferencer.infer_projects_for_conversation(conversation_id)

    return EnrichmentResult(
        conversation_id=conversation_id,
        field="projects",
        value=projects,
        confidence=0.80,
        cost=0.0,
        model="context/project_inference.py",
        timestamp=datetime.utcnow().isoformat(),
        partial=False
    )
```

### 1.5 Improve Multi-Subject Titles
**File**: `enrichment/sota_enricher.py:161-197`

**Current Prompt**: Generic title generation

**Solution**: Add multi-subject detection
```python
prompt = f"""Generate a clear, descriptive title for this conversation.

CONVERSATION:
{raw_text}

Title Generation Rules:
1. If conversation covers MULTIPLE distinct topics, use format: "Topic A + Topic B + Topic C"
   Example: "React Hooks + TypeScript Generics + Testing Strategy"
2. If single focused topic, use clear descriptive title
   Example: "Implementing Authentication with JWT"
3. Front-load the most important keywords
4. Keep under 80 characters
5. No quotes, no "This conversation is about..."

Title:"""
```

---

## Phase 2: API Integration

### 2.1 Intelligence API Endpoints
**File**: `cogrepo-ui/api_intelligence.py`

**Already exists! Just need to verify it works**

Test endpoints:
```bash
curl http://localhost:5001/api/intelligence/health
curl http://localhost:5001/api/intelligence/score/<convo_id>
curl http://localhost:5001/api/intelligence/recommendations/<convo_id>
curl http://localhost:5001/api/intelligence/clusters
```

### 2.2 Wire Hybrid Search
**File**: `cogrepo-ui/app.py` - Update `/api/search` endpoint

**Current**: Uses basic FTS5 search
**Goal**: Add semantic search using embeddings

```python
@app.route('/api/search')
def api_search():
    query = request.args.get('q', '')

    if not query:
        return jsonify({'results': []})

    # Hybrid search: FTS5 + Semantic
    from search.hybrid_search import HybridSearch

    searcher = HybridSearch(
        repository_path=get_repo_path(),
        embeddings_path="data/embeddings.npy"
    )

    results = searcher.search(query, limit=50)

    return jsonify({'results': results})
```

---

## Phase 3: UI Integration

### 3.1 Display Advanced Scores
**File**: `cogrepo-ui/static/js/app.js`

Add score display to conversation cards:
```javascript
// In renderConversationCard()
if (conv.advanced_score) {
    const scoreHtml = `
        <div class="score-badge">
            <span class="score-value">${conv.advanced_score.overall}/100</span>
            <span class="score-grade">${conv.advanced_score.grade}</span>
        </div>
    `;
}
```

### 3.2 Display Chains and Projects
**File**: `cogrepo-ui/static/js/app.js`

Add to conversation detail view:
```javascript
if (conv.chains && conv.chains.length > 0) {
    detailHtml += `
        <div class="chains-section">
            <h4>Related Conversations</h4>
            ${conv.chains.map(chain => `
                <a href="#" class="chain-link" data-id="${chain.id}">
                    ${chain.title}
                </a>
            `).join('')}
        </div>
    `;
}
```

### 3.3 Improve Popular Tags Display
**File**: `cogrepo-ui/static/js/app.js`

**Current**: Shows all 30K tags
**Goal**: Show top 50 by frequency, with good visual hierarchy

```javascript
async function loadPopularTags() {
    const response = await fetch('/api/tags');
    const data = await response.json();

    // Sort by frequency, take top 50
    const sortedTags = Object.entries(data.tags)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 50)
        .filter(([tag, count]) => count >= 5); // Min 5 occurrences

    // Render with size based on frequency
    const tagCloud = sortedTags.map(([tag, count]) => {
        const size = Math.min(count / 50, 2); // Scale size
        return `
            <button class="tag-btn" style="font-size: ${0.8 + size}em" data-tag="${tag}">
                ${tag} <span class="tag-count">${count}</span>
            </button>
        `;
    }).join('');

    document.getElementById('popular-tags').innerHTML = tagCloud;
}
```

---

## Phase 4: Testing

### 4.1 Enrichment Pipeline Test
```bash
# Test single conversation enrichment
python3 -c "
from enrichment.sota_enricher import SOTAEnricher
import asyncio

async def test():
    enricher = SOTAEnricher()
    # Load one conversation
    # Enrich with all new fields
    # Verify output structure

asyncio.run(test())
"
```

### 4.2 API Endpoint Tests
```bash
# Test each intelligence endpoint
for endpoint in health score recommendations clusters; do
    echo "Testing /api/intelligence/$endpoint"
    curl -s http://localhost:5001/api/intelligence/$endpoint | jq .
done
```

### 4.3 UI Smoke Test
1. Open http://localhost:5001
2. Search for a conversation
3. Click conversation card
4. Verify: advanced score, chains, projects, improved tags all display
5. Test tag filtering
6. Test score-based sorting

---

## Phase 5: Deployment

### 5.1 Re-enrich Existing Conversations
**CAUTION**: This will re-process 9363 conversations with new fields

```bash
# Backup first
cp data/enriched_repository.jsonl data/enriched_repository_backup_v3.jsonl

# Re-enrich with v3 features (will take ~2 hours)
python3 cogrepo_update.py --add-intelligence-fields
```

### 5.2 Git Commit
```bash
git add .
git commit -m "feat: Complete v3 intelligence layer integration

- Wire 1-100 scoring system
- Wire chain detection and project inference
- Fix tag generation (30K → ~500 quality tags)
- Improve multi-subject title detection
- Add hybrid semantic search
- Update UI with all v3 features
- Test all integrations

Closes #v3-integration"

git push
```

---

## Success Criteria

- [ ] Tags reduced from 30K to <1000 high-quality, reusable tags
- [ ] All conversations have `advanced_score` field (1-100 scale with grade)
- [ ] Chains detected and displayed in UI
- [ ] Projects inferred and displayed in UI
- [ ] Multi-subject conversations have "Topic A + Topic B" style titles
- [ ] Hybrid search works (FTS5 + semantic)
- [ ] Intelligence APIs all return 200
- [ ] UI displays all new features
- [ ] No existing functionality broken
- [ ] Documentation updated

---

## Rollback Plan

If anything breaks:
```bash
# Restore backup
cp data/enriched_repository_backup_v3.jsonl data/enriched_repository.jsonl

# Revert code
git revert HEAD

# Restart server
pkill -f "python3 app.py" && cd cogrepo-ui && python3 app.py
```

---

## Next Steps

1. ✅ Read this plan
2. Execute Phase 1 (Enrichment Pipeline)
3. Execute Phase 2 (APIs)
4. Execute Phase 3 (UI)
5. Execute Phase 4 (Testing)
6. Execute Phase 5 (Deployment)
