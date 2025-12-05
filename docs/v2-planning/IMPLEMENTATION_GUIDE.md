# CogRepo v2: Implementation Guide

> Step-by-step instructions for implementing each phase.
> Each section is designed to be self-contained and executable.

---

## Pre-Implementation Checklist

Before starting any phase:

```bash
# Verify you're in the right directory
cd /path/to/cogrepo

# Verify tests pass
pytest tests/ -v

# Verify current data integrity
python -c "
import json
with open('data/enriched_repository.jsonl') as f:
    count = sum(1 for _ in f)
print(f'Current conversations: {count}')
"

# Create backup
cp data/enriched_repository.jsonl data/enriched_repository.jsonl.backup.$(date +%Y%m%d)
```

---

## Phase 0: Foundation + Enrichment

### 0.1 Configuration System

**Goal:** Secure API key storage in `~/.cogrepo/config.yaml`

**Step 1: Update core/config.py**

```python
# Add to core/config.py

import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class AnthropicConfig:
    api_key: Optional[str] = None

    @classmethod
    def load(cls) -> 'AnthropicConfig':
        """Load API key from resolution chain."""
        # 1. Environment variable (highest priority)
        if os.getenv('ANTHROPIC_API_KEY'):
            return cls(api_key=os.getenv('ANTHROPIC_API_KEY'))

        # 2. User config (~/.cogrepo/config.yaml)
        user_config = Path.home() / '.cogrepo' / 'config.yaml'
        if user_config.exists():
            with open(user_config) as f:
                config = yaml.safe_load(f)
                if config and 'anthropic' in config:
                    return cls(api_key=config['anthropic'].get('api_key'))

        # 3. Project config (./.cogrepo/config.yaml)
        project_config = Path('.cogrepo/config.yaml')
        if project_config.exists():
            with open(project_config) as f:
                config = yaml.safe_load(f)
                if config and 'anthropic' in config:
                    return cls(api_key=config['anthropic'].get('api_key'))

        # 4. No key found
        return cls(api_key=None)

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)
```

**Step 2: Create cogrepo_setup.py**

```python
#!/usr/bin/env python3
"""Interactive setup wizard for CogRepo."""

import os
import stat
from pathlib import Path
import yaml

def setup():
    print("=" * 50)
    print("  CogRepo Setup Wizard")
    print("=" * 50)
    print()

    config_dir = Path.home() / '.cogrepo'
    config_file = config_dir / 'config.yaml'

    # Check existing
    if config_file.exists():
        print(f"Config already exists: {config_file}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return

    # Get API key
    print()
    print("Get your Anthropic API key from:")
    print("  https://console.anthropic.com/settings/keys")
    print()
    api_key = input("Enter your Anthropic API key: ").strip()

    if not api_key.startswith('sk-ant-'):
        print("Warning: Key doesn't look like an Anthropic key (should start with 'sk-ant-')")
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return

    # Create config directory
    config_dir.mkdir(mode=0o700, exist_ok=True)

    # Write config
    config = {
        'anthropic': {
            'api_key': api_key
        },
        'enrichment': {
            'model': 'claude-3-5-haiku-20241022',
            'batch_size': 10
        },
        'search': {
            'use_semantic': True,
            'default_limit': 50
        }
    }

    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Set permissions (owner read/write only)
    os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

    print()
    print(f"Config saved to: {config_file}")
    print(f"Permissions set to: 600 (owner-only)")
    print()
    print("Setup complete! You can now use CogRepo with AI enrichment.")

if __name__ == '__main__':
    setup()
```

**Step 3: Update .gitignore**

```bash
echo "# CogRepo config (contains secrets)
.cogrepo/
.env
.env.local" >> .gitignore
```

**Step 4: Verify**

```bash
python cogrepo_setup.py
python -c "from core.config import AnthropicConfig; c = AnthropicConfig.load(); print(f'API key configured: {c.has_api_key}')"
```

---

### 0.2 Zero-Token Enrichment

**Goal:** Extract maximum value without API calls

**Step 1: Create enrichment/zero_token.py**

```python
"""Zero-token enrichment - extract value without API calls."""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

@dataclass
class ZeroTokenMetrics:
    """Metrics extracted without any API calls."""

    # Code analysis
    has_code: bool = False
    code_languages: List[str] = field(default_factory=list)
    code_block_count: int = 0
    has_error_traces: bool = False

    # Conversation structure
    turn_count: int = 0
    question_count: int = 0
    question_types: List[str] = field(default_factory=list)
    avg_user_length: int = 0
    avg_assistant_length: int = 0

    # Content signals
    has_links: bool = False
    link_domains: List[str] = field(default_factory=list)
    link_urls: List[str] = field(default_factory=list)
    technical_terms: List[str] = field(default_factory=list)
    mentioned_files: List[str] = field(default_factory=list)

    # Temporal
    duration_minutes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ZeroTokenEnricher:
    """Extract enrichment without API calls."""

    # Code block pattern
    CODE_BLOCK_PATTERN = re.compile(r'```(\w*)\n[\s\S]*?```')

    # URL pattern
    URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')

    # File path patterns
    FILE_PATTERNS = [
        re.compile(r'[\w./\\-]+\.(py|js|ts|jsx|tsx|java|cpp|c|h|go|rs|rb|php|swift|kt)'),
        re.compile(r'[\w./\\-]+\.(json|yaml|yml|toml|xml|html|css|scss|sql)'),
        re.compile(r'[\w./\\-]+\.(md|txt|csv|log)'),
    ]

    # Error trace patterns
    ERROR_PATTERNS = [
        re.compile(r'Traceback \(most recent call last\)'),
        re.compile(r'Error: '),
        re.compile(r'Exception: '),
        re.compile(r'at [\w.]+\([\w.]+:\d+\)'),  # JavaScript stack
    ]

    # Question patterns
    QUESTION_PATTERNS = {
        'how-to': re.compile(r'\bhow (?:do|can|to|should)\b', re.I),
        'why': re.compile(r'\bwhy (?:is|does|do|are|was|were)\b', re.I),
        'what': re.compile(r'\bwhat (?:is|are|does|do)\b', re.I),
        'debug': re.compile(r'\b(?:error|bug|issue|problem|not working|fails?|broken)\b', re.I),
        'explain': re.compile(r'\b(?:explain|understand|clarify|meaning)\b', re.I),
    }

    # Common technical terms
    TECH_TERMS = [
        'api', 'database', 'server', 'client', 'frontend', 'backend',
        'async', 'await', 'promise', 'callback', 'function', 'class',
        'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'cloud',
        'python', 'javascript', 'typescript', 'react', 'vue', 'node',
        'sql', 'nosql', 'mongodb', 'postgres', 'redis', 'cache',
        'rest', 'graphql', 'grpc', 'websocket', 'http', 'https',
        'git', 'github', 'gitlab', 'ci/cd', 'pipeline', 'deploy',
        'test', 'unit test', 'integration', 'mock', 'fixture',
        'authentication', 'authorization', 'oauth', 'jwt', 'token',
    ]

    def extract(self, raw_text: str, messages: List[Dict] = None) -> ZeroTokenMetrics:
        """Extract all zero-token metrics from conversation."""
        metrics = ZeroTokenMetrics()

        # Code analysis
        code_blocks = self.CODE_BLOCK_PATTERN.findall(raw_text)
        metrics.code_block_count = len(code_blocks)
        metrics.has_code = metrics.code_block_count > 0
        metrics.code_languages = list(set(lang.lower() for lang in code_blocks if lang))

        # Error traces
        metrics.has_error_traces = any(p.search(raw_text) for p in self.ERROR_PATTERNS)

        # URLs and links
        urls = self.URL_PATTERN.findall(raw_text)
        metrics.has_links = len(urls) > 0
        metrics.link_urls = urls[:20]  # Limit to 20
        metrics.link_domains = list(set(
            re.search(r'https?://([^/]+)', url).group(1)
            for url in urls if re.search(r'https?://([^/]+)', url)
        ))[:10]

        # File paths
        mentioned_files = []
        for pattern in self.FILE_PATTERNS:
            mentioned_files.extend(pattern.findall(raw_text))
        metrics.mentioned_files = list(set(mentioned_files))[:20]

        # Questions
        question_count = raw_text.count('?')
        metrics.question_count = question_count

        detected_types = []
        for qtype, pattern in self.QUESTION_PATTERNS.items():
            if pattern.search(raw_text):
                detected_types.append(qtype)
        metrics.question_types = detected_types

        # Technical terms
        text_lower = raw_text.lower()
        found_terms = [term for term in self.TECH_TERMS if term in text_lower]
        metrics.technical_terms = found_terms

        # Message analysis (if provided)
        if messages:
            user_msgs = [m for m in messages if m.get('role') == 'user']
            asst_msgs = [m for m in messages if m.get('role') == 'assistant']

            metrics.turn_count = min(len(user_msgs), len(asst_msgs))

            if user_msgs:
                metrics.avg_user_length = sum(len(m.get('content', '')) for m in user_msgs) // len(user_msgs)
            if asst_msgs:
                metrics.avg_assistant_length = sum(len(m.get('content', '')) for m in asst_msgs) // len(asst_msgs)

        return metrics


def backfill_zero_token(jsonl_path: str, output_path: str = None):
    """Backfill zero-token metrics for all conversations."""
    import json
    from tqdm import tqdm

    output_path = output_path or jsonl_path
    enricher = ZeroTokenEnricher()

    # Read all conversations
    conversations = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))

    print(f"Processing {len(conversations)} conversations...")

    # Enrich each
    for conv in tqdm(conversations):
        raw_text = conv.get('raw_text', '')
        metrics = enricher.extract(raw_text)

        # Merge metrics into conversation
        conv.update(metrics.to_dict())

    # Write back
    with open(output_path, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    print(f"Done! Enriched {len(conversations)} conversations.")

    # Stats
    with_code = sum(1 for c in conversations if c.get('has_code'))
    with_links = sum(1 for c in conversations if c.get('has_links'))
    with_errors = sum(1 for c in conversations if c.get('has_error_traces'))

    print(f"\nStats:")
    print(f"  With code: {with_code} ({100*with_code/len(conversations):.1f}%)")
    print(f"  With links: {with_links} ({100*with_links/len(conversations):.1f}%)")
    print(f"  With errors: {with_errors} ({100*with_errors/len(conversations):.1f}%)")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        backfill_zero_token(sys.argv[1])
    else:
        print("Usage: python zero_token.py <jsonl_path>")
```

**Step 2: Run backfill**

```bash
python enrichment/zero_token.py data/enriched_repository.jsonl
```

**Step 3: Verify**

```bash
python -c "
import json
with open('data/enriched_repository.jsonl') as f:
    conv = json.loads(f.readline())
    print('New fields:', [k for k in conv.keys() if k.startswith('has_') or k.endswith('_count')])
"
```

---

### 0.3 Embeddings

**Goal:** Generate semantic vectors for similarity search

**Step 1: Install dependencies**

```bash
pip install sentence-transformers numpy
```

**Step 2: Create search/embeddings.py**

```python
"""Local embedding generation for semantic search."""

import numpy as np
from typing import List, Optional
from pathlib import Path
import json

class EmbeddingEngine:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def find_similar(
        self,
        query_vec: np.ndarray,
        corpus_vecs: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        """Find most similar vectors in corpus."""
        # Normalize
        query_norm = query_vec / np.linalg.norm(query_vec)
        corpus_norms = corpus_vecs / np.linalg.norm(corpus_vecs, axis=1, keepdims=True)

        # Compute similarities
        similarities = np.dot(corpus_norms, query_norm)

        # Get top k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]


def generate_embeddings(jsonl_path: str, output_path: str = None):
    """Generate embeddings for all conversations."""
    from tqdm import tqdm

    output_path = output_path or jsonl_path.replace('.jsonl', '_embeddings.npy')

    engine = EmbeddingEngine()

    # Load conversations
    texts = []
    ids = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            conv = json.loads(line)
            # Use summary + title for embedding
            text = f"{conv.get('generated_title', '')} {conv.get('summary_abstractive', '')}"
            texts.append(text)
            ids.append(conv.get('convo_id'))

    print(f"Generating embeddings for {len(texts)} conversations...")

    # Generate in batches
    embeddings = engine.embed_batch(texts, batch_size=64)

    # Save
    np.save(output_path, embeddings)

    # Save ID mapping
    id_path = output_path.replace('.npy', '_ids.json')
    with open(id_path, 'w') as f:
        json.dump(ids, f)

    print(f"Saved embeddings to: {output_path}")
    print(f"Saved ID mapping to: {id_path}")
    print(f"Embedding dimension: {embeddings.shape[1]}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        generate_embeddings(sys.argv[1])
    else:
        print("Usage: python embeddings.py <jsonl_path>")
```

**Step 3: Generate embeddings**

```bash
python search/embeddings.py data/enriched_repository.jsonl
```

---

### 0.4 Artifact Extraction

**Goal:** Extract reusable code, commands, solutions

**Step 1: Create enrichment/artifact_extractor.py**

```python
"""Extract reusable artifacts from conversations."""

import re
import json
from typing import List, Dict, Any
from dataclasses import dataclass, field, asdict
from anthropic import Anthropic

@dataclass
class Artifact:
    """A reusable piece extracted from a conversation."""
    artifact_id: str
    conversation_id: str
    artifact_type: str
    content: str
    language: str = ""
    description: str = ""
    use_case: str = ""
    tags: List[str] = field(default_factory=list)
    verified_working: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


EXTRACTION_PROMPT = '''Analyze this conversation and extract reusable artifacts.

<conversation>
{conversation}
</conversation>

Extract artifacts of these types:
- code_snippet: Reusable code (functions, classes, patterns)
- shell_command: Terminal commands that could be reused
- configuration: Config files, environment variables
- error_solution: Problem + solution pairs
- best_practice: Advice or patterns worth remembering

For each artifact, provide:
- type: One of the types above
- content: The actual artifact (exact code/command)
- language: Programming language if applicable
- description: What this does (1 sentence)
- use_case: When to use this
- tags: Relevant keywords

Respond with a JSON array of artifacts. Only include genuinely reusable items.
If no artifacts found, return an empty array [].

Example:
[
  {
    "type": "shell_command",
    "content": "docker run -d --name redis -p 6379:6379 redis:alpine",
    "language": "bash",
    "description": "Start Redis container with Alpine image",
    "use_case": "Local development caching",
    "tags": ["docker", "redis", "cache"]
  }
]'''


class ArtifactExtractor:
    """Extract artifacts using Claude Haiku."""

    def __init__(self, api_key: str = None):
        import os
        from core.config import AnthropicConfig

        if api_key:
            self.api_key = api_key
        else:
            config = AnthropicConfig.load()
            self.api_key = config.api_key

        if not self.api_key:
            raise ValueError("No API key configured. Run cogrepo_setup.py")

        self.client = Anthropic(api_key=self.api_key)

    def extract(self, conversation_id: str, raw_text: str) -> List[Artifact]:
        """Extract artifacts from a conversation."""
        # Truncate if too long
        if len(raw_text) > 12000:
            raw_text = raw_text[:6000] + "\n\n[...truncated...]\n\n" + raw_text[-6000:]

        prompt = EXTRACTION_PROMPT.format(conversation=raw_text)

        response = self.client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response
        text = response.content[0].text

        try:
            # Find JSON array in response
            match = re.search(r'\[[\s\S]*\]', text)
            if match:
                artifacts_data = json.loads(match.group())
            else:
                artifacts_data = []
        except json.JSONDecodeError:
            artifacts_data = []

        # Convert to Artifact objects
        artifacts = []
        for i, data in enumerate(artifacts_data):
            artifact = Artifact(
                artifact_id=f"{conversation_id}_art_{i}",
                conversation_id=conversation_id,
                artifact_type=data.get('type', 'unknown'),
                content=data.get('content', ''),
                language=data.get('language', ''),
                description=data.get('description', ''),
                use_case=data.get('use_case', ''),
                tags=data.get('tags', []),
                verified_working=True  # Assume working if in successful conversation
            )
            artifacts.append(artifact)

        return artifacts


def backfill_artifacts(jsonl_path: str, output_path: str = None):
    """Extract artifacts from all conversations."""
    from tqdm import tqdm
    import time

    output_path = output_path or jsonl_path.replace('.jsonl', '_artifacts.jsonl')

    extractor = ArtifactExtractor()

    # Load conversations
    conversations = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))

    print(f"Extracting artifacts from {len(conversations)} conversations...")

    all_artifacts = []

    for conv in tqdm(conversations):
        try:
            artifacts = extractor.extract(
                conv.get('convo_id', ''),
                conv.get('raw_text', '')
            )
            all_artifacts.extend(artifacts)

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"Error processing {conv.get('convo_id')}: {e}")

    # Save artifacts
    with open(output_path, 'w') as f:
        for artifact in all_artifacts:
            f.write(json.dumps(artifact.to_dict(), ensure_ascii=False) + '\n')

    print(f"\nExtracted {len(all_artifacts)} artifacts")
    print(f"Saved to: {output_path}")

    # Stats by type
    by_type = {}
    for a in all_artifacts:
        by_type[a.artifact_type] = by_type.get(a.artifact_type, 0) + 1

    print("\nBy type:")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        backfill_artifacts(sys.argv[1])
    else:
        print("Usage: python artifact_extractor.py <jsonl_path>")
```

**Step 2: Run extraction**

```bash
python enrichment/artifact_extractor.py data/enriched_repository.jsonl
```

---

### 0.5 - 0.7: Context System, Deep Analysis, Documentation

See [CONTINUATION_GUIDE.md](./CONTINUATION_GUIDE.md) for detailed instructions on these remaining Phase 0 tasks and subsequent phases.

---

## Quick Reference: Running Each Phase

```bash
# Phase 0.1: Config
python cogrepo_setup.py

# Phase 0.2: Zero-token
python enrichment/zero_token.py data/enriched_repository.jsonl

# Phase 0.3: Embeddings
python search/embeddings.py data/enriched_repository.jsonl

# Phase 0.4: Artifacts
python enrichment/artifact_extractor.py data/enriched_repository.jsonl

# Verify all
python cogrepo_verify.py --phase 0
```
