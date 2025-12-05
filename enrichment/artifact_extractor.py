"""
Artifact Extraction Module

Extracts reusable artifacts (code snippets, commands, solutions) from conversations
using Claude Haiku for cost-effective processing.

Artifact types:
- code_snippet: Reusable code blocks
- shell_command: Terminal commands
- configuration: Config files, env vars
- api_call: API examples, curl commands
- sql_query: Database queries
- error_solution: Problem + fix pairs
- best_practice: Advice and patterns

Usage:
    from enrichment.artifact_extractor import ArtifactExtractor

    extractor = ArtifactExtractor()
    artifacts = extractor.extract(conversation_id, raw_text)
"""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


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
    verified_working: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Extraction Prompt
# =============================================================================

EXTRACTION_PROMPT = '''Analyze this conversation and extract reusable artifacts.

<conversation>
{conversation}
</conversation>

Extract artifacts of these types ONLY if they are genuinely reusable:
- code_snippet: Reusable code (functions, classes, patterns) - NOT trivial one-liners
- shell_command: Terminal commands that solve a specific problem
- configuration: Config files, environment variables, settings
- error_solution: Problem + solution pairs (error message and fix)
- best_practice: Important advice or patterns worth remembering

For each artifact provide:
- type: One of the types above
- content: The actual artifact (exact code/command/solution)
- language: Programming language if applicable (python, bash, sql, javascript, etc.)
- description: What this does in ONE sentence
- use_case: When to use this
- tags: 2-4 relevant keywords

Rules:
- Only extract GENUINELY REUSABLE items someone would want to copy
- Skip trivial code like "print('hello')" or "import os"
- For error_solution, include BOTH the error and the fix
- If no artifacts found, return empty array []

Respond with ONLY a valid JSON array. Example:
[
  {{
    "type": "shell_command",
    "content": "docker run -d --name redis -p 6379:6379 redis:alpine",
    "language": "bash",
    "description": "Start Redis container with Alpine image",
    "use_case": "Local development caching",
    "tags": ["docker", "redis"]
  }},
  {{
    "type": "error_solution",
    "content": "Error: CORS policy blocked\\nFix: pip install flask-cors; CORS(app)",
    "language": "python",
    "description": "Fix CORS errors in Flask",
    "use_case": "When frontend can't reach Flask API",
    "tags": ["cors", "flask"]
  }}
]'''


class ArtifactExtractor:
    """
    Extract reusable artifacts from conversations using Claude Haiku.

    Uses Haiku for cost-effective extraction (~$0.001 per conversation).
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the extractor.

        Args:
            api_key: Anthropic API key (loads from config if not provided)
        """
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        """Lazy load the Anthropic client."""
        if self._client is not None:
            return self._client

        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic not installed. Install with: pip install anthropic"
            )

        # Get API key
        api_key = self.api_key

        if not api_key:
            # Try to load from config
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from core.config import get_config
                config = get_config()
                api_key = config.anthropic.api_key
            except Exception:
                pass

        if not api_key:
            import os
            api_key = os.getenv('ANTHROPIC_API_KEY')

        if not api_key:
            raise ValueError(
                "No API key found. Set ANTHROPIC_API_KEY or run cogrepo_setup.py"
            )

        self._client = Anthropic(api_key=api_key)
        return self._client

    def extract(
        self,
        conversation_id: str,
        raw_text: str,
        max_retries: int = 2
    ) -> List[Artifact]:
        """
        Extract artifacts from a conversation.

        Args:
            conversation_id: ID of the conversation
            raw_text: Full conversation text
            max_retries: Number of retries on failure

        Returns:
            List of extracted Artifact objects
        """
        if not raw_text or len(raw_text) < 100:
            return []

        # Truncate long conversations
        if len(raw_text) > 15000:
            # Keep beginning and end
            raw_text = raw_text[:7000] + "\n\n[...middle truncated...]\n\n" + raw_text[-7000:]

        prompt = EXTRACTION_PROMPT.format(conversation=raw_text)

        for attempt in range(max_retries + 1):
            try:
                response = self.client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=2000,
                    temperature=0.2,
                    messages=[{"role": "user", "content": prompt}]
                )

                text = response.content[0].text
                artifacts_data = self._parse_response(text)

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
                        verified_working=True
                    )
                    artifacts.append(artifact)

                return artifacts

            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                else:
                    print(f"Failed to extract artifacts: {e}")
                    return []

        return []

    def _parse_response(self, text: str) -> List[Dict]:
        """Parse JSON response, handling various formats."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code blocks
        code_block = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON array
        array_match = re.search(r'\[[\s\S]*\]', text)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass

        return []


# =============================================================================
# Local Extraction (No API)
# =============================================================================

class LocalArtifactExtractor:
    """
    Extract artifacts using regex patterns (no API calls).

    Less accurate than LLM extraction but completely free.
    Good for initial pass or when API is not available.
    """

    # Patterns for different artifact types
    PATTERNS = {
        'shell_command': [
            # Docker commands
            re.compile(r'docker\s+(?:run|build|compose|exec|pull|push)\s+[^\n]+'),
            # Git commands
            re.compile(r'git\s+(?:clone|pull|push|commit|checkout|branch|merge|rebase|stash)\s+[^\n]+'),
            # Package managers
            re.compile(r'(?:pip|npm|yarn|cargo|go)\s+(?:install|add|remove|update)\s+[^\n]+'),
            # Common CLI tools
            re.compile(r'(?:curl|wget|ssh|scp|rsync)\s+[^\n]+'),
        ],
        'sql_query': [
            re.compile(r'(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+[\s\S]+?(?:;|$)', re.I),
        ],
    }

    def extract(self, conversation_id: str, raw_text: str) -> List[Artifact]:
        """Extract artifacts using patterns."""
        artifacts = []
        seen_content = set()

        # Extract code blocks with language hints
        code_blocks = re.findall(r'```(\w*)\n([\s\S]*?)```', raw_text)

        for lang, content in code_blocks:
            content = content.strip()
            if len(content) < 20 or content in seen_content:
                continue

            seen_content.add(content)

            # Determine artifact type
            if lang in ('bash', 'sh', 'shell', 'zsh'):
                artifact_type = 'shell_command'
            elif lang == 'sql':
                artifact_type = 'sql_query'
            elif lang in ('json', 'yaml', 'yml', 'toml', 'ini'):
                artifact_type = 'configuration'
            else:
                artifact_type = 'code_snippet'

            artifacts.append(Artifact(
                artifact_id=f"{conversation_id}_local_{len(artifacts)}",
                conversation_id=conversation_id,
                artifact_type=artifact_type,
                content=content,
                language=lang or 'unknown',
                description="Extracted code block",
                use_case="",
                tags=[lang] if lang else []
            ))

        return artifacts


# =============================================================================
# Batch Processing
# =============================================================================

def extract_artifacts_batch(
    input_path: str,
    output_path: str = None,
    use_api: bool = True,
    rate_limit: float = 0.5
) -> Dict[str, Any]:
    """
    Extract artifacts from all conversations in a JSONL file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output artifacts JSONL (defaults to input_artifacts.jsonl)
        use_api: Use Claude API (True) or local extraction (False)
        rate_limit: Seconds between API calls

    Returns:
        Statistics about the extraction
    """
    input_path = Path(input_path)
    output_path = Path(output_path or str(input_path).replace('.jsonl', '_artifacts.jsonl'))

    # Load conversations
    conversations = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    print(f"Processing {len(conversations)} conversations...")

    # Create extractor
    if use_api:
        extractor = ArtifactExtractor()
    else:
        extractor = LocalArtifactExtractor()

    # Stats
    stats = {
        'total_conversations': len(conversations),
        'conversations_with_artifacts': 0,
        'total_artifacts': 0,
        'by_type': {},
    }

    all_artifacts = []

    try:
        from tqdm import tqdm
        iterator = tqdm(conversations, desc="Extracting")
    except ImportError:
        iterator = conversations

    for conv in iterator:
        convo_id = conv.get('convo_id', '')
        raw_text = conv.get('raw_text', '')

        try:
            artifacts = extractor.extract(convo_id, raw_text)

            if artifacts:
                stats['conversations_with_artifacts'] += 1
                stats['total_artifacts'] += len(artifacts)

                for artifact in artifacts:
                    atype = artifact.artifact_type
                    stats['by_type'][atype] = stats['by_type'].get(atype, 0) + 1
                    all_artifacts.append(artifact.to_dict())

            if use_api:
                time.sleep(rate_limit)

        except Exception as e:
            print(f"Error processing {convo_id}: {e}")

    # Save artifacts
    with open(output_path, 'w') as f:
        for artifact in all_artifacts:
            f.write(json.dumps(artifact, ensure_ascii=False) + '\n')

    print(f"\nExtracted {stats['total_artifacts']} artifacts from {stats['conversations_with_artifacts']} conversations")
    print(f"Saved to {output_path}")

    if stats['by_type']:
        print("\nBy type:")
        for atype, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
            print(f"  {atype}: {count}")

    return stats


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Extract artifacts from conversations")
    parser.add_argument('input', help="Input JSONL file")
    parser.add_argument('--output', '-o', help="Output artifacts file")
    parser.add_argument('--local', action='store_true', help="Use local extraction (no API)")
    parser.add_argument('--rate-limit', '-r', type=float, default=0.5,
                        help="Seconds between API calls")

    args = parser.parse_args()

    stats = extract_artifacts_batch(
        args.input,
        args.output,
        use_api=not args.local,
        rate_limit=args.rate_limit
    )
