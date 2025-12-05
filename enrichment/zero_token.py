"""
Zero-Token Enrichment Module

Extracts valuable metadata from conversations without any API calls.
All processing is done locally using regex and simple analysis.

Features extracted:
- Code detection (languages, block count)
- Error trace detection
- Question analysis (count, types)
- Link extraction (URLs, domains)
- Technical term detection
- File path mentions
- Conversation structure metrics

Usage:
    from enrichment.zero_token import ZeroTokenEnricher

    enricher = ZeroTokenEnricher()
    metrics = enricher.extract(raw_text)
    print(metrics.has_code, metrics.code_languages)
"""

import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import Counter
from pathlib import Path


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

    # Temporal (if available)
    duration_minutes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def merge_into(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Merge metrics into an existing conversation dict."""
        conversation.update(self.to_dict())
        return conversation


class ZeroTokenEnricher:
    """
    Extract maximum value from conversations without API calls.

    All processing is done locally using regex and simple text analysis.
    This is completely free and fast.
    """

    # ==========================================================================
    # Patterns
    # ==========================================================================

    # Code block pattern (matches ```language\ncode```)
    CODE_BLOCK_PATTERN = re.compile(r'```(\w*)\n([\s\S]*?)```', re.MULTILINE)

    # Inline code pattern (matches `code`)
    INLINE_CODE_PATTERN = re.compile(r'`[^`]+`')

    # URL pattern (comprehensive)
    URL_PATTERN = re.compile(
        r'https?://'  # http:// or https://
        r'(?:[\w-]+\.)+[\w-]+'  # domain
        r'(?:/[^\s<>"\'`\[\]{}|\\^]*)?',  # path (optional)
        re.IGNORECASE
    )

    # File path patterns
    FILE_PATTERNS = [
        # Python, JS, TS, etc.
        re.compile(r'\b[\w./\\-]+\.(?:py|js|ts|jsx|tsx|java|cpp|c|h|go|rs|rb|php|swift|kt)\b'),
        # Config files
        re.compile(r'\b[\w./\\-]+\.(?:json|yaml|yml|toml|xml|ini|cfg|conf)\b'),
        # Web files
        re.compile(r'\b[\w./\\-]+\.(?:html|css|scss|sass|less)\b'),
        # Data files
        re.compile(r'\b[\w./\\-]+\.(?:csv|sql|md|txt|log)\b'),
        # Shell scripts
        re.compile(r'\b[\w./\\-]+\.(?:sh|bash|zsh)\b'),
    ]

    # Error trace patterns
    ERROR_PATTERNS = [
        re.compile(r'Traceback \(most recent call last\)', re.IGNORECASE),
        re.compile(r'^\s*File "[^"]+", line \d+', re.MULTILINE),
        re.compile(r'(?:Error|Exception):\s+\S+', re.IGNORECASE),
        re.compile(r'at\s+[\w.$]+\s*\([^)]+:\d+:\d+\)'),  # JS stack trace
        re.compile(r'panic:\s+', re.IGNORECASE),  # Go panic
        re.compile(r'FATAL\s+ERROR:', re.IGNORECASE),
    ]

    # Question type patterns
    QUESTION_PATTERNS = {
        'how-to': re.compile(r'\bhow\s+(?:do|can|to|should|would|could)\b', re.I),
        'why': re.compile(r'\bwhy\s+(?:is|does|do|are|was|were|would|should)\b', re.I),
        'what': re.compile(r'\bwhat\s+(?:is|are|does|do|was|were)\b', re.I),
        'debug': re.compile(r'\b(?:error|bug|issue|problem|not\s+working|fails?|broken|wrong|fix)\b', re.I),
        'explain': re.compile(r'\b(?:explain|understand|clarify|meaning|means|difference)\b', re.I),
        'compare': re.compile(r'\b(?:vs\.?|versus|compare|comparison|better|worse|prefer)\b', re.I),
        'implement': re.compile(r'\b(?:implement|create|build|make|write|code)\b', re.I),
    }

    # Common technical terms (lowercase for matching)
    TECHNICAL_TERMS = {
        # Languages
        'python', 'javascript', 'typescript', 'java', 'golang', 'rust', 'ruby', 'php', 'swift', 'kotlin',
        # Frameworks
        'react', 'vue', 'angular', 'django', 'flask', 'fastapi', 'express', 'nextjs', 'rails', 'spring',
        # Databases
        'sql', 'postgresql', 'postgres', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'sqlite',
        # Cloud/DevOps
        'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'terraform', 'ansible', 'jenkins', 'github',
        # Concepts
        'api', 'rest', 'graphql', 'websocket', 'async', 'await', 'promise', 'callback',
        'authentication', 'authorization', 'oauth', 'jwt', 'cors', 'csrf',
        'cache', 'caching', 'cdn', 'load balancer', 'microservices', 'serverless',
        'git', 'ci/cd', 'pipeline', 'deployment', 'container', 'virtualization',
        'testing', 'unit test', 'integration test', 'tdd', 'bdd', 'mock', 'fixture',
        'machine learning', 'deep learning', 'neural network', 'nlp', 'llm',
    }

    # Language aliases (normalize to standard names)
    LANGUAGE_ALIASES = {
        'py': 'python',
        'js': 'javascript',
        'ts': 'typescript',
        'rb': 'ruby',
        'sh': 'bash',
        'shell': 'bash',
        'yml': 'yaml',
        'dockerfile': 'docker',
    }

    def __init__(self):
        """Initialize the enricher."""
        # Pre-compile technical terms pattern for efficiency
        terms_pattern = '|'.join(re.escape(term) for term in self.TECHNICAL_TERMS)
        self.tech_terms_pattern = re.compile(rf'\b({terms_pattern})\b', re.IGNORECASE)

    def extract(self, raw_text: str, messages: List[Dict] = None) -> ZeroTokenMetrics:
        """
        Extract all zero-token metrics from conversation text.

        Args:
            raw_text: Full conversation text
            messages: Optional list of message dicts with 'role' and 'content'

        Returns:
            ZeroTokenMetrics with all extracted data
        """
        metrics = ZeroTokenMetrics()

        if not raw_text:
            return metrics

        # Code analysis
        self._extract_code_metrics(raw_text, metrics)

        # Error traces
        self._extract_error_traces(raw_text, metrics)

        # URLs and links
        self._extract_links(raw_text, metrics)

        # File paths
        self._extract_file_paths(raw_text, metrics)

        # Questions
        self._extract_questions(raw_text, metrics)

        # Technical terms
        self._extract_technical_terms(raw_text, metrics)

        # Message structure analysis
        if messages:
            self._analyze_messages(messages, metrics)
        else:
            self._estimate_turns(raw_text, metrics)

        return metrics

    def _extract_code_metrics(self, text: str, metrics: ZeroTokenMetrics):
        """Extract code block information."""
        # Find all code blocks
        code_blocks = self.CODE_BLOCK_PATTERN.findall(text)

        metrics.code_block_count = len(code_blocks)
        metrics.has_code = metrics.code_block_count > 0

        # Extract and normalize languages
        languages = set()
        for lang, _ in code_blocks:
            if lang:
                lang = lang.lower().strip()
                # Normalize aliases
                lang = self.LANGUAGE_ALIASES.get(lang, lang)
                if lang:
                    languages.add(lang)

        metrics.code_languages = sorted(list(languages))

    def _extract_error_traces(self, text: str, metrics: ZeroTokenMetrics):
        """Detect error traces and stack traces."""
        for pattern in self.ERROR_PATTERNS:
            if pattern.search(text):
                metrics.has_error_traces = True
                break

    def _extract_links(self, text: str, metrics: ZeroTokenMetrics):
        """Extract URLs and domains."""
        urls = self.URL_PATTERN.findall(text)

        metrics.has_links = len(urls) > 0
        metrics.link_urls = urls[:20]  # Limit to 20

        # Extract unique domains
        domains = set()
        for url in urls:
            try:
                # Simple domain extraction
                domain = url.split('//')[1].split('/')[0]
                domain = domain.split(':')[0]  # Remove port
                domains.add(domain)
            except (IndexError, AttributeError):
                pass

        metrics.link_domains = sorted(list(domains))[:10]

    def _extract_file_paths(self, text: str, metrics: ZeroTokenMetrics):
        """Extract mentioned file paths."""
        files = set()

        for pattern in self.FILE_PATTERNS:
            matches = pattern.findall(text)
            files.update(matches)

        # Filter out common false positives
        filtered = [
            f for f in files
            if not f.startswith('http')
            and not f.startswith('//')
            and len(f) > 2
        ]

        metrics.mentioned_files = sorted(list(filtered))[:20]

    def _extract_questions(self, text: str, metrics: ZeroTokenMetrics):
        """Analyze questions in the text."""
        metrics.question_count = text.count('?')

        detected_types = []
        for qtype, pattern in self.QUESTION_PATTERNS.items():
            if pattern.search(text):
                detected_types.append(qtype)

        metrics.question_types = detected_types

    def _extract_technical_terms(self, text: str, metrics: ZeroTokenMetrics):
        """Extract technical terms mentioned."""
        matches = self.tech_terms_pattern.findall(text.lower())

        # Count occurrences and get unique terms
        term_counts = Counter(matches)

        # Return sorted by frequency, then alphabetically
        sorted_terms = sorted(
            term_counts.keys(),
            key=lambda t: (-term_counts[t], t)
        )

        metrics.technical_terms = sorted_terms[:30]  # Top 30 terms

    def _analyze_messages(self, messages: List[Dict], metrics: ZeroTokenMetrics):
        """Analyze message structure."""
        user_msgs = [m for m in messages if m.get('role') == 'user']
        asst_msgs = [m for m in messages if m.get('role') == 'assistant']

        metrics.turn_count = min(len(user_msgs), len(asst_msgs))

        if user_msgs:
            total_user = sum(len(m.get('content', '')) for m in user_msgs)
            metrics.avg_user_length = total_user // len(user_msgs)

        if asst_msgs:
            total_asst = sum(len(m.get('content', '')) for m in asst_msgs)
            metrics.avg_assistant_length = total_asst // len(asst_msgs)

    def _estimate_turns(self, text: str, metrics: ZeroTokenMetrics):
        """Estimate turn count from raw text."""
        # Count USER: and ASSISTANT: patterns
        user_pattern = re.compile(r'^(?:USER|Human|You):', re.MULTILINE | re.IGNORECASE)
        asst_pattern = re.compile(r'^(?:ASSISTANT|AI|Claude|ChatGPT):', re.MULTILINE | re.IGNORECASE)

        user_count = len(user_pattern.findall(text))
        asst_count = len(asst_pattern.findall(text))

        metrics.turn_count = min(user_count, asst_count)


# =============================================================================
# Batch Processing
# =============================================================================

def enrich_conversation(conv: Dict[str, Any], enricher: ZeroTokenEnricher = None) -> Dict[str, Any]:
    """
    Enrich a single conversation with zero-token metrics.

    Args:
        conv: Conversation dict with 'raw_text'
        enricher: Optional ZeroTokenEnricher instance (creates new if None)

    Returns:
        Conversation dict with added metrics
    """
    if enricher is None:
        enricher = ZeroTokenEnricher()

    raw_text = conv.get('raw_text', '')
    metrics = enricher.extract(raw_text)

    return metrics.merge_into(conv)


def backfill_zero_token(
    input_path: str,
    output_path: str = None,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Backfill zero-token metrics for all conversations in a JSONL file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file (defaults to input_path)
        show_progress: Show progress bar

    Returns:
        Statistics about the enrichment
    """
    import json

    output_path = output_path or input_path
    enricher = ZeroTokenEnricher()

    # Read all conversations
    conversations = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    print(f"Processing {len(conversations)} conversations...")

    # Stats
    stats = {
        'total': len(conversations),
        'with_code': 0,
        'with_links': 0,
        'with_errors': 0,
        'languages': Counter(),
        'domains': Counter(),
    }

    # Process each conversation
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(conversations, desc="Enriching")
        except ImportError:
            iterator = conversations
    else:
        iterator = conversations

    for conv in iterator:
        raw_text = conv.get('raw_text', '')
        metrics = enricher.extract(raw_text)
        metrics.merge_into(conv)

        # Update stats
        if metrics.has_code:
            stats['with_code'] += 1
            for lang in metrics.code_languages:
                stats['languages'][lang] += 1

        if metrics.has_links:
            stats['with_links'] += 1
            for domain in metrics.link_domains:
                stats['domains'][domain] += 1

        if metrics.has_error_traces:
            stats['with_errors'] += 1

    # Write results
    with open(output_path, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    print(f"\nDone! Enriched {len(conversations)} conversations.")
    print(f"  With code: {stats['with_code']} ({100*stats['with_code']/len(conversations):.1f}%)")
    print(f"  With links: {stats['with_links']} ({100*stats['with_links']/len(conversations):.1f}%)")
    print(f"  With errors: {stats['with_errors']} ({100*stats['with_errors']/len(conversations):.1f}%)")

    if stats['languages']:
        print(f"\nTop languages:")
        for lang, count in stats['languages'].most_common(5):
            print(f"    {lang}: {count}")

    return stats


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Zero-token conversation enrichment")
    parser.add_argument('input', help="Input JSONL file")
    parser.add_argument('--output', '-o', help="Output file (defaults to input)")
    parser.add_argument('--no-progress', action='store_true', help="Hide progress bar")

    args = parser.parse_args()

    stats = backfill_zero_token(
        args.input,
        args.output,
        show_progress=not args.no_progress
    )
