"""
Project Inference System for CogRepo v2

Auto-groups conversations by detecting project context from:
- File paths mentioned
- Repository names
- Technical stack signals
- Common terms and patterns

Zero-token cost: Uses only local pattern matching.

Usage:
    from context.project_inference import ProjectInferrer

    inferrer = ProjectInferrer()
    projects = inferrer.infer_projects(conversations)

    for project in projects:
        print(f"{project.name}: {len(project.conversation_ids)} conversations")
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, Counter
from pathlib import Path


@dataclass
class Project:
    """Represents an inferred project grouping."""
    name: str
    conversation_ids: List[str] = field(default_factory=list)
    file_paths: Set[str] = field(default_factory=set)
    technologies: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    keywords: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'conversation_ids': self.conversation_ids,
            'file_paths': list(self.file_paths),
            'technologies': list(self.technologies),
            'confidence': self.confidence,
            'keywords': list(self.keywords),
        }


class ProjectInferrer:
    """
    Infer project groupings from conversations.

    Uses multiple signals:
    1. File path patterns (e.g., /home/user/myproject/...)
    2. Repository names (e.g., "in the myrepo repository")
    3. Technology stack (e.g., React + TypeScript + Node)
    4. Common technical terms
    """

    # Patterns for extracting project signals
    FILE_PATH_PATTERN = re.compile(
        r'(?:^|[\s\'"(])(/(?:home|Users|var|opt|srv|projects?|repos?|work|code)/'
        r'[a-zA-Z0-9_.-]+/[a-zA-Z0-9_./+-]+)',
        re.MULTILINE
    )

    REPO_PATTERN = re.compile(
        r'(?:repository|repo|project|codebase)\s+(?:called\s+)?'
        r'["\']?([a-zA-Z][a-zA-Z0-9_-]+)["\']?',
        re.IGNORECASE
    )

    GIT_PATTERN = re.compile(
        r'(?:github\.com|gitlab\.com|bitbucket\.org)/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)',
        re.IGNORECASE
    )

    # Technology stacks that suggest related projects
    TECH_STACKS = {
        'react_frontend': {'react', 'jsx', 'tsx', 'webpack', 'vite', 'next.js', 'nextjs'},
        'vue_frontend': {'vue', 'vuex', 'nuxt', 'pinia'},
        'angular_frontend': {'angular', 'rxjs', 'ngrx'},
        'python_backend': {'flask', 'django', 'fastapi', 'sqlalchemy', 'celery'},
        'node_backend': {'express', 'nestjs', 'koa', 'prisma'},
        'rust_systems': {'rust', 'cargo', 'tokio', 'actix'},
        'go_systems': {'golang', 'gin', 'goroutine'},
        'data_science': {'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch'},
        'devops': {'docker', 'kubernetes', 'k8s', 'terraform', 'ansible'},
        'mobile': {'react native', 'flutter', 'swift', 'kotlin'},
    }

    def __init__(self, min_conversations: int = 2, min_confidence: float = 0.3):
        """
        Initialize the project inferrer.

        Args:
            min_conversations: Minimum conversations to form a project
            min_confidence: Minimum confidence score
        """
        self.min_conversations = min_conversations
        self.min_confidence = min_confidence

    def extract_signals(self, conversation: dict) -> dict:
        """
        Extract project signals from a single conversation.

        Args:
            conversation: Conversation dict with text content

        Returns:
            Dict of extracted signals
        """
        text = self._get_text(conversation)

        signals = {
            'project_roots': set(),
            'repo_names': set(),
            'git_repos': set(),
            'technologies': set(),
            'file_extensions': Counter(),
        }

        # Extract file paths and find project roots
        for match in self.FILE_PATH_PATTERN.finditer(text):
            path = match.group(1)
            root = self._extract_project_root(path)
            if root:
                signals['project_roots'].add(root)

            # Track file extensions
            ext = Path(path).suffix
            if ext:
                signals['file_extensions'][ext] += 1

        # Extract repository names
        for match in self.REPO_PATTERN.finditer(text):
            signals['repo_names'].add(match.group(1).lower())

        # Extract git URLs
        for match in self.GIT_PATTERN.finditer(text):
            signals['git_repos'].add(match.group(1).lower())

        # Detect technologies
        text_lower = text.lower()
        for tech_terms in self.TECH_STACKS.values():
            for term in tech_terms:
                if term in text_lower:
                    signals['technologies'].add(term)

        return signals

    def _get_text(self, conversation: dict) -> str:
        """Extract text content from conversation."""
        parts = []

        if 'raw_text' in conversation:
            parts.append(conversation['raw_text'])
        if 'generated_title' in conversation:
            parts.append(conversation['generated_title'])
        if 'summary_abstractive' in conversation:
            parts.append(conversation['summary_abstractive'])
        if 'file_paths' in conversation:
            parts.extend(conversation['file_paths'])

        return '\n'.join(parts)

    def _extract_project_root(self, path: str) -> Optional[str]:
        """
        Extract project root from a file path.

        Examples:
            /home/user/myproject/src/file.py -> /home/user/myproject
            /Users/dev/repos/api-server/pkg/... -> /Users/dev/repos/api-server
        """
        parts = Path(path).parts

        # Find likely project root (after common prefixes)
        prefixes = {'home', 'Users', 'var', 'opt', 'srv', 'projects', 'repos', 'work', 'code'}

        for i, part in enumerate(parts):
            if part in prefixes and i + 2 < len(parts):
                # Return the path up to and including the project directory
                if part in {'home', 'Users'}:
                    # /home/user/project or /Users/dev/project
                    if i + 2 < len(parts):
                        return '/'.join(parts[:i + 3])
                else:
                    # /var/repos/project
                    if i + 1 < len(parts):
                        return '/'.join(parts[:i + 2])

        return None

    def infer_projects(self, conversations: List[dict]) -> List[Project]:
        """
        Infer project groupings from a list of conversations.

        Args:
            conversations: List of conversation dicts

        Returns:
            List of Project objects, sorted by number of conversations
        """
        # Collect signals from all conversations
        convo_signals = {}
        for conv in conversations:
            convo_id = conv.get('convo_id', '')
            if convo_id:
                convo_signals[convo_id] = self.extract_signals(conv)

        # Group by project root
        root_to_convos: Dict[str, Set[str]] = defaultdict(set)
        root_to_techs: Dict[str, Set[str]] = defaultdict(set)

        for convo_id, signals in convo_signals.items():
            for root in signals['project_roots']:
                root_to_convos[root].add(convo_id)
                root_to_techs[root].update(signals['technologies'])

            for repo in signals['repo_names']:
                root_to_convos[f"repo:{repo}"].add(convo_id)
                root_to_techs[f"repo:{repo}"].update(signals['technologies'])

            for git_repo in signals['git_repos']:
                root_to_convos[f"git:{git_repo}"].add(convo_id)
                root_to_techs[f"git:{git_repo}"].update(signals['technologies'])

        # Merge overlapping projects
        projects = self._merge_projects(root_to_convos, root_to_techs)

        # Filter and sort
        filtered = [
            p for p in projects
            if len(p.conversation_ids) >= self.min_conversations
            and p.confidence >= self.min_confidence
        ]

        return sorted(filtered, key=lambda p: len(p.conversation_ids), reverse=True)

    def _merge_projects(
        self,
        root_to_convos: Dict[str, Set[str]],
        root_to_techs: Dict[str, Set[str]]
    ) -> List[Project]:
        """Merge projects with overlapping conversations."""
        # Find clusters of overlapping projects
        roots = list(root_to_convos.keys())
        merged = []
        used = set()

        for root in roots:
            if root in used:
                continue

            # Start a new cluster
            cluster_roots = {root}
            cluster_convos = set(root_to_convos[root])

            # Find overlapping roots
            changed = True
            while changed:
                changed = False
                for other_root in roots:
                    if other_root in cluster_roots:
                        continue

                    other_convos = root_to_convos[other_root]
                    # Merge if >50% overlap
                    overlap = len(cluster_convos & other_convos)
                    if overlap > 0 and overlap >= min(len(cluster_convos), len(other_convos)) * 0.5:
                        cluster_roots.add(other_root)
                        cluster_convos.update(other_convos)
                        changed = True

            used.update(cluster_roots)

            # Create project from cluster
            project = self._create_project(
                cluster_roots,
                cluster_convos,
                root_to_techs
            )
            merged.append(project)

        return merged

    def _create_project(
        self,
        roots: Set[str],
        convo_ids: Set[str],
        root_to_techs: Dict[str, Set[str]]
    ) -> Project:
        """Create a Project from merged roots."""
        # Determine best name
        name = self._choose_project_name(roots)

        # Collect technologies
        technologies = set()
        for root in roots:
            technologies.update(root_to_techs.get(root, set()))

        # Calculate confidence
        confidence = self._calculate_confidence(roots, convo_ids, technologies)

        return Project(
            name=name,
            conversation_ids=list(convo_ids),
            file_paths=set(r for r in roots if not r.startswith(('repo:', 'git:'))),
            technologies=technologies,
            confidence=confidence,
            keywords=self._extract_keywords(roots),
        )

    def _choose_project_name(self, roots: Set[str]) -> str:
        """Choose the best name for a project."""
        # Prefer git repos, then repo names, then path-based
        for root in roots:
            if root.startswith('git:'):
                return root[4:].split('/')[-1]

        for root in roots:
            if root.startswith('repo:'):
                return root[5:]

        # Use the shortest path-based name
        path_roots = [r for r in roots if not r.startswith(('repo:', 'git:'))]
        if path_roots:
            shortest = min(path_roots, key=len)
            return Path(shortest).name

        return 'unknown-project'

    def _calculate_confidence(
        self,
        roots: Set[str],
        convo_ids: Set[str],
        technologies: Set[str]
    ) -> float:
        """Calculate confidence score for a project grouping."""
        score = 0.0

        # More conversations = higher confidence
        n_convos = len(convo_ids)
        if n_convos >= 5:
            score += 0.3
        elif n_convos >= 3:
            score += 0.2
        elif n_convos >= 2:
            score += 0.1

        # Git/repo references = higher confidence
        if any(r.startswith(('git:', 'repo:')) for r in roots):
            score += 0.3

        # Consistent file paths = higher confidence
        path_roots = [r for r in roots if not r.startswith(('repo:', 'git:'))]
        if len(path_roots) >= 1:
            score += 0.2

        # Technology stack coherence
        if technologies:
            # Check if technologies belong to same stack
            for stack_name, stack_techs in self.TECH_STACKS.items():
                overlap = technologies & stack_techs
                if len(overlap) >= 2:
                    score += 0.2
                    break

        return min(score, 1.0)

    def _extract_keywords(self, roots: Set[str]) -> Set[str]:
        """Extract keywords from project roots."""
        keywords = set()

        for root in roots:
            if root.startswith('git:'):
                parts = root[4:].split('/')
                keywords.update(parts)
            elif root.startswith('repo:'):
                keywords.add(root[5:])
            else:
                # Path-based: extract project directory name
                keywords.add(Path(root).name)

        return keywords


def group_by_project(
    conversations: List[dict],
    min_conversations: int = 2
) -> Dict[str, List[dict]]:
    """
    Convenience function to group conversations by project.

    Args:
        conversations: List of conversation dicts
        min_conversations: Minimum conversations per project

    Returns:
        Dict mapping project name to list of conversations
    """
    inferrer = ProjectInferrer(min_conversations=min_conversations)
    projects = inferrer.infer_projects(conversations)

    # Create lookup
    convo_map = {c.get('convo_id', ''): c for c in conversations}

    result = {}
    for project in projects:
        result[project.name] = [
            convo_map[cid]
            for cid in project.conversation_ids
            if cid in convo_map
        ]

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Infer projects from conversations")
    parser.add_argument('input', help="Input JSONL file")
    parser.add_argument('--min-convos', '-m', type=int, default=2,
                        help="Minimum conversations per project")
    parser.add_argument('--output', '-o', help="Output JSON file (optional)")

    args = parser.parse_args()

    # Load conversations
    conversations = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    print(f"Loaded {len(conversations)} conversations")

    # Infer projects
    inferrer = ProjectInferrer(min_conversations=args.min_convos)
    projects = inferrer.infer_projects(conversations)

    print(f"\nFound {len(projects)} projects:")
    for project in projects:
        print(f"\n  {project.name}")
        print(f"    Conversations: {len(project.conversation_ids)}")
        print(f"    Technologies: {', '.join(project.technologies) or 'none detected'}")
        print(f"    Confidence: {project.confidence:.2f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump([p.to_dict() for p in projects], f, indent=2)
        print(f"\nSaved to {args.output}")
