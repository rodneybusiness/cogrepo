"""
CogRepo Multi-Topic Segmentation

Handles conversations that contain multiple distinct topics/subjects.
Many chatbot conversations naturally evolve through multiple topics.

Features:
- Segment conversations into distinct topics
- Identify topic transitions
- Tag each segment independently
- Handle topic threading and returns

Usage:
    from intelligence.topic_segmentation import TopicSegmenter

    segmenter = TopicSegmenter()
    segments = segmenter.segment(conversation)

    for segment in segments:
        print(f"Topic: {segment.topic}")
        print(f"Content: {segment.content[:100]}...")
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict


@dataclass
class TopicSegment:
    """A distinct topic segment within a conversation."""
    id: str
    topic: str
    subtopics: List[str]
    content: str
    start_turn: int
    end_turn: int
    technologies: List[str]
    concepts: List[str]
    is_code_focused: bool
    confidence: float
    related_segments: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'topic': self.topic,
            'subtopics': self.subtopics,
            'content_preview': self.content[:200] + '...' if len(self.content) > 200 else self.content,
            'start_turn': self.start_turn,
            'end_turn': self.end_turn,
            'technologies': self.technologies,
            'concepts': self.concepts,
            'is_code_focused': self.is_code_focused,
            'confidence': self.confidence,
            'related_segments': self.related_segments,
        }


@dataclass
class ConversationTopics:
    """All topics identified in a conversation."""
    conversation_id: str
    segments: List[TopicSegment]
    primary_topic: str
    topic_count: int
    has_topic_transitions: bool
    topic_flow: List[str]  # Ordered list of topic IDs
    topic_summary: str

    def to_dict(self) -> dict:
        return {
            'conversation_id': self.conversation_id,
            'segments': [s.to_dict() for s in self.segments],
            'primary_topic': self.primary_topic,
            'topic_count': self.topic_count,
            'has_topic_transitions': self.has_topic_transitions,
            'topic_flow': self.topic_flow,
            'topic_summary': self.topic_summary,
        }


class TopicSegmenter:
    """
    Segment conversations into distinct topics.

    Uses multiple signals:
    1. Explicit topic change markers ("new question", "different topic")
    2. Semantic shift detection (change in terminology)
    3. Code block boundaries (often indicate task boundaries)
    4. Question patterns (new questions often = new topics)
    """

    # Patterns indicating topic changes
    TOPIC_CHANGE_PATTERNS = [
        re.compile(r'(?:new|different|another|separate|unrelated)\s+(?:question|topic|issue|problem|thing)', re.I),
        re.compile(r'(?:can|could)\s+(?:I|we)\s+(?:also|now)\s+(?:ask|discuss|talk about)', re.I),
        re.compile(r'(?:moving|switching|changing)\s+(?:to|on to)\s+(?:a|another)', re.I),
        re.compile(r'(?:by the way|btw|also|separately|on another note)', re.I),
        re.compile(r'(?:one more thing|quick question|unrelated but)', re.I),
        re.compile(r'(?:thanks|thank you)[.!]*\s*(?:now|also|can)', re.I),
    ]

    # Technology patterns for classification
    TECH_PATTERNS = {
        'python': re.compile(r'\bpython\b|\.py\b|pip\s+install|import\s+\w+', re.I),
        'javascript': re.compile(r'\bjavascript\b|\.js\b|npm\b|node\b|const\s+\w+\s*=', re.I),
        'typescript': re.compile(r'\btypescript\b|\.ts\b|interface\s+\w+|type\s+\w+\s*=', re.I),
        'react': re.compile(r'\breact\b|jsx\b|tsx\b|usestate|useeffect|component', re.I),
        'database': re.compile(r'\bsql\b|database|query|select\s+\*|insert\s+into', re.I),
        'api': re.compile(r'\bapi\b|\brest\b|endpoint|fetch\(|axios|request', re.I),
        'docker': re.compile(r'\bdocker\b|container|dockerfile|docker-compose', re.I),
        'git': re.compile(r'\bgit\b|commit|branch|merge|pull request', re.I),
        'testing': re.compile(r'\btest\b|pytest|jest|unittest|mock|assert', re.I),
        'debugging': re.compile(r'\bdebug\b|error|exception|traceback|stack trace', re.I),
    }

    # Concept patterns
    CONCEPT_PATTERNS = {
        'architecture': re.compile(r'\barchitecture\b|design pattern|structure|organize', re.I),
        'performance': re.compile(r'\bperformance\b|optimiz|slow|fast|speed|memory', re.I),
        'security': re.compile(r'\bsecurity\b|auth|password|encrypt|token|vulnerability', re.I),
        'deployment': re.compile(r'\bdeploy\b|production|server|hosting|cloud', re.I),
        'data': re.compile(r'\bdata\b|dataset|csv|json|parse|transform', re.I),
        'ui': re.compile(r'\bui\b|ux|interface|frontend|css|style|layout', re.I),
    }

    def __init__(self, min_segment_length: int = 100):
        """
        Initialize topic segmenter.

        Args:
            min_segment_length: Minimum characters for a segment
        """
        self.min_segment_length = min_segment_length

    def segment(self, conversation: dict) -> ConversationTopics:
        """
        Segment a conversation into topics.

        Args:
            conversation: Conversation dict with raw_text or messages

        Returns:
            ConversationTopics object
        """
        convo_id = conversation.get('convo_id', '')

        # Get turns
        turns = self._extract_turns(conversation)
        if not turns:
            return self._empty_result(convo_id)

        # Find topic boundaries
        boundaries = self._detect_boundaries(turns)

        # Create segments
        segments = self._create_segments(turns, boundaries, convo_id)

        # Analyze relationships between segments
        self._find_related_segments(segments)

        # Determine primary topic
        primary_topic = self._determine_primary_topic(segments)

        # Create flow
        topic_flow = [s.id for s in segments]

        # Generate summary
        summary = self._generate_summary(segments)

        return ConversationTopics(
            conversation_id=convo_id,
            segments=segments,
            primary_topic=primary_topic,
            topic_count=len(segments),
            has_topic_transitions=len(segments) > 1,
            topic_flow=topic_flow,
            topic_summary=summary,
        )

    def _extract_turns(self, conversation: dict) -> List[Dict]:
        """Extract turns from conversation."""
        turns = []

        # Try messages first
        messages = conversation.get('messages', [])
        if messages:
            for i, msg in enumerate(messages):
                turns.append({
                    'turn': i,
                    'role': msg.get('role', 'unknown'),
                    'content': msg.get('content', ''),
                })
            return turns

        # Fall back to raw_text
        raw_text = conversation.get('raw_text', '')
        if not raw_text:
            return []

        # Try to parse role markers
        parts = re.split(r'\n(?=(?:Human|User|Assistant|Claude|AI):)', raw_text)

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            # Determine role
            if part.startswith(('Human:', 'User:')):
                role = 'user'
                content = re.sub(r'^(?:Human|User):\s*', '', part)
            elif part.startswith(('Assistant:', 'Claude:', 'AI:')):
                role = 'assistant'
                content = re.sub(r'^(?:Assistant|Claude|AI):\s*', '', part)
            else:
                role = 'unknown'
                content = part

            turns.append({
                'turn': i,
                'role': role,
                'content': content,
            })

        return turns

    def _detect_boundaries(self, turns: List[Dict]) -> List[int]:
        """Detect topic boundary indices."""
        boundaries = [0]  # Start is always a boundary

        for i, turn in enumerate(turns):
            if i == 0:
                continue

            content = turn['content']
            prev_content = turns[i - 1]['content']

            # Check for explicit topic change markers
            if self._has_topic_change_marker(content):
                if i not in boundaries:
                    boundaries.append(i)
                continue

            # Check for significant terminology shift
            if self._has_terminology_shift(prev_content, content):
                if i not in boundaries:
                    boundaries.append(i)
                continue

            # Check for user role with question after assistant response
            if turn['role'] == 'user' and i >= 2:
                if '?' in content and len(content) > 50:
                    prev_techs = self._extract_technologies(prev_content)
                    curr_techs = self._extract_technologies(content)
                    if prev_techs and curr_techs and not (prev_techs & curr_techs):
                        if i not in boundaries:
                            boundaries.append(i)

        return sorted(boundaries)

    def _has_topic_change_marker(self, content: str) -> bool:
        """Check if content has explicit topic change markers."""
        return any(p.search(content) for p in self.TOPIC_CHANGE_PATTERNS)

    def _has_terminology_shift(self, prev: str, curr: str) -> bool:
        """Detect significant shift in terminology."""
        prev_techs = self._extract_technologies(prev)
        curr_techs = self._extract_technologies(curr)

        prev_concepts = self._extract_concepts(prev)
        curr_concepts = self._extract_concepts(curr)

        # Major technology shift
        if prev_techs and curr_techs:
            overlap = len(prev_techs & curr_techs) / max(len(prev_techs | curr_techs), 1)
            if overlap < 0.2:
                return True

        # Major concept shift
        if prev_concepts and curr_concepts:
            overlap = len(prev_concepts & curr_concepts) / max(len(prev_concepts | curr_concepts), 1)
            if overlap < 0.2:
                return True

        return False

    def _extract_technologies(self, text: str) -> Set[str]:
        """Extract technology mentions."""
        techs = set()
        for tech, pattern in self.TECH_PATTERNS.items():
            if pattern.search(text):
                techs.add(tech)
        return techs

    def _extract_concepts(self, text: str) -> Set[str]:
        """Extract concept mentions."""
        concepts = set()
        for concept, pattern in self.CONCEPT_PATTERNS.items():
            if pattern.search(text):
                concepts.add(concept)
        return concepts

    def _create_segments(
        self,
        turns: List[Dict],
        boundaries: List[int],
        convo_id: str
    ) -> List[TopicSegment]:
        """Create topic segments from boundaries."""
        segments = []

        for i, start in enumerate(boundaries):
            # Determine end
            if i + 1 < len(boundaries):
                end = boundaries[i + 1]
            else:
                end = len(turns)

            # Get content for this segment
            segment_turns = turns[start:end]
            content = '\n\n'.join(t['content'] for t in segment_turns)

            if len(content) < self.min_segment_length and i > 0:
                # Merge with previous segment if too short
                if segments:
                    segments[-1].content += '\n\n' + content
                    segments[-1].end_turn = end - 1
                continue

            # Extract metadata
            technologies = list(self._extract_technologies(content))
            concepts = list(self._extract_concepts(content))

            # Determine topic name
            topic = self._determine_segment_topic(content, technologies, concepts)

            # Check if code-focused
            is_code_focused = bool(re.search(r'```[\s\S]*```', content))

            # Extract subtopics
            subtopics = self._extract_subtopics(content, topic)

            segments.append(TopicSegment(
                id=f"{convo_id}_seg_{i}",
                topic=topic,
                subtopics=subtopics,
                content=content,
                start_turn=start,
                end_turn=end - 1,
                technologies=technologies,
                concepts=concepts,
                is_code_focused=is_code_focused,
                confidence=0.8 if len(technologies) > 0 or len(concepts) > 0 else 0.5,
            ))

        return segments

    def _determine_segment_topic(
        self,
        content: str,
        technologies: List[str],
        concepts: List[str]
    ) -> str:
        """Determine the main topic for a segment."""
        # Priority order: technology, concept, generic

        if technologies:
            if len(technologies) == 1:
                return f"{technologies[0].title()} Development"
            else:
                return f"{technologies[0].title()} & {technologies[1].title()}"

        if concepts:
            return concepts[0].title()

        # Fall back to generic analysis
        if '?' in content:
            return "Question & Answer"
        elif re.search(r'```', content):
            return "Code Implementation"
        else:
            return "Discussion"

    def _extract_subtopics(self, content: str, main_topic: str) -> List[str]:
        """Extract subtopics from content."""
        subtopics = []

        # Check for specific patterns
        if re.search(r'\berror\b|\bfix\b|\bissue\b', content, re.I):
            subtopics.append("Troubleshooting")
        if re.search(r'\binstall\b|\bsetup\b|\bconfigure\b', content, re.I):
            subtopics.append("Setup")
        if re.search(r'\bexample\b|\bhow to\b|\btutorial\b', content, re.I):
            subtopics.append("Tutorial")
        if re.search(r'\brefactor\b|\bimprove\b|\boptimize\b', content, re.I):
            subtopics.append("Optimization")
        if re.search(r'\bexplain\b|\bunderstand\b|\bwhy\b', content, re.I):
            subtopics.append("Explanation")

        return subtopics[:3]  # Limit to 3 subtopics

    def _find_related_segments(self, segments: List[TopicSegment]):
        """Find relationships between segments (e.g., topic returns)."""
        for i, seg in enumerate(segments):
            for j, other in enumerate(segments):
                if i == j:
                    continue

                # Check technology overlap
                tech_overlap = set(seg.technologies) & set(other.technologies)
                if tech_overlap:
                    seg.related_segments.append(other.id)

    def _determine_primary_topic(self, segments: List[TopicSegment]) -> str:
        """Determine the primary/main topic."""
        if not segments:
            return "Unknown"

        # Longest segment or most technologies
        best = max(segments, key=lambda s: len(s.content) + len(s.technologies) * 100)
        return best.topic

    def _generate_summary(self, segments: List[TopicSegment]) -> str:
        """Generate a summary of topics."""
        if not segments:
            return "No topics identified"

        if len(segments) == 1:
            return f"Single topic: {segments[0].topic}"

        topics = [s.topic for s in segments]
        return f"Multi-topic conversation: {', '.join(topics)}"

    def _empty_result(self, convo_id: str) -> ConversationTopics:
        """Return empty result."""
        return ConversationTopics(
            conversation_id=convo_id,
            segments=[],
            primary_topic="Unknown",
            topic_count=0,
            has_topic_transitions=False,
            topic_flow=[],
            topic_summary="Unable to segment conversation",
        )


def segment_conversations(
    conversations: List[dict],
    min_segment_length: int = 100
) -> List[ConversationTopics]:
    """
    Segment multiple conversations.

    Args:
        conversations: List of conversation dicts
        min_segment_length: Minimum segment length

    Returns:
        List of ConversationTopics
    """
    segmenter = TopicSegmenter(min_segment_length)
    return [segmenter.segment(conv) for conv in conversations]


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Segment conversations by topic")
    parser.add_argument('input', help="Input JSONL file")
    parser.add_argument('--output', '-o', help="Output JSON file")

    args = parser.parse_args()

    # Load conversations
    conversations = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    print(f"Loaded {len(conversations)} conversations")

    # Segment
    results = segment_conversations(conversations)

    # Stats
    multi_topic = sum(1 for r in results if r.has_topic_transitions)
    avg_topics = sum(r.topic_count for r in results) / max(len(results), 1)

    print(f"\nSegmentation Results:")
    print(f"  Multi-topic conversations: {multi_topic} ({100*multi_topic/len(results):.0f}%)")
    print(f"  Average topics per conversation: {avg_topics:.1f}")

    # Show examples
    print("\nExamples of multi-topic conversations:")
    for result in results[:5]:
        if result.has_topic_transitions:
            print(f"\n  {result.conversation_id}:")
            print(f"    {result.topic_summary}")
            for seg in result.segments:
                print(f"      - {seg.topic} (turns {seg.start_turn}-{seg.end_turn})")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nSaved to {args.output}")
