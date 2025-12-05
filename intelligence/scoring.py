"""
CogRepo Advanced Scoring System

Replaces the simple 1-9 scale with a comprehensive 1-100 scoring system
with multiple dimensions and finer granularity.

Dimensions:
- Technical Depth (0-100): How deep/complex is the technical content
- Practical Value (0-100): How useful/reusable is the content
- Completeness (0-100): How complete is the conversation/solution
- Clarity (0-100): How clear and well-explained is the content
- Uniqueness (0-100): How unique/novel is the content

Overall Score: Weighted combination of dimensions

Usage:
    from intelligence.scoring import ConversationScorer

    scorer = ConversationScorer()
    scores = scorer.score(conversation)

    print(f"Overall: {scores.overall}")
    print(f"Technical Depth: {scores.technical_depth}")
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import Counter


@dataclass
class DimensionScore:
    """Score for a single dimension."""
    value: int  # 0-100
    confidence: float  # 0.0-1.0
    factors: List[str]  # Factors that contributed to this score

    def to_dict(self) -> dict:
        return {
            'value': self.value,
            'confidence': self.confidence,
            'factors': self.factors,
        }


@dataclass
class ConversationScore:
    """Comprehensive score for a conversation."""
    overall: int  # 0-100 weighted average

    # Individual dimensions
    technical_depth: DimensionScore
    practical_value: DimensionScore
    completeness: DimensionScore
    clarity: DimensionScore
    uniqueness: DimensionScore

    # Metadata
    grade: str  # A+, A, B+, B, C+, C, D, F
    percentile: Optional[float] = None  # Percentile in repository
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            'overall': self.overall,
            'grade': self.grade,
            'percentile': self.percentile,
            'summary': self.summary,
            'dimensions': {
                'technical_depth': self.technical_depth.to_dict(),
                'practical_value': self.practical_value.to_dict(),
                'completeness': self.completeness.to_dict(),
                'clarity': self.clarity.to_dict(),
                'uniqueness': self.uniqueness.to_dict(),
            }
        }

    @classmethod
    def from_legacy_score(cls, score_1_9: int) -> 'ConversationScore':
        """Convert legacy 1-9 score to new format."""
        # Map 1-9 to approximately 0-100
        overall = int((score_1_9 - 1) * 12.5)  # 1->0, 5->50, 9->100
        overall = max(0, min(100, overall))

        # Estimate dimensions from overall
        return cls(
            overall=overall,
            technical_depth=DimensionScore(overall, 0.5, ['converted from legacy']),
            practical_value=DimensionScore(overall, 0.5, ['converted from legacy']),
            completeness=DimensionScore(overall, 0.5, ['converted from legacy']),
            clarity=DimensionScore(overall, 0.5, ['converted from legacy']),
            uniqueness=DimensionScore(int(overall * 0.8), 0.3, ['converted from legacy']),
            grade=cls._score_to_grade(overall),
            summary="Score converted from legacy 1-9 scale"
        )

    @staticmethod
    def _score_to_grade(score: int) -> str:
        """Convert numeric score to letter grade.

        Grade scale (consistent with documentation):
        A+ (95+), A (90-94), A- (87-89)
        B+ (85-86), B (75-84), B- (70-74)
        C+ (65-69), C (50-64), C- (45-49)
        D (35-44), F (<35)
        """
        if score >= 95: return 'A+'
        if score >= 90: return 'A'
        if score >= 87: return 'A-'
        if score >= 85: return 'B+'
        if score >= 75: return 'B'
        if score >= 70: return 'B-'
        if score >= 65: return 'C+'
        if score >= 50: return 'C'
        if score >= 45: return 'C-'
        if score >= 35: return 'D'
        return 'F'


class ConversationScorer:
    """
    Score conversations on multiple dimensions.

    Uses heuristics and pattern analysis for zero-token scoring,
    with optional LLM enhancement.
    """

    # Dimension weights for overall score
    WEIGHTS = {
        'technical_depth': 0.25,
        'practical_value': 0.30,
        'completeness': 0.20,
        'clarity': 0.15,
        'uniqueness': 0.10,
    }

    # Technical patterns
    ADVANCED_TECH_PATTERNS = [
        re.compile(r'\basync\b.*\bawait\b', re.I),
        re.compile(r'\bgenerics?\b|\btype\s+parameter', re.I),
        re.compile(r'\brecursion\b|\brecursive\b', re.I),
        re.compile(r'\bconcurrency\b|\bparallel\b|\bthread', re.I),
        re.compile(r'\boptimiz\w+\b', re.I),
        re.compile(r'\balgorithm\b|\bdata structure\b', re.I),
        re.compile(r'\bdesign pattern\b|\barchitecture\b', re.I),
        re.compile(r'\bscalability\b|\bperformance\b', re.I),
        re.compile(r'\bsecurity\b|\bencryption\b|\bauth', re.I),
        re.compile(r'\bmicroservice\b|\bdistributed\b', re.I),
    ]

    PRACTICAL_PATTERNS = [
        re.compile(r'```\w*\n[\s\S]+```'),  # Code blocks
        re.compile(r'\bexample\b.*:|\bhere\'s how\b', re.I),
        re.compile(r'\bstep\s+\d+\b|\bfirst,?\s+\w+\b.*\bthen\b', re.I),
        re.compile(r'\binstall\b|\bsetup\b|\bconfigure\b', re.I),
        re.compile(r'\bcommand\b|\bterminal\b|\bbash\b', re.I),
        re.compile(r'`[^`]+`'),  # Inline code
        re.compile(r'\bfile\s*:\s*\S+', re.I),
        re.compile(r'https?://\S+'),  # URLs/links
    ]

    CLARITY_PATTERNS = [
        re.compile(r'\bin\s+(?:other\s+)?words\b', re.I),
        re.compile(r'\bfor\s+example\b|\be\.g\.\b|\bi\.e\.\b', re.I),
        re.compile(r'\bthis\s+means\b|\bspecifically\b', re.I),
        re.compile(r'\bstep\s+by\s+step\b', re.I),
        re.compile(r'\bsummary\b|\bto\s+summarize\b', re.I),
        re.compile(r'\bkey\s+(?:point|takeaway)\b', re.I),
        re.compile(r'^\s*[-*•]\s+', re.MULTILINE),  # Bullet points
        re.compile(r'^\s*\d+\.\s+', re.MULTILINE),  # Numbered lists
    ]

    def __init__(self, all_conversations: List[dict] = None):
        """
        Initialize scorer.

        Args:
            all_conversations: All conversations for percentile calculation
        """
        self.all_conversations = all_conversations or []
        self._tech_term_counts = self._build_term_frequency()

    def _build_term_frequency(self) -> Counter:
        """Build term frequency for uniqueness scoring."""
        counter = Counter()
        for conv in self.all_conversations:
            terms = conv.get('technical_terms', [])
            counter.update(terms)
        return counter

    def score(self, conversation: dict) -> ConversationScore:
        """
        Score a conversation on all dimensions.

        Args:
            conversation: Conversation dict

        Returns:
            ConversationScore object
        """
        text = self._get_text(conversation)

        # Score each dimension
        technical = self._score_technical_depth(text, conversation)
        practical = self._score_practical_value(text, conversation)
        completeness = self._score_completeness(text, conversation)
        clarity = self._score_clarity(text, conversation)
        uniqueness = self._score_uniqueness(text, conversation)

        # Calculate weighted overall
        overall = int(
            technical.value * self.WEIGHTS['technical_depth'] +
            practical.value * self.WEIGHTS['practical_value'] +
            completeness.value * self.WEIGHTS['completeness'] +
            clarity.value * self.WEIGHTS['clarity'] +
            uniqueness.value * self.WEIGHTS['uniqueness']
        )

        # Calculate grade
        grade = ConversationScore._score_to_grade(overall)

        # Generate summary
        summary = self._generate_summary(
            overall, technical, practical, completeness, clarity, uniqueness
        )

        return ConversationScore(
            overall=overall,
            technical_depth=technical,
            practical_value=practical,
            completeness=completeness,
            clarity=clarity,
            uniqueness=uniqueness,
            grade=grade,
            summary=summary,
        )

    def _get_text(self, conv: dict) -> str:
        """Extract text from conversation."""
        parts = []
        if 'raw_text' in conv:
            parts.append(conv['raw_text'])
        if 'generated_title' in conv:
            parts.append(conv['generated_title'])
        return '\n'.join(parts)

    def _score_technical_depth(
        self,
        text: str,
        conv: dict
    ) -> DimensionScore:
        """Score technical depth and complexity."""
        score = 30  # Base score
        factors = []

        # Check for advanced patterns
        advanced_count = sum(1 for p in self.ADVANCED_TECH_PATTERNS if p.search(text))
        if advanced_count >= 3:
            score += 30
            factors.append("Multiple advanced concepts")
        elif advanced_count >= 1:
            score += 15
            factors.append("Advanced concepts present")

        # Code complexity
        code_blocks = re.findall(r'```\w*\n([\s\S]*?)```', text)
        if code_blocks:
            total_lines = sum(len(block.split('\n')) for block in code_blocks)
            if total_lines > 50:
                score += 20
                factors.append("Substantial code examples")
            elif total_lines > 20:
                score += 10
                factors.append("Code examples present")

        # Technical terms
        tech_terms = conv.get('technical_terms', [])
        if len(tech_terms) > 10:
            score += 15
            factors.append("Rich technical vocabulary")
        elif len(tech_terms) > 5:
            score += 8
            factors.append("Good technical vocabulary")

        # Has error traces (debugging complexity)
        if conv.get('has_error_traces'):
            score += 5
            factors.append("Debugging content")

        # Cap at 100
        score = min(100, score)

        # Confidence based on length
        confidence = min(1.0, len(text) / 2000)

        return DimensionScore(
            value=score,
            confidence=confidence,
            factors=factors if factors else ["Basic technical content"]
        )

    def _score_practical_value(
        self,
        text: str,
        conv: dict
    ) -> DimensionScore:
        """Score practical/reusable value."""
        score = 25  # Base score
        factors = []

        # Check for practical patterns
        practical_count = sum(1 for p in self.PRACTICAL_PATTERNS if p.search(text))

        if practical_count >= 5:
            score += 35
            factors.append("Highly actionable content")
        elif practical_count >= 3:
            score += 20
            factors.append("Good practical examples")
        elif practical_count >= 1:
            score += 10
            factors.append("Some practical content")

        # Has code
        if conv.get('has_code'):
            score += 15
            factors.append("Contains reusable code")

        # Has working examples
        if re.search(r'\boutput\b.*:|result:|\bshould\s+(?:see|get)\b', text, re.I):
            score += 10
            factors.append("Shows expected output")

        # Solves a problem
        if re.search(r'\bsolution\b|\bsolve[sd]?\b|\bfix\b', text, re.I):
            score += 10
            factors.append("Problem-solution format")

        # Has links/references
        if conv.get('has_links'):
            score += 5
            factors.append("External references")

        score = min(100, score)
        confidence = 0.7 + (0.3 * min(1.0, practical_count / 5))

        return DimensionScore(
            value=score,
            confidence=confidence,
            factors=factors if factors else ["Limited practical content"]
        )

    def _score_completeness(
        self,
        text: str,
        conv: dict
    ) -> DimensionScore:
        """Score how complete the conversation/solution is."""
        score = 40  # Base score
        factors = []

        # Length indicates completeness
        word_count = len(text.split())
        if word_count > 2000:
            score += 25
            factors.append("Thorough coverage")
        elif word_count > 1000:
            score += 15
            factors.append("Good coverage")
        elif word_count > 500:
            score += 5
            factors.append("Moderate coverage")

        # Multiple turns indicate back-and-forth
        turn_count = conv.get('turn_count', 0)
        if turn_count >= 5:
            score += 10
            factors.append("Multiple exchanges")

        # Has conclusion/resolution
        if re.search(r'\bworks?\b|\bsuccess\b|\bcomplete\b|\bdone\b', text[-500:], re.I):
            score += 15
            factors.append("Reaches resolution")

        # Has summary
        if re.search(r'\bin summary\b|\bto summarize\b|\bkey takeaway', text, re.I):
            score += 10
            factors.append("Includes summary")

        score = min(100, score)
        confidence = 0.6 + (0.4 * min(1.0, word_count / 2000))

        return DimensionScore(
            value=score,
            confidence=confidence,
            factors=factors if factors else ["Basic coverage"]
        )

    def _score_clarity(
        self,
        text: str,
        conv: dict
    ) -> DimensionScore:
        """Score how clear and well-explained the content is."""
        score = 35  # Base score
        factors = []

        # Check clarity patterns
        clarity_count = sum(1 for p in self.CLARITY_PATTERNS if p.search(text))

        if clarity_count >= 5:
            score += 30
            factors.append("Excellent explanations")
        elif clarity_count >= 3:
            score += 20
            factors.append("Good explanations")
        elif clarity_count >= 1:
            score += 10
            factors.append("Some explanations")

        # Has examples
        example_count = len(re.findall(r'\bfor example\b|\be\.g\.\b', text, re.I))
        if example_count >= 2:
            score += 15
            factors.append("Multiple examples")
        elif example_count >= 1:
            score += 8
            factors.append("Examples provided")

        # Structured with lists
        has_lists = bool(re.search(r'^\s*[-*•]\s+|\n\d+\.\s+', text, re.MULTILINE))
        if has_lists:
            score += 10
            factors.append("Well-structured format")

        # Average sentence length (shorter = clearer)
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_len < 20:
                score += 5
                factors.append("Clear sentence structure")

        score = min(100, score)

        return DimensionScore(
            value=score,
            confidence=0.65 + (clarity_count * 0.05),
            factors=factors if factors else ["Average clarity"]
        )

    def _score_uniqueness(
        self,
        text: str,
        conv: dict
    ) -> DimensionScore:
        """Score how unique/novel the content is."""
        score = 50  # Base score (assume average uniqueness)
        factors = []

        # Check for unique term combinations
        terms = conv.get('technical_terms', [])

        if self._tech_term_counts and terms:
            # Calculate rarity of terms
            rare_terms = [
                t for t in terms
                if self._tech_term_counts.get(t, 0) <= 2
            ]

            if len(rare_terms) >= 3:
                score += 25
                factors.append("Unique topic combination")
            elif len(rare_terms) >= 1:
                score += 10
                factors.append("Some unique elements")

        # Novel code patterns
        if re.search(r'\bcustom\b|\bunique\b|\bnovel\b', text, re.I):
            score += 10
            factors.append("Novel approach mentioned")

        # Original solution
        if re.search(r'\bworkaround\b|\balternative\b|\binstead\b', text, re.I):
            score += 10
            factors.append("Alternative solution")

        # Edge case handling
        if re.search(r'\bedge case\b|\bcorner case\b|\bunusual\b', text, re.I):
            score += 10
            factors.append("Handles edge cases")

        score = min(100, max(0, score))

        # Lower confidence for uniqueness (hard to measure)
        confidence = 0.4 + (0.1 * len(factors))

        return DimensionScore(
            value=score,
            confidence=min(0.7, confidence),
            factors=factors if factors else ["Standard content"]
        )

    def _generate_summary(
        self,
        overall: int,
        technical: DimensionScore,
        practical: DimensionScore,
        completeness: DimensionScore,
        clarity: DimensionScore,
        uniqueness: DimensionScore
    ) -> str:
        """Generate a human-readable summary."""
        strengths = []
        areas_to_improve = []

        dimensions = [
            ('Technical Depth', technical.value),
            ('Practical Value', practical.value),
            ('Completeness', completeness.value),
            ('Clarity', clarity.value),
            ('Uniqueness', uniqueness.value),
        ]

        for name, value in dimensions:
            if value >= 75:
                strengths.append(name.lower())
            elif value <= 40:
                areas_to_improve.append(name.lower())

        summary_parts = []

        if overall >= 80:
            summary_parts.append("Excellent conversation")
        elif overall >= 60:
            summary_parts.append("Good conversation")
        elif overall >= 40:
            summary_parts.append("Average conversation")
        else:
            summary_parts.append("Basic conversation")

        if strengths:
            summary_parts.append(f"Strong in {', '.join(strengths[:2])}")

        if areas_to_improve:
            summary_parts.append(f"Could improve {', '.join(areas_to_improve[:2])}")

        return ". ".join(summary_parts) + "."


def score_conversations(
    conversations: List[dict],
    calculate_percentiles: bool = True
) -> List[Tuple[str, ConversationScore]]:
    """
    Score multiple conversations.

    Args:
        conversations: List of conversation dicts
        calculate_percentiles: Whether to calculate percentile rankings

    Returns:
        List of (convo_id, score) tuples
    """
    scorer = ConversationScorer(conversations)

    results = []
    for conv in conversations:
        score = scorer.score(conv)
        results.append((conv.get('convo_id', ''), score))

    # Calculate percentiles
    if calculate_percentiles and len(results) > 1:
        sorted_scores = sorted([s.overall for _, s in results])
        for convo_id, score in results:
            rank = sorted_scores.index(score.overall) + 1
            score.percentile = (rank / len(results)) * 100

    return results


def convert_legacy_scores(conversations: List[dict]) -> List[dict]:
    """
    Convert legacy 1-9 scores to new 1-100 format.

    Args:
        conversations: Conversations with legacy scores

    Returns:
        Conversations with new score format
    """
    for conv in conversations:
        legacy = conv.get('score', conv.get('brilliance_score', {}).get('score'))

        if legacy and isinstance(legacy, (int, float)) and legacy <= 10:
            new_score = ConversationScore.from_legacy_score(int(legacy))
            conv['score_v2'] = new_score.to_dict()
            # Keep legacy for backward compatibility
            conv['score_legacy'] = legacy

    return conversations


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Score conversations with new 1-100 system")
    parser.add_argument('input', help="Input JSONL file")
    parser.add_argument('--output', '-o', help="Output JSONL file")
    parser.add_argument('--top', '-t', type=int, default=10, help="Show top N conversations")

    args = parser.parse_args()

    # Load conversations
    conversations = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    print(f"Scoring {len(conversations)} conversations...")

    # Score
    results = score_conversations(conversations)

    # Stats
    scores = [s.overall for _, s in results]
    avg = sum(scores) / len(scores)

    print(f"\nScore Distribution:")
    print(f"  Average: {avg:.1f}")
    print(f"  Min: {min(scores)}")
    print(f"  Max: {max(scores)}")

    # Show top
    sorted_results = sorted(results, key=lambda x: x[1].overall, reverse=True)
    print(f"\nTop {args.top} Conversations:")
    for convo_id, score in sorted_results[:args.top]:
        print(f"  {score.grade} ({score.overall}): {convo_id}")
        print(f"    {score.summary}")

    # Save
    if args.output:
        scored = []
        for conv in conversations:
            cid = conv.get('convo_id', '')
            for rid, score in results:
                if rid == cid:
                    conv['score_v2'] = score.to_dict()
                    break
            scored.append(conv)

        with open(args.output, 'w') as f:
            for conv in scored:
                f.write(json.dumps(conv) + '\n')
        print(f"\nSaved to {args.output}")
