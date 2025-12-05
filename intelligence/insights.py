"""
CogRepo Insights Engine

Generates insights and analytics from the conversation repository:
- Trend analysis over time
- Technology adoption patterns
- Topic evolution
- Quality metrics
- Activity patterns

Usage:
    from intelligence.insights import InsightsEngine

    engine = InsightsEngine(conversations)

    # Get overall insights
    insights = engine.generate_insights()

    # Get trends for a specific topic
    trends = engine.get_topic_trends('python')
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import math


@dataclass
class Insight:
    """A single insight or finding."""
    id: str
    category: str  # 'trend', 'pattern', 'anomaly', 'summary'
    title: str
    description: str
    score: float  # Importance/confidence score
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'category': self.category,
            'title': self.title,
            'description': self.description,
            'score': self.score,
            'data': self.data,
        }


@dataclass
class TrendData:
    """Time-series trend data."""
    name: str
    data_points: List[Tuple[str, int]]  # (date, count)
    total: int
    growth_rate: float  # Percentage change
    trend_direction: str  # 'rising', 'falling', 'stable'

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'data_points': [{'date': d, 'count': c} for d, c in self.data_points],
            'total': self.total,
            'growth_rate': self.growth_rate,
            'trend_direction': self.trend_direction,
        }


class InsightsEngine:
    """
    Generates insights from conversation data.

    Analyzes:
    - Technology trends
    - Topic evolution
    - Quality patterns
    - Activity patterns
    - Anomalies
    """

    def __init__(self, conversations: List[dict]):
        """
        Initialize insights engine.

        Args:
            conversations: List of conversation dicts
        """
        self.conversations = conversations
        self._parsed_convos = self._parse_conversations()

    def _parse_conversations(self) -> List[Dict]:
        """Parse conversations with normalized timestamps."""
        parsed = []

        for conv in self.conversations:
            timestamp_str = conv.get('created_at', conv.get('timestamp', ''))
            timestamp = self._parse_timestamp(timestamp_str)

            parsed.append({
                'convo_id': conv.get('convo_id', ''),
                'timestamp': timestamp,
                'date': timestamp.date() if timestamp else None,
                'week': timestamp.isocalendar()[:2] if timestamp else None,
                'month': (timestamp.year, timestamp.month) if timestamp else None,
                'tags': conv.get('tags', []),
                'code_languages': conv.get('code_languages', []),
                'technical_terms': conv.get('technical_terms', []),
                'primary_domain': conv.get('primary_domain', ''),
                'score': conv.get('score', 50),
                'has_code': conv.get('has_code', False),
                'has_error_traces': conv.get('has_error_traces', False),
                'source': conv.get('source', 'unknown'),
            })

        return [p for p in parsed if p['timestamp']]

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string."""
        if not timestamp_str:
            return None

        for fmt in [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
        ]:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        return None

    def generate_insights(self) -> List[Insight]:
        """
        Generate comprehensive insights.

        Returns:
            List of Insight objects sorted by score
        """
        insights = []

        # Summary insights
        insights.extend(self._summary_insights())

        # Trend insights
        insights.extend(self._trend_insights())

        # Pattern insights
        insights.extend(self._pattern_insights())

        # Anomaly insights
        insights.extend(self._anomaly_insights())

        # Sort by score
        insights.sort(key=lambda i: i.score, reverse=True)
        return insights

    def _summary_insights(self) -> List[Insight]:
        """Generate summary statistics insights."""
        insights = []

        if not self._parsed_convos:
            return insights

        # Total conversations
        total = len(self._parsed_convos)
        with_code = sum(1 for c in self._parsed_convos if c['has_code'])
        with_errors = sum(1 for c in self._parsed_convos if c['has_error_traces'])
        avg_score = sum(c['score'] for c in self._parsed_convos) / total

        insights.append(Insight(
            id='summary_overview',
            category='summary',
            title='Repository Overview',
            description=f'{total} conversations, {with_code} with code ({100*with_code/total:.0f}%)',
            score=1.0,
            data={
                'total': total,
                'with_code': with_code,
                'with_errors': with_errors,
                'average_score': round(avg_score, 1),
            }
        ))

        # Top technologies
        tech_counts = Counter()
        for conv in self._parsed_convos:
            tech_counts.update(conv['code_languages'])

        if tech_counts:
            top_tech = tech_counts.most_common(5)
            insights.append(Insight(
                id='top_technologies',
                category='summary',
                title='Top Technologies',
                description=f'Most used: {", ".join(t[0] for t in top_tech[:3])}',
                score=0.9,
                data={'technologies': [{'name': t, 'count': c} for t, c in top_tech]}
            ))

        # Top domains
        domain_counts = Counter(c['primary_domain'] for c in self._parsed_convos if c['primary_domain'])
        if domain_counts:
            top_domains = domain_counts.most_common(5)
            insights.append(Insight(
                id='top_domains',
                category='summary',
                title='Top Domains',
                description=f'Focus areas: {", ".join(d[0] for d in top_domains[:3])}',
                score=0.85,
                data={'domains': [{'name': d, 'count': c} for d, c in top_domains]}
            ))

        return insights

    def _trend_insights(self) -> List[Insight]:
        """Generate trend-based insights."""
        insights = []

        # Get monthly trends for technologies
        tech_trends = self.get_technology_trends(period='month')

        for trend in tech_trends[:5]:  # Top 5 trending
            if trend.growth_rate > 50:
                insights.append(Insight(
                    id=f'trend_{trend.name}',
                    category='trend',
                    title=f'{trend.name.title()} is Rising',
                    description=f'{trend.growth_rate:.0f}% increase in recent period',
                    score=min(0.9, 0.5 + trend.growth_rate / 200),
                    data=trend.to_dict()
                ))
            elif trend.growth_rate < -30:
                insights.append(Insight(
                    id=f'decline_{trend.name}',
                    category='trend',
                    title=f'{trend.name.title()} Usage Declining',
                    description=f'{abs(trend.growth_rate):.0f}% decrease in recent period',
                    score=0.6,
                    data=trend.to_dict()
                ))

        # Activity trends
        activity_trend = self._get_activity_trend()
        if activity_trend:
            insights.append(activity_trend)

        return insights

    def _pattern_insights(self) -> List[Insight]:
        """Generate pattern-based insights."""
        insights = []

        # Debug session patterns
        error_rate = sum(1 for c in self._parsed_convos if c['has_error_traces']) / max(len(self._parsed_convos), 1)
        if error_rate > 0.3:
            insights.append(Insight(
                id='high_debug_rate',
                category='pattern',
                title='High Debug Activity',
                description=f'{error_rate*100:.0f}% of conversations involve debugging',
                score=0.7,
                data={'error_rate': error_rate}
            ))

        # Code intensity
        code_rate = sum(1 for c in self._parsed_convos if c['has_code']) / max(len(self._parsed_convos), 1)
        if code_rate > 0.7:
            insights.append(Insight(
                id='code_heavy',
                category='pattern',
                title='Code-Heavy Repository',
                description=f'{code_rate*100:.0f}% of conversations contain code',
                score=0.65,
                data={'code_rate': code_rate}
            ))

        # Source distribution
        source_counts = Counter(c['source'] for c in self._parsed_convos)
        if len(source_counts) > 1:
            dominant = source_counts.most_common(1)[0]
            dominance = dominant[1] / len(self._parsed_convos)
            if dominance > 0.8:
                insights.append(Insight(
                    id='source_dominance',
                    category='pattern',
                    title=f'{dominant[0].title()} Dominant',
                    description=f'{dominance*100:.0f}% of conversations from {dominant[0]}',
                    score=0.5,
                    data={'source_distribution': dict(source_counts)}
                ))

        return insights

    def _anomaly_insights(self) -> List[Insight]:
        """Detect anomalies in the data."""
        insights = []

        if len(self._parsed_convos) < 10:
            return insights

        # Quality anomalies
        scores = [c['score'] for c in self._parsed_convos]
        mean_score = sum(scores) / len(scores)
        std_score = math.sqrt(sum((s - mean_score) ** 2 for s in scores) / len(scores))

        low_quality = sum(1 for s in scores if s < mean_score - 2 * std_score)
        if low_quality > 0:
            insights.append(Insight(
                id='quality_outliers',
                category='anomaly',
                title='Quality Outliers Detected',
                description=f'{low_quality} conversations with unusually low quality scores',
                score=0.6,
                data={'count': low_quality, 'threshold': mean_score - 2 * std_score}
            ))

        # Activity spikes
        daily_counts = defaultdict(int)
        for conv in self._parsed_convos:
            if conv['date']:
                daily_counts[conv['date']] += 1

        if daily_counts:
            counts = list(daily_counts.values())
            mean_daily = sum(counts) / len(counts)
            max_daily = max(counts)

            if max_daily > mean_daily * 3:
                peak_date = max(daily_counts, key=daily_counts.get)
                insights.append(Insight(
                    id='activity_spike',
                    category='anomaly',
                    title='Activity Spike Detected',
                    description=f'Unusual activity on {peak_date}: {max_daily} conversations',
                    score=0.55,
                    data={'date': str(peak_date), 'count': max_daily, 'average': mean_daily}
                ))

        return insights

    def _get_activity_trend(self) -> Optional[Insight]:
        """Get overall activity trend."""
        monthly_counts = defaultdict(int)

        for conv in self._parsed_convos:
            if conv['month']:
                monthly_counts[conv['month']] += 1

        if len(monthly_counts) < 2:
            return None

        sorted_months = sorted(monthly_counts.keys())

        # Compare recent to earlier
        mid = len(sorted_months) // 2
        early_avg = sum(monthly_counts[m] for m in sorted_months[:mid]) / max(mid, 1)
        late_avg = sum(monthly_counts[m] for m in sorted_months[mid:]) / max(len(sorted_months) - mid, 1)

        if early_avg > 0:
            growth = (late_avg - early_avg) / early_avg * 100
        else:
            growth = 100 if late_avg > 0 else 0

        if abs(growth) > 20:
            direction = 'increasing' if growth > 0 else 'decreasing'
            return Insight(
                id='activity_trend',
                category='trend',
                title=f'Activity is {direction.title()}',
                description=f'{abs(growth):.0f}% change in activity level',
                score=0.75,
                data={
                    'growth_rate': growth,
                    'early_average': early_avg,
                    'recent_average': late_avg
                }
            )

        return None

    def get_technology_trends(
        self,
        period: str = 'month',
        limit: int = 20
    ) -> List[TrendData]:
        """
        Get technology adoption trends.

        Args:
            period: 'week' or 'month'
            limit: Maximum technologies to return

        Returns:
            List of TrendData objects
        """
        # Group by period
        period_key = 'week' if period == 'week' else 'month'
        tech_by_period: Dict[Any, Counter] = defaultdict(Counter)

        for conv in self._parsed_convos:
            period_val = conv[period_key]
            if period_val:
                tech_by_period[period_val].update(conv['code_languages'])

        if not tech_by_period:
            return []

        # Get all technologies
        all_techs = Counter()
        for period_counter in tech_by_period.values():
            all_techs.update(period_counter)

        trends = []
        sorted_periods = sorted(tech_by_period.keys())

        for tech, total in all_techs.most_common(limit):
            # Build time series
            data_points = []
            for period_val in sorted_periods:
                period_str = f"{period_val[0]}-{period_val[1]:02d}" if period == 'month' else f"{period_val[0]}-W{period_val[1]:02d}"
                count = tech_by_period[period_val].get(tech, 0)
                data_points.append((period_str, count))

            # Calculate growth
            if len(data_points) >= 2:
                mid = len(data_points) // 2
                early = sum(c for _, c in data_points[:mid]) / max(mid, 1)
                late = sum(c for _, c in data_points[mid:]) / max(len(data_points) - mid, 1)

                if early > 0:
                    growth = (late - early) / early * 100
                else:
                    growth = 100 if late > 0 else 0

                if growth > 10:
                    direction = 'rising'
                elif growth < -10:
                    direction = 'falling'
                else:
                    direction = 'stable'
            else:
                growth = 0
                direction = 'stable'

            trends.append(TrendData(
                name=tech,
                data_points=data_points,
                total=total,
                growth_rate=growth,
                trend_direction=direction
            ))

        # Sort by growth rate
        trends.sort(key=lambda t: t.growth_rate, reverse=True)
        return trends

    def get_topic_trends(
        self,
        topic: str,
        period: str = 'month'
    ) -> TrendData:
        """
        Get trend data for a specific topic.

        Args:
            topic: Topic/technology name
            period: 'week' or 'month'

        Returns:
            TrendData for the topic
        """
        topic_lower = topic.lower()
        period_key = 'week' if period == 'week' else 'month'

        period_counts = defaultdict(int)
        total = 0

        for conv in self._parsed_convos:
            # Check if topic appears
            has_topic = (
                topic_lower in [t.lower() for t in conv['code_languages']] or
                topic_lower in [t.lower() for t in conv['tags']] or
                topic_lower in [t.lower() for t in conv['technical_terms']]
            )

            if has_topic:
                period_val = conv[period_key]
                if period_val:
                    period_counts[period_val] += 1
                    total += 1

        # Build time series
        sorted_periods = sorted(period_counts.keys())
        data_points = []

        for period_val in sorted_periods:
            period_str = f"{period_val[0]}-{period_val[1]:02d}" if period == 'month' else f"{period_val[0]}-W{period_val[1]:02d}"
            data_points.append((period_str, period_counts[period_val]))

        # Calculate growth
        if len(data_points) >= 2:
            mid = len(data_points) // 2
            early = sum(c for _, c in data_points[:mid]) / max(mid, 1)
            late = sum(c for _, c in data_points[mid:]) / max(len(data_points) - mid, 1)
            growth = ((late - early) / early * 100) if early > 0 else (100 if late > 0 else 0)
            direction = 'rising' if growth > 10 else ('falling' if growth < -10 else 'stable')
        else:
            growth = 0
            direction = 'stable'

        return TrendData(
            name=topic,
            data_points=data_points,
            total=total,
            growth_rate=growth,
            trend_direction=direction
        )

    def get_quality_over_time(self, period: str = 'month') -> List[Tuple[str, float]]:
        """Get average quality score over time."""
        period_key = 'week' if period == 'week' else 'month'
        period_scores: Dict[Any, List[int]] = defaultdict(list)

        for conv in self._parsed_convos:
            period_val = conv[period_key]
            if period_val:
                period_scores[period_val].append(conv['score'])

        results = []
        for period_val in sorted(period_scores.keys()):
            scores = period_scores[period_val]
            avg = sum(scores) / len(scores)
            period_str = f"{period_val[0]}-{period_val[1]:02d}" if period == 'month' else f"{period_val[0]}-W{period_val[1]:02d}"
            results.append((period_str, avg))

        return results

    def export_dashboard_data(self) -> dict:
        """Export all data needed for a dashboard."""
        return {
            'insights': [i.to_dict() for i in self.generate_insights()],
            'technology_trends': [t.to_dict() for t in self.get_technology_trends()],
            'quality_over_time': [
                {'period': p, 'score': s}
                for p, s in self.get_quality_over_time()
            ],
            'summary': {
                'total_conversations': len(self._parsed_convos),
                'with_code': sum(1 for c in self._parsed_convos if c['has_code']),
                'with_errors': sum(1 for c in self._parsed_convos if c['has_error_traces']),
                'average_score': sum(c['score'] for c in self._parsed_convos) / max(len(self._parsed_convos), 1),
            }
        }


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate insights from conversations")
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

    # Generate insights
    engine = InsightsEngine(conversations)
    insights = engine.generate_insights()

    print(f"\nGenerated {len(insights)} insights:")
    for insight in insights[:10]:
        print(f"\n  [{insight.category.upper()}] {insight.title}")
        print(f"    {insight.description}")
        print(f"    Score: {insight.score:.2f}")

    if args.output:
        dashboard = engine.export_dashboard_data()
        with open(args.output, 'w') as f:
            json.dump(dashboard, f, indent=2)
        print(f"\nSaved dashboard data to {args.output}")
