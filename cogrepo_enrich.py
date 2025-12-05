#!/usr/bin/env python3
"""
CogRepo v2 Enrichment Pipeline

Master orchestration script for running the full enrichment pipeline.
Processes conversations through all enrichment tiers.

Enrichment Tiers:
1. Zero-token: Local pattern extraction (code, links, terms)
2. Embeddings: Generate semantic embeddings
3. Artifacts: Extract code, commands, solutions (requires API key)
4. Context: Infer projects and conversation chains
5. Database: Import to SQLite with FTS5

Usage:
    # Full pipeline
    python cogrepo_enrich.py data/conversations.jsonl

    # Specific phases
    python cogrepo_enrich.py data/conversations.jsonl --phase zero-token
    python cogrepo_enrich.py data/conversations.jsonl --phase embeddings
    python cogrepo_enrich.py data/conversations.jsonl --phase database

    # Skip expensive operations
    python cogrepo_enrich.py data/conversations.jsonl --no-artifacts

    # Dry run
    python cogrepo_enrich.py data/conversations.jsonl --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


def load_conversations(input_path: str) -> List[dict]:
    """Load conversations from JSONL file."""
    conversations = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    return conversations


def save_conversations(conversations: List[dict], output_path: str):
    """Save conversations to JSONL file."""
    with open(output_path, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')


def run_zero_token(conversations: List[dict], verbose: bool = False) -> List[dict]:
    """Run zero-token enrichment on all conversations."""
    from enrichment.zero_token import ZeroTokenEnricher

    print("\n[Phase 1/5] Zero-Token Enrichment")
    print("-" * 40)

    enricher = ZeroTokenEnricher()
    enriched = []
    stats = {'has_code': 0, 'has_links': 0, 'has_errors': 0}

    for i, conv in enumerate(conversations):
        text = conv.get('raw_text', '')

        # Extract metrics
        metrics = enricher.extract(text)
        conv.update(metrics.to_dict())

        # Update stats
        if metrics.has_code:
            stats['has_code'] += 1
        if metrics.has_links:
            stats['has_links'] += 1
        if metrics.has_error_traces:
            stats['has_errors'] += 1

        enriched.append(conv)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(conversations)}")

    print(f"\n  Results:")
    print(f"    Conversations with code: {stats['has_code']}")
    print(f"    Conversations with links: {stats['has_links']}")
    print(f"    Conversations with errors: {stats['has_errors']}")

    return enriched


def run_embeddings(conversations: List[dict], output_dir: str, verbose: bool = False) -> bool:
    """Generate embeddings for all conversations."""
    print("\n[Phase 2/5] Embedding Generation")
    print("-" * 40)

    try:
        from search.embeddings import EmbeddingEngine, EmbeddingStore
        engine = EmbeddingEngine()

        # Prepare text for embedding (title + summary)
        texts = []
        ids = []

        for conv in conversations:
            title = conv.get('generated_title', '')
            summary = conv.get('summary_abstractive', '')
            text = f"{title}. {summary}" if summary else title
            texts.append(text or conv.get('raw_text', '')[:500])
            ids.append(conv.get('convo_id', ''))

        if not texts:
            print("  No texts to embed")
            return False

        # Generate embeddings
        print(f"  Generating embeddings for {len(texts)} conversations...")
        embeddings = engine.embed_batch(texts, show_progress=verbose)

        # Save
        store = EmbeddingStore(output_dir)
        store.save(embeddings, ids)

        print(f"  Saved {len(embeddings)} embeddings ({embeddings.shape[1]}D)")
        return True

    except ImportError as e:
        print(f"  Skipping: sentence-transformers not installed")
        print(f"  Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def run_artifacts(
    conversations: List[dict],
    use_api: bool = True,
    verbose: bool = False
) -> List[dict]:
    """Extract artifacts from conversations."""
    from enrichment.artifact_extractor import ArtifactExtractor, LocalArtifactExtractor

    print("\n[Phase 3/5] Artifact Extraction")
    print("-" * 40)

    if use_api:
        try:
            from core.config import get_config
            config = get_config()
            if not config.has_api_key:
                print("  No API key, using local extraction")
                use_api = False
        except Exception:
            print("  Config error, using local extraction")
            use_api = False

    extractor = ArtifactExtractor() if use_api else LocalArtifactExtractor()
    print(f"  Using: {'API (Claude Haiku)' if use_api else 'Local regex'}")

    total_artifacts = 0

    for i, conv in enumerate(conversations):
        # Skip if already has artifacts
        if conv.get('artifacts') and len(conv['artifacts']) > 0:
            continue

        text = conv.get('raw_text', '')
        convo_id = conv.get('convo_id', '')

        try:
            artifacts = extractor.extract(convo_id, text)
            conv['artifacts'] = [a.to_dict() for a in artifacts]
            total_artifacts += len(artifacts)
        except Exception as e:
            if verbose:
                print(f"  Error extracting artifacts for {convo_id}: {e}")
            conv['artifacts'] = []

        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(conversations)}")

    print(f"  Extracted {total_artifacts} artifacts total")
    return conversations


def run_context(conversations: List[dict], verbose: bool = False) -> dict:
    """Infer project groupings and conversation chains."""
    from context.project_inference import ProjectInferrer
    from context.chain_detection import ChainDetector

    print("\n[Phase 4/5] Context Analysis")
    print("-" * 40)

    # Infer projects
    print("  Detecting projects...")
    inferrer = ProjectInferrer()
    projects = inferrer.infer_projects(conversations)

    print(f"    Found {len(projects)} projects:")
    for project in projects[:5]:
        print(f"      - {project.name} ({len(project.conversation_ids)} conversations)")
    if len(projects) > 5:
        print(f"      ... and {len(projects) - 5} more")

    # Detect chains
    print("\n  Detecting conversation chains...")
    detector = ChainDetector()
    chains = detector.detect_chains(conversations)

    print(f"    Found {len(chains)} chains:")
    for chain in chains[:5]:
        print(f"      - {chain.chain_type}: {len(chain.conversation_ids)} conversations")
    if len(chains) > 5:
        print(f"      ... and {len(chains) - 5} more")

    return {
        'projects': [p.to_dict() for p in projects],
        'chains': [c.to_dict() for c in chains],
    }


def run_database(
    conversations: List[dict],
    db_path: str,
    verbose: bool = False
) -> bool:
    """Import conversations to SQLite database."""
    from database.repository import ConversationRepository

    print("\n[Phase 5/5] Database Import")
    print("-" * 40)

    repo = ConversationRepository(db_path)

    print(f"  Importing {len(conversations)} conversations to {db_path}...")

    # Use batch import if available
    try:
        imported = repo.import_from_list(conversations)
        print(f"  Imported {imported} conversations")
    except AttributeError:
        # Fallback to individual saves
        for i, conv in enumerate(conversations):
            repo.save(conv)
            if verbose and (i + 1) % 50 == 0:
                print(f"    Saved {i + 1}/{len(conversations)}")
        print(f"  Imported {len(conversations)} conversations")

    # Get stats
    stats = repo.get_stats()
    print(f"\n  Database stats:")
    print(f"    Total conversations: {stats.get('total', 0)}")
    print(f"    With code: {stats.get('with_code', 0)}")
    print(f"    Total artifacts: {stats.get('total_artifacts', 0)}")

    return True


def run_full_pipeline(
    input_path: str,
    output_dir: str = None,
    phases: List[str] = None,
    no_artifacts: bool = False,
    dry_run: bool = False,
    verbose: bool = False
) -> dict:
    """
    Run the full enrichment pipeline.

    Args:
        input_path: Path to input JSONL file
        output_dir: Output directory (defaults to same as input)
        phases: Specific phases to run (default: all)
        no_artifacts: Skip artifact extraction
        dry_run: Show what would be done without doing it
        verbose: Show detailed progress

    Returns:
        Statistics dict
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir or input_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default phases
    all_phases = ['zero-token', 'embeddings', 'artifacts', 'context', 'database']
    phases = phases or all_phases

    if no_artifacts and 'artifacts' in phases:
        phases.remove('artifacts')

    print("=" * 50)
    print("CogRepo v2 Enrichment Pipeline")
    print("=" * 50)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Phases: {', '.join(phases)}")

    if dry_run:
        print("\n[DRY RUN - No changes will be made]")
        return {}

    # Load conversations
    print(f"\nLoading conversations from {input_path}...")
    conversations = load_conversations(input_path)
    print(f"Loaded {len(conversations)} conversations")

    stats = {
        'input_count': len(conversations),
        'phases_run': [],
        'start_time': datetime.now().isoformat(),
    }

    # Run phases
    if 'zero-token' in phases:
        conversations = run_zero_token(conversations, verbose)
        stats['phases_run'].append('zero-token')

    if 'embeddings' in phases:
        success = run_embeddings(conversations, str(output_dir), verbose)
        if success:
            stats['phases_run'].append('embeddings')

    if 'artifacts' in phases:
        conversations = run_artifacts(conversations, use_api=True, verbose=verbose)
        stats['phases_run'].append('artifacts')

    context_data = None
    if 'context' in phases:
        context_data = run_context(conversations, verbose)
        stats['phases_run'].append('context')
        stats['projects_found'] = len(context_data.get('projects', []))
        stats['chains_found'] = len(context_data.get('chains', []))

    if 'database' in phases:
        db_path = output_dir / 'cogrepo.db'
        success = run_database(conversations, str(db_path), verbose)
        if success:
            stats['phases_run'].append('database')

    # Save enriched JSONL
    output_jsonl = output_dir / 'enriched_repository.jsonl'
    print(f"\nSaving enriched data to {output_jsonl}...")
    save_conversations(conversations, output_jsonl)

    # Save context data if generated
    if context_data:
        context_file = output_dir / 'context_analysis.json'
        with open(context_file, 'w') as f:
            json.dump(context_data, f, indent=2)
        print(f"Saved context analysis to {context_file}")

    stats['end_time'] = datetime.now().isoformat()
    stats['output_count'] = len(conversations)

    print("\n" + "=" * 50)
    print("Pipeline Complete!")
    print("=" * 50)
    print(f"Phases completed: {', '.join(stats['phases_run'])}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="CogRepo v2 Enrichment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('input', help="Input JSONL file")
    parser.add_argument('--output', '-o', help="Output directory")
    parser.add_argument('--phase', '-p', action='append',
                        choices=['zero-token', 'embeddings', 'artifacts', 'context', 'database'],
                        help="Run specific phase (can be repeated)")
    parser.add_argument('--no-artifacts', action='store_true',
                        help="Skip artifact extraction (saves API cost)")
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help="Show what would be done without doing it")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Show detailed progress")
    parser.add_argument('--generate-sample', action='store_true',
                        help="Generate sample data for testing")

    args = parser.parse_args()

    # Handle sample generation
    if args.generate_sample:
        from tests.fixtures.sample_data import generate_sample_dataset

        print("Generating sample dataset...")
        output_path = args.input if args.input != 'sample' else 'data/sample_conversations.jsonl'

        # Create directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        conversations = generate_sample_dataset(count=50)

        # Save to file
        with open(output_path, 'w') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')

        print(f"Generated {len(conversations)} sample conversations to {output_path}")
        print(f"\nRun enrichment with: python cogrepo_enrich.py {output_path}")
        return

    # Check input exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("\nTo generate sample data: python cogrepo_enrich.py sample --generate-sample")
        sys.exit(1)

    # Run pipeline
    stats = run_full_pipeline(
        input_path=args.input,
        output_dir=args.output,
        phases=args.phase,
        no_artifacts=args.no_artifacts,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    # Print summary
    if stats and not args.dry_run:
        print(f"\nEnriched {stats.get('output_count', 0)} conversations")


if __name__ == '__main__':
    main()
