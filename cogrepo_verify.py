#!/usr/bin/env python3
"""
CogRepo v2 Verification Script

Validates the CogRepo installation and data integrity.
Checks all components and reports their status.

Usage:
    # Full verification
    python cogrepo_verify.py

    # Quick status check
    python cogrepo_verify.py --status

    # Check specific component
    python cogrepo_verify.py --check config
    python cogrepo_verify.py --check database
    python cogrepo_verify.py --check embeddings
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class Verifier:
    """Verify CogRepo installation and data."""

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.results: Dict[str, Tuple[bool, str]] = {}

    def check_all(self) -> Dict[str, Tuple[bool, str]]:
        """Run all verification checks."""
        self.check_config()
        self.check_dependencies()
        self.check_database()
        self.check_embeddings()
        self.check_data_files()
        return self.results

    def check_config(self) -> Tuple[bool, str]:
        """Check configuration system."""
        try:
            from core.config import get_config

            config = get_config()

            if config.has_api_key:
                result = (True, "API key configured")
            else:
                result = (False, "No API key found (optional for zero-token enrichment)")

            self.results['config'] = result
            return result

        except ImportError as e:
            result = (False, f"Import error: {e}")
            self.results['config'] = result
            return result
        except Exception as e:
            result = (False, f"Error: {e}")
            self.results['config'] = result
            return result

    def check_dependencies(self) -> Tuple[bool, str]:
        """Check required dependencies."""
        missing = []
        optional_missing = []

        # Core dependencies
        try:
            import pydantic
        except ImportError:
            missing.append('pydantic')

        try:
            import yaml
        except ImportError:
            missing.append('pyyaml')

        # Optional dependencies
        try:
            import sentence_transformers
        except ImportError:
            optional_missing.append('sentence-transformers')

        try:
            import anthropic
        except ImportError:
            optional_missing.append('anthropic')

        try:
            import numpy
        except ImportError:
            missing.append('numpy')

        if missing:
            result = (False, f"Missing required: {', '.join(missing)}")
        elif optional_missing:
            result = (True, f"Core OK, optional missing: {', '.join(optional_missing)}")
        else:
            result = (True, "All dependencies installed")

        self.results['dependencies'] = result
        return result

    def check_database(self) -> Tuple[bool, str]:
        """Check database status."""
        db_path = self.data_dir / 'cogrepo.db'

        if not db_path.exists():
            result = (False, f"Database not found at {db_path}")
            self.results['database'] = result
            return result

        try:
            from database.repository import ConversationRepository

            repo = ConversationRepository(str(db_path))
            stats = repo.get_stats()

            total = stats.get('total', 0)
            with_code = stats.get('with_code', 0)

            result = (True, f"{total} conversations, {with_code} with code")
            self.results['database'] = result
            return result

        except Exception as e:
            result = (False, f"Database error: {e}")
            self.results['database'] = result
            return result

    def check_embeddings(self) -> Tuple[bool, str]:
        """Check embeddings status."""
        embeddings_file = self.data_dir / 'embeddings.npy'
        ids_file = self.data_dir / 'embedding_ids.json'

        if not embeddings_file.exists():
            result = (False, "Embeddings not generated")
            self.results['embeddings'] = result
            return result

        try:
            import numpy as np
            import json

            embeddings = np.load(embeddings_file)

            if ids_file.exists():
                with open(ids_file) as f:
                    ids = json.load(f)
                count = len(ids)
            else:
                count = embeddings.shape[0]

            result = (True, f"{count} embeddings ({embeddings.shape[1]}D)")
            self.results['embeddings'] = result
            return result

        except Exception as e:
            result = (False, f"Embeddings error: {e}")
            self.results['embeddings'] = result
            return result

    def check_data_files(self) -> Tuple[bool, str]:
        """Check data files."""
        jsonl_path = self.data_dir / 'enriched_repository.jsonl'

        if not jsonl_path.exists():
            # Check for sample data
            sample_path = self.data_dir / 'sample_conversations.jsonl'
            if sample_path.exists():
                result = (True, f"Sample data available at {sample_path}")
            else:
                result = (False, "No data files found")
            self.results['data_files'] = result
            return result

        try:
            import json

            count = 0
            with_code = 0
            with_artifacts = 0

            with open(jsonl_path) as f:
                for line in f:
                    if line.strip():
                        conv = json.loads(line)
                        count += 1
                        if conv.get('has_code'):
                            with_code += 1
                        if conv.get('artifacts'):
                            with_artifacts += 1

            result = (True, f"{count} conversations, {with_code} with code, {with_artifacts} with artifacts")
            self.results['data_files'] = result
            return result

        except Exception as e:
            result = (False, f"Data file error: {e}")
            self.results['data_files'] = result
            return result

    def print_status(self):
        """Print status summary."""
        print("\n" + "=" * 50)
        print("CogRepo v2 Status")
        print("=" * 50 + "\n")

        all_ok = True
        for component, (status, message) in self.results.items():
            icon = "[OK]" if status else "[!!]"
            print(f"  {icon} {component.title()}: {message}")
            if not status and component not in ['config']:  # config is optional
                all_ok = False

        print("\n" + "-" * 50)
        if all_ok:
            print("  Status: Ready")
        else:
            print("  Status: Issues found (see above)")
        print()


def get_phase_status() -> Dict[str, str]:
    """Get implementation phase status."""
    phases = {}
    data_dir = Path('data')

    # Phase 0.1: Config
    try:
        from core.config import get_config
        config = get_config()
        phases['0.1 Config'] = 'Complete' if Path.home().joinpath('.cogrepo/config.yaml').exists() or config.has_api_key else 'Not configured'
    except:
        phases['0.1 Config'] = 'Not configured'

    # Phase 0.2: Zero-token
    try:
        from enrichment.zero_token import ZeroTokenEnricher
        phases['0.2 Zero-token'] = 'Complete'
    except:
        phases['0.2 Zero-token'] = 'Not installed'

    # Phase 0.3: Embeddings
    try:
        from search.embeddings import EmbeddingEngine
        if (data_dir / 'embeddings.npy').exists():
            phases['0.3 Embeddings'] = 'Complete'
        else:
            phases['0.3 Embeddings'] = 'Engine ready, no data'
    except:
        phases['0.3 Embeddings'] = 'Not installed'

    # Phase 0.4: Artifacts
    try:
        from enrichment.artifact_extractor import ArtifactExtractor
        phases['0.4 Artifacts'] = 'Complete'
    except:
        phases['0.4 Artifacts'] = 'Not installed'

    # Phase 0.5: Context
    try:
        from context.project_inference import ProjectInferrer
        from context.chain_detection import ChainDetector
        phases['0.5 Context'] = 'Complete'
    except:
        phases['0.5 Context'] = 'Not installed'

    # Phase 1: Database
    if (data_dir / 'cogrepo.db').exists():
        phases['1 Database'] = 'Complete'
    else:
        try:
            from database.repository import ConversationRepository
            phases['1 Database'] = 'Ready, no data'
        except:
            phases['1 Database'] = 'Not installed'

    # Phase 2: Hybrid Search
    try:
        from search.hybrid_search import HybridSearcher
        phases['2 Hybrid Search'] = 'Complete'
    except:
        phases['2 Hybrid Search'] = 'Not installed'

    return phases


def main():
    parser = argparse.ArgumentParser(description="CogRepo v2 Verification")
    parser.add_argument('--status', '-s', action='store_true',
                        help="Quick status check")
    parser.add_argument('--check', '-c',
                        choices=['config', 'dependencies', 'database', 'embeddings', 'data'],
                        help="Check specific component")
    parser.add_argument('--data-dir', '-d', default='data',
                        help="Data directory path")
    parser.add_argument('--phases', '-p', action='store_true',
                        help="Show implementation phase status")

    args = parser.parse_args()

    if args.phases:
        print("\n" + "=" * 50)
        print("CogRepo v2 Phase Status")
        print("=" * 50 + "\n")

        phases = get_phase_status()
        for phase, status in phases.items():
            icon = "[OK]" if 'Complete' in status else "[--]"
            print(f"  {icon} {phase}: {status}")
        print()
        return

    verifier = Verifier(args.data_dir)

    if args.check:
        check_map = {
            'config': verifier.check_config,
            'dependencies': verifier.check_dependencies,
            'database': verifier.check_database,
            'embeddings': verifier.check_embeddings,
            'data': verifier.check_data_files,
        }

        status, message = check_map[args.check]()
        icon = "[OK]" if status else "[!!]"
        print(f"{icon} {args.check.title()}: {message}")
        sys.exit(0 if status else 1)

    # Full verification
    verifier.check_all()
    verifier.print_status()

    # Return non-zero if critical checks failed
    critical = ['dependencies', 'database']
    for check in critical:
        if check in verifier.results:
            status, _ = verifier.results[check]
            if not status and check != 'database':  # database is OK to be missing
                sys.exit(1)


if __name__ == '__main__':
    main()
