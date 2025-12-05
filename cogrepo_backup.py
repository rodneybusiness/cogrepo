#!/usr/bin/env python3
"""
CogRepo Backup & Restore System

Provides one-click backup and restore for:
- SQLite database
- JSONL data files
- Embeddings
- Configuration

Usage:
    # Create backup
    python cogrepo_backup.py backup

    # Create backup with custom name
    python cogrepo_backup.py backup --name my-backup

    # List backups
    python cogrepo_backup.py list

    # Restore from backup
    python cogrepo_backup.py restore backup_20241201_120000.tar.gz

    # Verify backup integrity
    python cogrepo_backup.py verify backup_20241201_120000.tar.gz
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

class BackupConfig:
    """Backup configuration and paths."""

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.data_dir = self.base_dir / 'data'
        self.backup_dir = self.base_dir / 'backups'

        # Files to backup
        self.backup_files = [
            'data/enriched_repository.jsonl',
            'data/cogrepo.db',
            'data/embeddings.npy',
            'data/embedding_ids.json',
            'data/context_analysis.json',
        ]

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Backup Operations
# =============================================================================

def calculate_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_file_stats(file_path: Path) -> Dict:
    """Get file statistics."""
    if not file_path.exists():
        return {'exists': False}

    stat = file_path.stat()
    return {
        'exists': True,
        'size_bytes': stat.st_size,
        'size_mb': round(stat.st_size / 1024 / 1024, 2),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'checksum': calculate_checksum(file_path)
    }


def create_backup(config: BackupConfig, name: str = None, verbose: bool = False) -> Tuple[bool, str]:
    """
    Create a backup of all CogRepo data.

    Args:
        config: Backup configuration
        name: Optional backup name
        verbose: Show detailed progress

    Returns:
        Tuple of (success, backup_path or error message)
    """
    # Generate backup name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = name or f'backup_{timestamp}'
    backup_path = config.backup_dir / f'{backup_name}.tar.gz'

    print(f"\n{'='*50}")
    print(f"  CogRepo Backup")
    print(f"{'='*50}")
    print(f"\n  Output: {backup_path}")
    print()

    # Collect files to backup
    files_to_backup = []
    manifest = {
        'created': datetime.now().isoformat(),
        'version': '2.0.0',
        'files': {}
    }

    for rel_path in config.backup_files:
        file_path = config.base_dir / rel_path
        if file_path.exists():
            stats = get_file_stats(file_path)
            manifest['files'][rel_path] = stats
            files_to_backup.append((file_path, rel_path))
            print(f"  [+] {rel_path} ({stats['size_mb']} MB)")
        else:
            if verbose:
                print(f"  [-] {rel_path} (not found, skipping)")

    if not files_to_backup:
        return False, "No files found to backup"

    # Create temporary manifest file
    manifest_path = config.backup_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Create tarball
    print(f"\n  Creating archive...")

    try:
        with tarfile.open(backup_path, 'w:gz') as tar:
            # Add manifest first
            tar.add(manifest_path, arcname='manifest.json')

            # Add data files
            for file_path, arcname in files_to_backup:
                tar.add(file_path, arcname=arcname)
                if verbose:
                    print(f"    Added: {arcname}")

        # Clean up manifest
        manifest_path.unlink()

        # Get backup size
        backup_size = backup_path.stat().st_size / 1024 / 1024

        print(f"\n  Backup complete!")
        print(f"  Size: {backup_size:.2f} MB")
        print(f"  Files: {len(files_to_backup)}")
        print(f"  Path: {backup_path}")

        return True, str(backup_path)

    except Exception as e:
        if manifest_path.exists():
            manifest_path.unlink()
        return False, f"Backup failed: {str(e)}"


def list_backups(config: BackupConfig) -> List[Dict]:
    """List all available backups."""
    backups = []

    for backup_file in sorted(config.backup_dir.glob('*.tar.gz'), reverse=True):
        stat = backup_file.stat()

        # Try to read manifest
        manifest = None
        try:
            with tarfile.open(backup_file, 'r:gz') as tar:
                manifest_member = tar.getmember('manifest.json')
                manifest_file = tar.extractfile(manifest_member)
                if manifest_file:
                    manifest = json.load(manifest_file)
        except Exception:
            pass

        backups.append({
            'name': backup_file.name,
            'path': str(backup_file),
            'size_mb': round(stat.st_size / 1024 / 1024, 2),
            'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'manifest': manifest
        })

    return backups


def verify_backup(config: BackupConfig, backup_path: str) -> Tuple[bool, Dict]:
    """
    Verify backup integrity.

    Returns:
        Tuple of (is_valid, details)
    """
    backup_file = Path(backup_path)
    if not backup_file.exists():
        return False, {'error': 'Backup file not found'}

    result = {
        'path': str(backup_file),
        'size_mb': round(backup_file.stat().st_size / 1024 / 1024, 2),
        'files': [],
        'errors': []
    }

    try:
        with tarfile.open(backup_file, 'r:gz') as tar:
            # Check for manifest
            try:
                manifest_member = tar.getmember('manifest.json')
                manifest_file = tar.extractfile(manifest_member)
                if manifest_file:
                    manifest = json.load(manifest_file)
                    result['manifest'] = manifest
            except KeyError:
                result['errors'].append('Missing manifest.json')

            # List and verify files
            for member in tar.getmembers():
                if member.name == 'manifest.json':
                    continue

                result['files'].append({
                    'name': member.name,
                    'size': member.size,
                    'type': 'file' if member.isfile() else 'other'
                })

        result['valid'] = len(result['errors']) == 0 and len(result['files']) > 0
        return result['valid'], result

    except tarfile.TarError as e:
        return False, {'error': f'Invalid tar archive: {str(e)}'}
    except Exception as e:
        return False, {'error': f'Verification failed: {str(e)}'}


def restore_backup(
    config: BackupConfig,
    backup_path: str,
    force: bool = False,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Restore from a backup.

    Args:
        config: Backup configuration
        backup_path: Path to backup file
        force: Overwrite existing files without confirmation
        verbose: Show detailed progress

    Returns:
        Tuple of (success, message)
    """
    backup_file = Path(backup_path)
    if not backup_file.exists():
        return False, f"Backup not found: {backup_path}"

    print(f"\n{'='*50}")
    print(f"  CogRepo Restore")
    print(f"{'='*50}")
    print(f"\n  Source: {backup_file}")
    print()

    # Verify backup first
    is_valid, details = verify_backup(config, backup_path)
    if not is_valid:
        return False, f"Invalid backup: {details.get('error', 'Unknown error')}"

    # Check for existing files
    existing_files = []
    for rel_path in config.backup_files:
        file_path = config.base_dir / rel_path
        if file_path.exists():
            existing_files.append(rel_path)

    if existing_files and not force:
        print("  WARNING: The following files will be overwritten:")
        for f in existing_files:
            print(f"    - {f}")
        print()
        response = input("  Continue? [y/N]: ").strip().lower()
        if response != 'y':
            return False, "Restore cancelled by user"

    # Create backup of current data before restore
    print("\n  Creating safety backup of current data...")
    safety_backup_name = f'pre_restore_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    create_backup(config, name=safety_backup_name, verbose=False)

    # Extract backup
    print("\n  Restoring files...")
    try:
        with tarfile.open(backup_file, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name == 'manifest.json':
                    continue

                # Create parent directories
                dest_path = config.base_dir / member.name
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract file
                tar.extract(member, config.base_dir)
                if verbose:
                    print(f"    Restored: {member.name}")
                else:
                    print(f"  [+] {member.name}")

        print(f"\n  Restore complete!")
        print(f"  Safety backup: {safety_backup_name}.tar.gz")

        return True, "Restore completed successfully"

    except Exception as e:
        return False, f"Restore failed: {str(e)}"


def cleanup_old_backups(config: BackupConfig, keep: int = 5) -> int:
    """
    Remove old backups, keeping the most recent ones.

    Args:
        config: Backup configuration
        keep: Number of backups to keep

    Returns:
        Number of backups removed
    """
    backups = list_backups(config)

    if len(backups) <= keep:
        return 0

    removed = 0
    for backup in backups[keep:]:
        try:
            Path(backup['path']).unlink()
            removed += 1
            print(f"  Removed: {backup['name']}")
        except Exception as e:
            print(f"  Failed to remove {backup['name']}: {e}")

    return removed


# =============================================================================
# Database-Specific Operations
# =============================================================================

def backup_database(config: BackupConfig, output_path: str = None) -> Tuple[bool, str]:
    """
    Create a hot backup of the SQLite database.

    Uses SQLite's backup API for consistency.
    """
    db_path = config.data_dir / 'cogrepo.db'
    if not db_path.exists():
        return False, "Database not found"

    output_path = output_path or str(config.backup_dir / f'cogrepo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')

    try:
        # Use SQLite backup API
        source = sqlite3.connect(str(db_path))
        dest = sqlite3.connect(output_path)

        source.backup(dest)

        source.close()
        dest.close()

        return True, output_path

    except Exception as e:
        return False, f"Database backup failed: {str(e)}"


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CogRepo Backup & Restore System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create a backup')
    backup_parser.add_argument('--name', '-n', help='Backup name (default: auto-generated)')
    backup_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # List command
    list_parser = subparsers.add_parser('list', help='List available backups')

    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('backup', help='Backup file to restore')
    restore_parser.add_argument('--force', '-f', action='store_true', help='Overwrite without confirmation')
    restore_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify backup integrity')
    verify_parser.add_argument('backup', help='Backup file to verify')

    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Remove old backups')
    cleanup_parser.add_argument('--keep', '-k', type=int, default=5, help='Number of backups to keep')

    args = parser.parse_args()
    config = BackupConfig()

    if args.command == 'backup':
        success, result = create_backup(config, name=args.name, verbose=args.verbose)
        sys.exit(0 if success else 1)

    elif args.command == 'list':
        backups = list_backups(config)
        if not backups:
            print("No backups found.")
        else:
            print(f"\n{'='*60}")
            print(f"  Available Backups ({len(backups)})")
            print(f"{'='*60}\n")
            for b in backups:
                print(f"  {b['name']}")
                print(f"    Size: {b['size_mb']} MB")
                print(f"    Created: {b['created']}")
                if b['manifest']:
                    files = len(b['manifest'].get('files', {}))
                    print(f"    Files: {files}")
                print()

    elif args.command == 'restore':
        # Find backup file
        backup_path = args.backup
        if not Path(backup_path).exists():
            # Try in backups directory
            backup_path = str(config.backup_dir / args.backup)

        success, message = restore_backup(config, backup_path, force=args.force, verbose=args.verbose)
        print(f"\n  {message}")
        sys.exit(0 if success else 1)

    elif args.command == 'verify':
        backup_path = args.backup
        if not Path(backup_path).exists():
            backup_path = str(config.backup_dir / args.backup)

        is_valid, details = verify_backup(config, backup_path)

        print(f"\n{'='*50}")
        print(f"  Backup Verification")
        print(f"{'='*50}\n")

        if is_valid:
            print(f"  Status: VALID")
            print(f"  Size: {details['size_mb']} MB")
            print(f"  Files: {len(details['files'])}")
            for f in details['files']:
                print(f"    - {f['name']} ({f['size']} bytes)")
        else:
            print(f"  Status: INVALID")
            print(f"  Error: {details.get('error', 'Unknown')}")

        sys.exit(0 if is_valid else 1)

    elif args.command == 'cleanup':
        removed = cleanup_old_backups(config, keep=args.keep)
        print(f"\n  Removed {removed} old backup(s), keeping {args.keep} most recent.")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
