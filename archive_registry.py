"""
Archive Registry for CogRepo

Central management system for tracking LLM conversation archives.
Enables smart incremental processing by tracking:
- Registered archive files (ChatGPT, Claude, Gemini)
- Processing state and cursors for each archive
- File change detection
- Last processed timestamps

This is the key to "process only what's new" functionality.
"""

import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, field, asdict
from uuid import uuid4


@dataclass
class ProcessingCursor:
    """
    Tracks where we left off processing an archive.

    This enables resuming from the exact point of last processing
    instead of re-reading the entire archive.
    """
    last_external_id: Optional[str] = None  # Last conversation ID processed
    last_timestamp: Optional[str] = None    # Last conversation timestamp
    byte_offset: int = 0                     # File position (for JSONL)
    conversation_count: int = 0              # Total conversations processed from this archive

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingCursor':
        return cls(**data)


@dataclass
class Archive:
    """
    Represents a registered LLM conversation archive.

    Each of your three LLMs (ChatGPT, Claude, Gemini) would have
    one Archive entry tracking its export file.
    """
    id: str                                  # Unique archive ID
    name: str                                # Display name ("ChatGPT Main")
    source: Literal["OpenAI", "Anthropic", "Google"]  # Platform
    file_path: str                           # Path to export file

    # File tracking
    file_hash: Optional[str] = None          # SHA256 of file (change detection)
    file_size: int = 0                       # File size in bytes
    last_modified: Optional[str] = None      # File modification time

    # Processing state
    total_conversations: int = 0             # Total conversations in file
    processed_conversations: int = 0         # Conversations we've processed
    pending_conversations: int = 0           # New since last sync

    # Timing
    registered_at: Optional[str] = None      # When archive was registered
    last_sync_at: Optional[str] = None       # Last successful sync
    last_sync_duration: float = 0            # Seconds for last sync

    # Processing cursor
    cursor: ProcessingCursor = field(default_factory=ProcessingCursor)

    # Settings
    auto_sync: bool = True                   # Include in quick sync
    enabled: bool = True                     # Is this archive active

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['cursor'] = self.cursor.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Archive':
        cursor_data = data.pop('cursor', {})
        archive = cls(**data)
        archive.cursor = ProcessingCursor.from_dict(cursor_data)
        return archive


@dataclass
class FileChangeReport:
    """Report of changes detected in an archive file."""
    archive_name: str
    has_changed: bool
    change_type: Optional[str] = None  # "new_content", "modified", "replaced", "unchanged"
    size_change: int = 0               # Bytes added (positive) or removed (negative)
    estimated_new_conversations: int = 0
    details: str = ""


class ArchiveRegistry:
    """
    Central registry for managing LLM conversation archives.

    This is the "brain" that tracks your three archives and enables
    smart incremental processing.

    Usage:
        registry = ArchiveRegistry()

        # Register your archives (one-time)
        registry.register("ChatGPT", "chatgpt", "~/exports/chatgpt.json")
        registry.register("Claude", "claude", "~/exports/claude.jsonl")
        registry.register("Gemini", "gemini", "~/exports/gemini.json")

        # Check what's new
        changes = registry.detect_all_changes()

        # Get only new conversations
        for conv in registry.get_new_conversations("ChatGPT"):
            process(conv)

        # Update cursor after processing
        registry.update_cursor("ChatGPT", last_id, last_timestamp)
    """

    def __init__(self, registry_file: str = "data/archive_registry.json"):
        """
        Initialize archive registry.

        Args:
            registry_file: Path to registry file (relative to cogrepo root)
        """
        self.registry_file = Path(__file__).parent / registry_file
        self.archives: Dict[str, Archive] = {}
        self.settings: Dict[str, Any] = {
            "quick_sync_enabled": True,
            "auto_detect_updates": True,
            "default_enrich": True
        }
        self._load()

    def _load(self):
        """Load registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for name, archive_data in data.get("archives", {}).items():
                    self.archives[name] = Archive.from_dict(archive_data)

                self.settings = data.get("settings", self.settings)

            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load registry: {e}")

    def save(self):
        """Save registry to file."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "last_updated": datetime.now().isoformat(),
            "archives": {
                name: archive.to_dict()
                for name, archive in self.archives.items()
            },
            "settings": self.settings
        }

        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def register(
        self,
        name: str,
        source: str,
        file_path: str,
        auto_sync: bool = True
    ) -> Archive:
        """
        Register a new archive for tracking.

        Args:
            name: Display name for the archive ("ChatGPT Main")
            source: Source platform ("chatgpt", "claude", "gemini")
            file_path: Path to the export file
            auto_sync: Include in quick sync operations

        Returns:
            Archive object

        Example:
            registry.register("ChatGPT", "chatgpt", "~/exports/conversations.json")
        """
        # Normalize source
        source_map = {
            "chatgpt": "OpenAI",
            "openai": "OpenAI",
            "claude": "Anthropic",
            "anthropic": "Anthropic",
            "gemini": "Google",
            "google": "Google"
        }
        normalized_source = source_map.get(source.lower(), source)

        # Resolve file path
        resolved_path = str(Path(file_path).expanduser().resolve())

        # Check if file exists
        if not Path(resolved_path).exists():
            raise FileNotFoundError(f"Archive file not found: {file_path}")

        # Get file info
        file_stat = os.stat(resolved_path)
        file_hash = self._compute_file_hash(resolved_path)

        # Create archive
        archive = Archive(
            id=str(uuid4()),
            name=name,
            source=normalized_source,
            file_path=resolved_path,
            file_hash=file_hash,
            file_size=file_stat.st_size,
            last_modified=datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            registered_at=datetime.now().isoformat(),
            auto_sync=auto_sync
        )

        # Count conversations in file
        archive.total_conversations = self._count_conversations(archive)
        archive.pending_conversations = archive.total_conversations

        # Store
        self.archives[name] = archive
        self.save()

        return archive

    def unregister(self, name: str) -> bool:
        """
        Remove an archive from the registry.

        Args:
            name: Archive name to remove

        Returns:
            True if removed, False if not found
        """
        if name in self.archives:
            del self.archives[name]
            self.save()
            return True
        return False

    def get_archive(self, name: str) -> Optional[Archive]:
        """Get an archive by name."""
        return self.archives.get(name)

    def list_archives(self) -> List[Archive]:
        """Get list of all registered archives."""
        return list(self.archives.values())

    def detect_changes(self, name: str) -> FileChangeReport:
        """
        Detect if an archive file has changed since last sync.

        This is the key to knowing "what's new" without re-reading everything.

        Args:
            name: Archive name

        Returns:
            FileChangeReport with change details
        """
        archive = self.archives.get(name)
        if not archive:
            return FileChangeReport(
                archive_name=name,
                has_changed=False,
                details="Archive not found"
            )

        # Check if file still exists
        if not Path(archive.file_path).exists():
            return FileChangeReport(
                archive_name=name,
                has_changed=False,
                change_type="missing",
                details=f"File not found: {archive.file_path}"
            )

        # Get current file info
        file_stat = os.stat(archive.file_path)
        current_size = file_stat.st_size
        current_modified = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

        # Quick check: file size unchanged = likely unchanged
        if current_size == archive.file_size:
            # Double-check with hash for certainty
            current_hash = self._compute_file_hash(archive.file_path)
            if current_hash == archive.file_hash:
                return FileChangeReport(
                    archive_name=name,
                    has_changed=False,
                    change_type="unchanged",
                    details="File unchanged since last sync"
                )

        # File changed - determine type of change
        size_diff = current_size - archive.file_size

        if size_diff > 0:
            # File grew - likely new conversations appended
            change_type = "new_content"
            # Estimate new conversations based on size increase
            avg_conv_size = archive.file_size / max(archive.total_conversations, 1)
            estimated_new = int(size_diff / avg_conv_size) if avg_conv_size > 0 else 0
            details = f"File grew by {self._format_size(size_diff)} (~{estimated_new} new conversations)"
        elif size_diff < 0:
            # File shrunk - likely replaced/modified
            change_type = "replaced"
            estimated_new = 0
            details = f"File shrunk by {self._format_size(abs(size_diff))} - may need full re-sync"
        else:
            # Same size but different content
            change_type = "modified"
            estimated_new = 0
            details = "File content changed (same size)"

        # Count actual new conversations
        current_total = self._count_conversations(archive)
        actual_new = max(0, current_total - archive.processed_conversations)

        return FileChangeReport(
            archive_name=name,
            has_changed=True,
            change_type=change_type,
            size_change=size_diff,
            estimated_new_conversations=actual_new,
            details=details
        )

    def detect_all_changes(self) -> List[FileChangeReport]:
        """
        Detect changes for all registered archives.

        Returns:
            List of FileChangeReports for each archive
        """
        reports = []
        for name in self.archives:
            reports.append(self.detect_changes(name))
        return reports

    def update_after_sync(
        self,
        name: str,
        conversations_processed: int,
        last_external_id: Optional[str] = None,
        last_timestamp: Optional[str] = None,
        byte_offset: int = 0,
        duration_seconds: float = 0
    ):
        """
        Update archive state after a successful sync.

        Args:
            name: Archive name
            conversations_processed: Number of new conversations processed
            last_external_id: External ID of last processed conversation
            last_timestamp: Timestamp of last processed conversation
            byte_offset: File position (for JSONL resumption)
            duration_seconds: How long the sync took
        """
        archive = self.archives.get(name)
        if not archive:
            return

        # Update file tracking
        if Path(archive.file_path).exists():
            file_stat = os.stat(archive.file_path)
            archive.file_hash = self._compute_file_hash(archive.file_path)
            archive.file_size = file_stat.st_size
            archive.last_modified = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

        # Update counts
        archive.processed_conversations += conversations_processed
        archive.total_conversations = self._count_conversations(archive)
        archive.pending_conversations = max(0, archive.total_conversations - archive.processed_conversations)

        # Update cursor
        archive.cursor.last_external_id = last_external_id
        archive.cursor.last_timestamp = last_timestamp
        archive.cursor.byte_offset = byte_offset
        archive.cursor.conversation_count = archive.processed_conversations

        # Update timing
        archive.last_sync_at = datetime.now().isoformat()
        archive.last_sync_duration = duration_seconds

        self.save()

    def get_status(self) -> Dict[str, Any]:
        """
        Get overall status of all archives.

        Returns formatted status for display.
        """
        total_conversations = 0
        total_pending = 0
        archives_needing_sync = []

        archive_statuses = []
        for name, archive in self.archives.items():
            total_conversations += archive.processed_conversations
            total_pending += archive.pending_conversations

            # Check for changes
            changes = self.detect_changes(name)

            status = {
                "name": name,
                "source": archive.source,
                "processed": archive.processed_conversations,
                "pending": changes.estimated_new_conversations if changes.has_changed else 0,
                "last_sync": archive.last_sync_at,
                "has_changes": changes.has_changed,
                "change_details": changes.details,
                "enabled": archive.enabled,
                "auto_sync": archive.auto_sync
            }
            archive_statuses.append(status)

            if changes.has_changed and archive.auto_sync:
                archives_needing_sync.append(name)

        return {
            "total_conversations": total_conversations,
            "total_pending": total_pending,
            "archive_count": len(self.archives),
            "archives_needing_sync": archives_needing_sync,
            "archives": archive_statuses
        }

    def _compute_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _count_conversations(self, archive: Archive) -> int:
        """
        Count total conversations in archive file.

        Uses quick estimation based on file structure.
        """
        try:
            file_path = Path(archive.file_path)

            if file_path.suffix.lower() == '.jsonl':
                # JSONL: count lines
                with open(file_path, 'r', encoding='utf-8') as f:
                    return sum(1 for line in f if line.strip())
            else:
                # JSON: load and count array length
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return len(data)
                    else:
                        return 1  # Single conversation
        except Exception:
            return 0

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size for display."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


# Convenience functions for CLI usage
def get_registry() -> ArchiveRegistry:
    """Get the default archive registry."""
    return ArchiveRegistry()


def quick_status():
    """Print quick status of all archives."""
    registry = get_registry()
    status = registry.get_status()

    print(f"\n{'='*60}")
    print(f"COGREPO ARCHIVE STATUS")
    print(f"{'='*60}")
    print(f"Total Conversations: {status['total_conversations']}")
    print(f"Pending Sync: {status['total_pending']}")
    print(f"Archives: {status['archive_count']}")

    if status['archives_needing_sync']:
        print(f"\nArchives needing sync: {', '.join(status['archives_needing_sync'])}")

    print(f"\n{'─'*60}")
    for arch in status['archives']:
        sync_indicator = "!" if arch['has_changes'] else " "
        print(f"{sync_indicator} {arch['name']:<15} {arch['source']:<10} "
              f"Processed: {arch['processed']} | Pending: {arch['pending']}")
        if arch['has_changes']:
            print(f"  └─ {arch['change_details']}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    quick_status()
