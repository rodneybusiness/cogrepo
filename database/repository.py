"""
SQLite Database Repository for CogRepo

Provides indexed storage with:
- Full-text search via FTS5
- Efficient filtering and pagination
- Transaction support
- Migration from JSONL

Usage:
    from database.repository import ConversationRepository

    repo = ConversationRepository()
    repo.save(conversation_dict)
    results = repo.search("python async", limit=20)
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager


# =============================================================================
# Schema
# =============================================================================

SCHEMA = '''
-- Main conversations table
CREATE TABLE IF NOT EXISTS conversations (
    convo_id TEXT PRIMARY KEY,
    external_id TEXT,
    timestamp TEXT,
    source TEXT,
    raw_text TEXT,

    -- Tier 1: AI Enrichment
    generated_title TEXT,
    summary_abstractive TEXT,
    summary_extractive TEXT,
    primary_domain TEXT,
    tags TEXT,  -- JSON array
    key_topics TEXT,  -- JSON array
    brilliance_score TEXT,  -- JSON object
    key_insights TEXT,  -- JSON array
    status TEXT,
    future_potential TEXT,  -- JSON object
    score INTEGER,
    score_reasoning TEXT,

    -- Tier 2: Zero-token
    has_code BOOLEAN DEFAULT 0,
    code_languages TEXT,  -- JSON array
    code_block_count INTEGER DEFAULT 0,
    has_error_traces BOOLEAN DEFAULT 0,
    turn_count INTEGER DEFAULT 0,
    question_count INTEGER DEFAULT 0,
    question_types TEXT,  -- JSON array
    has_links BOOLEAN DEFAULT 0,
    link_domains TEXT,  -- JSON array
    technical_terms TEXT,  -- JSON array
    mentioned_files TEXT,  -- JSON array

    -- Tier 5: Context
    project_id TEXT,
    chain_id TEXT,
    chain_position INTEGER,

    -- Metadata
    metadata TEXT,  -- JSON object
    enrichment_version TEXT DEFAULT '2.0',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Full-text search index
CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
    generated_title,
    summary_abstractive,
    raw_text,
    tags,
    key_topics,
    key_insights,
    content=conversations,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
    INSERT INTO conversations_fts(rowid, generated_title, summary_abstractive, raw_text, tags, key_topics, key_insights)
    VALUES (NEW.rowid, NEW.generated_title, NEW.summary_abstractive, NEW.raw_text, NEW.tags, NEW.key_topics, NEW.key_insights);
END;

CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, generated_title, summary_abstractive, raw_text, tags, key_topics, key_insights)
    VALUES ('delete', OLD.rowid, OLD.generated_title, OLD.summary_abstractive, OLD.raw_text, OLD.tags, OLD.key_topics, OLD.key_insights);
END;

CREATE TRIGGER IF NOT EXISTS conversations_au AFTER UPDATE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, generated_title, summary_abstractive, raw_text, tags, key_topics, key_insights)
    VALUES ('delete', OLD.rowid, OLD.generated_title, OLD.summary_abstractive, OLD.raw_text, OLD.tags, OLD.key_topics, OLD.key_insights);
    INSERT INTO conversations_fts(rowid, generated_title, summary_abstractive, raw_text, tags, key_topics, key_insights)
    VALUES (NEW.rowid, NEW.generated_title, NEW.summary_abstractive, NEW.raw_text, NEW.tags, NEW.key_topics, NEW.key_insights);
END;

-- Artifacts table
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    artifact_type TEXT,
    content TEXT,
    language TEXT,
    description TEXT,
    use_case TEXT,
    tags TEXT,  -- JSON array
    verified_working BOOLEAN DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(convo_id)
);

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    project_name TEXT,
    technologies TEXT,  -- JSON array
    keywords TEXT,  -- JSON array
    first_conversation TEXT,
    last_conversation TEXT,
    conversation_count INTEGER,
    project_summary TEXT,
    status TEXT DEFAULT 'active',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Conversation chains
CREATE TABLE IF NOT EXISTS chains (
    chain_id TEXT PRIMARY KEY,
    title TEXT,
    starting_point TEXT,
    resolution TEXT,
    key_learnings TEXT,  -- JSON array
    conversation_ids TEXT,  -- JSON array
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_conversations_source ON conversations(source);
CREATE INDEX IF NOT EXISTS idx_conversations_domain ON conversations(primary_domain);
CREATE INDEX IF NOT EXISTS idx_conversations_score ON conversations(score);
CREATE INDEX IF NOT EXISTS idx_conversations_has_code ON conversations(has_code);
CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);
CREATE INDEX IF NOT EXISTS idx_artifacts_conversation ON artifacts(conversation_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);
'''


class ConversationRepository:
    """
    Repository for conversation storage and retrieval.

    Uses SQLite with FTS5 for efficient full-text search.
    """

    def __init__(self, db_path: str = None):
        """
        Initialize the repository.

        Args:
            db_path: Path to SQLite database. Defaults to data/cogrepo.db
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / 'data' / 'cogrepo.db'

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with self._connection() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def save(self, conversation: Dict[str, Any]) -> bool:
        """
        Save or update a conversation.

        Args:
            conversation: Conversation dict with at least 'convo_id'

        Returns:
            True if successful
        """
        # Serialize JSON fields
        json_fields = [
            'tags', 'key_topics', 'brilliance_score', 'key_insights',
            'future_potential', 'code_languages', 'question_types',
            'link_domains', 'technical_terms', 'mentioned_files', 'metadata'
        ]

        data = conversation.copy()
        for field in json_fields:
            if field in data and not isinstance(data[field], str):
                data[field] = json.dumps(data[field])

        # Build column list dynamically based on data
        columns = []
        values = []
        placeholders = []

        for key, value in data.items():
            if key in self._get_column_names():
                columns.append(key)
                values.append(value)
                placeholders.append('?')

        columns.append('updated_at')
        values.append(datetime.now().isoformat())
        placeholders.append('?')

        sql = f'''
            INSERT OR REPLACE INTO conversations ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        '''

        with self._connection() as conn:
            conn.execute(sql, values)

        return True

    def get(self, convo_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID.

        Args:
            convo_id: Conversation ID

        Returns:
            Conversation dict or None
        """
        with self._connection() as conn:
            row = conn.execute(
                'SELECT * FROM conversations WHERE convo_id = ?',
                (convo_id,)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def delete(self, convo_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            convo_id: Conversation ID

        Returns:
            True if deleted
        """
        with self._connection() as conn:
            cursor = conn.execute(
                'DELETE FROM conversations WHERE convo_id = ?',
                (convo_id,)
            )
            return cursor.rowcount > 0

    def count(self, filters: Dict[str, Any] = None) -> int:
        """Get total conversation count with optional filters."""
        filters = filters or {}

        # Build WHERE clause
        where_parts = []
        params = []

        if filters.get('source'):
            where_parts.append('source = ?')
            params.append(filters['source'])
        if filters.get('domain'):
            where_parts.append('primary_domain = ?')
            params.append(filters['domain'])
        if filters.get('has_code'):
            where_parts.append('has_code = 1')

        where_clause = ' AND '.join(where_parts) if where_parts else '1=1'

        with self._connection() as conn:
            row = conn.execute(f'SELECT COUNT(*) FROM conversations WHERE {where_clause}', params).fetchone()
            return row[0]

    # =========================================================================
    # Search
    # =========================================================================

    def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Full-text search with optional filters.

        Args:
            query: Search query
            limit: Maximum results
            offset: Skip first N results
            filters: Optional filters (source, domain, score, has_code, etc.)

        Returns:
            List of matching conversations
        """
        filters = filters or {}

        # Build WHERE clause for filters
        where_parts = []
        params = []

        if filters.get('source'):
            where_parts.append('c.source = ?')
            params.append(filters['source'])

        if filters.get('domain'):
            where_parts.append('c.primary_domain = ?')
            params.append(filters['domain'])

        if filters.get('min_score'):
            where_parts.append('c.score >= ?')
            params.append(filters['min_score'])

        if filters.get('has_code'):
            where_parts.append('c.has_code = 1')

        if filters.get('has_links'):
            where_parts.append('c.has_links = 1')

        where_clause = ' AND '.join(where_parts) if where_parts else '1=1'

        if query and query.strip():
            # FTS search
            sql = f'''
                SELECT c.*, bm25(conversations_fts) as rank
                FROM conversations c
                JOIN conversations_fts fts ON c.rowid = fts.rowid
                WHERE conversations_fts MATCH ? AND {where_clause}
                ORDER BY rank
                LIMIT ? OFFSET ?
            '''
            params = [query] + params + [limit, offset]
        else:
            # No query, just filters
            sql = f'''
                SELECT c.*
                FROM conversations c
                WHERE {where_clause}
                ORDER BY c.timestamp DESC
                LIMIT ? OFFSET ?
            '''
            params = params + [limit, offset]

        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def search_by_tag(self, tag: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search conversations by tag."""
        with self._connection() as conn:
            rows = conn.execute('''
                SELECT * FROM conversations
                WHERE tags LIKE ?
                ORDER BY score DESC, timestamp DESC
                LIMIT ?
            ''', (f'%"{tag}"%', limit)).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = 'timestamp DESC'
    ) -> List[Dict[str, Any]]:
        """Get all conversations with pagination."""
        with self._connection() as conn:
            rows = conn.execute(f'''
                SELECT * FROM conversations
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
            ''', (limit, offset)).fetchall()

        return [self._row_to_dict(row) for row in rows]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        with self._connection() as conn:
            stats = {}

            # Total count
            stats['total'] = conn.execute(
                'SELECT COUNT(*) FROM conversations'
            ).fetchone()[0]

            # By source
            rows = conn.execute('''
                SELECT source, COUNT(*) as count
                FROM conversations
                GROUP BY source
            ''').fetchall()
            stats['by_source'] = {row['source']: row['count'] for row in rows}

            # By domain
            rows = conn.execute('''
                SELECT primary_domain, COUNT(*) as count
                FROM conversations
                GROUP BY primary_domain
                ORDER BY count DESC
                LIMIT 10
            ''').fetchall()
            stats['by_domain'] = {row['primary_domain']: row['count'] for row in rows}

            # Score distribution
            rows = conn.execute('''
                SELECT score, COUNT(*) as count
                FROM conversations
                WHERE score IS NOT NULL
                GROUP BY score
                ORDER BY score
            ''').fetchall()
            stats['by_score'] = {row['score']: row['count'] for row in rows}

            # With code
            stats['with_code'] = conn.execute(
                'SELECT COUNT(*) FROM conversations WHERE has_code = 1'
            ).fetchone()[0]

            return stats

    def get_tag_cloud(self, limit: int = 50) -> List[Tuple[str, int]]:
        """Get most common tags with counts."""
        with self._connection() as conn:
            rows = conn.execute('SELECT tags FROM conversations').fetchall()

        # Count tags
        tag_counts = {}
        for row in rows:
            if row['tags']:
                try:
                    tags = json.loads(row['tags'])
                    for tag in tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass

        # Sort by count
        sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])
        return sorted_tags[:limit]

    # =========================================================================
    # Artifacts
    # =========================================================================

    def save_artifact(self, artifact: Dict[str, Any]) -> bool:
        """Save an artifact."""
        data = artifact.copy()
        if 'tags' in data and not isinstance(data['tags'], str):
            data['tags'] = json.dumps(data['tags'])

        sql = '''
            INSERT OR REPLACE INTO artifacts
            (artifact_id, conversation_id, artifact_type, content, language,
             description, use_case, tags, verified_working)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        with self._connection() as conn:
            conn.execute(sql, (
                data.get('artifact_id'),
                data.get('conversation_id'),
                data.get('artifact_type'),
                data.get('content'),
                data.get('language'),
                data.get('description'),
                data.get('use_case'),
                data.get('tags'),
                data.get('verified_working', True)
            ))

        return True

    def get_artifacts(
        self,
        conversation_id: str = None,
        artifact_type: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get artifacts with optional filters."""
        where_parts = []
        params = []

        if conversation_id:
            where_parts.append('conversation_id = ?')
            params.append(conversation_id)

        if artifact_type:
            where_parts.append('artifact_type = ?')
            params.append(artifact_type)

        where_clause = ' AND '.join(where_parts) if where_parts else '1=1'

        with self._connection() as conn:
            rows = conn.execute(f'''
                SELECT * FROM artifacts
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
            ''', params + [limit]).fetchall()

        return [dict(row) for row in rows]

    # =========================================================================
    # Migration
    # =========================================================================

    def import_from_jsonl(self, jsonl_path: str, show_progress: bool = True) -> int:
        """
        Import conversations from JSONL file.

        Args:
            jsonl_path: Path to JSONL file
            show_progress: Show progress bar

        Returns:
            Number of imported conversations
        """
        # Read all conversations
        conversations = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))

        print(f"Importing {len(conversations)} conversations...")

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(conversations, desc="Importing")
            except ImportError:
                iterator = conversations
        else:
            iterator = conversations

        count = 0
        for conv in iterator:
            self.save(conv)
            count += 1

        print(f"Imported {count} conversations")
        return count

    def import_from_list(self, conversations: List[Dict], show_progress: bool = True) -> int:
        """
        Import conversations from a list of dicts.

        Args:
            conversations: List of conversation dictionaries
            show_progress: Show progress

        Returns:
            Number of imported conversations
        """
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(conversations, desc="Importing")
            except ImportError:
                iterator = conversations
        else:
            iterator = conversations

        count = 0
        for conv in iterator:
            self.save(conv)
            count += 1

        return count

    def export_to_jsonl(self, jsonl_path: str) -> int:
        """
        Export all conversations to JSONL file.

        Args:
            jsonl_path: Path to output JSONL file

        Returns:
            Number of exported conversations
        """
        conversations = self.get_all(limit=100000)

        with open(jsonl_path, 'w') as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')

        print(f"Exported {len(conversations)} conversations to {jsonl_path}")
        return len(conversations)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_column_names(self) -> set:
        """Get valid column names."""
        with self._connection() as conn:
            cursor = conn.execute('PRAGMA table_info(conversations)')
            return {row[1] for row in cursor.fetchall()}

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a dictionary with JSON parsing."""
        data = dict(row)

        # Parse JSON fields
        json_fields = [
            'tags', 'key_topics', 'brilliance_score', 'key_insights',
            'future_potential', 'code_languages', 'question_types',
            'link_domains', 'technical_terms', 'mentioned_files', 'metadata'
        ]

        for field in json_fields:
            if field in data and data[field]:
                try:
                    data[field] = json.loads(data[field])
                except (json.JSONDecodeError, TypeError):
                    pass

        return data


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="CogRepo Database Management")
    parser.add_argument('action', choices=['import', 'export', 'stats', 'search'],
                        help="Action to perform")
    parser.add_argument('--file', '-f', help="JSONL file for import/export")
    parser.add_argument('--query', '-q', help="Search query")
    parser.add_argument('--db', help="Database path")

    args = parser.parse_args()

    repo = ConversationRepository(args.db)

    if args.action == 'import':
        if not args.file:
            print("Error: --file required for import")
        else:
            repo.import_from_jsonl(args.file)

    elif args.action == 'export':
        if not args.file:
            print("Error: --file required for export")
        else:
            repo.export_to_jsonl(args.file)

    elif args.action == 'stats':
        stats = repo.get_stats()
        print(json.dumps(stats, indent=2))

    elif args.action == 'search':
        if not args.query:
            print("Error: --query required for search")
        else:
            results = repo.search(args.query, limit=10)
            for r in results:
                print(f"- {r['generated_title']} (score: {r.get('score', 'N/A')})")
