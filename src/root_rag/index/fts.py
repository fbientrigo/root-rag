"""SQLite FTS5 lexical search backend for ROOT CODE retrieval."""
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List

from root_rag.index.schemas import Chunk

logger = logging.getLogger(__name__)


def check_fts5_available() -> bool:
    """Check if SQLite FTS5 module is available.
    
    Returns:
        True if FTS5 is available, False otherwise.
    """
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content)")
        conn.close()
        return True
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        logger.warning(f"FTS5 not available: {e}")
        return False


def create_fts5_db(db_path: Path) -> None:
    """Create FTS5 SQLite database with chunks virtual table.
    
    Args:
        db_path: Path where SQLite database will be created.
        
    Raises:
        sqlite3.OperationalError: If FTS5 table creation fails.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(exist_ok=True, parents=True)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create FTS5 virtual table with indexed and unindexed columns
    # Indexed columns: content, file_path, symbol_path, doc_origin (used for scoring)
    # Unindexed columns: metadata preserved but not scored (more efficient)
    cursor.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content,
            file_path,
            symbol_path,
            doc_origin,
            chunk_id UNINDEXED,
            start_line UNINDEXED,
            end_line UNINDEXED,
            root_ref UNINDEXED,
            resolved_commit UNINDEXED,
            language UNINDEXED,
            index_schema_version UNINDEXED
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Created FTS5 database at {db_path}")


def insert_chunks_into_fts(db_path: Path, chunks: List[Chunk]) -> Dict[str, int]:
    """Insert chunks into FTS5 database.
    
    Args:
        db_path: Path to FTS5 SQLite database.
        chunks: List of Chunk objects to insert (will be sorted for determinism).
        
    Returns:
        Dict with keys "inserted" and "errors" (counts).
    """
    db_path = Path(db_path)
    
    # Sort chunks deterministically: file_path, start_line, end_line, chunk_id
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (c.file_path, c.start_line, c.end_line, c.chunk_id)
    )
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    inserted = 0
    errors = 0
    
    for chunk in sorted_chunks:
        try:
            cursor.execute("""
                INSERT INTO chunks_fts (
                    content,
                    file_path,
                    symbol_path,
                    doc_origin,
                    chunk_id,
                    start_line,
                    end_line,
                    root_ref,
                    resolved_commit,
                    language,
                    index_schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.content,
                chunk.file_path,
                chunk.symbol_path or "",
                chunk.doc_origin,
                chunk.chunk_id,
                chunk.start_line,
                chunk.end_line,
                chunk.root_ref,
                chunk.resolved_commit,
                chunk.language,
                chunk.index_schema_version,
            ))
            inserted += 1
        except sqlite3.Error as e:
            logger.error(f"Failed to insert chunk {chunk.chunk_id}: {e}")
            errors += 1
    
    conn.commit()
    conn.close()
    logger.info(f"Inserted {inserted} chunks into FTS5 (errors: {errors})")
    
    return {"inserted": inserted, "errors": errors}


def build_fts_index(db_path: Path, chunks: List[Chunk]) -> Dict:
    """Build FTS5 index from chunks.
    
    Orchestrator function that creates DB and inserts chunks.
    
    Args:
        db_path: Path where FTS5 SQLite database will be created.
        chunks: List of Chunk objects to index.
        
    Returns:
        Dict with keys: db_path (str), chunk_count (int), created_at (str), status (str).
    """
    from datetime import datetime
    
    try:
        create_fts5_db(db_path)
        stats = insert_chunks_into_fts(db_path, chunks)
        
        created_at = datetime.utcnow().isoformat() + "Z"
        logger.info(f"FTS5 index ready: {db_path} ({stats['inserted']} chunks)")
        
        return {
            "db_path": str(db_path),
            "chunk_count": stats["inserted"],
            "created_at": created_at,
            "status": "success" if stats["errors"] == 0 else "partial",
        }
    except Exception as e:
        logger.error(f"Failed to build FTS5 index: {e}")
        return {
            "db_path": str(db_path),
            "chunk_count": 0,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "status": "failed",
        }
