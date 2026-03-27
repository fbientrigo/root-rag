"""Lexical retrieval using SQLite FTS5."""
import logging
import re
import sqlite3
from pathlib import Path
from typing import List

from root_rag.retrieval.models import EvidenceCandidate

logger = logging.getLogger(__name__)


def _sanitize_fts_query(query: str) -> str:
    """Sanitize query string for FTS5 MATCH operator.
    
    FTS5 treats certain characters as special (: for column specifiers,
    parentheses for grouping, quotes for phrases). We need to escape or
    quote the query to avoid syntax errors.
    
    Strategy: Remove stop words and punctuation, quote terms with special chars.
    
    Args:
        query: Raw query string from user
        
    Returns:
        FTS5-safe query string
    """
    # Remove common question words and punctuation
    query = re.sub(r'\b(where|what|how|is|are|was|were|the|a|an)\b', ' ', query, flags=re.IGNORECASE)
    query = re.sub(r'[?!.,;]', ' ', query)
    
    # Split on whitespace and process each word
    words = query.split()
    quoted_words = []
    
    for word in words:
        word = word.strip()
        if not word:
            continue
            
        # Skip FTS5 operators
        if word.upper() in ('AND', 'OR', 'NOT', 'NEAR'):
            quoted_words.append(word)
            continue
        
        # Quote words with special chars (: - * ( ) ")
        if re.search(r'[:\(\)\*\-"]', word):
            # Escape internal quotes and wrap in quotes
            word = '"' + word.replace('"', '""') + '"'
        
        quoted_words.append(word)
    
    if not quoted_words:
        return '""'  # Empty search
    
    return ' '.join(quoted_words)


def lexical_search(
    db_path: Path,
    query: str,
    top_k: int = 10,
) -> List[EvidenceCandidate]:
    """Search FTS5 index for query and return ranked evidence.
    
    Args:
        db_path: Path to FTS5 SQLite database
        query: Search query string (will be used with MATCH operator)
        top_k: Maximum number of results to return
        
    Returns:
        List of EvidenceCandidate objects, ranked by BM25 score
    """
    db_path = Path(db_path)
    
    if not db_path.exists():
        logger.warning(f"Database does not exist: {db_path}")
        return []
    
    results = []
    
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Sanitize query for FTS5: quote if it contains special chars
        # FTS5 special chars: : ( ) " * - AND OR NOT NEAR
        fts_query = _sanitize_fts_query(query)
        
        # Use FTS5 MATCH operator with BM25 scoring
        # Query all indexed and unindexed columns
        cursor.execute("""
            SELECT 
                chunk_id,
                file_path,
                start_line,
                end_line,
                symbol_path,
                doc_origin,
                language,
                root_ref,
                resolved_commit,
                bm25(chunks_fts) as score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score, file_path, start_line
            LIMIT ?
        """, (fts_query, top_k))
        
        for row in cursor.fetchall():
            evidence = EvidenceCandidate(
                chunk_id=row["chunk_id"],
                file_path=row["file_path"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                symbol_path=row["symbol_path"] if row["symbol_path"] else None,
                doc_origin=row["doc_origin"],
                language=row["language"],
                root_ref=row["root_ref"],
                resolved_commit=row["resolved_commit"],
                score=row["score"],
            )
            results.append(evidence)
        
        conn.close()
        
        logger.info(
            f"Lexical search for '{query}': {len(results)} results "
            f"(top_k={top_k})"
        )
        
    except sqlite3.Error as e:
        logger.error(f"FTS5 search failed: {e}")
        return []
    
    return results
