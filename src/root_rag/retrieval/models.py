"""Data models for retrieval module."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvidenceCandidate:
    """A single piece of evidence (chunk) returned from retrieval."""
    
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    symbol_path: Optional[str]
    doc_origin: str
    language: str
    root_ref: str
    resolved_commit: str
    score: float
