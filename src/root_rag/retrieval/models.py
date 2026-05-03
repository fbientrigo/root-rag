"""Data models for retrieval module."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def classify_source_type(file_path: str) -> str:
    """Classify evidence source type from repository-relative path.

    Precedence (highest to lowest):
    1) artifact
    2) doc
    3) macro
    4) python
    5) cpp
    """
    normalized = file_path.replace("\\", "/")
    normalized_lower = normalized.lower()
    suffix = Path(normalized_lower).suffix
    filename = Path(normalized_lower).name

    # 1) Artifacts override all other categories.
    if normalized_lower.startswith("data/indexes_") or normalized_lower.startswith("data/processed/"):
        return "artifact"
    if suffix in {".sqlite", ".jsonl"}:
        return "artifact"
    if filename == "manifest.json" or filename.endswith("_manifest.json") or filename == "semantic_manifest.json":
        return "artifact"

    # 2) Documentation.
    if filename.startswith("readme") or suffix == ".md":
        return "doc"

    # 3) FairShip workflow macros.
    if normalized_lower.startswith("macro/") and suffix in {".py", ".c"}:
        return "macro"

    # 4) Python orchestration/scripts outside macro/.
    if suffix == ".py":
        return "python"

    # 5) C/C++ implementation and headers.
    if suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}:
        return "cpp"

    # Conservative fallback for non-source/index-adjacent material.
    return "artifact"


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

    @property
    def source_type(self) -> str:
        return classify_source_type(self.file_path)
