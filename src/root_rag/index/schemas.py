"""Chunk and IndexManifest schemas with validation."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Chunk(BaseModel):
    """Canonical chunk schema for indexed ROOT evidence.
    
    All chunks MUST comply with invariants in docs/spec/index_schema.md:
    - line ranges are 1-indexed and inclusive
    - file_path is repo-relative with POSIX (/) separators
    - content matches exact lines from file [start_line, end_line]
    - root_ref and resolved_commit must match corpus version
    - chunk_id is deterministic and stable
    """

    # === Required fields ===
    chunk_id: str = Field(..., description="Stable, deterministic chunk identifier")
    root_ref: str = Field(..., description="User-requested ROOT reference (tag/branch/commit)")
    resolved_commit: str = Field(..., description="Immutable commit SHA")
    file_path: str = Field(..., description="Repository-relative path (POSIX, normalized)")
    language: str = Field(..., description="Language identifier: cpp, c, h, txt, etc.")
    start_line: int = Field(..., ge=1, description="Inclusive start line (1-indexed)")
    end_line: int = Field(..., ge=1, description="Inclusive end line (1-indexed)")
    content: str = Field(..., description="Exact text slice from [start_line, end_line]")
    doc_origin: str = Field(
        ...,
        description="Origin category: source_header, source_impl, doxygen_comment, etc.",
    )
    index_schema_version: str = Field(default="1.0.0", description="Chunk schema version")

    # === Strongly recommended (MVP includes these) ===
    symbol_path: Optional[str] = Field(default=None, description="Symbol path if known, e.g., TTree::Draw")
    has_doxygen: bool = Field(default=False, description="Whether chunk has Doxygen markers")

    @field_validator("end_line")
    @classmethod
    def validate_end_line_vs_start(cls, v: int, info) -> int:
        """Ensure end_line >= start_line."""
        if "start_line" in info.data and v < info.data["start_line"]:
            raise ValueError(f"end_line ({v}) must be >= start_line ({info.data['start_line']})")
        return v

    @field_validator("content")
    @classmethod
    def validate_content_nonempty(cls, v: str) -> str:
        """Ensure content is not empty and within reasonable bounds."""
        if not v or not v.strip():
            raise ValueError("content must not be empty or whitespace-only")
        if len(v) > 1_000_000:  # 1MB max
            raise ValueError("content exceeds maximum length (1MB)")
        return v

    @field_validator("file_path")
    @classmethod
    def validate_file_path_relative_posix(cls, v: str) -> str:
        """Ensure file_path is relative and uses POSIX separators."""
        if v.startswith("/") or v.startswith("\\"):
            raise ValueError(f"file_path must be relative, got: {v}")
        if "\\" in v:
            raise ValueError(f"file_path must use POSIX separators (/), got: {v}")
        if "." == v or v.startswith("../"):
            raise ValueError(f"file_path must not escape repo root, got: {v}")
        return v

    @field_validator("doc_origin")
    @classmethod
    def validate_doc_origin(cls, v: str) -> str:
        """Ensure doc_origin is a known category."""
        valid_origins = {
            "source_header",
            "source_impl",
            "doxygen_comment",
            "reference_doc",
            "tutorial_doc",
        }
        if v not in valid_origins:
            raise ValueError(
                f"doc_origin must be one of {valid_origins}, got: {v}"
            )
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Ensure language is lowercase identifier."""
        if not v or not v.islower():
            raise ValueError(f"language must be lowercase identifier, got: {v}")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": "a8f8b9abc",
                "root_ref": "v6-32-00",
                "resolved_commit": "0123456789abcdef0123456789abcdef01234567",
                "file_path": "tree/tree/inc/TTree.h",
                "language": "cpp",
                "start_line": 210,
                "end_line": 245,
                "content": "virtual Long64_t Draw(...);",
                "doc_origin": "source_header",
                "index_schema_version": "1.0.0",
                "symbol_path": "TTree::Draw",
                "has_doxygen": True,
            }
        }
    )

    @classmethod
    def compute_chunk_id(
        cls,
        root_ref: str,
        resolved_commit: str,
        file_path: str,
        start_line: int,
        end_line: int,
    ) -> str:
        """Compute deterministic chunk ID from provenance tuple.
        
        Uses SHA256 of a canonical string representation, takes first 12 chars.
        This ensures identical inputs always produce identical IDs.
        """
        provenance = f"{root_ref}:{resolved_commit}:{file_path}:{start_line}:{end_line}"
        digest = hashlib.sha256(provenance.encode()).hexdigest()
        return digest[:12]

    def to_jsonl_line(self) -> str:
        """Serialize chunk to JSONL line (JSON + newline)."""
        return self.model_dump_json() + "\n"

    @classmethod
    def from_file_slice(
        cls,
        file_path: str,
        start_line: int,
        end_line: int,
        content: str,
        root_ref: str,
        resolved_commit: str,
        language: str,
        doc_origin: str,
        symbol_path: Optional[str] = None,
        has_doxygen: bool = False,
    ) -> "Chunk":
        """Factory method to construct chunk from file slice metadata."""
        chunk_id = cls.compute_chunk_id(root_ref, resolved_commit, file_path, start_line, end_line)
        return cls(
            chunk_id=chunk_id,
            root_ref=root_ref,
            resolved_commit=resolved_commit,
            file_path=file_path,
            language=language,
            start_line=start_line,
            end_line=end_line,
            content=content,
            doc_origin=doc_origin,
            symbol_path=symbol_path,
            has_doxygen=has_doxygen,
        )


class IndexManifest(BaseModel):
    """Index metadata schema following versioning and determinism contracts.
    
    Records the provenance of an FTS5 lexical index and its configuration.
    Enables reproducible index rebuilds and version integrity validation.
    """

    # === Index identification ===
    index_id: str = Field(..., description="Unique index identifier (deterministic from corpus + timestamp)")
    corpus_id: str = Field(..., description="Deterministic corpus identifier (root_ref + commit SHA[:12])")

    # === Corpus and versioning ===
    root_ref: str = Field(..., description="User-requested ROOT reference (tag/branch/commit)")
    resolved_commit: str = Field(..., description="Immutable commit SHA resolved from root_ref")
    corpus_url: str = Field(..., description="Repository URL")

    # === Artifact paths ===
    chunks_path: str = Field(..., description="Path to chunks.jsonl file")
    fts_db_path: str = Field(..., description="Path to FTS5 SQLite database")

    # === Versions and configuration ===
    schema_version: str = Field(default="1.0.0", description="Index manifest schema version")
    index_schema_version: str = Field(default="1.0.0", description="Chunk schema version (from chunks)")

    # === Statistics ===
    chunk_count: int = Field(..., ge=0, description="Total chunks in index")
    file_count: int = Field(..., ge=0, description="Total unique source files indexed")
    retrieval_modes: list = Field(default_factory=lambda: ["lexical"], description="Available retrieval modes")

    # === Metadata ===
    created_at: str = Field(..., description="Index creation timestamp (ISO8601)")
    tool_version: str = Field(default="0.1.0", description="root-rag tool version")

    @staticmethod
    def compute_corpus_id(root_ref: str, resolved_commit: str) -> str:
        """Compute deterministic corpus ID from reference and commit.

        Format: {root_ref}__{commit[:12]}
        Example: v6-32-00__0123456789ab
        """
        commit_short = resolved_commit[:12]
        return f"{root_ref}__{commit_short}"

    @staticmethod
    def compute_index_id(corpus_id: str, created_at: str) -> str:
        """Compute unique index ID from corpus and timestamp.

        Format: {corpus_id}__<timestamp_compact>
        Example: v6-32-00__0123456789ab__20260227T235900Z
        """
        # Extract YYYYMMDDTHHMMSSZ format from ISO8601
        # created_at example: "2026-02-27T23:59:00Z"
        timestamp_compact = created_at.replace("-", "").replace(":", "").replace(".", "")
        return f"{corpus_id}__{timestamp_compact}"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return self.model_dump()

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "IndexManifest":
        """Load manifest from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "index_id": "v6-32-00__0123456789ab__20260227T235900Z",
                "corpus_id": "v6-32-00__0123456789ab",
                "root_ref": "v6-32-00",
                "resolved_commit": "0123456789abcdefabcdefabcdefabcdefabcdef",
                "corpus_url": "https://github.com/root-project/root.git",
                "chunks_path": "data/processed/chunks/v6-32-00__0123456789ab/chunks.jsonl",
                "fts_db_path": "data/indexes/v6-32-00__0123456789ab__20260227T235900Z/fts.sqlite",
                "schema_version": "1.0.0",
                "index_schema_version": "1.0.0",
                "chunk_count": 42,
                "file_count": 5,
                "retrieval_modes": ["lexical"],
                "created_at": "2026-02-27T23:59:00Z",
                "tool_version": "0.1.0",
            }
        }
    )
