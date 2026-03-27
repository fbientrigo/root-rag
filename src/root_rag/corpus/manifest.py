"""Manifest model for corpus metadata and versioning."""
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Manifest(BaseModel):
    """Corpus manifest with versioning metadata.
    
    This model ensures reproducibility by capturing:
    - What ref was requested (root_ref)
    - What commit was resolved (resolved_commit)
    - Where it's stored locally (local_path)
    - When it was fetched (fetched_at)
    - The tool version that created it (tool_version)
    """

    schema_version: str = Field(default="corpus_manifest_v1")
    repo_url: str = Field(..., description="Source repository URL")
    root_ref: str = Field(..., description="Requested branch/tag/commit")
    resolved_commit: str = Field(..., description="Resolved immutable commit SHA")
    local_path: str = Field(..., description="Absolute or relative path to local corpus")
    fetched_at: str = Field(..., description="ISO8601 timestamp of fetch")
    dirty: bool = Field(default=False, description="True if corpus has uncommitted changes")
    tool_version: str = Field(..., description="Version of root-rag that created this manifest")

    @field_validator("resolved_commit")
    @classmethod
    def validate_commit_sha(cls, v: str) -> str:
        """Ensure resolved_commit looks like a valid git SHA."""
        if len(v) < 7 or len(v) > 40:
            raise ValueError(
                f"resolved_commit must be 7-40 hex characters; got '{v}' (len={len(v)})"
            )
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError(
                f"resolved_commit must be hexadecimal; got '{v}'"
            )
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "schema_version": "corpus_manifest_v1",
                "repo_url": "https://github.com/root-project/root.git",
                "root_ref": "v6-32-00",
                "resolved_commit": "abc123def456abc123def456abc123def456abc1",
                "local_path": "/home/user/.cache/root-rag/root__abc123de/repo",
                "fetched_at": "2026-02-27T10:30:00+00:00",
                "dirty": False,
                "tool_version": "0.0.1",
            }
        }
    )

    def save(self, path: Path) -> None:
        """Write manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Load manifest from JSON file."""
        path = Path(path)
        return cls.model_validate_json(path.read_text())

    def to_dict(self) -> dict:
        """Convert to dictionary (for logging, display)."""
        return self.model_dump()
