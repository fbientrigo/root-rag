"""S1 semantic retrieval helpers: local embeddings, FAISS exact search, and hybrid fusion."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

from root_rag.retrieval.models import EvidenceCandidate

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CAMEL_RE = re.compile(r"[A-Z][a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|$)")
SYMBOL_HINT_RE = re.compile(r"(::|->|/|\\|[#<>{}\[\]();]|[A-Za-z_][A-Za-z0-9_]*\()")


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "S1 semantic retrieval requires optional dependency 'numpy'. "
            "Install the project extra for S1 support."
        ) from exc
    return np


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "S1 semantic retrieval requires optional dependency 'faiss-cpu'. "
            "Install the project extra for S1 support."
        ) from exc
    return faiss


class LocalEmbedder(Protocol):
    """Local-only embedding backend contract for S1."""

    model_name: str

    def embedding_dimension(self) -> int:
        """Return embedding vector dimension."""

    def embed(self, texts: Sequence[str]):
        """Return a normalized float32 numpy array of shape [n, dim]."""


@dataclass
class SentenceTransformerLocalEmbedder:
    """Local sentence-transformers embedder for S1 artifacts and query search."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 16
    local_files_only: bool = False
    _model: Optional[object] = None
    _dim: Optional[int] = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "S1 semantic retrieval requires optional dependency 'sentence-transformers'. "
                "Install the project extra for S1 support."
            ) from exc
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            local_files_only=self.local_files_only,
        )
        self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embedding_dimension(self) -> int:
        if self._dim is not None:
            return self._dim
        self._load_model()
        assert self._dim is not None
        return self._dim

    def embed(self, texts: Sequence[str]):
        np = _import_numpy()
        model = self._load_model()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        return normalize_vectors(vectors)


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_row_for_evidence(row: dict) -> dict:
    """Normalize chunk/corpus row variants used across index and benchmark paths."""
    start_line = row.get("start_line")
    end_line = row.get("end_line")
    if start_line is None or end_line is None:
        line_range = row.get("line_range", [1, 1])
        if line_range:
            start_line = int(line_range[0])
            end_line = int(line_range[1]) if len(line_range) > 1 else int(line_range[0])
        else:
            start_line = 1
            end_line = 1

    return {
        "chunk_id": row["chunk_id"],
        "file_path": row.get("file_path", ""),
        "start_line": int(start_line),
        "end_line": int(end_line),
        "symbol_path": row.get("symbol_path"),
        "doc_origin": row.get("doc_origin") or row.get("provenance", "benchmark_artifact"),
        "language": row.get("language") or _infer_language_from_path(row.get("file_path", "")),
        "root_ref": row.get("root_ref", "benchmark_artifact"),
        "resolved_commit": row.get("resolved_commit", "benchmark_artifact"),
        "text": row.get("text") or row.get("content") or "",
        "headers_used": list(row.get("headers_used", [])),
    }


def build_embedding_text(row: dict) -> str:
    """Build deterministic semantic text from available chunk metadata."""
    normalized = normalize_row_for_evidence(row)
    sections = [
        f"path: {normalized['file_path']}",
        f"language: {normalized['language']}",
        f"lines: {normalized['start_line']}-{normalized['end_line']}",
    ]
    if normalized["symbol_path"]:
        sections.append(f"symbol: {normalized['symbol_path']}")
    if normalized["doc_origin"]:
        sections.append(f"origin: {normalized['doc_origin']}")
    headers_used = normalized.get("headers_used", [])
    if headers_used:
        sections.append(f"headers: {', '.join(sorted(str(item) for item in headers_used))}")
    body = normalized["text"].strip()
    if body:
        sections.append("content:")
        sections.append(body)
    return "\n".join(section for section in sections if section.strip())


def _infer_language_from_path(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}:
        return "cpp"
    if suffix in {".py"}:
        return "python"
    return "txt"


def normalize_vectors(vectors):
    """L2-normalize dense vectors for cosine-via-inner-product search."""
    np = _import_numpy()
    if vectors.ndim != 2:
        raise ValueError("expected 2D array of vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = vectors / norms
    return np.asarray(normalized, dtype=np.float32)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


@dataclass
class SemanticIndexManifest:
    """Manifest for S1 semantic artifacts."""

    schema_version: str
    model_name: str
    embedding_dimension: int
    normalization: str
    corpus_source_identifier: str
    corpus_path: str
    corpus_sha256: str
    row_count: int
    faiss_index_type: str
    index_path: str
    records_path: str
    vectors_path: str
    created_at: str
    build_backend: str
    python_version: str
    platform: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "normalization": self.normalization,
            "corpus_source_identifier": self.corpus_source_identifier,
            "corpus_path": self.corpus_path,
            "corpus_sha256": self.corpus_sha256,
            "row_count": self.row_count,
            "faiss_index_type": self.faiss_index_type,
            "index_path": self.index_path,
            "records_path": self.records_path,
            "vectors_path": self.vectors_path,
            "created_at": self.created_at,
            "build_backend": self.build_backend,
            "python_version": self.python_version,
            "platform": self.platform,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SemanticIndexManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def load_semantic_records(records_path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    for line in Path(records_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["chunk_id"]] = row
    return rows


def load_corpus_rows(corpus_path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in Path(corpus_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_semantic_index_artifacts(
    *,
    corpus_rows: Sequence[dict],
    corpus_path: Path,
    output_dir: Path,
    embedder: LocalEmbedder,
    corpus_source_identifier: str,
) -> SemanticIndexManifest:
    """Build deterministic S1 semantic artifacts from chunk/corpus rows."""
    np = _import_numpy()
    faiss = _import_faiss()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted((normalize_row_for_evidence(row) for row in corpus_rows), key=lambda row: row["chunk_id"])
    texts = [build_embedding_text(row) for row in ordered_rows]
    vectors = embedder.embed(texts)
    vectors = normalize_vectors(np.asarray(vectors, dtype=np.float32))
    if len(ordered_rows) != int(vectors.shape[0]):
        raise ValueError("row/vector count mismatch while building semantic index")
    vector_dim = int(vectors.shape[1])

    faiss_index = faiss.IndexFlatIP(vector_dim)
    faiss_index.add(vectors)

    vectors_path = output_dir / "vectors.npy"
    index_path = output_dir / "index.faiss"
    records_path = output_dir / "records.jsonl"
    manifest_path = output_dir / "semantic_manifest.json"

    np.save(vectors_path, vectors)
    faiss.write_index(faiss_index, str(index_path))
    with records_path.open("w", encoding="utf-8") as handle:
        for row, text in zip(ordered_rows, texts):
            materialized = dict(row)
            materialized["embedding_text"] = text
            handle.write(json.dumps(materialized, sort_keys=True) + "\n")

    manifest = SemanticIndexManifest(
        schema_version="1.0.0",
        model_name=embedder.model_name,
        embedding_dimension=vector_dim,
        normalization="l2",
        corpus_source_identifier=corpus_source_identifier,
        corpus_path=str(Path(corpus_path)),
        corpus_sha256=_file_sha256(Path(corpus_path)),
        row_count=len(ordered_rows),
        faiss_index_type="IndexFlatIP",
        index_path=str(index_path),
        records_path=str(records_path),
        vectors_path=str(vectors_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        build_backend="sentence_transformers_local",
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )
    manifest.save(manifest_path)
    return manifest


@dataclass
class SemanticFaissSearcher:
    """Load and search S1 semantic artifacts using exact FAISS inner-product search."""

    manifest_path: Path
    embedder: LocalEmbedder
    _manifest: Optional[SemanticIndexManifest] = None
    _records: Optional[List[dict]] = None
    _records_by_chunk_id: Optional[Dict[str, dict]] = None
    _index: Optional[object] = None

    @property
    def manifest(self) -> SemanticIndexManifest:
        if self._manifest is None:
            self._manifest = SemanticIndexManifest.load(self.manifest_path)
        return self._manifest

    def _load_index(self):
        if self._index is None:
            faiss = _import_faiss()
            self._index = faiss.read_index(self.manifest.index_path)
        return self._index

    @property
    def records(self) -> List[dict]:
        if self._records is None:
            records_by_chunk = load_semantic_records(Path(self.manifest.records_path))
            ordered = sorted(records_by_chunk.values(), key=lambda row: row["chunk_id"])
            self._records = ordered
            self._records_by_chunk_id = {row["chunk_id"]: row for row in ordered}
        return self._records

    @property
    def records_by_chunk_id(self) -> Dict[str, dict]:
        _ = self.records
        assert self._records_by_chunk_id is not None
        return self._records_by_chunk_id

    def search(self, query: str, top_k: int) -> List[EvidenceCandidate]:
        np = _import_numpy()
        if top_k <= 0:
            return []
        query_vector = self.embedder.embed([query])
        query_vector = normalize_vectors(np.asarray(query_vector, dtype=np.float32))
        search_k = min(int(top_k), len(self.records))
        if search_k <= 0:
            return []
        scores, ids = self._load_index().search(query_vector, search_k)
        results: List[EvidenceCandidate] = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0:
                continue
            row = self.records[idx]
            results.append(
                EvidenceCandidate(
                    chunk_id=row["chunk_id"],
                    file_path=row["file_path"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    symbol_path=row.get("symbol_path"),
                    doc_origin=row["doc_origin"],
                    language=row["language"],
                    root_ref=row["root_ref"],
                    resolved_commit=row["resolved_commit"],
                    score=float(score),
                )
            )
        return results


def is_symbol_like_query(query: str) -> bool:
    """Conservative symbol/query guardrail heuristic for S1 fusion."""
    stripped = query.strip()
    if not stripped:
        return False
    tokens = TOKEN_RE.findall(stripped)
    if not tokens:
        return True
    if SYMBOL_HINT_RE.search(stripped):
        return True
    if any(token.count("_") >= 1 or len(token) >= 18 for token in tokens):
        return True
    if any(CAMEL_RE.search(token) for token in tokens):
        return True
    punctuation = sum(1 for ch in stripped if not ch.isalnum() and not ch.isspace())
    return (punctuation / max(len(stripped), 1)) >= 0.15


def fuse_ranked_results(
    *,
    lexical_results: Sequence[EvidenceCandidate],
    semantic_results: Sequence[EvidenceCandidate],
    top_k: int,
    rrf_k: int = 60,
    lexical_weight: float = 0.45,
    semantic_weight: float = 0.55,
    symbol_safe: bool = False,
    lexical_pin_count: int = 3,
) -> List[EvidenceCandidate]:
    """Fuse lexical and semantic ranked lists using deterministic weighted RRF."""
    if top_k <= 0:
        return []

    by_chunk: Dict[str, EvidenceCandidate] = {}
    fused_scores: Dict[str, float] = {}

    for rank, row in enumerate(lexical_results, start=1):
        by_chunk.setdefault(row.chunk_id, row)
        fused_scores[row.chunk_id] = fused_scores.get(row.chunk_id, 0.0) + lexical_weight / (rrf_k + rank)

    for rank, row in enumerate(semantic_results, start=1):
        by_chunk.setdefault(row.chunk_id, row)
        fused_scores[row.chunk_id] = fused_scores.get(row.chunk_id, 0.0) + semantic_weight / (rrf_k + rank)

    ordered_chunk_ids = [
        chunk_id
        for chunk_id, _score in sorted(
            fused_scores.items(),
            key=lambda item: (-item[1], by_chunk[item[0]].file_path, by_chunk[item[0]].start_line, item[0]),
        )
    ]

    final_chunk_ids: List[str] = []
    if symbol_safe:
        for row in lexical_results[:lexical_pin_count]:
            if row.chunk_id not in final_chunk_ids:
                final_chunk_ids.append(row.chunk_id)

    for chunk_id in ordered_chunk_ids:
        if chunk_id not in final_chunk_ids:
            final_chunk_ids.append(chunk_id)
        if len(final_chunk_ids) >= top_k:
            break

    return [by_chunk[chunk_id] for chunk_id in final_chunk_ids[:top_k]]
